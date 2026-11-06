"""Признаки для одного инструмента за один день.

Главное правило: признак в момент t считается ТОЛЬКО по данным с ts <= t.
Поэтому здесь:
  * только окна "назад" (скользящая сумма через cumsum: S[i] - S[i-k]);
  * только forward-fill (протягивание последнего известного значения вперёд);
  * никаких нормировок по всему дню, shift(-k), bfill, центрированных окон.

Чтобы окна в начале дня не были пустыми, берём ещё и ПРЕДЫДУЩИЙ день
(организаторы его дают для "прогрева"). Считаем всё на склеенных двух днях,
а на выход отдаём только точки текущего дня.

Все расчёты идут на сетке 100 мс (как bbo и trades), а потом берём каждую
5-ю точку — это и есть сетка прогноза 500 мс.
"""
import datetime as dt

import numpy as np

from common import (BBO_MS, GRID_MS, TRADE_MS, align_to_grid, grid_points,
                    quote_grid, read_day)

# окна в барах по 100 мс: 0.5 с, 2 с, 10 с, 60 с
W_SHORT, W_MID, W_LONG, W_VOL = 5, 20, 100, 600
WINDOWS = [W_SHORT, W_MID, W_LONG]

STEP = GRID_MS // BBO_MS          # 5 баров по 100 мс в одном шаге 500 мс
BARS_PER_DAY = 86_400_000 // BBO_MS

# "ведущий" инструмент для меж-рыночных признаков; для самого BTC берём ETH
LEADER = "BTCUSDT_PERP"
LEADER_FOR_LEADER = "ETHUSDT_PERP"

EPS = 1e-12


# ------------------------------------------------------------------ имена
def feature_names():
    names = ["imb1", "micro_bps", "spread_bps", "depth_imb_5", "depth_imb_all"]
    names += [f"ofi_{w}" for w in WINDOWS]
    names += [f"tflow_{w}" for w in WINDOWS]
    names += [f"ret_{w}" for w in WINDOWS]
    names += [f"lead_ret_{w}" for w in (W_SHORT, W_MID)]
    names += ["vol_bps"]
    # сигнал × волатильность: в бурные периоды прогноз должен быть крупнее
    names += ["imb1_x_vol", "ofi_5_x_vol", "tflow_5_x_vol", "depth_imb_5_x_vol"]
    return names


FEATURE_NAMES = feature_names()


# ------------------------------------------------------- маленькие помощники
def prev_date(date):
    return (dt.date.fromisoformat(date) - dt.timedelta(days=1)).isoformat()


def ffill(a):
    """Протянуть последнее известное (не NaN) значение вперёд. Строго причинно."""
    ok = np.isfinite(a)
    idx = np.where(ok, np.arange(len(a)), -1)
    np.maximum.accumulate(idx, out=idx)
    return np.where(idx >= 0, a[np.maximum(idx, 0)], np.nan)


def rsum(x, k):
    """Сумма за последние k баров, включая текущий. x не должен содержать NaN.

    cumsum считается последовательно слева направо, поэтому значение в точке i
    зависит только от x[0..i] — обрезка данных после i его не меняет.
    """
    c = np.cumsum(x)
    out = c.copy()
    out[k:] = c[k:] - c[:-k]
    return out


def lag(x, k):
    """x[i-k] (значение k баров назад); в начале — NaN."""
    out = np.full_like(x, np.nan)
    out[k:] = x[:-k]
    return out


# ------------------------------------------------------------- загрузка
# common.py (файл организаторов) не меняем. Его функции падают на пустом
# parquet-файле, поэтому оборачиваем вызовы здесь: пустой файл = данных нет.
def read_nonempty(root, table, date, inst=None, columns=None):
    t = read_day(root, table, date, inst, columns=columns)
    return t if (t is not None and t.num_rows > 0) else None


def safe_quote_grid(root, inst, date):
    if read_nonempty(root, "book", date, inst, columns=["ts"]) is None:
        return None
    try:
        return quote_grid(root, inst, date)
    except (IndexError, ValueError):
        return None


def quotes_two_days(root, inst, date, cache=None):
    """Котировки на сетке 100 мс за (вчера + сегодня). NaN, где данных нет."""
    key = ("q", inst, date)
    if cache is not None and key in cache:
        return cache[key]
    parts = []
    for d in (prev_date(date), date):
        q = safe_quote_grid(root, inst, d)
        if q is None:
            g = grid_points(d, BBO_MS)
            nan = np.full(len(g), np.nan)
            q = (g, nan, nan, nan, nan)
        parts.append(q)
    ts, bid, ask, bq, aq = (np.concatenate([p[k] for p in parts]) for k in range(5))
    # некорректные котировки выкидываем и протягиваем предыдущие
    bad = ~((bid > 0) & (ask > 0) & (ask >= bid))
    for a in (bid, ask, bq, aq):
        a[bad] = np.nan
    bid, ask, bq, aq = ffill(bid), ffill(ask), ffill(bq), ffill(aq)
    out = (ts, bid, ask, bq, aq)
    if cache is not None:
        cache[key] = out
    return out


def trades_two_days(root, inst, date):
    """Объёмы сделок покупателей/продавцов на сетке 100 мс; пустые бары = 0."""
    cols = ["buy_qty", "sell_qty"]
    res = {c: [] for c in cols}
    for d in (prev_date(date), date):
        g = grid_points(d, TRADE_MS)
        t = read_nonempty(root, "trades", d, inst, columns=["ts"] + cols)
        idx = align_to_grid(t, g)
        hit = idx >= 0
        for c in cols:
            arr = np.zeros(len(g))
            if t is not None:
                v = t[c].to_numpy().astype("float64")
                arr[hit] = v[idx[hit]]
            res[c].append(np.nan_to_num(arr))
    return np.concatenate(res["buy_qty"]), np.concatenate(res["sell_qty"])


def book_depth(root, inst, date):
    """Объёмы по корзинам стакана на сетке 500 мс текущего дня (NaN, где строки нет)."""
    g = grid_points(date, GRID_MS)
    bid_cols = [f"bid_v{i:02d}" for i in range(1, 12)]
    ask_cols = [f"ask_v{i:02d}" for i in range(1, 12)]
    t = read_nonempty(root, "book", date, inst, columns=["ts"] + bid_cols + ask_cols)
    B = np.full((len(g), 11), np.nan)
    A = np.full((len(g), 11), np.nan)
    if t is not None:
        idx = align_to_grid(t, g)
        hit = idx >= 0
        for k in range(11):
            B[hit, k] = t[bid_cols[k]].to_numpy().astype("float64")[idx[hit]]
            A[hit, k] = t[ask_cols[k]].to_numpy().astype("float64")[idx[hit]]
    return B, A


# ------------------------------------------------------------- признаки
def compute_features(root, inst, date, cache=None):
    """Матрица признаков (172_800 × len(FEATURE_NAMES)) для сетки 500 мс дня `date`.

    NaN заменяются нулём в самом конце (так же это делает и обучение, и прогноз).
    """
    ts, bid, ask, bq, aq = quotes_two_days(root, inst, date, cache)
    mid = (bid + ask) / 2
    logmid = np.log(mid)

    # --- верх стакана
    imb1 = (bq - aq) / (bq + aq + EPS)
    micro = (bid * aq + ask * bq) / (bq + aq + EPS)
    micro_bps = (micro - mid) / mid * 1e4
    spread_bps = (ask - bid) / mid * 1e4

    # --- Order Flow Imbalance (Cont, Kukanov, Stoikov 2014) на шаге 100 мс
    pb, pa_, pbq, paq = lag(bid, 1), lag(ask, 1), lag(bq, 1), lag(aq, 1)
    e = (np.where(bid >= pb, bq, 0) - np.where(bid <= pb, pbq, 0)
         - np.where(ask <= pa_, aq, 0) + np.where(ask >= pa_, paq, 0))
    e = np.nan_to_num(e)
    depth = np.nan_to_num((bq + aq) / 2)

    # --- поток агрессивных сделок
    buy, sell = trades_two_days(root, inst, date)

    # --- доходности mid (в б.п.) за 100 мс; NaN -> 0 для накопления
    r1 = np.nan_to_num((logmid - lag(logmid, 1)) * 1e4)

    # --- волатильность за последние 60 с (б.п.)
    vol = np.sqrt(np.maximum(rsum(r1 * r1, W_VOL), 0))

    # --- ведущий инструмент (BTC; для BTC — ETH)
    leader = LEADER_FOR_LEADER if inst == LEADER else LEADER
    _, lb, la, _, _ = quotes_two_days(root, leader, date, cache)
    lmid = np.log((lb + la) / 2)

    cols = {
        "imb1": imb1,
        "micro_bps": micro_bps,
        "spread_bps": spread_bps,
        "vol_bps": vol,
    }
    for w in WINDOWS:
        cols[f"ofi_{w}"] = rsum(e, w) / (rsum(depth, w) / w + EPS)
        b, s = rsum(buy, w), rsum(sell, w)
        cols[f"tflow_{w}"] = (b - s) / (b + s + EPS)
        cols[f"ret_{w}"] = (logmid - lag(logmid, w)) * 1e4
    for w in (W_SHORT, W_MID):
        cols[f"lead_ret_{w}"] = (lmid - lag(lmid, w)) * 1e4

    # --- берём только точки текущего дня на сетке 500 мс
    sel = BARS_PER_DAY + np.arange(0, BARS_PER_DAY, STEP)
    out = {k: v[sel] for k, v in cols.items()}

    # --- глубина по корзинам (уже на сетке 500 мс)
    B, A = book_depth(root, inst, date)
    b5, a5 = B[:, :5].sum(1), A[:, :5].sum(1)
    out["depth_imb_5"] = (b5 - a5) / (b5 + a5 + EPS)
    ba, aa = B.sum(1), A.sum(1)
    out["depth_imb_all"] = (ba - aa) / (ba + aa + EPS)

    # --- взаимодействия с волатильностью
    for s in ("imb1", "ofi_5", "tflow_5", "depth_imb_5"):
        out[f"{s}_x_vol"] = out[s] * out["vol_bps"]

    X = np.column_stack([out[n] for n in FEATURE_NAMES])
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
