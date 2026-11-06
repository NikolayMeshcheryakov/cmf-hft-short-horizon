"""Блок признаков v2 — дополнение к 21 признаку из features.py (тот файл не меняется,
чтобы старый кэш и модель lgbm_v1 остались рабочими).

Правила те же: в момент t только данные с ts <= t; окна только назад; протягивание вперёд.
Статистики «по предыдущему дню» (шаг цены, β, разброс доходностей) считаются целиком по
вчерашнему дню — он полностью в прошлом, и проверка с обрезкой текущего дня их не меняет.

Группы (окна в барах по 100 мс):
  1. Динамика верха стакана: d_imb1, q_depl_1/5, mid_age_s, bid_dir, ask_dir, side_last,
     imb1_mean_20/50, quote_age_s, quote_upd_20
  2. Сделки: timp_5/20/80 (сила удара), age_buy_s, age_sell_s, big_buy, big_sell,
     ntrades_5, ntflow_5
  3. Спред в фактических шагах цены: spread_steps, wide
  4. Спот (USDT): basis_bps, basis_dev, dbasis_5/20, spot_ret_5/20, spot_imb1
  5. Межрыночные: missed_btc/eth/sol/pepe («пропущенное движение» соседа), mkt_ex_self
  6. Ближняя глубина (корзины book, раз в 500 мс): depth_flow_1/4, depth_imb_12, depth_shape
  7. Волатильность: vol_ratio (2 с к 60 с)
"""
import numpy as np

from common import TARGET_INSTRUMENTS, TRADE_MS, align_to_grid, grid_points
from features import (BARS_PER_DAY, EPS, STEP, book_depth, ffill, lag, prev_date,
                      quotes_two_days, read_nonempty, rsum)

SHORT = {"BTCUSDT_PERP": "btc", "ETHUSDT_PERP": "eth", "SOLUSDT_PERP": "sol", "1000PEPEUSDT_PERP": "pepe"}
# спот к USDT и множитель цены: 1000PEPE-перп = 1000 × PEPE-спот
SPOT = {"BTCUSDT_PERP": ("BTCUSDT_SPOT", 1.0), "ETHUSDT_PERP": ("ETHUSDT_SPOT", 1.0),
        "SOLUSDT_PERP": ("SOLUSDT_SPOT", 1.0), "1000PEPEUSDT_PERP": ("PEPEUSDT_SPOT", 1000.0)}

AGE_CAP_BARS = 50          # возраст обрезаем на 5 с
TRADE_AGE_CAP_S = 10.0


def v2_names():
    n = ["d_imb1", "q_depl_1", "q_depl_5", "mid_age_s", "bid_dir", "ask_dir", "side_last",
         "imb1_mean_20", "imb1_mean_50", "quote_age_s", "quote_upd_20"]
    n += ["timp_5", "timp_20", "timp_80", "age_buy_s", "age_sell_s", "big_buy", "big_sell",
          "ntrades_5", "ntflow_5"]
    n += ["spread_steps", "wide"]
    n += ["basis_bps", "basis_dev", "dbasis_5", "dbasis_20", "spot_ret_5", "spot_ret_20", "spot_imb1"]
    n += [f"missed_{SHORT[i]}" for i in TARGET_INSTRUMENTS] + ["mkt_ex_self"]
    n += ["depth_flow_1", "depth_flow_4", "depth_imb_12", "depth_shape"]
    n += ["vol_ratio"]
    return n


V2_NAMES = v2_names()


# ------------------------------------------------------------------ помощники
def nan0(x):
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def bars_since(event):
    """Сколько баров прошло с последнего True (0 — в этом баре). До первого события — NaN."""
    n = len(event)
    idx = np.where(event, np.arange(n), -1)
    np.maximum.accumulate(idx, out=idx)
    out = (np.arange(n) - idx).astype(float)
    out[idx < 0] = np.nan
    return out


def last_index(event):
    n = len(event)
    idx = np.where(event, np.arange(n), -1)
    np.maximum.accumulate(idx, out=idx)
    return idx


def last_nonzero(x):
    """Последнее ненулевое значение x (протянуто вперёд); до первого — 0."""
    y = np.where(x != 0, x, np.nan)
    return nan0(ffill(y))


def rmax(x, k):
    """Максимум за последние k баров (включая текущий)."""
    out = x.copy()
    for j in range(1, k):
        out[j:] = np.maximum(out[j:], x[:-j])
    return out


def rmean_valid(x, k, min_frac=0.5):
    """Среднее за k баров по конечным значениям; NaN, если их меньше min_frac·k."""
    ok = np.isfinite(x)
    s, c = rsum(np.where(ok, x, 0.0), k), rsum(ok.astype(float), k)
    return np.where(c >= min_frac * k, s / np.maximum(c, 1), np.nan)


def logret(logp, k):
    return (logp - lag(logp, k)) * 1e4


def trades_full_two_days(root, inst, date):
    """Все колонки сделок на сетке 100 мс за (вчера + сегодня); пустые бары = 0 / NaN."""
    num = ["buy_qty", "sell_qty", "buy_n", "sell_n", "buy_max_qty", "sell_max_qty"]
    px = ["buy_vwap", "sell_vwap", "buy_last_ms", "sell_last_ms"]
    out = {c: [] for c in num + px + ["ts"]}
    for d in (prev_date(date), date):
        g = grid_points(d, TRADE_MS)
        t = read_nonempty(root, "trades", d, inst, columns=["ts"] + num + px)
        idx = align_to_grid(t, g)
        hit = idx >= 0
        for c in num + px:
            arr = np.zeros(len(g)) if c in num else np.full(len(g), np.nan)
            if t is not None and c in t.column_names:
                v = t[c].to_numpy().astype("float64")          # null → NaN
                arr[hit] = v[idx[hit]]
            if c in num:
                arr = nan0(arr)
            out[c].append(arr)
        out["ts"].append(g.astype("float64"))
    return {c: np.concatenate(v) for c, v in out.items()}


def prev_day_part(x):
    return x[:BARS_PER_DAY]


# ------------------------------------------------------------------ признаки
def compute_v2(root, inst, date, cache=None):
    """Матрица (172_800 × len(V2_NAMES)) на сетке 500 мс дня `date`. NaN → 0 в самом конце."""
    cache = {} if cache is None else cache
    ts, bid, ask, bq, aq = quotes_two_days(root, inst, date, cache)
    mid = (bid + ask) / 2
    logmid = np.log(mid)
    imb = (bq - aq) / (bq + aq + EPS)
    c = {}

    # ---- 1. динамика верха стакана
    pb, pa, pbq, paq = lag(bid, 1), lag(ask, 1), lag(bq, 1), lag(aq, 1)
    c["d_imb1"] = imb - lag(imb, 1)
    with np.errstate(invalid="ignore"):
        da = np.where(ask == pa, np.maximum(paq - aq, 0) / (paq + EPS), 0.0)   # тает очередь ask
        db = np.where(bid == pb, np.maximum(pbq - bq, 0) / (pbq + EPS), 0.0)   # тает очередь bid
    qd = nan0(da - db)
    c["q_depl_1"] = qd
    c["q_depl_5"] = rsum(qd, 5)
    with np.errstate(invalid="ignore"):
        mid_chg = np.r_[False, mid[1:] != mid[:-1]] & np.isfinite(mid) & np.isfinite(lag(mid, 1))
    c["mid_age_s"] = np.minimum(np.nan_to_num(bars_since(mid_chg), nan=AGE_CAP_BARS), AGE_CAP_BARS) / 10
    sb, sa = nan0(np.sign(bid - pb)), nan0(np.sign(ask - pa))
    c["bid_dir"], c["ask_dir"] = last_nonzero(sb), last_nonzero(sa)
    ib, ia = last_index(sb != 0), last_index(sa != 0)
    c["side_last"] = np.sign(ib - ia).astype(float)          # +1: последним двигался bid, −1: ask
    imb0 = nan0(imb)
    c["imb1_mean_20"] = rsum(imb0, 20) / 20
    c["imb1_mean_50"] = rsum(imb0, 50) / 50
    with np.errstate(invalid="ignore"):
        q_chg = np.r_[False, (bid[1:] != bid[:-1]) | (ask[1:] != ask[:-1]) |
                      (bq[1:] != bq[:-1]) | (aq[1:] != aq[:-1])] & np.isfinite(mid)
    c["quote_age_s"] = np.minimum(np.nan_to_num(bars_since(q_chg), nan=AGE_CAP_BARS), AGE_CAP_BARS) / 10
    c["quote_upd_20"] = rsum(q_chg.astype(float), 20)

    # ---- 2. сделки
    t = trades_full_two_days(root, inst, date)
    midp = lag(mid, 1)                                       # mid на начало бара сделок
    with np.errstate(invalid="ignore", divide="ignore"):
        okp = np.isfinite(midp) & (midp > 0)
        bimp = np.where((t["buy_qty"] > 0) & np.isfinite(t["buy_vwap"]) & okp,
                        (t["buy_vwap"] - midp) / midp * 1e4, 0.0)
        simp = np.where((t["sell_qty"] > 0) & np.isfinite(t["sell_vwap"]) & okp,
                        (midp - t["sell_vwap"]) / midp * 1e4, 0.0)
    num = nan0(t["buy_qty"] * bimp - t["sell_qty"] * simp)
    vol_q = t["buy_qty"] + t["sell_qty"]
    for w in (5, 20, 80):
        c[f"timp_{w}"] = rsum(num, w) / (rsum(vol_q, w) + EPS)
    for side in ("buy", "sell"):
        evt = np.where(t[f"{side}_n"] > 0, t["ts"] - nan0(t[f"{side}_last_ms"]), np.nan)
        last_evt = ffill(evt)
        age = (ts - last_evt) / 1000
        c[f"age_{side}_s"] = np.minimum(np.nan_to_num(age, nan=TRADE_AGE_CAP_S), TRADE_AGE_CAP_S)
    n_all = t["buy_n"] + t["sell_n"]
    n600 = rsum(n_all, 600)
    avg_size = np.where(n600 >= 10, rsum(vol_q, 600) / np.maximum(n600, 1), np.nan)
    for side in ("buy", "sell"):
        mx = rmax(nan0(t[f"{side}_max_qty"]), 5)
        c[f"big_{side}"] = np.log1p(mx / avg_size)
    n5b, n5s = rsum(t["buy_n"], 5), rsum(t["sell_n"], 5)
    c["ntrades_5"] = np.log1p(n5b + n5s)
    c["ntflow_5"] = (n5b - n5s) / (n5b + n5s + EPS)

    # ---- 3. спред в фактических шагах цены (шаг — медиана спреда за вчера, не тик из меты)
    spread = ask - bid
    ps = prev_day_part(spread)
    ps = ps[np.isfinite(ps) & (ps > 0)]
    if len(ps) > 1000:
        step = float(np.median(ps))
    else:
        # вчерашнего дня нет — запасной вариант: наименьший положительный спред, виденный
        # до момента t (накопленный минимум, причинно). Грубее медианы, но без заглядывания вперёд.
        pos = np.where(np.isfinite(spread) & (spread > 0), spread, np.inf)
        step = np.minimum.accumulate(pos)
        step = np.where(np.isfinite(step), step, np.nan)
    c["spread_steps"] = spread / step
    c["wide"] = (c["spread_steps"] > 1.5).astype(float)

    # ---- 4. спот
    sname, mult = SPOT[inst]
    _, sbid, sask, sbq, saq = quotes_two_days(root, sname, date, cache)
    with np.errstate(invalid="ignore", divide="ignore"):
        slog = np.log((sbid + sask) / 2 * mult)
    basis = (logmid - slog) * 1e4
    c["basis_bps"] = basis
    c["basis_dev"] = basis - rmean_valid(basis, 600)
    c["dbasis_5"] = basis - lag(basis, 5)
    c["dbasis_20"] = basis - lag(basis, 20)
    c["spot_ret_5"] = logret(slog, 5)
    c["spot_ret_20"] = logret(slog, 20)
    c["spot_imb1"] = (sbq - saq) / (sbq + saq + EPS)

    # ---- 5. межрыночные: доходность соседа минус β × своя (β — по вчерашнему дню)
    r_self = logret(logmid, 5)
    xs = prev_day_part(r_self)[::5]
    norm_sum = np.zeros_like(r_self)
    norm_cnt = np.zeros_like(r_self)
    for j in TARGET_INSTRUMENTS:
        name = f"missed_{SHORT[j]}"
        if j == inst:
            c[name] = np.zeros_like(r_self)
            continue
        _, jb, ja, _, _ = quotes_two_days(root, j, date, cache)
        r_j = logret(np.log((jb + ja) / 2), 5)
        ys = prev_day_part(r_j)[::5]
        m = np.isfinite(xs) & np.isfinite(ys)
        if m.sum() > 1000 and np.var(xs[m]) > 0:
            beta = float(np.cov(xs[m], ys[m])[0, 1] / np.var(xs[m], ddof=1))
            sd = float(np.std(ys[m]))
        else:
            beta, sd = np.nan, np.nan
        c[name] = r_j - beta * r_self
        if np.isfinite(sd) and sd > 0:
            z = r_j / sd
            ok = np.isfinite(z)
            norm_sum += np.where(ok, z, 0.0)
            norm_cnt += ok
    c["mkt_ex_self"] = np.where(norm_cnt > 0, norm_sum / np.maximum(norm_cnt, 1), np.nan)

    # ---- 7. волатильность: короткая к минутной
    r1 = nan0((logmid - lag(logmid, 1)) * 1e4)
    v20 = np.sqrt(rsum(r1 * r1, 20) / 20)
    v600 = np.sqrt(rsum(r1 * r1, 600) / 600)
    c["vol_ratio"] = np.where(v600 > 0, v20 / np.maximum(v600, EPS), np.nan)

    # ---- точки текущего дня на сетке 500 мс
    sel = BARS_PER_DAY + np.arange(0, BARS_PER_DAY, STEP)
    out = {k: v[sel] for k, v in c.items()}

    # ---- 6. ближняя глубина (book раз в 500 мс; берём и вчера для прогрева окон)
    BA = [book_depth(root, inst, d) for d in (prev_date(date), date)]
    B, A = np.vstack([x[0] for x in BA]), np.vstack([x[1] for x in BA])
    sel2 = np.arange(0, 2 * BARS_PER_DAY, STEP)              # 500-мс точки за два дня
    mid5 = mid[sel2]
    nb, na = B[:, :3].sum(1), A[:, :3].sum(1)                # корзины 0–1.5 б.п. от mid
    with np.errstate(invalid="ignore"):
        same_mid = mid5 == lag(mid5, 1)                      # корзины считаются от mid: сравнимы,
    flow = nan0(np.where(same_mid, (nb - lag(nb, 1)) - (na - lag(na, 1)), 0.0))   # только если mid стоял
    norm = rmean_valid(nb + na, 20)
    half = slice(BARS_PER_DAY // STEP, None)
    out["depth_flow_1"] = (flow / (norm + EPS))[half]
    out["depth_flow_4"] = (rsum(flow, 4) / (norm + EPS))[half]
    b12, a12 = B[:, :2].sum(1), A[:, :2].sum(1)
    bf, af = B[:, 4:].sum(1), A[:, 4:].sum(1)
    imb12 = (b12 - a12) / (b12 + a12 + EPS)
    out["depth_imb_12"] = imb12[half]
    out["depth_shape"] = (imb12 - (bf - af) / (bf + af + EPS))[half]

    X = np.column_stack([out[n] for n in V2_NAMES])
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
