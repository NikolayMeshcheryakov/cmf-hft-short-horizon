"""Блок признаков v3 — рынки того же актива в других стейблкоинах (дополнение к base и v2).

Подсказка организаторов (README): курсы из `stable` нужны, чтобы привести цены к общему
знаменателю (BTCUSDC_SPOT · mid(USDCUSDT)); отношение самих цен BTC — это и есть сигнал.

Главная величина — РАЗРЫВ между приведённым альтернативным рынком и опорным:
    gap_S = (ln(mid_alt_S · mid(S→USDT)) − ln mid_ref) · 1e4,   S ∈ {USDC, FDUSD, USD1}
  опорный рынок: для BTC/ETH/SOL — спот к USDT (базис «перп − спот USDT» уже есть в v2,
  поэтому базис к S = он + gap_S, и новая информация именно в gap_S);
  для PEPE: альтернативный рынок — перп 1000PEPEUSDC, опорный — сам 1000PEPEUSDT.

Доступность (avail_S = 0, разрывы = 0): у альтернативного ИЛИ опорного рынка не было валидной
котировки за последнюю секунду (нет файла дня, пары ещё нет — USD1 до декабря 2025, пары обнулены
из-за отклонения стейблкоина > 50 б.п.), курс старше FX_MAX_AGE_S или сам курс отклонился > 50 б.п.
Протянутая вперёд старая цена никогда не выдаётся за актуальную. Изменения (dgap, OFI) считаются
только по окну, где рынок доступен непрерывно, — через разрыв доступности ничего не сравнивается.

Признаки (окна в барах по 100 мс):
  USDC и FDUSD: gap_S, dgap_S_5, dgap_S_20, gapdev_S (от среднего за 60 с),
                edge_S (исполнимый разрыв по bid/ask), avail_S
  USD1:         gap_usd1, dgap_usd1_20, avail_usd1
  рынок USDC:   alt_imb1, alt_ofi_5 (для PEPE — второго перпа)
  курсы:        fx_dev_usdc, fx_dev_fdusd (отклонение от 1, б.п.), fx_age_usdc, fx_age_fdusd (с),
                fx_flow_usdc (поток сделок пары USDCUSDT за 5 с до момента прогноза)
  спот к USDT:  sbasis, sbasis_dev, dsbasis_5/20, sret_5/20, simb1, savail — те же величины, что
                basis_*/spot_* в v2, но со свежестью котировок перпа и спота (в v2 спот протягивался
                без ограничения возраста). В экспериментах группу v2 можно выкинуть через --drop.
"""
import numpy as np

from common import BBO_MS, grid_points, read_day
from features import BARS_PER_DAY, EPS, STEP, lag, prev_date, rsum
from features_v2 import rmean_valid

ALT = {
    "BTCUSDT_PERP": {"ref": "BTCUSDT_SPOT", "usdc": "BTCUSDC_SPOT", "fdusd": "BTCFDUSD_SPOT", "usd1": "BTCUSD1_SPOT"},
    "ETHUSDT_PERP": {"ref": "ETHUSDT_SPOT", "usdc": "ETHUSDC_SPOT", "fdusd": "ETHFDUSD_SPOT", "usd1": "ETHUSD1_SPOT"},
    "SOLUSDT_PERP": {"ref": "SOLUSDT_SPOT", "usdc": "SOLUSDC_SPOT", "fdusd": "SOLFDUSD_SPOT", "usd1": None},
    "1000PEPEUSDT_PERP": {"ref": "1000PEPEUSDT_PERP", "usdc": "1000PEPEUSDC_PERP", "fdusd": None, "usd1": None},
}
FX_PAIR = {"usdc": "USDCUSDT", "fdusd": "FDUSDUSDT", "usd1": "USD1USDT"}
# спот к USDT и множитель цены перпа к споту (1000PEPE-перп = 1000 × PEPE-спот)
SPOT_USDT = {"BTCUSDT_PERP": ("BTCUSDT_SPOT", 1.0), "ETHUSDT_PERP": ("ETHUSDT_SPOT", 1.0),
             "SOLUSDT_PERP": ("SOLUSDT_SPOT", 1.0), "1000PEPEUSDT_PERP": ("PEPEUSDT_SPOT", 1000.0)}
FX_MAX_AGE_S = 3.0
DEPEG_BPS = 50.0
FX_AGE_CAP_S = 60.0
QUOTE_MAX_AGE_BARS = 10      # котировка рынка считается живой, если валидная строка была не позже 1 с назад
FLOW_WINDOW_MS = 5_000


def v3_names():
    n = []
    for s in ("usdc", "fdusd"):
        n += [f"gap_{s}", f"dgap_{s}_5", f"dgap_{s}_20", f"gapdev_{s}", f"edge_{s}", f"avail_{s}"]
    n += ["gap_usd1", "dgap_usd1_20", "avail_usd1"]
    n += ["alt_imb1", "alt_ofi_5"]
    n += ["fx_dev_usdc", "fx_dev_fdusd", "fx_age_usdc", "fx_age_fdusd", "fx_flow_usdc"]
    # спот к USDT заново, со свежестью котировок (замена группы basis_*/spot_* из v2, где спот протягивался)
    n += ["sbasis", "sbasis_dev", "dsbasis_5", "dsbasis_20", "sret_5", "sret_20", "simb1", "savail"]
    return n


V3_NAMES = v3_names()


def raw_rows_day(root, inst, d):
    """Сырые строки book + bbo за день, ДО протягивания (как их склеивает common.quote_grid):
    → (ts, bid, ask, bq, aq) отсортированные по ts, или None."""
    cols = ["ts", "bid_px", "ask_px", "bid_qty", "ask_qty"]
    parts = []
    for table in ("book", "bbo"):
        t = read_day(root, table, d, inst, columns=cols)
        if t is not None and t.num_rows:
            parts.append([t[c].to_numpy().astype("float64") for c in cols])
    if not parts:
        return None
    arr = [np.concatenate([p[k] for p in parts]) for k in range(5)]
    order = np.argsort(arr[0], kind="stable")
    return [a[order] for a in arr]


def quotes_raw_two_days(root, inst, date, cache=None):
    """Котировки на сетке 100 мс за (вчера + сегодня) + флаг `fresh`.

    Цены протягиваются вперёд (последняя валидная строка с ts <= t). `fresh` = последняя
    ВАЛИДНАЯ сырая строка book/bbo пришла не раньше QUOTE_MAX_AGE_BARS·100 мс назад.
    Возраст считается по времени настоящих строк, а не по протянутой сетке: если рынок замолчал
    посреди дня (нет строк), пропал на весь день или его строки пустые (маскировка) — fresh=False."""
    key = ("qr", inst, date)
    if cache is not None and key in cache:
        return cache[key]
    g = np.concatenate([grid_points(d, BBO_MS) for d in (prev_date(date), date)]).astype("float64")
    rows = [r for r in (raw_rows_day(root, inst, d) for d in (prev_date(date), date)) if r is not None]
    nan = np.full(len(g), np.nan)
    if not rows:
        out = (nan, nan.copy(), nan.copy(), nan.copy(), np.zeros(len(g), bool))
    else:
        ts, bid, ask, bq, aq = (np.concatenate([r[k] for r in rows]) for k in range(5))
        with np.errstate(invalid="ignore"):
            ok = (np.isfinite(bid) & np.isfinite(ask) & (bid > 0) & (ask >= bid)
                  & np.isfinite(bq) & np.isfinite(aq) & (bq >= 0) & (aq >= 0))
        ts, bid, ask, bq, aq = ts[ok], bid[ok], ask[ok], bq[ok], aq[ok]
        i = np.searchsorted(ts, g, "right") - 1               # последняя валидная строка с ts <= t
        has = i >= 0
        ii = np.maximum(i, 0)
        take = lambda a: np.where(has, a[ii], np.nan) if len(a) else nan.copy()
        last_ts = take(ts)
        fresh = has & (g - last_ts <= QUOTE_MAX_AGE_BARS * BBO_MS)
        out = (take(bid), take(ask), take(bq), take(aq), fresh)
    if cache is not None:
        cache[key] = out
    return out


def fx_two_days(root, date, pair, cache=None):
    """Курс стейблкоина на сетке 100 мс за (вчера + сегодня): последняя строка `stable` с ts <= t.
    → dict(mid, bid, ask, age_s, flow5); NaN, где данных нет."""
    key = ("fx", pair, date)
    if cache is not None and key in cache:
        return cache[key]
    cols = ["ts", "pair", "bid_px", "ask_px", "buy_qty", "sell_qty", "stale_ms"]
    parts = []
    for d in (prev_date(date), date):
        t = read_day(root, "stable", d, None, columns=cols)
        if t is None or t.num_rows == 0:
            continue
        pr = np.asarray(t["pair"].to_numpy()).astype(str)
        m = pr == pair
        if m.any():
            parts.append({c: t[c].to_numpy()[m].astype("float64") for c in cols if c != "pair"})
    g = np.concatenate([grid_points(d, BBO_MS) for d in (prev_date(date), date)]).astype("float64")
    nan = np.full(len(g), np.nan)
    if not parts:
        out = {"mid": nan, "bid": nan.copy(), "ask": nan.copy(), "age_s": nan.copy(), "flow5": nan.copy()}
    else:
        r = {c: np.concatenate([p[c] for p in parts]) for c in parts[0]}
        order = np.argsort(r["ts"], kind="stable")
        r = {c: v[order] for c, v in r.items()}
        b, s = np.nan_to_num(r["buy_qty"]), np.nan_to_num(r["sell_qty"])
        cb, cs = np.r_[0.0, np.cumsum(b)], np.r_[0.0, np.cumsum(s)]
        i = np.searchsorted(r["ts"], g, "right") - 1         # последняя строка с ts <= t
        k0 = np.searchsorted(r["ts"], g - FLOW_WINDOW_MS, "right")   # первая строка с ts > t − 5 с
        bs, ss = cb[i + 1] - cb[k0], cs[i + 1] - cs[k0]     # строки в (t − 5 с, t] — окно от МОМЕНТА ПРОГНОЗА
        flow5 = (bs - ss) / (bs + ss + EPS)
        ok = i >= 0
        ii = np.maximum(i, 0)
        take = lambda a: np.where(ok, a[ii], np.nan)
        bid, ask = take(r["bid_px"]), take(r["ask_px"])
        st = take(r["stale_ms"])
        st = np.where(np.isfinite(st) & (st >= 0), st, np.nan)   # неизвестный возраст курса — не «свежий»
        age = (st + (g - take(r["ts"]))) / 1000
        f5 = np.where(ok & (bs + ss > 0) & (age <= FX_MAX_AGE_S), flow5, np.nan)   # нет сделок / устаревший курс
        out = {"mid": (bid + ask) / 2, "bid": bid, "ask": ask, "age_s": age, "flow5": f5}
    if cache is not None:
        cache[key] = out
    return out


def compute_v3(root, inst, date, cache=None):
    """Матрица (172_800 × len(V3_NAMES)) на сетке 500 мс дня `date`. NaN → 0 в самом конце."""
    cache = {} if cache is None else cache
    spec = ALT[inst]
    rb, ra, _, _, ref_fresh = quotes_raw_two_days(root, spec["ref"], date, cache)
    with np.errstate(invalid="ignore", divide="ignore"):
        ref_log = np.log((rb + ra) / 2)
    n = len(ref_log)
    c = {}
    fx = {s: fx_two_days(root, date, FX_PAIR[s], cache) for s in FX_PAIR}

    for s in ("usdc", "fdusd", "usd1"):
        name = spec[s]
        if name is None:
            gap = np.full(n, np.nan)
            avail = np.zeros(n, bool)
            ab = aa = abq = aaq = np.full(n, np.nan)
        else:
            ab, aa, abq, aaq, alt_fresh = quotes_raw_two_days(root, name, date, cache)
            f = fx[s]
            with np.errstate(invalid="ignore", divide="ignore"):
                avail = (alt_fresh & ref_fresh & np.isfinite(ab) & np.isfinite(aa) & np.isfinite(ref_log)
                         & np.isfinite(f["mid"]) & (f["age_s"] <= FX_MAX_AGE_S)
                         & (np.abs(f["mid"] - 1) * 1e4 <= DEPEG_BPS))
                gap = np.where(avail, (np.log((ab + aa) / 2 * f["mid"]) - ref_log) * 1e4, np.nan)
        c[f"gap_{s}"] = gap
        c[f"avail_{s}"] = avail.astype(float)
        av = avail.astype(float)
        cont = lambda k: rsum(av, k + 1) >= k + 1           # доступен весь отрезок [t−k, t] без разрыва
        dg = lambda k: np.where(cont(k), gap - lag(gap, k), np.nan)
        if s == "usd1":
            c["dgap_usd1_20"] = dg(20)
            continue
        c[f"dgap_{s}_5"] = dg(5)
        c[f"dgap_{s}_20"] = dg(20)
        c[f"gapdev_{s}"] = np.where(cont(600), gap - rmean_valid(gap, 600), np.nan)   # среднее — только по
                                                                                    # непрерывной минуте
        if name is not None:
            f = fx[s]
            with np.errstate(invalid="ignore", divide="ignore"):
                bid_syn, ask_syn = ab * f["bid"], aa * f["ask"]
                e_up = np.maximum(0.0, np.log(bid_syn / ra))          # альтернатива дороже опорного ask
                e_dn = np.maximum(0.0, np.log(rb / ask_syn))          # опорный bid дороже альтернативы
                c[f"edge_{s}"] = np.where(avail, (e_up - e_dn) * 1e4, np.nan)
        else:
            c[f"edge_{s}"] = np.full(n, np.nan)
        if s == "usdc":
            ok = avail
            imb = (abq - aaq) / (abq + aaq + EPS)
            c["alt_imb1"] = np.where(ok, imb, np.nan)
            pb, pa, pbq, paq = lag(ab, 1), lag(aa, 1), lag(abq, 1), lag(aaq, 1)
            with np.errstate(invalid="ignore"):
                e = (np.where(ab >= pb, abq, 0) - np.where(ab <= pb, pbq, 0)
                     - np.where(aa <= pa, aaq, 0) + np.where(aa >= pa, paq, 0))
            pair_ok = ok & np.r_[False, ok[:-1]]                # и текущий, и предыдущий бар — живые
            e = np.where(pair_ok & np.isfinite(e), e, 0.0)
            depth = np.nan_to_num((abq + aaq) / 2)
            c["alt_ofi_5"] = np.where(cont(5), rsum(e, 5) / (rsum(depth, 5) / 5 + EPS), np.nan)

    for s in ("usdc", "fdusd"):
        c[f"fx_dev_{s}"] = np.where(fx[s]["age_s"] <= FX_MAX_AGE_S, (fx[s]["mid"] - 1) * 1e4, np.nan)  # без
                                                                                    # устаревшего курса
        c[f"fx_age_{s}"] = np.minimum(np.nan_to_num(fx[s]["age_s"], nan=FX_AGE_CAP_S), FX_AGE_CAP_S)
    c["fx_flow_usdc"] = fx["usdc"]["flow5"]

    # ---- спот к USDT со свежестью (опорный перп — сам целевой инструмент)
    pb_, pa_, _, _, perp_fresh = quotes_raw_two_days(root, inst, date, cache)
    sname, mult = SPOT_USDT[inst]
    sb_, sa_, sbq_, saq_, spot_fresh = quotes_raw_two_days(root, sname, date, cache)
    with np.errstate(invalid="ignore", divide="ignore"):
        plog = np.log((pb_ + pa_) / 2)
        slog = np.log((sb_ + sa_) / 2 * mult)
    sav = perp_fresh & spot_fresh & np.isfinite(plog) & np.isfinite(slog)
    sbasis = np.where(sav, (plog - slog) * 1e4, np.nan)
    sv = sav.astype(float)
    scont = lambda k: rsum(sv, k + 1) >= k + 1
    c["sbasis"] = sbasis
    c["sbasis_dev"] = np.where(scont(600), sbasis - rmean_valid(sbasis, 600), np.nan)
    c["dsbasis_5"] = np.where(scont(5), sbasis - lag(sbasis, 5), np.nan)
    c["dsbasis_20"] = np.where(scont(20), sbasis - lag(sbasis, 20), np.nan)
    spf = spot_fresh.astype(float)
    for k in (5, 20):
        c[f"sret_{k}"] = np.where(rsum(spf, k + 1) >= k + 1, (slog - lag(slog, k)) * 1e4, np.nan)
    c["simb1"] = np.where(spot_fresh, (sbq_ - saq_) / (sbq_ + saq_ + EPS), np.nan)
    c["savail"] = sv

    sel = BARS_PER_DAY + np.arange(0, BARS_PER_DAY, STEP)
    X = np.column_stack([c[k][sel] for k in V3_NAMES])
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
