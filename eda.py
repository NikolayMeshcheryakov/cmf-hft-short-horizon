"""EDA на выборке дней. Ничего не обучает — только смотрит на данные.

    python eda.py --data-dir D:\\hft\\data --days 12 --out eda_out

Блоки (каждый — свой CSV в --out, сводка — eda_report.md):
  1. inventory   — какие дни есть, покрытие периода, дыры
  2. quality     — пропуски/невалидные/скрещенные котировки, пропуски строк book,
                   самые долгие периоды без изменения цены и без новых строк book, бары сделок
  3. targets     — СЫРЫЕ таргеты (до обрезки): нули, хвосты, доля обрезки; часы суток (RMS, std, покрытие)
  4. micro       — тик (из meta/daily, если есть; иначе робастная оценка), спред в тиках, частота изменения mid
  5. ic          — дневная корреляция признак↔таргет РОВНО как в метрике (metric.day_corr),
                   бутстрэп-интервал по дням, поправка Benjamini–Hochberg, проверка знака
                   «первая половина задаёт знак → доля дней второй половины с тем же знаком»,
                   IC на таргете, нормированном на прошлую волатильность
     ic_regimes  — IC внутри режимов: волатильность (терцили дня), спред 1 тик / шире
     alpha_decay — IC признака, взятого на 0.5…8 с раньше: как быстро умирает сигнал
  6. leadlag     — кросс-корреляции 100-мс доходностей с контролем устаревания:
                   доля обновлений у обоих, корреляция только на барах, где ведомый сдвинулся,
                   асимметрия (+k) − (−k) с бутстрэп-интервалом; участки без строк book выкинуты
  7. autocorr    — автокорреляция 100-мс и 500-мс доходностей mid, отдельно при спреде 1 тик и шире

Всё это РАЗВЕДКА: дни выборки потом не являются независимой проверкой.
"""
import argparse
import datetime as dt
import os

import numpy as np
import pyarrow.parquet as pq

from common import HORIZONS, TARGET_INSTRUMENTS, dates, grid_points, load_clip, read_day
from features import FEATURE_NAMES, compute_features, read_nonempty, safe_quote_grid
from metric import day_corr, load_targets

PERIOD = ("2025-09-15", "2026-09-14")
TABLES = ["book", "bbo", "trades", "targets"]
PAIRS = [  # (ведущий, ведомый) — отрицательные лаги показывают обратное направление
    ("BTCUSDT_PERP", "ETHUSDT_PERP"), ("BTCUSDT_PERP", "SOLUSDT_PERP"),
    ("BTCUSDT_PERP", "1000PEPEUSDT_PERP"), ("ETHUSDT_PERP", "SOLUSDT_PERP"),
    ("BTCUSDT_SPOT", "BTCUSDT_PERP"), ("ETHUSDT_SPOT", "ETHUSDT_PERP"),
    ("SOLUSDT_SPOT", "SOLUSDT_PERP"), ("PEPEUSDT_SPOT", "1000PEPEUSDT_PERP"),
    ("BTCUSDT_SPOT", "ETHUSDT_PERP"), ("1000PEPEUSDC_PERP", "1000PEPEUSDT_PERP"),
]
LAGS = list(range(-10, 11))                      # бары по 100 мс
DECAY_FEATS = ["imb1", "micro_bps", "depth_imb_5", "ofi_5", "tflow_5", "ret_5", "lead_ret_5"]
DECAY_LAGS = [0, 1, 2, 4, 8, 16]                 # шаги по 500 мс
BOOK_GAP_MS = 1500                               # нет новой строки book дольше → считаем дырой
N = 864_000                                      # точек 100 мс в сутках
RNG = np.random.default_rng(0)


# ============================================================ helpers
def write_csv(path, rows):
    if not rows:
        return
    cols = list(dict.fromkeys(k for r in rows for k in r))   # объединение ключей всех строк
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(_fmt(r.get(c, "")) for c in cols) + "\n")


def _fmt(v):
    if isinstance(v, (float, np.floating)):
        return "" if not np.isfinite(v) else f"{v:.4g}"
    return str(v)


def md_table(rows, cols=None, max_rows=60):
    if not rows:
        return "_нет данных_\n"
    cols = cols or list(dict.fromkeys(k for r in rows for k in r))
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows[:max_rows]:
        out.append("| " + " | ".join(_fmt(r.get(c, "")) for c in cols) + " |")
    if len(rows) > max_rows:
        out.append(f"\n_… ещё {len(rows) - max_rows} строк в CSV_")
    return "\n".join(out) + "\n"


def pcorr(a, b):
    """Пирсон по конечным парам; NaN, если данных мало или дисперсия нулевая."""
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 100:
        return np.nan
    a, b = a[m], b[m]
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def boot_mean(v, B=2000):
    """→ (среднее, lo95, hi95, p).

    Интервал — бутстрэп по дням (только описание разброса).
    p — sign-flip тест нулевой гипотезы «истинный IC = 0» (знаки дневных IC случайны,
    распределение симметрично вокруг нуля). Точный перебор при n ≤ 16, иначе Монте-Карло.
    Дни считаются независимыми — при зависимых днях p занижено, это ориентир, а не доказательство.
    """
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    n = len(v)
    if n < 2:
        return (float(v.mean()) if n else np.nan), np.nan, np.nan, np.nan
    m = RNG.integers(0, n, size=(B, n))
    bm = v[m].mean(1)
    obs = abs(v.mean())
    if n <= 16:
        signs = ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1) * 2 - 1
    else:
        signs = RNG.choice([-1, 1], size=(20000, n))
    null = np.abs((signs * v).mean(1))
    hits = int(np.count_nonzero(null >= obs - 1e-15))
    p = hits / len(null) if n <= 16 else (hits + 1) / (len(null) + 1)   # Монте-Карло: p не может быть 0
    return float(v.mean()), float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5)), p


def bh_qvalues(p):
    """Benjamini–Hochberg q-values (NaN пропускаются)."""
    p = np.asarray(p, float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return q
    pv = p[ok]
    order = np.argsort(pv)
    n = len(pv)
    ranked = pv[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(ranked, 1.0)
    q[ok] = out
    return q


def longest_run(mask):
    """Самая длинная серия подряд идущих True."""
    if not mask.any():
        return 0
    m = np.r_[False, mask, False].astype(np.int8)
    d = np.diff(m)
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return int((ends - starts).max())


def list_days(root, table, inst):
    d = os.path.join(root, table, inst) if inst else os.path.join(root, table)
    if not os.path.isdir(d):
        return set()
    return {f[:-8] for f in os.listdir(d) if f.endswith(".parquet")}


def instruments(root):
    """Все инструменты, у которых есть папка хотя бы в одной таблице."""
    out = set()
    for t in ("book", "bbo", "trades", "targets"):
        d = os.path.join(root, t)
        if os.path.isdir(d):
            out |= {x for x in os.listdir(d) if os.path.isdir(os.path.join(d, x))}
    return sorted(out)


# ============================================================ выбор дней
def pick_days(root, n, start, end):
    """Дни, где есть таргеты ВСЕХ 4 инструментов; равномерно по календарю."""
    common_days = None
    for inst in TARGET_INSTRUMENTS:
        s = list_days(root, "targets", inst)
        common_days = s if common_days is None else common_days & s
    avail = sorted(d for d in (common_days or set()) if start <= d <= end)
    if not avail:
        raise SystemExit(f"Нет дней с таргетами всех 4 инструментов в {start}..{end}. "
                         f"Для другого периода задайте --start/--end.")
    if len(avail) <= n:
        print(f"ВНИМАНИЕ: доступно только {len(avail)} дней — берём все")
        return avail
    cal = dates(start, end)
    targets = [cal[i] for i in np.linspace(0, len(cal) - 1, n).round().astype(int)]
    av = np.array([dt.date.fromisoformat(d).toordinal() for d in avail])
    chosen = []
    for t in targets:                           # ближайший ещё не выбранный доступный день
        dist = np.abs(av - dt.date.fromisoformat(t).toordinal()).astype(float)
        for c in chosen:
            dist[avail.index(c)] = np.inf
        chosen.append(avail[int(np.argmin(dist))])
    return sorted(chosen)


# ============================================================ 1. inventory
def inventory(root, start, end):
    rows = []
    period = dates(start, end)
    for inst in instruments(root):
        for table in TABLES:
            have = list_days(root, table, inst)
            if not have and table == "targets":
                continue
            inside = [d for d in period if d in have]
            first, last = (min(have), max(have)) if have else ("", "")
            span = [d for d in period if have and first <= d <= last]
            holes = [d for d in span if d not in have]
            rows.append({"instrument": inst, "table": table, "days": len(have),
                         "first": first, "last": last,
                         "coverage_pct_period": 100 * len(inside) / len(period),
                         "missing_in_period": len(period) - len(inside),
                         "holes_inside_first_last": len(holes),
                         "hole_examples": " ".join(holes[:5])})
    st = list_days(root, "stable", None)
    inside = [d for d in period if d in st]
    rows.append({"instrument": "(stable)", "table": "stable", "days": len(st),
                 "first": min(st) if st else "", "last": max(st) if st else "",
                 "coverage_pct_period": 100 * len(inside) / len(period),
                 "missing_in_period": len(period) - len(inside),
                 "holes_inside_first_last": "", "hole_examples": ""})
    return rows


# ============================================================ 2+4. quality & micro
def quality_from_arrays(bid, ask, bq, aq, book_ts, trade_ts, grid100):
    """Чистая функция (удобно тестировать). Все доли — от полной 100-мс сетки, если не сказано иное."""
    n = len(bid)
    finite = np.isfinite(bid) & np.isfinite(ask) & np.isfinite(bq) & np.isfinite(aq)
    pos = finite & (bid > 0) & (ask > 0) & (bq >= 0) & (aq >= 0)
    crossed, locked = pos & (ask < bid), pos & (ask == bid)
    valid = pos & (ask > bid)
    pair_ok = valid[1:] & valid[:-1]
    px_chg = pair_ok & ((bid[1:] != bid[:-1]) | (ask[1:] != ask[:-1]))
    any_chg = px_chg | (pair_ok & ((bq[1:] != bq[:-1]) | (aq[1:] != aq[:-1])))
    pairs = int(pair_ok.sum())

    # строки book: ожидаем каждые 500 мс
    g0 = grid100[0]
    bts = np.unique(book_ts[(book_ts >= g0) & (book_ts < g0 + 86_400_000) & (book_ts % 500 == 0)]) \
        if book_ts is not None and len(book_ts) else np.array([], dtype=np.int64)
    edges = np.r_[g0 - 500, bts, g0 + 86_400_000]
    longest_book_gap = (np.diff(edges).max() - 500) / 1000 if len(edges) > 1 else np.nan

    # бары сделок: уникальные метки на 100-мс сетке внутри дня
    if trade_ts is None:
        tr_present, tr_unique, tr_dup, tr_off = False, 0, 0, 0
    else:
        on_grid = (trade_ts >= g0) & (trade_ts < g0 + 86_400_000) & (trade_ts % 100 == 0)
        tr_present = True
        tr_unique = len(np.unique(trade_ts[on_grid]))
        tr_dup = len(trade_ts) - len(np.unique(trade_ts))          # повторы одной метки
        tr_off = int((~on_grid).sum())                              # вне дня или не кратно 100 мс

    return {
        "missing_share": float((~finite).mean()),
        "nonpositive_share": float((finite & ~pos).mean()),
        "crossed_share": float(crossed.mean()),
        "locked_share": float(locked.mean()),
        "valid_share": float(valid.mean()),
        "price_change_share_100ms": float(px_chg.sum() / pairs) if pairs else np.nan,
        "quote_change_share_100ms": float(any_chg.sum() / pairs) if pairs else np.nan,
        # серии неизменных интервалов; NaN/невалидное РАЗРЫВАЕТ серию
        "longest_no_price_change_s": longest_run(pair_ok & ~px_chg) * 0.1,
        "longest_no_quote_change_s": longest_run(pair_ok & ~any_chg) * 0.1,
        "book_rows_on_grid": int(len(bts)),
        "book_rows_missing": int(172_800 - len(bts)),
        "longest_book_gap_s": float(longest_book_gap),
        "trades_file_present": tr_present,
        "trade_bar_share_100ms": tr_unique / n,
        "trade_duplicate_rows": int(tr_dup),
        "trade_offgrid_rows": tr_off,
    }


def tick_from_meta(root, inst, day):
    """Пытаемся найти размер тика в meta/daily (схема заранее неизвестна — ищем колонку с 'tick')."""
    cands = [os.path.join(root, "meta", "daily", f"{day}__{inst}.parquet"),
             os.path.join(root, "meta", "daily.parquet")]
    for p in cands:
        if not os.path.isfile(p):
            continue
        try:
            t = pq.read_table(p)
        except Exception:
            continue
        cols = [c for c in t.column_names if "tick" in c.lower()]
        if not cols:
            continue
        d = t.to_pydict()
        idx = list(range(t.num_rows))
        if "instrument" in d:
            idx = [i for i in idx if str(d["instrument"][i]) == inst]
        for k in ("date", "day"):
            if k in d:
                idx = [i for i in idx if str(d[k][i])[:10] == day]
        if idx:
            try:
                return float(d[cols[0]][idx[0]]), cols[0]
            except (TypeError, ValueError):
                pass
    return np.nan, ""


def tick_estimate(bid, ask):
    """Запасная оценка, если тика нет в мете. Кандидаты — шаги bid, шаги ask и сам спред
    (спред часто ровно 1 тик, даже если цена прыгает через тик). Берём частые значения
    (≥5% от самого частого) и из них наименьшее. Это оценка, а не контрактный тик."""
    valid = np.isfinite(bid) & np.isfinite(ask) & (bid > 0) & (ask > bid)
    if valid.sum() < 2:
        return np.nan
    med = np.nanmedian((bid + ask)[valid] / 2)
    ok = valid[1:] & valid[:-1]
    d = np.r_[np.abs(np.diff(bid))[ok], np.abs(np.diff(ask))[ok], (ask - bid)[valid]]
    d = d[d > 1e-8 * med]
    if not len(d):
        return np.nan
    vals, cnt = np.unique(np.round(d / med, 12), return_counts=True)
    keep = vals[cnt >= 0.05 * cnt.max()]
    return float(keep.min() * med)


def micro_from_arrays(bid, ask, tick):
    valid = np.isfinite(bid) & np.isfinite(ask) & (bid > 0) & (ask > bid)
    mid = (bid + ask) / 2
    pair_ok = valid[1:] & valid[:-1]
    mid_chg = pair_ok & (mid[1:] != mid[:-1])
    out = {"median_price": float(np.nanmedian(mid[valid])) if valid.any() else np.nan,
           "mid_change_share_100ms": float(mid_chg.sum() / pair_ok.sum()) if pair_ok.any() else np.nan,
           "tick_bps": np.nan, "spread_1tick_share": np.nan, "spread_mean_ticks": np.nan,
           "spread_nonint_share": np.nan}
    if np.isfinite(tick) and tick > 0 and valid.any():
        st = (ask - bid)[valid] / tick
        out.update({"tick_bps": tick / out["median_price"] * 1e4,
                    "spread_1tick_share": float((np.abs(st - 1) < 0.01).mean()),
                    "spread_mean_ticks": float(st.mean()),
                    "spread_nonint_share": float((np.abs(st - np.round(st)) > 0.01).mean())})
    return out


def regime_masks(vol, obs, okq, qb, qa, tick):
    """Маски режимов на 500-мс сетке. obs — котировка валидна и строка book есть.
    Волатильность — терцили по наблюдаемым точкам (одинаковые значения не уезжают в «высокую»)."""
    n = len(vol)
    if np.isfinite(tick) and tick > 0:
        sp = np.where(okq, (qa - qb) / tick, np.nan)
        one, wider = obs & (np.abs(sp - 1) < 0.01), obs & (sp > 1.01)
    else:
        one = wider = np.zeros(n, bool)
    if obs.sum() > 100:
        q1, q2 = np.nanpercentile(vol[obs], [33.3, 66.7])
        lo, mid, hi = obs & (vol <= q1), obs & (vol > q1) & (vol <= q2), obs & (vol > q2)
    else:
        lo = mid = hi = np.zeros(n, bool)
    return {"vol_low": lo, "vol_mid": mid, "vol_high": hi, "spread_1tick": one, "spread_wider": wider,
            "fresh_book": obs, "no_book_row_or_bad_quote": ~obs}


# ============================================================ 3. targets
def target_stats(raw_by_inst, clip):
    rows, hourly = [], []
    for inst, days in raw_by_inst.items():
        for h in HORIZONS:
            y_all = np.concatenate([d[h] for d in days]).astype(float)
            fin = np.isfinite(y_all)
            y = y_all[fin]
            if not len(y):
                continue
            lo, hi = clip[(inst, h)]
            yb = y * 1e4
            m, s = yb.mean(), yb.std()
            q = np.percentile(yb, [0.1, 1, 50, 99, 99.9])
            rows.append({"instrument": inst, "horizon": h, "coverage": float(fin.mean()),
                         "zero_share": float((y == 0).mean()), "std_bps": float(s),
                         "excess_kurtosis": float(((yb - m) ** 4).mean() / s ** 4 - 3) if s > 0 else np.nan,
                         "p0.1": q[0], "p1": q[1], "p50": q[2], "p99": q[3], "p99.9": q[4],
                         "min_bps": float(yb.min()), "max_bps": float(yb.max()),
                         "clip_lo_bps": lo * 1e4, "clip_hi_bps": hi * 1e4,
                         "clipped_share": float(((y < lo) | (y > hi)).mean())})
        stack = np.vstack([d["y_500ms"] for d in days]).astype(float) * 1e4
        per_hour = stack.reshape(stack.shape[0], 24, -1)
        for hr in range(24):
            v = per_hour[:, hr, :]
            fin = np.isfinite(v)
            hourly.append({"instrument": inst, "hour_utc": hr, "coverage": float(fin.mean()),
                           "rms_y500ms_bps": float(np.sqrt(np.mean(v[fin] ** 2))) if fin.any() else np.nan,
                           "std_y500ms_bps": float(np.std(v[fin])) if fin.any() else np.nan})
    return rows, hourly


# ============================================================ 5. IC
def ic_summary(daily, days, B):
    rows = []
    half = len(days) // 2
    for (inst, h, f), v in daily.items():
        v = np.array(v, float)
        if np.isfinite(v).sum() < 2:
            continue
        mean, lo, hi, p = boot_mean(v, B)
        a, b = v[:half], v[half:]
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        sign1 = np.sign(a.mean()) if len(a) else 0.0
        rows.append({"instrument": inst, "horizon": h, "feature": f, "ic_mean": mean,
                     "ci95_lo": lo, "ci95_hi": hi, "p_boot": p, "days": int(np.isfinite(v).sum()),
                     "ic_first_half": float(a.mean()) if len(a) else np.nan,
                     "ic_second_half": float(b.mean()) if len(b) else np.nan,
                     "n_first": len(a), "n_second": len(b),
                     "second_half_same_sign_share": float((np.sign(b) == sign1).mean())
                     if len(b) and sign1 != 0 else np.nan})
    q = bh_qvalues([r["p_boot"] for r in rows])
    for r, qq in zip(rows, q):
        r["q_bh"] = float(qq)
    rows.sort(key=lambda r: (r["instrument"], r["horizon"], -abs(r["ic_mean"])))
    return rows


# ============================================================ 6+7. returns, lead-lag, autocorr
def returns_100ms(q, book_ts):
    """100-мс лог-доходности mid (б.п.). NaN там, где котировка невалидна или давно не было строки book."""
    if q is None:
        return None, None
    g, bid, ask, _, _ = q
    valid = np.isfinite(bid) & np.isfinite(ask) & (bid > 0) & (ask > bid)
    lm = np.where(valid, np.log((bid + ask) / 2), np.nan)
    if book_ts is not None and len(book_ts):
        bts = np.sort(book_ts)
        i = np.searchsorted(bts, g, "right") - 1
        age = np.where(i >= 0, g - bts[np.maximum(i, 0)], np.inf)
        lm[age > BOOK_GAP_MS] = np.nan
    else:
        lm[:] = np.nan
    r = np.full(len(lm), np.nan)
    r[1:] = (lm[1:] - lm[:-1]) * 1e4
    moved = np.zeros(len(lm), bool)
    moved[1:] = np.isfinite(r[1:]) & (r[1:] != 0)
    return r, moved


def lagged(lead, follow, k, mask=None):
    """corr(lead[t-k], follow[t]); k > 0 — ведущий раньше. mask — по времени ведомого."""
    if k > 0:
        a, b, m = lead[:-k], follow[k:], (None if mask is None else mask[k:])
    elif k < 0:
        a, b, m = lead[-k:], follow[:k], (None if mask is None else mask[:k])
    else:
        a, b, m = lead, follow, mask
    if m is not None:
        a, b = a[m], b[m]
    return pcorr(a, b)


# ============================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--days", type=int, default=12)
    ap.add_argument("--start", default=PERIOD[0])
    ap.add_argument("--end", default=PERIOD[1])
    ap.add_argument("--out", default="eda_out")
    ap.add_argument("--no-ic", action="store_true", help="пропустить блок 5 (самый долгий)")
    ap.add_argument("--all-instruments", action="store_true",
                    help="качество для всех 20 инструментов (по умолчанию — цели и участники пар)")
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    root = a.data_dir

    days = pick_days(root, a.days, a.start, a.end)
    clip = load_clip(root)
    print(f"дни выборки ({len(days)}): {', '.join(days)}")
    rep = [f"# EDA\n\nДни выборки ({len(days)}): {', '.join(days)}\n\n"
           "Это разведка на небольшой выборке дней: цифры — ориентир, а не доказательство. "
           "Отобранные здесь признаки потом надо проверять на днях, которые сюда не попали.\n"]

    # ---------------------------------------------------------- 1
    inv = inventory(root, a.start, a.end)
    write_csv(f"{a.out}/inventory.csv", inv)
    rep += ["\n## 1. Какие данные есть\n",
            "coverage_pct_period — доля дней периода, за которые есть файл; "
            "holes_inside_first_last — дыры между первым и последним днём инструмента "
            "(пропуски до листинга сюда не входят).\n\n", md_table(inv, max_rows=100)]
    print("1/7 inventory")

    # ---------------------------------------------------------- 2+4 и 6+7 (общий проход по дням)
    pair_insts = sorted({x for p in PAIRS for x in p})
    q_insts = instruments(root) if a.all_instruments else sorted(set(TARGET_INSTRUMENTS) | set(pair_insts))
    qual, micro = [], []
    ll = {}                                   # (lead, fol, kind, k) -> [по дням]
    upd = {}                                  # inst -> [доля 100-мс баров с движением mid]
    ac = {}                                   # (inst, grid, cond, k) -> [по дням]
    for di, day in enumerate(days):
        g100 = grid_points(day, 100)
        Q, R, MV, BT = {}, {}, {}, {}
        for inst in q_insts:
            q = safe_quote_grid(root, inst, day)
            # сделки: отличаем «файла нет» (None) от «файл есть, но пустой» (пустой массив)
            t = read_day(root, "trades", day, inst, columns=["ts"])
            tts = t["ts"].to_numpy() if t is not None else None
            if q is None:
                qual.append({"instrument": inst, "day": day, "book_available": False,
                             "missing_share": 1.0, "trades_file_present": t is not None})
                continue
            b = read_nonempty(root, "book", day, inst, columns=["ts"])
            bts = b["ts"].to_numpy() if b is not None else None
            _, bid, ask, bq, aq = q
            row = {"instrument": inst, "day": day, "book_available": True}
            row.update(quality_from_arrays(bid, ask, bq, aq, bts, tts, g100))
            qual.append(row)
            tm, tcol = tick_from_meta(root, inst, day)
            mid = (bid + ask) / 2
            te = tick_estimate(bid, ask)
            tick = tm if np.isfinite(tm) else te
            mrow = {"instrument": inst, "day": day, "tick_meta": tm, "tick_estimate": te,
                    "tick_source": tcol or "estimate (ненадёжно)",
                    "tick_mismatch": bool(np.isfinite(tm) and np.isfinite(te) and abs(tm - te) > 1e-9 * max(tm, te))}
            mrow.update(micro_from_arrays(bid, ask, tick))
            micro.append(mrow)
            Q[inst], BT[inst] = q, bts
            R[inst], MV[inst] = returns_100ms(q, bts)
            upd.setdefault(inst, []).append(float(MV[inst].mean()))

        # 6. lead-lag
        for lead, fol in PAIRS:
            if lead not in R or fol not in R:
                continue
            for k in LAGS:
                ll.setdefault((lead, fol, "all", k), [np.nan] * len(days))[di] = lagged(R[lead], R[fol], k)
                ll.setdefault((lead, fol, "fol_moved", k), [np.nan] * len(days))[di] = \
                    lagged(R[lead], R[fol], k, MV[fol])

        # 7. автокорреляция: 100 мс (по состоянию спреда в t−1) и 500 мс
        for inst in TARGET_INSTRUMENTS:
            if inst not in R:
                continue
            r = R[inst]
            _, bid, ask, _, _ = Q[inst]
            tick = next((m["tick_meta"] if np.isfinite(m["tick_meta"]) else m["tick_estimate"])
                        for m in micro if m["instrument"] == inst and m["day"] == day)
            one = np.zeros(len(r), bool)
            wide = np.zeros(len(r), bool)
            if np.isfinite(tick):
                okp = np.isfinite(bid) & np.isfinite(ask) & (ask > bid)
                st = np.where(okp, (ask - bid) / tick, np.nan)
                one[1:] = np.abs(st[:-1] - 1) < 0.01
                wide[1:] = st[:-1] > 1.01
            for k in (1, 2, 5, 10):
                ac.setdefault((inst, "100ms", "all", k), [np.nan] * len(days))[di] = lagged(r, r, k)
                ac.setdefault((inst, "100ms", "spread=1tick", k), [np.nan] * len(days))[di] = lagged(r, r, k, one)
                ac.setdefault((inst, "100ms", "spread>1tick", k), [np.nan] * len(days))[di] = lagged(r, r, k, wide)
            r5 = np.nansum(r.reshape(-1, 5), axis=1)
            r5[np.isnan(r.reshape(-1, 5)).any(1)] = np.nan
            for k in (1, 2, 4):
                ac.setdefault((inst, "500ms", "all", k), [np.nan] * len(days))[di] = lagged(r5, r5, k)
        print(f"   день {di + 1}/{len(days)} {day}: качество, lead-lag, автокорреляция")

    write_csv(f"{a.out}/quality.csv", qual)
    write_csv(f"{a.out}/micro.csv", micro)

    def summarize(rows, keys, worst=None):
        out = []
        for inst in sorted({r["instrument"] for r in rows}):
            rr = [r for r in rows if r["instrument"] == inst]
            s = {"instrument": inst, "days": len(rr)}
            for k in keys:
                v = np.array([r.get(k, np.nan) for r in rr], float)
                s[k] = float(np.nanmedian(v)) if np.isfinite(v).any() else np.nan
            for k in (worst or []):
                v = np.array([r.get(k, np.nan) for r in rr], float)
                s["worst_" + k] = float(np.nanmax(v)) if np.isfinite(v).any() else np.nan
            out.append(s)
        return out

    q_sum = summarize(qual, ["missing_share", "nonpositive_share", "crossed_share", "locked_share",
                             "book_rows_missing", "price_change_share_100ms", "trade_bar_share_100ms"],
                      worst=["longest_book_gap_s", "longest_no_price_change_s", "trade_duplicate_rows",
                             "trade_offgrid_rows"])
    m_sum = summarize(micro, ["median_price", "tick_meta", "tick_estimate", "tick_bps",
                              "spread_1tick_share", "spread_mean_ticks", "spread_nonint_share",
                              "mid_change_share_100ms"])
    for s in m_sum:
        s["tick_mismatch_days"] = sum(m["tick_mismatch"] for m in micro if m["instrument"] == s["instrument"])
    rep += ["\n## 2. Качество данных (медиана по дням; worst_* — худший день)\n",
            "Доли — от полной 100-мс сетки. longest_book_gap_s — самый долгий промежуток без новой строки "
            "book сверх нормальных 0.5 с: это признак сбоя коллектора. longest_no_price_change_s — самая "
            "длинная серия без изменения bid/ask при валидных котировках: это может быть и нормальное затишье, "
            "смотреть вместе с book_gap. Невалидные точки разрывают серии и не считаются изменениями.\n\n",
            md_table(q_sum),
            "\n## 4. Микроструктура (медиана по дням)\n",
            "Тик берётся из meta/daily, если там нашлась колонка с 'tick', иначе — робастная оценка по частым "
            "шагам bid/ask и по самому спреду (может завысить тик, если цена ходит только через тик — "
            "тогда выводы о доле однотиковых спредов ненадёжны). tick_mismatch_days — дни, где мета и оценка разошлись (стоит посмотреть). "
            "Крупный тик (спред почти всегда 1 тик, mid меняется редко) — дисбаланс очереди особенно информативен.\n\n",
            md_table(m_sum)]
    print("2,4/7 quality + micro")

    # ---------------------------------------------------------- 3 + 5
    raw_by_inst = {i: [] for i in TARGET_INSTRUMENTS}
    no_clip = {(i, h): (-np.inf, np.inf) for i in TARGET_INSTRUMENTS for h in HORIZONS}
    ic_d, icn_d, reg_d, dec_d = {}, {}, {}, {}
    fidx = {f: j for j, f in enumerate(FEATURE_NAMES)}
    tick_day = {(m["instrument"], m["day"]): (m["tick_meta"] if np.isfinite(m["tick_meta"]) else m["tick_estimate"])
                for m in micro}
    ic_rows_long = []
    for di, day in enumerate(days):
        cache = {}
        for inst in TARGET_INSTRUMENTS:
            raw = load_targets(root, inst, day, no_clip)
            if raw is None:
                continue
            raw_by_inst[inst].append({h: raw[h].astype(np.float32) for h in HORIZONS})
            if a.no_ic:
                continue
            ys = {h: np.clip(raw[h], *clip[(inst, h)]) for h in HORIZONS}  # как в метрике
            X = compute_features(root, inst, day, cache)
            vol = X[:, fidx["vol_bps"]]
            sig = np.maximum(vol, 0.1) * 1e-4                    # прошлая волатильность, с полом
            # Режимы считаем ТОЛЬКО по реально наблюдаемым котировкам: в X пропуски заменены нулями,
            # и «спред = 0» иначе попал бы в режим «1 тик». Берём котировки и строки book напрямую.
            q = safe_quote_grid(root, inst, day)
            b = read_nonempty(root, "book", day, inst, columns=["ts"])
            g500 = grid_points(day)
            if q is not None:
                _, qb, qa, _, _ = q
                qb, qa = qb[::5], qa[::5]                        # точки 500-мс сетки
                okq = np.isfinite(qb) & np.isfinite(qa) & (qb > 0) & (qa > qb)
            else:
                qb = qa = np.full(len(g500), np.nan)
                okq = np.zeros(len(g500), bool)
            fresh = np.isin(g500, b["ts"].to_numpy()) if b is not None else np.zeros(len(g500), bool)
            obs = okq & fresh                                    # котировка валидна и строка book есть
            regimes = regime_masks(vol, obs, okq, qb, qa, tick_day.get((inst, day), np.nan))
            for h in HORIZONS:
                y = ys[h]
                for f, j in fidx.items():
                    x = X[:, j]
                    v = day_corr(x, y)                            # ровно как метрика
                    ic_d.setdefault((inst, h, f), [np.nan] * len(days))[di] = v
                    icn_d.setdefault((inst, h, f), [np.nan] * len(days))[di] = day_corr(x, y / sig)
                    ic_rows_long.append({"instrument": inst, "horizon": h, "feature": f, "day": day, "ic": v})
                    for rg, m in regimes.items():
                        if m.sum() > 1000:
                            reg_d.setdefault((inst, h, f, rg), [np.nan] * len(days))[di] = day_corr(x[m], y[m])
                for f in DECAY_FEATS:
                    x = X[:, fidx[f]]
                    for L in DECAY_LAGS:
                        # первые L прогнозов = 0, как «пропущенный прогноз» в метрике
                        xl = x if L == 0 else np.r_[np.zeros(L), x[:-L]]
                        dec_d.setdefault((inst, h, f, L), [np.nan] * len(days))[di] = day_corr(xl, y)
        print(f"   день {di + 1}/{len(days)} {day}: таргеты" + ("" if a.no_ic else " + IC"))

    t_rows, hourly = target_stats({k: v for k, v in raw_by_inst.items() if v}, clip)
    write_csv(f"{a.out}/targets.csv", t_rows)
    write_csv(f"{a.out}/targets_hourly.csv", hourly)
    peak = []
    for inst in TARGET_INSTRUMENTS:
        hh = [r for r in hourly if r["instrument"] == inst and np.isfinite(r["std_y500ms_bps"])
              and r["coverage"] > 0.5]
        if hh:
            hi = max(hh, key=lambda r: r["std_y500ms_bps"])
            lo = min(hh, key=lambda r: r["std_y500ms_bps"])
            peak.append({"instrument": inst, "most_volatile_hour_utc": hi["hour_utc"],
                         "std_max_bps": hi["std_y500ms_bps"], "calmest_hour_utc": lo["hour_utc"],
                         "std_min_bps": lo["std_y500ms_bps"],
                         "ratio": hi["std_y500ms_bps"] / max(lo["std_y500ms_bps"], 1e-12)})
    rep += ["\n## 3. Таргеты — СЫРЫЕ, до обрезки (б.п.)\n",
            "clip_lo/hi — пороги метрики, clipped_share — какая доля точек будет обрезана. "
            "coverage — доля точек с таргетом.\n\n", md_table(t_rows),
            "\n### Часы суток (std таргета 500 мс; учитываются часы с покрытием > 50%; полная таблица — "
            "targets_hourly.csv, там же RMS)\n", md_table(peak)]
    print("3/7 targets")

    if not a.no_ic:
        write_csv(f"{a.out}/ic_daily.csv", ic_rows_long)
        ic = ic_summary(ic_d, days, a.boot)
        icn = {(r["instrument"], r["horizon"], r["feature"]): r["ic_mean"] for r in ic_summary(icn_d, days, 200)}
        for r in ic:
            r["ic_volnorm_target"] = icn.get((r["instrument"], r["horizon"], r["feature"]), np.nan)
        write_csv(f"{a.out}/ic.csv", ic)
        top = []
        for inst in TARGET_INSTRUMENTS:
            for h in HORIZONS:
                top += [r for r in ic if r["instrument"] == inst and r["horizon"] == h][:5]
        # режимы
        reg_rows = []
        for (inst, h, f, rg), v in reg_d.items():
            v = np.array(v, float)
            reg_rows.append({"instrument": inst, "horizon": h, "feature": f, "regime": rg,
                             "ic_mean": float(np.nanmean(v)) if np.isfinite(v).any() else np.nan,
                             "days": int(np.isfinite(v).sum())})
        write_csv(f"{a.out}/ic_regimes.csv", reg_rows)
        reg_view = []
        for inst in TARGET_INSTRUMENTS:
            for f in ("imb1", "ofi_5", "tflow_5", "lead_ret_5"):
                r = {"instrument": inst, "feature": f}
                for rg in ("vol_low", "vol_mid", "vol_high", "spread_1tick", "spread_wider",
                           "fresh_book", "no_book_row_or_bad_quote"):
                    m = [x for x in reg_rows if x["instrument"] == inst and x["horizon"] == "y_500ms"
                         and x["feature"] == f and x["regime"] == rg]
                    r[rg] = m[0]["ic_mean"] if m else np.nan
                reg_view.append(r)
        # alpha decay
        dec_rows = []
        for (inst, h, f, L), v in dec_d.items():
            v = np.array(v, float)
            dec_rows.append({"instrument": inst, "horizon": h, "feature": f, "lag_s": L * 0.5,
                             "ic_mean": float(np.nanmean(v)) if np.isfinite(v).any() else np.nan,
                             "days": int(np.isfinite(v).sum())})
        write_csv(f"{a.out}/alpha_decay.csv", dec_rows)
        dec_view = []
        for inst in TARGET_INSTRUMENTS:
            for f in DECAY_FEATS:
                r = {"instrument": inst, "feature": f}
                for L in DECAY_LAGS:
                    m = [x for x in dec_rows if x["instrument"] == inst and x["horizon"] == "y_500ms"
                         and x["feature"] == f and x["lag_s"] == L * 0.5]
                    r[f"lag_{L * 0.5:g}s"] = m[0]["ic_mean"] if m else np.nan
                dec_view.append(r)

        rep += ["\n## 5. Сила признаков (IC)\n",
                "IC — средняя дневная корреляция, посчитанная функцией метрики (постоянный признак за день = 0). "
                "ci95 — бутстрэп по дням (описание разброса); p_boot — sign-flip тест «IC = 0» по дневным IC "
                "(дни считаются независимыми, так что p скорее занижено); q_bh — поправка Benjamini–Hochberg на все "
                f"{len(ic)} проверок (q < 0.05 — переживает множественные сравнения). "
                "При малом числе дней интервалы широкие, выводы предварительные. "
                "second_half_same_sign_share — знак задаёт первая половина дней, считаем, в какой доле дней "
                "второй половины он повторился. ic_volnorm_target — IC с таргетом, делённым на прошлую "
                "волатильность: если он сильно ниже ic_mean, сигнал держится на бурных периодах. "
                "Признаки *_x_vol сравнивать с их базовой версией именно по этой колонке. "
                "Полная таблица — ic.csv, по дням — ic_daily.csv.\n\n",
                md_table(top, ["instrument", "horizon", "feature", "ic_mean", "ci95_lo", "ci95_hi", "q_bh",
                               "ic_volnorm_target", "ic_first_half", "ic_second_half",
                               "second_half_same_sign_share"], max_rows=80),
                "\n### IC по режимам (горизонт 500 мс; все горизонты и признаки — ic_regimes.csv)\n",
                "Режимы строятся только по точкам с валидной котировкой и строкой book (fresh_book); "
                "no_book_row_or_bad_quote — точки, где этого нет: если IC там сильно другой, признаки "
                "на устаревших котировках ведут себя иначе. Терцили волатильности — внутри дня "
                "(допустимо в разведке, но не как признак модели).\n\n", md_table(reg_view),
                "\n### Затухание сигнала (горизонт 500 мс; все — alpha_decay.csv)\n",
                "IC признака, взятого на lag секунд раньше момента прогноза (посчитано как метрика: "
                "первые точки без значения = 0). Скорость падения — ориентир, сколько живёт сигнал; "
                "часть падения может объясняться автокорреляцией самого признака.\n\n", md_table(dec_view)]
        print("5/7 IC")

    # ---------------------------------------------------------- 6 + 7 отчёт
    ll_rows = []
    for (lead, fol, kind, k), v in ll.items():
        v = np.array(v, float)
        ll_rows.append({"leader": lead, "follower": fol, "sample": kind, "lag_100ms": k,
                        "corr": float(np.nanmean(v)) if np.isfinite(v).any() else np.nan,
                        "days": int(np.isfinite(v).sum())})
    write_csv(f"{a.out}/leadlag.csv", ll_rows)
    ll_sum = []
    for lead, fol in PAIRS:
        def get(kind, k):
            return np.array(ll.get((lead, fol, kind, k), [np.nan]), float)
        if not np.isfinite(get("all", 0)).any():
            continue
        r = {"leader": lead, "follower": fol, "days": int(np.isfinite(get("all", 0)).sum()),
             "upd_leader": float(np.nanmean(upd.get(lead, [np.nan]))),
             "upd_follower": float(np.nanmean(upd.get(fol, [np.nan]))),
             "lag0": float(np.nanmean(get("all", 0)))}
        for k in (1, 2, 5):
            r[f"+{k}"] = float(np.nanmean(get("all", k)))
        for k in (1, 2):
            r[f"-{k}"] = float(np.nanmean(get("all", -k)))
        for k in (1, 2):
            m, lo, hi, _ = boot_mean(get("all", k) - get("all", -k), 1000)
            r[f"asym{k}"], r[f"asym{k}_lo"], r[f"asym{k}_hi"] = m, lo, hi
        r["+1_fol_moved"] = float(np.nanmean(get("fol_moved", 1)))
        r["+2_fol_moved"] = float(np.nanmean(get("fol_moved", 2)))
        ll_sum.append(r)
    rep += ["\n## 6. Связи рынков (корреляции 100-мс доходностей со сдвигом)\n",
            "+k — доходность ведущего на k×100 мс раньше против доходности ведомого сейчас; −k — наоборот. "
            "asymK = (+K) − (−K) с 95% бутстрэп-интервалом по дням (K = 1, 2). upd_* — доля 100-мс баров, где mid "
            "сдвинулся. *_fol_moved — та же корреляция только на барах, где ведомый реально сдвинулся "
            "(это диагностика порядка обновлений, а не доказательство: если ведомый просто позже реагирует "
            "на общую новость, связь останется и здесь).\n\n"
            "Как читать: асимметрия показывает, что котировки одного рынка обновляются раньше. Это может быть "
            "просто тем, что второй рынок реже обновляется (устаревшие цены), и само по себе не доказывает "
            "прогнозной силы. Признак полезен, если связь видна и на барах, где ведомый двигался, и если "
            "дальше он поднимает SCORE модели. Участки, где book молчал дольше 1.5 с, выкинуты.\n\n",
            md_table(ll_sum)]

    ac_rows = []
    for (inst, grid, cond, k), v in ac.items():
        v = np.array(v, float)
        ac_rows.append({"instrument": inst, "grid": grid, "condition": cond, "lag": k,
                        "autocorr": float(np.nanmean(v)) if np.isfinite(v).any() else np.nan,
                        "days": int(np.isfinite(v).sum())})
    write_csv(f"{a.out}/autocorr.csv", ac_rows)
    ac_view = []
    for inst in TARGET_INSTRUMENTS:
        for grid, cond in (("100ms", "all"), ("100ms", "spread=1tick"), ("100ms", "spread>1tick"), ("500ms", "all")):
            r = {"instrument": inst, "grid": grid, "condition": cond}
            for x in ac_rows:
                if x["instrument"] == inst and x["grid"] == grid and x["condition"] == cond:
                    r[f"lag{x['lag']}"] = x["autocorr"]
            ac_view.append(r)
    rep += ["\n## 7. Автокорреляция доходностей mid\n",
            "Отрицательная на лаге 1 — mid чаще возвращается, чем продолжает. На mid это не классический "
            "bid-ask bounce (он у цен сделок), а скорее дискретность тика, поочерёдное движение bid и ask или "
            "краткий разворот котировок. Разбивка по спреду в t−1: если эффект живёт только при спреде шире "
            "тика (mid стоит «между» тиками и возвращается), это механика стакана. Если на 500 мс эффект "
            "исчезает — это шум микроструктуры, а не торгуемый разворот.\n\n", md_table(ac_view)]
    print("6,7/7 lead-lag + autocorr")

    with open(f"{a.out}/eda_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(rep))
    print(f"\nГотово. Отчёт: {a.out}/eda_report.md (таблицы — CSV рядом)")


if __name__ == "__main__":
    from textio import utf8_console
    utf8_console()
    main()
