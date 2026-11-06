"""Фейковые данные в ТОМ ЖЕ формате, что у организаторов — чтобы проверить,
что весь конвейер запускается, пока настоящие данные качаются.

    python make_fake_data.py --out ./fake_data --start 2026-01-01 --days 5

В фейке специально зашиты сигналы (дисбаланс, поток сделок, BTC ведёт остальных),
поэтому корреляции тут будут нереально высокими. Это только тест кода.
"""
import argparse
import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from common import HORIZONS, TARGET_INSTRUMENTS, dates, grid_points

SPEC = {  # цена, тик, типичная глубина L1 в базовой валюте
    "BTCUSDT_PERP": (100_000.0, 0.1, 5.0),
    "ETHUSDT_PERP": (3_500.0, 0.01, 50.0),
    "SOLUSDT_PERP": (200.0, 0.01, 500.0),
    "1000PEPEUSDT_PERP": (0.012, 0.000001, 2e6),
}
H_BARS = {"y_500ms": 5, "y_2s": 20, "y_8s": 80}
# спот к USDT для каждой монеты: (имя, множитель цены относительно перпа, тик)
SPOT = {"BTCUSDT_PERP": ("BTCUSDT_SPOT", 1.0, 0.01), "ETHUSDT_PERP": ("ETHUSDT_SPOT", 1.0, 0.01),
        "SOLUSDT_PERP": ("SOLUSDT_SPOT", 1.0, 0.01), "1000PEPEUSDT_PERP": ("PEPEUSDT_SPOT", 1e-3, 1e-9)}
# рынки в других стейблкоинах (как у организаторов): споты к USDC/FDUSD/USD1 и перп 1000PEPEUSDC
ALT = {"BTCUSDT_PERP": [("BTCUSDC_SPOT", 0.01), ("BTCFDUSD_SPOT", 0.01), ("BTCUSD1_SPOT", 0.01)],
       "ETHUSDT_PERP": [("ETHUSDC_SPOT", 0.01), ("ETHFDUSD_SPOT", 0.01), ("ETHUSD1_SPOT", 0.01)],
       "SOLUSDT_PERP": [("SOLUSDC_SPOT", 0.01), ("SOLFDUSD_SPOT", 0.01)],
       "1000PEPEUSDT_PERP": [("1000PEPEUSDC_PERP", 0.000001)]}


def ar1(rng, n, phi, sd):
    e = rng.normal(0, sd, n)
    try:
        from scipy.signal import lfilter      # быстро, если scipy есть
        return lfilter([1.0], [1.0, -phi], e)
    except ImportError:
        x = np.empty(n)
        x[0] = e[0]
        for i in range(1, n):                 # медленно, но работает без scipy
            x[i] = phi * x[i - 1] + e[i]
        return x


def write(t, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pq.write_table(pa.table(t), path, compression="zstd")


def write_market(out, inst, day, ts, on500, bid, ask, bq, aq, mid, imb, flow, depth0, rng):
    """book (500 мс), bbo (только изменения, не на 500 мс), trades (бары 100 мс) за один день."""
    per_day = len(ts)
    b = {"ts": ts[on500], "bid_px": bid[on500], "ask_px": ask[on500],
         "bid_qty": bq[on500], "ask_qty": aq[on500]}
    m = on500.sum()
    for k in range(1, 12):
        scale = depth0 * mid[on500] * k
        b[f"bid_v{k:02d}"] = np.round(scale * np.exp(rng.normal(0, .3, m)) * (1 + 0.3 * imb[on500]), 2)
        b[f"ask_v{k:02d}"] = np.round(scale * np.exp(rng.normal(0, .3, m)) * (1 - 0.3 * imb[on500]), 2)
    write(b, f"{out}/book/{inst}/{day}.parquet")
    q = np.column_stack([bid, ask, bq, aq])
    changed = np.r_[True, (q[1:] != q[:-1]).any(1)]
    keep = changed & ~on500
    write({"ts": ts[keep], "bid_px": bid[keep], "ask_px": ask[keep],
           "bid_qty": bq[keep], "ask_qty": aq[keep]}, f"{out}/bbo/{inst}/{day}.parquet")
    lam = depth0 * 0.05
    buy = np.where(rng.random(per_day) < 0.3 * (1 + flow), rng.exponential(lam, per_day), 0)
    sell = np.where(rng.random(per_day) < 0.3 * (1 - flow), rng.exponential(lam, per_day), 0)
    buy, sell = np.round(buy, 3), np.round(sell, 3)
    k = (buy > 0) | (sell > 0)
    nb, ns = (buy[k] > 0).astype("int64"), (sell[k] > 0).astype("int64")
    last = rng.integers(0, 100, k.sum()).astype(float)
    write({"ts": ts[k], "buy_qty": buy[k], "sell_qty": sell[k],
           "buy_vwap": np.where(nb > 0, ask[k], np.nan),
           "sell_vwap": np.where(ns > 0, bid[k], np.nan),
           "buy_n": nb, "sell_n": ns,
           "buy_max_qty": np.where(nb > 0, buy[k], np.nan),
           "sell_max_qty": np.where(ns > 0, sell[k], np.nan),
           "buy_last_ms": np.where(nb > 0, last, np.nan),
           "sell_last_ms": np.where(ns > 0, last, np.nan)},
          f"{out}/trades/{inst}/{day}.parquet")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./fake_data")
    ap.add_argument("--start", default="2026-01-01")
    ap.add_argument("--days", type=int, default=5)
    ap.add_argument("--train-days", type=int, default=3,
                    help="по первым N дням считаются пороги обрезки таргета")
    a = ap.parse_args()
    rng = np.random.default_rng(42)

    import datetime as dt
    d0 = dt.date.fromisoformat(a.start)
    days = dates(a.start, (d0 + dt.timedelta(days=a.days - 1)).isoformat())
    per_day = 864_000
    n = per_day * (len(days) + 1)          # +1 день, чтобы посчитать таргеты в конце
    ts_all = np.concatenate([grid_points(d, 100) for d in days] +
                            [grid_points(days[-1], 100) + 86_400_000])

    btc_r = None
    clip_rows = {"instrument": [], "horizon": [], "lo": [], "hi": []}
    for inst in TARGET_INSTRUMENTS:
        px0, tick, depth0 = SPEC[inst]
        print("генерирую", inst)
        imb = np.tanh(ar1(rng, n, 0.97, 0.25))
        flow = np.tanh(ar1(rng, n, 0.9, 0.3))
        vol = 0.4 * np.exp(np.repeat(ar1(rng, n // 6000 + 1, 0.95, 0.15), 6000)[:n])
        r = vol * (rng.normal(0, 1, n) + 0.6 * np.roll(imb, 1) + 0.4 * np.roll(flow, 1))
        if btc_r is None:
            btc_r = r.copy()
        else:
            r += 0.5 * np.roll(btc_r, 2)    # BTC ведёт остальных на 200 мс
        fair = px0 * np.exp(np.cumsum(r) * 1e-4)
        bid = np.floor(fair / tick) * tick
        ask = bid + tick
        D = depth0 * np.exp(rng.normal(0, 0.3, n))
        hold = rng.random(n) < 0.6          # чаще всего объёмы не меняются
        D = np.where(hold, np.roll(D, 1), D)
        bq = np.round(D * (1 + imb) / 2 + 1e-3, 3)
        aq = np.round(D * (1 - imb) / 2 + 1e-3, 3)
        mid = (bid + ask) / 2
        basis = ar1(rng, n, 0.999, 0.05)        # медленно гуляющий базис перп−спот, б.п.
        gap2 = ar1(rng, n, 0.995, 0.08)         # разрыв цен одного актива в разных стейблкоинах, б.п.

        # targets на сетке 500 мс: ln(mid(t+H)/mid(t))
        lm = np.log(mid)
        tgt = {}
        for h, k in H_BARS.items():
            y = np.full(n, np.nan)
            y[:-k] = lm[k:] - lm[:-k]
            tgt[h] = y

        for di, day in enumerate(days):
            s = slice(di * per_day, (di + 1) * per_day)
            ts = ts_all[s]
            on500 = ts % 500 == 0
            write_market(a.out, inst, day, ts, on500, bid[s], ask[s], bq[s], aq[s], mid[s],
                         imb[s], flow[s], depth0, rng)
            sname, mult, stick = SPOT[inst]                 # спот: та же цена с медленным базисом
            sf = fair[s] * mult * np.exp(basis[s] * 1e-4)
            sb = np.floor(sf / stick) * stick
            sa = sb + stick
            write_market(a.out, sname, day, ts, on500, sb, sa, bq[s], aq[s], (sb + sa) / 2,
                         imb[s], flow[s], depth0, rng)
            for aname, atick in ALT[inst]:                   # тот же актив, свой небольшой разрыв цены
                af = (sf / mult if aname.endswith("_PERP") else sf) * np.exp(gap2[s] * 1e-4)
                ab = np.floor(af / atick) * atick
                write_market(a.out, aname, day, ts, on500, ab, ab + atick, bq[s] * 0.3, aq[s] * 0.3,
                             ab + atick / 2, imb[s], flow[s], depth0 * 0.3, rng)
            # --- targets
            write({"ts": ts[on500], **{h: tgt[h][s][on500] for h in HORIZONS}},
                  f"{a.out}/targets/{inst}/{day}.parquet")

        tr = slice(0, a.train_days * per_day)
        for h in HORIZONS:
            y = tgt[h][tr][ts_all[tr] % 500 == 0]
            lo, hi = np.nanpercentile(y, [0.1, 99.9])
            clip_rows["instrument"].append(inst)
            clip_rows["horizon"].append(h)
            clip_rows["lo"].append(float(lo))
            clip_rows["hi"].append(float(hi))

    # --- stable (раз в секунду, 3 пары)
    for day in days:
        g = grid_points(day, 1000)
        rows = {k: [] for k in ("ts", "pair", "bid_px", "ask_px", "bid_qty",
                                "ask_qty", "buy_qty", "sell_qty", "stale_ms")}
        for pair in ("USDCUSDT", "FDUSDUSDT", "USD1USDT"):
            px = 1 + rng.normal(0, 1e-4, len(g))
            rows["ts"].append(g)
            rows["pair"].append(np.full(len(g), pair))
            rows["bid_px"].append(px - 5e-5)
            rows["ask_px"].append(px + 5e-5)
            for c in ("bid_qty", "ask_qty", "buy_qty", "sell_qty"):
                rows[c].append(np.round(rng.exponential(1e5, len(g)), 2))
            rows["stale_ms"].append(np.zeros(len(g), dtype="int64"))
        write({k: np.concatenate(v) for k, v in rows.items()}, f"{a.out}/stable/{day}.parquet")

    write({k: np.array(v) for k, v in clip_rows.items()}, f"{a.out}/meta/target_clip.parquet")
    print(f"готово: {a.out}  (дни {days[0]} .. {days[-1]})")


if __name__ == "__main__":
    from textio import utf8_console
    utf8_console()
    main()
