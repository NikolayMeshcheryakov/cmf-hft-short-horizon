"""«Грязные» тестовые данные: берём чистый фейк (make_fake_data.py) и портим так, как бывает в жизни.

    python make_dirty_data.py --src ./fake_data --out ./fake_dirty

Что ломается (дни — по порядку дней в src: d0, d1, d2, d3, d4):
  BTC   d1  — сбой коллектора: 10 минут нет ни строк book, ни bbo
  BTC   d2  — в trades дубликаты строк и строки не на 100-мс сетке
  ETH   d1  — час со спредом 3 тика; несколько скрещенных котировок; нулевая цена; NaN в объёме
  ETH   d0  — первый час таргетов NaN
  SOL   d2  — файлов book и bbo нет совсем (таргеты есть)
  SOL   d1  — файла trades нет
  PEPE  d1  — 2 часа цена стоит (нулевая волатильность), bbo в это время пустой
  PEPE  d3  — файл trades есть, но пустой
  PEPE  d4  — нет файла таргетов
  stable d2 — файла нет
  BTC-спот  d2 — нет ни book, ни bbo, ни trades
  PEPE-спот d1 — час без строк book и bbo
  USD1-пары   d0, d1 — ещё не листинговались (файлов нет)
  USDC        d3 — 30 минут курс USDCUSDT 1.007 (> 50 б.п.), все пары к USDC в это время пустые (null)
  FDUSD       d4 — час курс FDUSDUSDT не обновляется (stale_ms растёт)
"""
import argparse
import os
import shutil

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def rd(p):
    return pq.read_table(p)


def wr(t, p):
    pq.write_table(t, p, compression="zstd")


def to_dict(t):
    return {c: t[c].to_numpy().copy() for c in t.column_names}


def from_dict(d):
    return pa.table(d)


def keep_rows(p, mask):
    t = rd(p)
    wr(t.filter(pa.array(mask)), p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if os.path.exists(a.out):
        shutil.rmtree(a.out)
    shutil.copytree(a.src, a.out)
    R = a.out
    days = sorted(f[:-8] for f in os.listdir(f"{R}/book/BTCUSDT_PERP"))
    assert len(days) >= 5, "нужно минимум 5 дней в src"
    d0, d1, d2, d3, d4 = days[:5]
    P = lambda t, i, d: f"{R}/{t}/{i}/{d}.parquet"

    # BTC d1: 10 минут без book и bbo (с 03:00 до 03:10)
    for tb in ("book", "bbo"):
        t = to_dict(rd(P(tb, "BTCUSDT_PERP", d1)))
        t0 = t["ts"].min() // 86_400_000 * 86_400_000
        keep_rows(P(tb, "BTCUSDT_PERP", d1), ~((t["ts"] >= t0 + 3 * 3_600_000) & (t["ts"] < t0 + 3 * 3_600_000 + 600_000)))

    # BTC d2: дубликаты и строки вне сетки в trades
    t = to_dict(rd(P("trades", "BTCUSDT_PERP", d2)))
    dup = {c: np.r_[v, v[:50]] for c, v in t.items()}
    dup["ts"] = np.r_[t["ts"], t["ts"][:50]]
    dup["ts"][-10:] += 37                                     # 10 строк не на сетке
    order = np.argsort(dup["ts"], kind="stable")
    wr(from_dict({c: v[order] for c, v in dup.items()}), P("trades", "BTCUSDT_PERP", d2))

    # ETH d1: спред 3 тика на час, скрещенные котировки, нулевая цена, NaN объём
    for tb in ("book", "bbo"):
        t = to_dict(rd(P(tb, "ETHUSDT_PERP", d1)))
        t0 = t["ts"].min() // 86_400_000 * 86_400_000
        hr = (t["ts"] >= t0 + 5 * 3_600_000) & (t["ts"] < t0 + 6 * 3_600_000)
        t["ask_px"] = np.where(hr, t["ask_px"] + 0.02, t["ask_px"])
        idx = np.flatnonzero(~hr)[1000:1010]
        t["ask_px"][idx[:5]] = t["bid_px"][idx[:5]] - 0.01      # скрещенный стакан
        t["bid_px"][idx[5:7]] = 0.0                             # нулевая цена
        t["bid_qty"] = t["bid_qty"].astype(float)
        t["bid_qty"][idx[7:]] = np.nan                          # NaN объём
        wr(from_dict(t), P(tb, "ETHUSDT_PERP", d1))

    # ETH d0: первый час таргетов NaN
    t = to_dict(rd(P("targets", "ETHUSDT_PERP", d0)))
    for h in ("y_500ms", "y_2s", "y_8s"):
        t[h][:7200] = np.nan
    wr(from_dict(t), P("targets", "ETHUSDT_PERP", d0))

    # SOL d2: нет book и bbo; SOL d1: нет trades
    os.remove(P("book", "SOLUSDT_PERP", d2))
    os.remove(P("bbo", "SOLUSDT_PERP", d2))
    os.remove(P("trades", "SOLUSDT_PERP", d1))

    # PEPE d1: 2 часа цена стоит (с 10:00 до 12:00), bbo в это время пустой
    t = to_dict(rd(P("book", "1000PEPEUSDT_PERP", d1)))
    t0 = t["ts"].min() // 86_400_000 * 86_400_000
    win = (t["ts"] >= t0 + 10 * 3_600_000) & (t["ts"] < t0 + 12 * 3_600_000)
    first = np.flatnonzero(win)[0]
    for c in ("bid_px", "ask_px", "bid_qty", "ask_qty"):
        t[c][win] = t[c][first]
    wr(from_dict(t), P("book", "1000PEPEUSDT_PERP", d1))
    b = to_dict(rd(P("bbo", "1000PEPEUSDT_PERP", d1)))
    keep_rows(P("bbo", "1000PEPEUSDT_PERP", d1),
              ~((b["ts"] >= t0 + 10 * 3_600_000) & (b["ts"] < t0 + 12 * 3_600_000)))

    # PEPE d3: пустой trades; PEPE d4: нет таргетов
    t = rd(P("trades", "1000PEPEUSDT_PERP", d3))
    wr(t.filter(pa.array(np.zeros(t.num_rows, bool))), P("trades", "1000PEPEUSDT_PERP", d3))
    os.remove(P("targets", "1000PEPEUSDT_PERP", d4))

    # спот: BTC d2 нет совсем; PEPE d1 — час без book/bbo (с 07:00 до 08:00)
    for tb in ("book", "bbo", "trades"):
        p = P(tb, "BTCUSDT_SPOT", d2)
        if os.path.isfile(p):
            os.remove(p)
    for tb in ("book", "bbo"):
        p = P(tb, "PEPEUSDT_SPOT", d1)
        if os.path.isfile(p):
            t = to_dict(rd(p))
            t0 = t["ts"].min() // 86_400_000 * 86_400_000
            keep_rows(p, ~((t["ts"] >= t0 + 7 * 3_600_000) & (t["ts"] < t0 + 8 * 3_600_000)))

    # USD1 d0, d1: пар ещё нет
    for inst in ("BTCUSD1_SPOT", "ETHUSD1_SPOT"):
        for day in (d0, d1):
            for tb in ("book", "bbo", "trades"):
                p = P(tb, inst, day)
                if os.path.isfile(p):
                    os.remove(p)

    # USDC d3: отклонение курса > 50 б.п. на 30 минут (06:00–06:30) — пары к USDC обнулены
    sp3 = f"{R}/stable/{d3}.parquet"
    if os.path.isfile(sp3):
        t = to_dict(rd(sp3))
        t0 = t["ts"].min() // 86_400_000 * 86_400_000
        win = (t["ts"] >= t0 + 6 * 3_600_000) & (t["ts"] < t0 + 6 * 3_600_000 + 1_800_000)
        pr = np.asarray(t["pair"]).astype(str)
        for col, v in (("bid_px", 1.0069), ("ask_px", 1.0071)):
            t[col] = np.where(win & (pr == "USDCUSDT"), v, t[col])
        wr(from_dict(t), sp3)
        for inst in ("BTCUSDC_SPOT", "ETHUSDC_SPOT", "SOLUSDC_SPOT", "1000PEPEUSDC_PERP"):
            for tb in ("book", "bbo"):
                p = P(tb, inst, d3)
                if not os.path.isfile(p):
                    continue
                q = to_dict(rd(p))
                w = (q["ts"] >= t0 + 6 * 3_600_000) & (q["ts"] < t0 + 6 * 3_600_000 + 1_800_000)
                for col in ("bid_px", "ask_px", "bid_qty", "ask_qty"):
                    q[col] = np.where(w, np.nan, q[col].astype(float))
                wr(from_dict(q), p)

    # FDUSD d4: час без обновления курса (09:00–10:00): цена стоит, stale_ms растёт
    sp4 = f"{R}/stable/{d4}.parquet"
    if os.path.isfile(sp4):
        t = to_dict(rd(sp4))
        t0 = t["ts"].min() // 86_400_000 * 86_400_000
        pr = np.asarray(t["pair"]).astype(str)
        win = (t["ts"] >= t0 + 9 * 3_600_000) & (t["ts"] < t0 + 10 * 3_600_000) & (pr == "FDUSDUSDT")
        if win.any():
            first = np.flatnonzero(win)[0]
            for col in ("bid_px", "ask_px"):
                t[col] = np.where(win, t[col][first], t[col])
            t["stale_ms"] = np.where(win, t["ts"] - t["ts"][first], t["stale_ms"]).astype("int64")
            wr(from_dict(t), sp4)

    # stable d2: нет файла
    sp = f"{R}/stable/{d2}.parquet"
    if os.path.isfile(sp):
        os.remove(sp)
    print(f"грязные данные: {R}  (дни {d0} … {d4})")


if __name__ == "__main__":
    from textio import utf8_console
    utf8_console()
    main()
