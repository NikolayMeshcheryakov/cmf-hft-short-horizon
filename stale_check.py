"""Как часто рынки «молчат»: доля времени без свежей котировки (последняя валидная строка book/bbo
старше 1 с) и самая длинная тишина. Нужно, чтобы решить, насколько важна проверка свежести
для признаков на споте и соседних рынках.

    py -3.13 stale_check.py --data-dir D:\\hft\\data --days 12

Выход: таблица по рынкам (медиана и худший день) + stale_check.csv рядом.
"""
import argparse
import os

import numpy as np

from common import TARGET_INSTRUMENTS, dates
from features import BARS_PER_DAY
from features_v3 import ALT, quotes_raw_two_days

SPOTS = ["BTCUSDT_SPOT", "ETHUSDT_SPOT", "SOLUSDT_SPOT", "PEPEUSDT_SPOT"]


def longest_false(m):
    if m.all():
        return 0
    x = np.r_[1, m.astype(int), 1]
    d = np.diff(x)
    starts, ends = np.flatnonzero(d == -1), np.flatnonzero(d == 1)
    return int((ends - starts).max()) if len(starts) else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--days", type=int, default=12)
    ap.add_argument("--start", default="2025-09-16")
    ap.add_argument("--end", default="2026-09-14")
    ap.add_argument("--out", default="stale_check.csv")
    a = ap.parse_args()
    all_days = dates(a.start, a.end)
    days = [all_days[int(i)] for i in np.linspace(0, len(all_days) - 1, a.days)]
    markets = list(TARGET_INSTRUMENTS) + SPOTS + sorted({v for s in ALT.values() for k, v in s.items()
                                                          if v and k != "ref" and v not in TARGET_INSTRUMENTS})
    rows = []
    for d in days:
        for m in markets:
            fresh = quotes_raw_two_days(a.data_dir, m, d)[4][BARS_PER_DAY:]      # только сегодня
            rows.append({"market": m, "day": d, "stale_share": float(1 - fresh.mean()),
                         "longest_stale_s": longest_false(fresh) / 10})
        print(f"  {d}: готово")
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("market,day,stale_share,longest_stale_s\n")
        for r in rows:
            f.write(f"{r['market']},{r['day']},{r['stale_share']:.6f},{r['longest_stale_s']:.1f}\n")
    print(f"\n{'рынок':20} {'доля без свежей котировки':>28} {'худший день':>12} {'макс. тишина, с':>16}")
    for m in markets:
        rr = [r for r in rows if r["market"] == m]
        sh = np.array([r["stale_share"] for r in rr])
        print(f"{m:20} {np.median(sh):28.4%} {sh.max():12.4%} {max(r['longest_stale_s'] for r in rr):16.1f}")
    print(f"\nпо дням — {os.path.abspath(a.out)}")


if __name__ == "__main__":
    from textio import utf8_console
    utf8_console()
    main()
