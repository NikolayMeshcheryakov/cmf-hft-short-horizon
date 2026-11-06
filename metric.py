"""Метрика ровно как у организаторов.

1. Таргет обрезается по порогам из meta/target_clip.parquet.
2. Пирсон считается отдельно для каждого (инструмент, горизонт) и КАЖДОГО дня.
   Пропущенный прогноз (NaN или нет строки) = 0.
3. Среднее по дням, потом среднее по 12 комбинациям.

Оценить папку с прогнозами predict.py (файлы <дата>.parquet):
    python metric.py --data-dir ./data --pred-dir ./preds --dates 2026-07-15 2026-09-14
"""
import argparse
import os

import numpy as np
import pyarrow.parquet as pq

from common import (HORIZONS, PREDICTION_COLUMNS, TARGET_INSTRUMENTS, dates,
                    grid_points, load_clip, read_day)


def load_targets(root, inst, date, clip):
    """Таргеты дня на полной сетке 500 мс (NaN, где таргета нет), уже обрезанные."""
    g = grid_points(date)
    t = read_day(root, "targets", date, inst, columns=["ts"] + HORIZONS)
    if t is None:
        return None
    ts = t["ts"].to_numpy()
    i = np.searchsorted(g, ts)
    ok = (i < len(g)) & (g[np.minimum(i, len(g) - 1)] == ts)
    out = {}
    for h in HORIZONS:
        y = np.full(len(g), np.nan)
        y[i[ok]] = t[h].to_numpy().astype("float64")[ok]
        lo, hi = clip[(inst, h)]
        out[h] = np.clip(y, lo, hi)
    return out


def day_corr(pred, y):
    """Пирсон за один день. pred: пропуски -> 0. Берём точки, где таргет есть."""
    pred = np.nan_to_num(np.asarray(pred, dtype="float64"), nan=0.0)
    m = np.isfinite(y)
    p, t = pred[m], y[m]
    if len(t) < 2 or p.std() == 0 or t.std() == 0:
        return 0.0
    return float(np.corrcoef(p, t)[0, 1])


class Scorer:
    """Копит дневные корреляции и печатает итоговую таблицу."""

    def __init__(self):
        self.rho = {(i, h): [] for i in TARGET_INSTRUMENTS for h in HORIZONS}

    def add(self, inst, h, value):
        self.rho[(inst, h)].append(value)

    def table(self, title=""):
        lines = [title] if title else []
        lines.append(f"{'instrument':20} {'horizon':9} {'corr':>8} {'SE':>8} {'days':>5}")
        cells, ndays = [], set()
        for (inst, h), v in self.rho.items():
            if not v:
                continue
            v = np.array(v)
            se = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else float("nan")
            cells.append(v.mean())
            ndays.add(len(v))
            lines.append(f"{inst:20} {h:9} {v.mean():8.4f} {se:8.4f} {len(v):5d}")
        total = float(np.mean(cells)) if len(cells) == 12 else float("nan")
        lines.append(f"\nSCORE (среднее по 12 ячейкам): {total:.4f}")
        if len(cells) != 12 or len(ndays) > 1:
            lines.append(f"ВНИМАНИЕ: покрытие неполное (ячеек {len(cells)} из 12, дней по ячейкам "
                         f"{sorted(ndays)}) — это не полноценный SCORE")
        return "\n".join(lines), total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--dates", nargs=2, required=True, metavar=("START", "END"))
    a = ap.parse_args()

    clip = load_clip(a.data_dir)
    sc = Scorer()
    missing = []
    for date in dates(*a.dates):
        path = os.path.join(a.pred_dir, f"{date}.parquet")
        P = pq.read_table(path) if os.path.isfile(path) else None
        if P is None:
            missing.append(date)                     # нет файла = все прогнозы дня равны 0
        g = grid_points(date)
        for inst in TARGET_INSTRUMENTS:
            ys = load_targets(a.data_dir, inst, date, clip)
            if ys is None:
                continue
            for col, h in zip(PREDICTION_COLUMNS, HORIZONS):
                pred = np.zeros(len(g))              # нет прогноза -> 0
                if P is not None:
                    p_ts, p_inst = P["ts"].to_numpy(), P["instrument"].to_numpy().astype(str)
                    m = p_inst == inst
                    i = np.searchsorted(g, p_ts[m])
                    ok = (i < len(g)) & (g[np.minimum(i, len(g) - 1)] == p_ts[m])
                    pred[i[ok]] = P[col].to_numpy()[m][ok]
                sc.add(inst, h, day_corr(pred, ys[h]))
    print(sc.table()[0])
    if missing:
        print(f"ВНИМАНИЕ: нет файлов прогноза за {len(missing)} дн. ({', '.join(missing[:5])}…) — "
              f"посчитаны как нулевой прогноз (корреляция 0)")


if __name__ == "__main__":
    from textio import utf8_console
    utf8_console()
    main()
