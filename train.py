"""Обучение линейной модели (Ridge) на признаках из features.py и оценка на валидации.

Отдельная модель на каждую из 12 пар (инструмент × горизонт).
Данные не влезают в память, поэтому обучение идёт "потоком": по одному дню
копим суммы X'X и X'y, а в конце решаем систему уравнений.

Примеры:
  # повторить baseline (только дисбаланс L1) — для сравнения
  python train.py --data-dir ./data --features imb1 --out model_baseline.json

  # все признаки, каждый 5-й день трейна (быстро), оценка на валидации
  python train.py --data-dir ./data --day-step 5 --out model.json

Даты по умолчанию — из условия: трейн 2025-09-15..2026-07-14, валид 2026-07-15..2026-09-14.
"""
import argparse
import datetime as dt
import json
import time

import numpy as np

from common import HORIZONS, TARGET_INSTRUMENTS, dates, load_clip
from features import FEATURE_NAMES, compute_features
from fversion import FEATURES_HASH, FHASH
from metric import Scorer, day_corr, load_targets

# День, где дисперсия таргета меньше этого (std < 0.01 б.п. в лог-доходности), считаем вырожденным
# (цена почти не двигалась или данных нет) и в обучение не берём: иначе вес 1/var взлетает до небес.
MIN_DAY_VAR = 1e-12
DELAY_MS, MAX_H_MS = 10, 8_000


def purge_mask(day, rows, valid_start):
    """Маска строк дня, которые можно брать в обучение.

    Если день — последний перед началом валидации, его последние точки имеют таргет,
    который заканчивается уже в валидации (t + 10 мс + 8 с). Такие строки выкидываем.
    rows — номера точек 500-мс сетки дня."""
    nxt = (dt.date.fromisoformat(day) + dt.timedelta(days=1)).isoformat()
    keep = np.ones(len(rows), bool)
    if nxt == valid_start:
        keep = np.asarray(rows) * 500 + DELAY_MS + MAX_H_MS < 86_400_000
    return keep


class Accumulator:
    """Взвешенные суммы для линейной регрессии: n, Σx, Σxx', Σy, Σxy."""

    def __init__(self, k):
        self.n = 0.0
        self.sx = np.zeros(k)
        self.sxx = np.zeros((k, k))
        self.sy = 0.0
        self.sxy = np.zeros(k)

    def add(self, X, y, w):
        self.n += w * len(y)
        self.sx += w * X.sum(0)
        self.sxx += w * (X.T @ X)
        self.sy += w * y.sum()
        self.sxy += w * (X.T @ y)

    def solve(self, lam):
        """Ridge на стандартизованных признаках; возвращает (a, b) в исходных единицах.

        lam — сила регуляризации (0 = обычный OLS). Масштаб: доля от диагонали
        корреляционной матрицы, т.е. lam=0.01 — "мягко", lam=1 — "сильно".
        """
        n = self.n
        if not n > 0:
            raise ValueError("нет наблюдений для обучения")
        mx, my = self.sx / n, self.sy / n
        C = self.sxx / n - np.outer(mx, mx)
        cxy = self.sxy / n - mx * my
        sd = np.sqrt(np.clip(np.diag(C), 0, None))
        sd[sd == 0] = 1.0
        R = C / np.outer(sd, sd)
        r = cxy / sd
        bz = np.linalg.solve(R + lam * np.eye(len(r)), r)
        b = bz / sd
        a = my - mx @ b
        return float(a), b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--train", nargs=2, default=["2025-09-15", "2026-07-14"])
    ap.add_argument("--valid", nargs=2, default=["2026-07-15", "2026-09-14"])
    ap.add_argument("--day-step", type=int, default=1,
                    help="брать каждый N-й день трейна (для скорости)")
    ap.add_argument("--valid-step", type=int, default=1,
                    help="брать каждый N-й день валидации")
    ap.add_argument("--features", nargs="*", default=None,
                    help="подмножество признаков (по умолчанию все)")
    ap.add_argument("--lam", type=float, default=0.01, help="сила Ridge")
    ap.add_argument("--no-equal-days", action="store_true",
                    help="не выравнивать вес дней (по умолчанию каждый день весит одинаково)")
    ap.add_argument("--out", default="model.json")
    a = ap.parse_args()

    feats = a.features or FEATURE_NAMES
    cols = [FEATURE_NAMES.index(f) for f in feats]
    clip = load_clip(a.data_dir)
    train_days = dates(*a.train)[::a.day_step]
    valid_days = dates(*a.valid)[::a.valid_step]
    print(f"признаков: {len(feats)} | дней трейна: {len(train_days)} | "
          f"дней валидации: {len(valid_days)}")

    model = {"features": feats, "lam": a.lam, "train": a.train,
             "day_step": a.day_step, "coef": {}, "precision": "float64",
             "fhash": FHASH, "features_hash": FEATURES_HASH}      # predict.py сверит версию признаков

    # ---------------------------------------------------------- обучение
    for inst in TARGET_INSTRUMENTS:
        acc = {h: Accumulator(len(cols)) for h in HORIZONS}
        t0 = time.time()
        for date in train_days:
            ys = load_targets(a.data_dir, inst, date, clip)
            if ys is None:
                continue
            X = compute_features(a.data_dir, inst, date)[:, cols]
            keep = purge_mask(date, np.arange(len(X)), a.valid[0])   # таргеты не залезают в валидацию
            for h in HORIZONS:
                m = np.isfinite(ys[h]) & keep
                if m.sum() < 1000:
                    continue
                y = ys[h][m]
                v = float(y.var())
                if not np.isfinite(v) or v < MIN_DAY_VAR:          # вырожденный день — пропускаем
                    continue
                # вес дня: 1/дисперсия таргета -> тихие и бурные дни весят примерно одинаково.
                # Эвристика в духе метрики (она усредняет корреляции по дням), а не её прямая оптимизация.
                w = 1.0 if a.no_equal_days else 1.0 / v
                acc[h].add(X[m], y, w)
        model["coef"][inst] = {}
        for h in HORIZONS:
            if not acc[h].n > 0:
                print(f"  {inst} {h}: нет данных")
                continue
            ai, b = acc[h].solve(a.lam)
            model["coef"][inst][h] = {"a": ai, "b": b.tolist()}
        print(f"  обучено {inst} за {time.time() - t0:.0f} с")

    with open(a.out, "w") as f:
        json.dump(model, f, indent=1)
    print(f"модель сохранена: {a.out}")

    # ---------------------------------------------------------- валидация
    sc = Scorer()
    for inst in TARGET_INSTRUMENTS:
        for date in valid_days:
            ys = load_targets(a.data_dir, inst, date, clip)
            if ys is None:
                continue
            X = compute_features(a.data_dir, inst, date)[:, cols]
            for h in HORIZONS:
                c = model["coef"][inst].get(h)
                if c is None:
                    continue
                pred = c["a"] + X @ np.array(c["b"])
                sc.add(inst, h, day_corr(pred, ys[h]))
    text, _ = sc.table("\nВАЛИДАЦИЯ")
    print(text)

    # для интереса: коэффициенты одной из 12 моделей
    c = model["coef"].get("BTCUSDT_PERP", {}).get("y_500ms")
    if c:
        print("\nкоэффициенты BTCUSDT_PERP / y_500ms:")
        for n, v in zip(feats, c["b"]):
            print(f"  {n:20} {v:+.3e}")


if __name__ == "__main__":
    from textio import utf8_console
    utf8_console()
    main()
