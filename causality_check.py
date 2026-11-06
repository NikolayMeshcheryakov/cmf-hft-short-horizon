"""Проверка на заглядывание в будущее — копия проверки организаторов.

Что делает:
  1. Считает прогноз на полном дне.
  2. Несколько раз обрезает данные этого дня в случайный момент T
     (оставляет только строки с ts <= T), снова считает прогноз.
  3. Прогнозы для всех t <= T обязаны совпасть ТОЧНО. Иначе — дисквалификация.

    python causality_check.py --data-dir ./data --date 2026-07-20 --trials 3

Запускать после каждого нового признака.
Временные файлы кладутся рядом с папкой данных (_causality_tmp) и удаляются.
"""
import argparse
import os
import shutil

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from common import HORIZONS, TARGET_INSTRUMENTS, grid_points
from features import prev_date
from features import FEATURE_NAMES, compute_features
from features_v2 import V2_NAMES, compute_v2
from features_v3 import V3_NAMES, compute_v3
from predict import DEFAULT_MODEL, load_model, predict_day

TABLES = ["book", "bbo", "trades"]       # по папке на инструмент
# targets/ не копируем: у организаторов его тоже нет. meta/ прогнозу не нужна.


def link_or_copy(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        os.link(src, dst)                # жёсткая ссылка: мгновенно и без места
    except OSError:
        shutil.copy2(src, dst)


def truncate(src, dst, T):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    t = pq.read_table(src)
    mask = t["ts"].to_numpy() <= T
    pq.write_table(t.filter(pa.array(mask)), dst, compression="zstd")


def build_truncated(root, tmp, date, T):
    prev = prev_date(date)
    for table in TABLES:
        base = os.path.join(root, table)
        if not os.path.isdir(base):
            continue
        for inst in os.listdir(base):
            for d, cut in ((prev, False), (date, True)):
                src = os.path.join(base, inst, f"{d}.parquet")
                if not os.path.isfile(src):
                    continue
                dst = os.path.join(tmp, table, inst, f"{d}.parquet")
                truncate(src, dst, T) if cut else link_or_copy(src, dst)
    for d, cut in ((prev, False), (date, True)):
        src = os.path.join(root, "stable", f"{d}.parquet")
        if os.path.isfile(src):
            dst = os.path.join(tmp, "stable", f"{d}.parquet")
            truncate(src, dst, T) if cut else link_or_copy(src, dst)


def all_features(root, date):
    """{инструмент: матрица 172_800 × (21 + 39 + 30)} — все признаки base + v2 + v3."""
    cache, out = {}, {}
    for inst in TARGET_INSTRUMENTS:
        out[inst] = np.hstack([compute_features(root, inst, date, cache), compute_v2(root, inst, date, cache),
                               compute_v3(root, inst, date, cache)])
    return out


def feature_markets():
    """Все рынки, которые читают признаки: 4 перпа, споты к USDT, рынки в других стейблкоинах."""
    from features_v3 import ALT
    out = list(TARGET_INSTRUMENTS) + ["BTCUSDT_SPOT", "ETHUSDT_SPOT", "SOLUSDT_SPOT", "PEPEUSDT_SPOT"]
    for alts in ALT.values():
        out += [v for v in alts.values() if v and v not in out]
    return out


def special_times(root, date, grid, rng=None, n_market=8):
    """Опасные моменты обрезки: сразу после полуночи, ровно на точке сетки и за 1 мс до неё,
    на границе 100 мс и секунды (обновление курса stable), а также за 1 мс до изменения котировки
    и ровно на нём — у РАЗНЫХ рынков (до n_market штук из всех рынков признаков, выбор по rng)
    и у курса stable."""
    t0 = int(grid[0])
    ts = [t0 + 1, t0 + 1_000, t0 + 60_000, int(grid[7200]), int(grid[7200]) - 1, int(grid[7200]) + 99,
          t0 + 3_600_000 + 999, t0 + 3_600_000 + 1_000]
    rng = rng or np.random.default_rng(0)
    cand = []
    for k, inst in enumerate(feature_markets()):
        p = os.path.join(root, "bbo", inst, f"{date}.parquet")
        if not os.path.isfile(p):
            continue
        b = pq.read_table(p, columns=["ts"])["ts"].to_numpy()
        b = b[(b > t0 + 60_000) & (b < t0 + 86_340_000)]
        if len(b) > 10:
            c = int(b[int(rng.integers(0, len(b)))])
            cand.append((inst, c))
    sp = os.path.join(root, "stable", f"{date}.parquet")
    if os.path.isfile(sp):
        s = pq.read_table(sp, columns=["ts"])["ts"].to_numpy()
        s = s[(s > t0 + 60_000) & (s < t0 + 86_340_000)]
        if len(s):
            cand.append(("stable", int(s[int(rng.integers(0, len(s)))])))
    pick = sorted(rng.choice(len(cand), min(n_market, len(cand)), replace=False)) if cand else []
    for i in pick:
        ts += [cand[i][1] - 1, cand[i][1]]
    return ts


def check_features(a, grid, rng):
    """Причинность КАЖДОГО признака: значения для t <= T на обрезанных данных обязаны точно
    совпасть с полными. Это строже проверки прогнозов: признак с нулевым весом в модели
    прогноз не испортит, но смотреть вперёд всё равно не должен."""
    names = FEATURE_NAMES + V2_NAMES + V3_NAMES
    full = all_features(a.data_dir, a.date)
    tmp = os.path.join(os.path.dirname(os.path.abspath(a.data_dir)), "_causality_tmp")
    Ts = special_times(a.data_dir, a.date, grid, rng) + [int(grid[0] + rng.integers(60_000, 86_400_000))
                                                          for _ in range(a.trials)]
    ok = True
    for T in Ts:
        shutil.rmtree(tmp, ignore_errors=True)
        try:
            build_truncated(a.data_dir, tmp, a.date, T)
            cut = all_features(tmp, a.date)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        m = grid <= T
        bad = []
        for inst in TARGET_INSTRUMENTS:
            x, y = full[inst][m], cut[inst][m]
            diff = ~((x == y) | (np.isnan(x) & np.isnan(y)))
            for j in np.flatnonzero(diff.any(0)):
                bad.append(f"{inst}:{names[j]} (первая точка №{int(np.argmax(diff[:, j]))})")
        ok &= not bad
        print(f"признаки, T = {(T - grid[0]) / 3_600_000:7.4f} ч, точек до T: {m.sum():,} -> "
              f"{'OK' if not bad else 'ОШИБКА: ' + ', '.join(bad[:8])}")
        del cut                                    # не держим две обрезанные копии разом
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--date", required=True)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--special", action="store_true",
                    help="для прогнозов модели: кроме случайных T ещё и опасные моменты (как в --features)")
    ap.add_argument("--special-markets", type=int, default=8,
                    help="--special: у скольких рынков взять момент изменения котировки")
    ap.add_argument("--features", action="store_true",
                    help="проверять каждый признак (base + v2 + v3), а не только прогнозы модели")
    a = ap.parse_args()
    if a.features:
        grid = grid_points(a.date)
        ok = check_features(a, grid, np.random.default_rng(a.seed))
        print("\nИТОГ:", "OK, утечек не найдено" if ok else "ЕСТЬ РАСХОЖДЕНИЯ — какой-то признак смотрит в будущее!")
        if not ok:
            raise SystemExit(1)
        return

    model = load_model(a.model)
    rng = np.random.default_rng(a.seed)
    grid = grid_points(a.date)
    full = predict_day(a.data_dir, a.date, model)
    tmp = os.path.join(os.path.dirname(os.path.abspath(a.data_dir)), "_causality_tmp")

    all_ok = True
    Ts = [int(grid[0] + rng.integers(60_000, 86_400_000)) for _ in range(a.trials)]   # случайные T до мс
    if a.special:
        Ts += special_times(a.data_dir, a.date, grid, rng, a.special_markets)
    for k, T in enumerate(Ts):
        shutil.rmtree(tmp, ignore_errors=True)
        try:
            build_truncated(a.data_dir, tmp, a.date, T)
            cut = predict_day(tmp, a.date, model)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        m = grid <= T
        worst = 0.0
        for inst in TARGET_INSTRUMENTS:
            for h in HORIZONS:
                x, y = full[inst][h][m], cut[inst][h][m]
                if not np.array_equal(x, y):
                    all_ok = False
                    d = np.abs(x - y)
                    first = int(np.argmax(d > 0))
                    worst = max(worst, float(d.max()))
                    print(f"  НЕ СОВПАЛО: {inst} {h}, первая точка расхождения "
                          f"№{first} (ts={grid[m][first]}), max |diff|={d.max():.3e}")
        hhmm = (T - grid[0]) / 3_600_000
        print(f"проба {k + 1}: T = {hhmm:5.2f} ч от начала дня, "
              f"точек до T: {m.sum():,} -> {'OK' if worst == 0 else 'ОШИБКА'}")

    print("\nИТОГ:", "OK, утечек не найдено" if all_ok else
          "ЕСТЬ РАСХОЖДЕНИЯ — какой-то признак смотрит в будущее!")
    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    from textio import utf8_console
    utf8_console()
    main()
