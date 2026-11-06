"""SCORE моделей на днях ТРЕЙНА (то, на чём модели учились) — для таблицы «трейн / валидация».

Берёт признаки и таргеты из кэша (дни трейна там прорежены: каждая 5-я точка 500-мс сетки), считает прогноз
сохранённых моделей (model.json — одиночных и ансамблей) и метрику задания: корреляция Пирсона по дню для
каждой пары «инструмент × горизонт» (таргет обрезан по перцентилям из meta), среднее по дням, затем по 12 парам.

  py -3.13 report.py --cache-dir D:\\hft\\cache --data-dir D:\\hft\\data --start 2025-09-15 --end 2026-07-14 ^
      --day-step 3 --model val15_lgbm=models\\val15_lgbm\\model.json --model ens15=models\\ens15_val\\model.json
"""
import argparse
import os

import numpy as np

import predict as pr
from cache import cached_names, load_day
from common import HORIZONS, TARGET_INSTRUMENTS, dates, load_clip
from experiment import clip_y
from metric import day_corr


def member_pred(sub, inst, X, names, n, gap):
    """{горизонт: прогноз} одной (не ансамбль) модели; GRU — с учётом шага строк (прореженные дни трейна)."""
    if sub.get("type") != "gru":
        return pr.single_predict(sub, inst, X, names, n)
    k = sub["inst_index"].get(inst)
    if k is None:
        return {h: np.zeros(n) for h in HORIZONS}
    Xf = X[:, [names.index(f) for f in sub["features"]]].astype(np.float32)
    P = sub["net"].predict(Xf, k, row_gap=gap)
    return {h: np.nan_to_num(P[:, j]) for j, h in enumerate(HORIZONS)}


def model_pred(m, inst, X, names, n, gap, memo):
    """memo: {имя модели: прогноз} — участник ансамбля, уже посчитанный отдельно, не считается заново."""
    if m.get("type") != "ensemble":
        key = m.get("name")
        if key is None:
            return member_pred(m, inst, X, names, n, gap)
        if key not in memo:
            memo[key] = member_pred(m, inst, X, names, n, gap)
        return memo[key]
    out = {h: np.zeros(n) for h in HORIZONS}
    for mb, sub in zip(m["members"], m["loaded"]):
        ps = model_pred(sub, inst, X, names, n, gap, memo)
        for h in HORIZONS:
            c = mb["coef"].get(inst, {}).get(h, 0.0)
            if c:
                out[h] = out[h] + c * ps[h]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--block", default="base+v2+v3")
    ap.add_argument("--start", default="2025-09-15")
    ap.add_argument("--end", default="2026-07-14")
    ap.add_argument("--day-step", type=int, default=3, help="каждый N-й день (быстрее; 3 — ~100 дней трейна)")
    ap.add_argument("--model", action="append", required=True, help="имя=путь к model.json (можно много раз)")
    ap.add_argument("--out", default=os.path.join("results", "train_scores.csv"))
    a = ap.parse_args()

    models = {}
    for spec in a.model:
        nm, path = spec.split("=", 1)
        models[nm] = pr.load_model(path)
    names = cached_names(a.cache_dir, a.block)
    clip = load_clip(a.data_dir)
    days = dates(a.start, a.end)[::a.day_step]
    corr = {nm: {} for nm in models}                # nm -> (inst, h) -> [дневные корреляции]
    for i, d in enumerate(days):
        for inst in TARGET_INSTRUMENTS:
            r = load_day(a.cache_dir, a.block, inst, d)
            if r is None:
                continue
            X, Y, rws = r
            Y = clip_y(Y, inst, clip)
            gap = int(rws[1] - rws[0]) if len(rws) > 1 else 5
            X = np.asarray(X, np.float64)
            memo = {}
            for nm, m in models.items():
                P = model_pred(m, inst, X, names, len(X), gap, memo)
                for j, h in enumerate(HORIZONS):
                    c = day_corr(P[h], Y[:, j])
                    if np.isfinite(c):
                        corr[nm].setdefault((inst, h), []).append(c)
        print(f"  день {i + 1}/{len(days)} {d}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("model,score," + ",".join(f"{i}|{h}" for i in TARGET_INSTRUMENTS for h in HORIZONS) + "\n")
        print(f"\nSCORE на трейне {a.start}..{a.end} (каждый {a.day_step}-й день, точки кэша; модели этот период видели)")
        for nm in models:
            cells = [float(np.mean(corr[nm][(i, h)])) if corr[nm].get((i, h)) else np.nan
                     for i in TARGET_INSTRUMENTS for h in HORIZONS]
            score = float(np.nanmean(cells))
            print(f"ТРЕЙН {nm:16} {score:.4f}")
            f.write(f"{nm},{score:.5f}," + ",".join(f"{c:.5f}" for c in cells) + "\n")
    print(f"записано: {a.out}")


if __name__ == "__main__":
    from textio import utf8_console
    utf8_console()
    main()
