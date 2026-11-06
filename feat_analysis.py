"""Анализ признаков по кэшу: какие дублируют друг друга и что каждый добавляет сверх imb1.

    py -3.13 feat_analysis.py --cache-dir D:\\hft\\cache --data-dir D:\\hft\\data --out D:\\hft\\feat_out

Смотрим ТОЛЬКО дни трейна (по умолчанию каждый 7-й): валидация остаётся честной проверкой.

1. Сходство признаков: корреляция каждой пары признаков, посчитанная внутри дня и усреднённая
   по дням (как и метрика, всё «по дням»). Группы почти одинаковых признаков (|r| ≥ --thr,
   в среднем по инструментам).
2. Что признак добавляет сверх imb1: внутри каждого дня убираем из признака часть, линейно
   повторяющую imb1 (остаток регрессии на imb1), и считаем корреляцию остатка с таргетом
   (функцией метрики). Около нуля → признак почти дубль imb1. Регрессия строится по тому же дню —
   для анализа это допустимо, для модели — нет.
3. Готовые команды experiment.py --drop для проверки групп выкидыванием.
4. Горизонт 8 с: ранняя и поздняя часть. y_8s = y_500ms + (y_8s − y_500ms). Ковариация признака
   с y_8s в точности раскладывается на ковариацию с первыми 0.5 с и с остатком (на сырых, не
   обрезанных таргетах). late_share ≈ 0 → признак предсказывает 8 с только через первые 0.5 с,
   и улучшать 8 с надо через более точный прогноз первых движений, а не медленными признаками.

Выход: feat_report.md, feat_corr_<инструмент>.csv, feat_ic.csv, feat_summary.csv, feat_late.csv.
"""
import argparse
import os

import numpy as np

from cache import TRAIN, cached_names, check_cache_version, load_day
from common import HORIZONS, TARGET_INSTRUMENTS, dates, load_clip
from eda import boot_mean, md_table, write_csv
from metric import day_corr

BASE = "imb1"
GROUPS = {
    "ofi": ["ofi_5", "ofi_20", "ofi_100", "ofi_5_x_vol"],
    "trades": ["tflow_5", "tflow_20", "tflow_100", "tflow_5_x_vol"],
    "returns": ["ret_5", "ret_20", "ret_100"],
    "lead": ["lead_ret_5", "lead_ret_20"],
    "depth": ["depth_imb_5", "depth_imb_all", "depth_imb_5_x_vol"],
    "x_vol": ["imb1_x_vol", "ofi_5_x_vol", "tflow_5_x_vol", "depth_imb_5_x_vol"],
    "micro": ["micro_bps"],
}


def day_corr_matrix(X):
    """Корреляции столбцов за день; у постоянного столбца все корреляции = 0 (как в метрике)."""
    Z = X - X.mean(0)
    sd = Z.std(0)
    ok = sd > 0
    Z[:, ok] /= sd[ok]
    Z[:, ~ok] = 0.0
    C = (Z.T @ Z) / len(Z)
    np.fill_diagonal(C, 1.0)
    return C


def resid_on(x, b):
    """Остаток x после линейной регрессии на b (с константой) внутри дня."""
    bc = b - b.mean()
    v = float(bc @ bc)
    if v == 0:
        return x - x.mean()
    return x - x.mean() - (float((x - x.mean()) @ bc) / v) * bc


def clusters(names, C, thr):
    """Связные группы: признаки соединены, если |r| ≥ thr."""
    parent = list(range(len(names)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if abs(C[i, j]) >= thr:
                parent[find(i)] = find(j)
    out = {}
    for i in range(len(names)):
        out.setdefault(find(i), []).append(i)
    return [g for g in out.values() if len(g) > 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--data-dir", required=True, help="нужна только meta/target_clip.parquet")
    ap.add_argument("--block", default="base")
    ap.add_argument("--start", default=TRAIN[0])
    ap.add_argument("--end", default=TRAIN[1])
    ap.add_argument("--day-step", type=int, default=7)
    ap.add_argument("--thr", type=float, default=0.9, help="порог |r| для группы похожих признаков")
    ap.add_argument("--ref", default="fa_ridge_ref",
                    help="имя опорного прогона в командах --drop (первая команда его и считает)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.end >= "2026-07-15":
        print("ВНИМАНИЕ: в анализ попадают дни валидации — выбор признаков по ним сделает валидацию нечестной")
    os.makedirs(a.out, exist_ok=True)

    names = cached_names(a.cache_dir, a.block)
    if names is None:
        raise SystemExit("кэш пуст — сначала запустите cache.py")
    check_cache_version(a.cache_dir, a.block)
    clip = load_clip(a.data_dir)
    k = len(names)
    ib = names.index(BASE) if BASE in names else None
    days = dates(a.start, a.end)[::a.day_step]

    Cs, ic, icr = {}, {}, {}           # ic[(inst, h, f)] = [дневные значения]
    cov_e, cov_l, ic_late = {}, {}, {}  # раскладка 8 с: (inst, f) → суммы ковариаций / дневные IC остатка
    i5, i8 = HORIZONS.index("y_500ms"), HORIZONS.index("y_8s")
    for inst in TARGET_INSTRUMENTS:
        Csum, nd = np.zeros((k, k)), 0
        for d in days:
            r = load_day(a.cache_dir, a.block, inst, d)
            if r is None:
                continue
            X, Y, _ = r
            X = np.nan_to_num(X.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
            Csum += day_corr_matrix(X)
            nd += 1
            for j, h in enumerate(HORIZONS):
                y = np.clip(Y[:, j].astype(np.float64), *clip[(inst, h)])
                for fi, f in enumerate(names):
                    ic.setdefault((inst, h, f), []).append(day_corr(X[:, fi], y))
                    if ib is not None and fi != ib:
                        icr.setdefault((inst, h, f), []).append(day_corr(resid_on(X[:, fi], X[:, ib]), y))
            y5, y8 = Y[:, i5].astype(np.float64), Y[:, i8].astype(np.float64)
            m = np.isfinite(y5) & np.isfinite(y8)
            if m.sum() > 100:
                late = y8[m] - y5[m]
                e0, l0 = y5[m] - y5[m].mean(), late - late.mean()
                late_c = np.clip(late, *clip[(inst, "y_8s")])
                for fi, f in enumerate(names):
                    x = X[m, fi] - X[m, fi].mean()
                    cov_e[(inst, f)] = cov_e.get((inst, f), 0.0) + float(x @ e0) / m.sum()
                    cov_l[(inst, f)] = cov_l.get((inst, f), 0.0) + float(x @ l0) / m.sum()
                    ic_late.setdefault((inst, f), []).append(day_corr(X[m, fi], late_c))
        if nd == 0:
            print(f"  {inst}: нет дней в кэше — пропущен")
            continue
        Cs[inst] = Csum / nd
        print(f"  {inst}: {nd} дн.")
        write_csv(os.path.join(a.out, f"feat_corr_{inst}.csv"),
                  [{"feature": f, **{g: Cs[inst][i, j] for j, g in enumerate(names)}} for i, f in enumerate(names)])
    if not Cs:
        raise SystemExit("ни одного дня в кэше за выбранный период")

    # ---- таблицы IC
    rows = []
    for (inst, h, f), v in sorted(ic.items()):
        m, lo, hi, _ = boot_mean(v)
        row = {"instrument": inst, "horizon": h, "feature": f, "ic": m, "ic_lo": lo, "ic_hi": hi}
        if (inst, h, f) in icr:
            mr, lor, hir, _ = boot_mean(icr[(inst, h, f)])
            row.update({"ic_beyond_imb1": mr, "beyond_lo": lor, "beyond_hi": hir})
        rows.append(row)
    write_csv(os.path.join(a.out, "feat_ic.csv"), rows)

    Cavg = np.mean([Cs[i] for i in Cs], axis=0)
    summ = []
    for fi, f in enumerate(names):
        mine = [r for r in rows if r["feature"] == f]
        s = {"feature": f, "ic_mean_12": float(np.mean([r["ic"] for r in mine])),
             "ic_beyond_imb1_mean_12": (float(np.mean([r["ic_beyond_imb1"] for r in mine]))
                                       if "ic_beyond_imb1" in mine[0] else np.nan),
             "corr_with_imb1": float(Cavg[fi, ib]) if ib is not None else np.nan}
        others = [(abs(Cavg[fi, j]), names[j]) for j in range(k) if j != fi]
        if others:
            mx = max(others)
            s.update({"most_similar": mx[1], "max_abs_corr": mx[0]})
        summ.append(s)
    summ.sort(key=lambda s: -abs(s["ic_mean_12"]))
    write_csv(os.path.join(a.out, "feat_summary.csv"), summ)

    # ---- группы похожих
    grp_rows = []
    for g in clusters(names, Cavg, a.thr):
        members = sorted(g, key=lambda i: -abs(next(s["ic_mean_12"] for s in summ if s["feature"] == names[i])))
        sub = Cavg[np.ix_(g, g)]
        grp_rows.append({"group": " / ".join(names[i] for i in members),
                         "min_abs_corr_inside": float(np.abs(sub[~np.eye(len(g), dtype=bool)]).min()),
                         "strongest_ic": names[members[0]]})

    # ---- раскладка 8 с
    late_rows = []
    for (inst, f), ce in sorted(cov_e.items()):
        cl = cov_l[(inst, f)]
        mi, lo, hi, _ = boot_mean(ic_late[(inst, f)])
        tot = ce + cl
        late_rows.append({"instrument": inst, "feature": f,
                          "ic_8s": float(np.mean(ic[(inst, "y_8s", f)])) if (inst, "y_8s", f) in ic else np.nan,
                          "ic_late": mi, "ic_late_lo": lo, "ic_late_hi": hi,
                          "late_share": cl / tot if abs(tot) > 1e-30 else np.nan})
    write_csv(os.path.join(a.out, "feat_late.csv"), late_rows)
    late_summ = []
    for f in names:
        rr = [r for r in late_rows if r["feature"] == f]
        if rr:
            late_summ.append({"feature": f, "ic_8s_mean": float(np.nanmean([r["ic_8s"] for r in rr])),
                              "ic_late_mean": float(np.nanmean([r["ic_late"] for r in rr])),
                              "late_share_mean": float(np.nanmean([r["late_share"] for r in rr])),
                              "late_share_min": float(np.nanmin([r["late_share"] for r in rr])),
                              "late_share_max": float(np.nanmax([r["late_share"] for r in rr]))})
    late_summ.sort(key=lambda r: -abs(r["ic_8s_mean"]))

    # ---- команды для шага 3
    groups = {g: [f for f in fs if f in names] for g, fs in GROUPS.items()}
    groups = {g: fs for g, fs in groups.items() if fs}
    cmd = ["[Console]::OutputEncoding = [System.Text.Encoding]::UTF8",
           "cd D:\\hft\\hft_pipeline\\hft_pipeline",
           # тот же набор признаков, что анализировался, и внутренняя проверка (валидация не трогается)
           f'$base = "--cache-dir","D:\\hft\\cache","--data-dir","D:\\hft\\data","--block","{a.block}","--internal"',
           "$runs = @(",
           f'  @("--name","{a.ref}","--model","ridge"),']
    cmd += [f'  @("--name","drop_{g}","--model","ridge","--ref","{a.ref}","--drop",'
            + ",".join(f'"{f}"' for f in fs) + ")" + ("," if n < len(groups) - 1 else "")
            for n, (g, fs) in enumerate(groups.items())]
    cmd += [")", "foreach ($r in $runs) {",
            "  py -3.13 experiment.py @base @r 2>&1 | Tee-Object -FilePath drop_log.txt -Append",
            '  if ($LASTEXITCODE -ne 0) { Write-Host "ОСТАНОВ: ошибка в $($r[1])"; break }', "}"]

    pd = sorted(((abs(Cavg[i, j]), names[i], names[j]) for i in range(k) for j in range(i + 1, k)), reverse=True)
    rep = [f"# Анализ признаков\n\nДни трейна {a.start}..{a.end}, каждый {a.day_step}-й "
           f"({len(days)} дн. на инструмент). Валидация не использовалась.\n",
           "## 1. Самые похожие пары (|r| внутри дня, среднее по дням и инструментам)\n",
           md_table([{"feature_a": x, "feature_b": y, "abs_corr": c} for c, x, y in pd[:20]]),
           f"\n### Группы почти одинаковых признаков (|r| ≥ {a.thr})\n",
           "Первым в группе стоит признак с самым сильным IC. Остальные — кандидаты на удаление, "
           "но решает только шаг 3 (выкидывание и SCORE).\n\n",
           md_table(grp_rows) if grp_rows else "_таких групп нет_\n",
           "\n## 2. Сила признака и что он добавляет сверх imb1 (среднее по 12 парам)\n",
           "ic_mean_12 — корреляция признака с таргетом; ic_beyond_imb1_mean_12 — то же для остатка "
           "признака после вычитания части, повторяющей imb1. Остаток ≈ 0 → признак почти дубль imb1. "
           "vol_bps и spread_bps сами по себе дают ≈ 0 — они нужны как множители и для деревьев, "
           "по этой таблице их не убирать. По парам с интервалами — feat_ic.csv.\n\n",
           md_table(summ),
           "\n## 3. Проверка групп выкидыванием (Ridge, внутренняя проверка)\n",
           f"Первая строка — опорный прогон `{a.ref}` на всех признаках блока `{a.block}`, остальные — без одной "
           f"группы, сравнение с ним (`--internal`: валидация не трогается). Если SCORE не падает "
           "(разница в пределах шума) — группа лишняя для Ridge. Для LightGBM вывод может быть другим.\n\n",
           "```\n" + "\n".join(cmd) + "\n```\n",
           "\n## 4. Горизонт 8 с: что признак предсказывает — первые 0.5 с или остаток\n",
           "y_8s = y_500ms + остаток (0.5→8 с). late_share — доля ковариации признака с y_8s, которая "
           "приходится на остаток (сырые таргеты, раскладка точная). ic_late — корреляция признака с "
           "остатком (обрезанным по порогам 8 с). late_share ≈ 0 и ic_late ≈ 0 → на 8 с признак работает "
           "только через первые 0.5 с. late_share < 0 → после первых 0.5 с движение частично откатывается. "
           "По монетам — feat_late.csv.\n\n",
           md_table(late_summ)]
    with open(os.path.join(a.out, "feat_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(rep))
    print(f"Готово: {os.path.join(a.out, 'feat_report.md')}")


if __name__ == "__main__":
    from textio import utf8_console
    utf8_console()
    main()
