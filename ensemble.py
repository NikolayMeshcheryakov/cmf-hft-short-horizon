"""Ансамбль: подбор весов по прогнозам моделей на днях проверки и сборка model.json для predict.py.

Прогнозы моделей сохраняет experiment.py --save-preds ПАПКА (а масштаб прогнозов —
results/<имя>_predstd.json). Метрика по каждому дню инвариантна к масштабу, поэтому прогноз
каждой модели сначала делится на её масштаб (std на последних 10% дней трейна — константа,
не зависит ни от валидации, ни от будущего), а потом складывается с весами w ≥ 0, Σw = 1.

1) Подбор (на внутренней проверке, валидация не трогается):
   py -3.13 ensemble.py fit --preds-dir D:\\hft\\preds\\internal --members int_v3_lgbm int_cb int_pool int_mlp int_v3_ridge --out results\\ens_weights.json
   Веса ищутся перебором по сетке с шагом 0.05 так, чтобы максимизировать саму метрику (средняя
   дневная корреляция). Три варианта: общие веса, свои веса на горизонт, свои на каждую из 12 пар.
   Какой брать — решает честная проверка: веса подбираются на половине дней (чёт/нечёт) и
   проверяются на другой половине. Выбирается вариант с лучшей такой оценкой.

2) Проверка на других прогнозах (например, на валидации моделей, обученных на полном трейне):
   py -3.13 ensemble.py eval --preds-dir D:\\hft\\preds\\valid --weights results\\ens_weights.json --members v_lgbm v_cb v_pool v_mlp v_ridge
   (участники — в том же порядке, что при подборе; масштаб берётся у каждой модели свой;
    участника с нулевым весом везде можно не обучать и передать «-»)

3) Сборка модели для predict.py:
   py -3.13 ensemble.py build --weights results\\ens_weights.json --models models\\lgbm\\model.json models\\cb\\model.json ... --out models\\ens
"""
import argparse
import itertools
import json
import os
import shutil

import numpy as np

from common import HORIZONS, TARGET_INSTRUMENTS

CELLS = [(i, h) for i in TARGET_INSTRUMENTS for h in HORIZONS]


# ============================================================ данные
def load_stats(preds_dir, members, results):
    """Для каждой пары и дня — достаточные статистики: c[m] = cov(p_m/s_m, y), S[m,n] = cov(...),
    vy = var(y) по точкам с таргетом (прогноз-пропуск = 0, как в метрике).
    → {(inst, h): (days, C (D×M), S (D×M×M), vy (D))}."""
    scale = {}
    for m in members:
        if m == "-":
            continue
        sp = os.path.join(results, f"{m}_predstd.json")
        if not os.path.isfile(sp):
            raise SystemExit(f"нет {sp} — эксперимент {m} запускался старым experiment.py")
        with open(sp) as f:
            scale[m] = json.load(f)
    load_stats.ident = {m: (scale[m].get("_kind"), scale[m].get("_feat")) for m in scale}
    out = {}
    for inst in TARGET_INSTRUMENTS:
        ydir = os.path.join(preds_dir, "_y", inst)
        if not os.path.isdir(ydir):
            raise SystemExit(f"нет {ydir}: в {preds_dir} не сохранялись прогнозы (experiment.py --save-preds)")
        days = sorted(f[:-4] for f in os.listdir(ydir) if f.endswith(".npy"))
        for m in members:
            if m == "-":
                continue
            md = os.path.join(preds_dir, m, inst)
            have = set(f[:-4] for f in os.listdir(md)) if os.path.isdir(md) else set()
            if set(days) - have:
                raise SystemExit(f"{m} {inst}: нет прогнозов за {len(set(days) - have)} дн. "
                                 f"(например {sorted(set(days) - have)[0]}) — модель не досчитана?")
        per = {h: ([], [], [], []) for h in HORIZONS}
        for d in days:
            Y = np.load(os.path.join(ydir, f"{d}.npy")).astype(np.float64)
            Ps = [np.zeros(Y.shape) if m == "-" else
                  np.load(os.path.join(preds_dir, m, inst, f"{d}.npy")).astype(np.float64) for m in members]
            for P in Ps:
                if P.shape != Y.shape:
                    raise SystemExit(f"{inst} {d}: размер прогнозов не совпадает с таргетом")
            for j, h in enumerate(HORIZONS):
                y = Y[:, j]
                ok = np.isfinite(y)
                if ok.sum() < 2:
                    continue
                cols = []
                for m, P in zip(members, Ps):
                    s = None if m == "-" else scale[m].get(inst, {}).get(h)
                    p = np.nan_to_num(P[ok, j], nan=0.0)
                    cols.append(p / s if s else np.zeros_like(p))
                A = np.column_stack(cols)
                A = A - A.mean(0)
                yc = y[ok] - y[ok].mean()
                per[h][0].append(d)
                per[h][1].append(A.T @ yc / len(yc))
                per[h][2].append(A.T @ A / len(yc))
                per[h][3].append(float(yc @ yc / len(yc)))
        for h in HORIZONS:
            dd, C, S, V = per[h]
            out[(inst, h)] = (dd, np.array(C), np.array(S), np.array(V))
    return out


def day_corrs(W, C, S, V):
    """W (P×M) → корреляции (P×D). Как metric.day_corr: нулевой разброс → 0."""
    num = W @ C.T
    den2 = np.einsum("pm,dmn,pn->pd", W, S, W) * V[None, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        r = num / np.sqrt(den2)
    r[~(den2 > 0)] = 0.0
    return r


def simplex(M, step):
    k = int(round(1 / step))
    pts = [c for c in itertools.product(range(k + 1), repeat=M - 1) if sum(c) <= k]
    return np.array([list(c) + [k - sum(c)] for c in pts], dtype=float) / k


# ============================================================ подбор
def score_of(stats, W_by_cell, day_sel=None):
    """SCORE (среднее по 12 парам среднего по дням) для весов {(inst,h): w}."""
    cells = {}
    for c in CELLS:
        dd, C, S, V = stats[c]
        sel = np.ones(len(dd), bool) if day_sel is None else np.array([day_sel(d) for d in dd])
        r = day_corrs(W_by_cell[c][None, :], C[sel], S[sel], V[sel])[0]
        cells[c] = float(r.mean()) if len(r) else np.nan
    return float(np.mean([cells[c] for c in CELLS])), cells


def fit_weights(stats, grid, mode, day_sel=None):
    """mode: global | horizon | cell. → {(inst,h): w}."""
    groups = {"global": [CELLS], "horizon": [[c for c in CELLS if c[1] == h] for h in HORIZONS],
              "cell": [[c] for c in CELLS]}[mode]
    W = {}
    for g in groups:
        tot = np.zeros(len(grid))
        for c in g:
            dd, C, S, V = stats[c]
            sel = np.ones(len(dd), bool) if day_sel is None else np.array([day_sel(d) for d in dd])
            if sel.sum():
                tot += day_corrs(grid, C[sel], S[sel], V[sel]).mean(1)
        best = grid[int(np.argmax(tot))]
        for c in g:
            W[c] = best
    return W


def cmd_fit(a):
    members = a.members
    stats = load_stats(a.preds_dir, members, a.results)
    M = len(members)
    grid = simplex(M, a.step)
    print(f"участников {M}, вариантов весов {len(grid):,}, дней {len(stats[CELLS[0]][0])}")
    print(f"\n{'модель':24} {'SCORE':>8}")
    single = {}
    for k, m in enumerate(members):
        w = np.eye(M)[k]
        single[m] = score_of(stats, {c: w for c in CELLS})[0]
        print(f"{m:24} {single[m]:8.4f}")
    eq = score_of(stats, {c: np.full(M, 1 / M) for c in CELLS})[0]
    print(f"{'поровну':24} {eq:8.4f}")
    days = sorted(stats[CELLS[0]][0])
    odd = {d: i % 2 for i, d in enumerate(days)}
    res = {}
    print(f"\n{'веса':10} {'на всех днях':>13} {'честно (чёт/нечёт)':>20}")
    for mode in ("global", "horizon", "cell"):
        full = score_of(stats, fit_weights(stats, grid, mode))[0]
        cv = []
        for f in (0, 1):
            W = fit_weights(stats, grid, mode, lambda d, f=f: odd.get(d, 0) == f)
            cv.append(score_of(stats, W, lambda d, f=f: odd.get(d, 0) != f)[0])
        res[mode] = float(np.mean(cv))
        print(f"{mode:10} {full:13.4f} {res[mode]:20.4f}")
    best_single = max(single.values())
    cv_single = []
    for f in (0, 1):                      # лучший одиночный — тоже честно: выбран на половине дней
        pick = max(range(M), key=lambda k: score_of(stats, {c: np.eye(M)[k] for c in CELLS},
                                                    lambda d, f=f: odd.get(d, 0) == f)[0])
        cv_single.append(score_of(stats, {c: np.eye(M)[pick] for c in CELLS}, lambda d, f=f: odd.get(d, 0) != f)[0])
    cv_single = float(np.mean(cv_single))
    mode = max(res, key=res.get)
    W = fit_weights(stats, grid, mode)
    print(f"\nлучший одиночный: {best_single:.4f} (честно {cv_single:.4f});  "
          f"ансамбль «{mode}»: честно {res[mode]:.4f}  → прирост {res[mode] - cv_single:+.4f}")
    print("\nвеса:")
    for c in CELLS:
        print(f"  {c[0]:18} {c[1]:8} " + "  ".join(f"{m}={w:.2f}" for m, w in zip(members, W[c])))
    used = [m for k, m in enumerate(members) if any(W[c][k] > 0 for c in CELLS)]
    print(f"\nв ансамбль вошли: {', '.join(used)}")
    with open(a.out, "w") as f:
        json.dump({"members": members, "mode": mode, "cv_score": res[mode], "cv_best_single": cv_single,
                   "kinds": [load_stats.ident[m][0] for m in members],
                   "feats": [load_stats.ident[m][1] for m in members],
                   "weights": {i: {h: W[(i, h)].tolist() for h in HORIZONS} for i in TARGET_INSTRUMENTS}},
                  f, indent=1)
    print(f"записано: {a.out}")


def load_weights(path, names):
    with open(path) as f:
        w = json.load(f)
    if len(w["members"]) != len(names):
        raise SystemExit(f"в {path} {len(w['members'])} участников ({w['members']}), передано {len(names)} — "
                         f"нужен тот же состав и порядок")
    for k, nm in enumerate(names):
        if nm == "-" and any(w["weights"][i][h][k] > 0 for i, h in CELLS):
            raise SystemExit(f"участник №{k + 1} ({w['members'][k]}) пропущен «-», но его вес не 0")
    return w


def check_ident(w, k, name, kind, feat):
    """Участник №k должен быть той же модели (тип) на том же наборе признаков, что при подборе весов —
    иначе вес модели A достался бы модели B."""
    wk = (w.get("kinds") or [None] * len(w["members"]))[k]
    wf = (w.get("feats") or [None] * len(w["members"]))[k]
    if wk is None:
        print(f"  ВНИМАНИЕ: в весах нет типа участника №{k + 1} (подобраны старым ensemble.py) — порядок не проверен")
        return
    if kind != wk:
        raise SystemExit(f"участник №{k + 1}: {name} — модель «{kind}», а вес подбирался для «{wk}» "
                         f"({w['members'][k]}). Порядок участников перепутан?")
    if feat is not None and wf is not None and feat != wf:
        raise SystemExit(f"участник №{k + 1}: {name} обучен на другом наборе признаков, чем {w['members'][k]}")


def cmd_eval(a):
    w = load_weights(a.weights, a.members)
    stats = load_stats(a.preds_dir, a.members, a.results)
    for k, m in enumerate(a.members):
        if m != "-":
            check_ident(w, k, m, *load_stats.ident[m])
    M = len(a.members)
    W = {(i, h): np.array(w["weights"][i][h]) for i, h in CELLS}
    print(f"{'модель':24} {'SCORE':>8}   (дней {len(stats[CELLS[0]][0])})")
    for k, m in enumerate(a.members):
        if m != "-":
            print(f"{m:24} {score_of(stats, {c: np.eye(M)[k] for c in CELLS})[0]:8.4f}")
    s, cells = score_of(stats, W)
    print(f"{'АНСАМБЛЬ':24} {s:8.4f}")
    for c in CELLS:
        print(f"  {c[0]:18} {c[1]:8} {cells[c]:.4f}")


def cmd_build(a):
    w = load_weights(a.weights, a.models)
    os.makedirs(a.out, exist_ok=True)
    members = []
    for k, mp in enumerate(a.models):
        if mp == "-":
            continue
        with open(mp) as f:
            meta = json.load(f)
        from experiment import feats_id
        check_ident(w, k, mp, meta.get("type"), feats_id(meta["features"]))
        ps = meta.get("pred_std")
        if not ps:
            raise SystemExit(f"{mp}: нет pred_std — модель сохранена старым experiment.py, переобучите")
        coef = {}
        for i, h in CELLS:
            wk = float(w["weights"][i][h][k])
            s = ps.get(i, {}).get(h)
            if wk > 0 and s:
                coef.setdefault(i, {})[h] = wk / s
        if not coef:
            print(f"  {mp}: вес 0 везде — в ансамбль не входит")
            continue
        dst = os.path.join(a.out, "members", str(len(members)))
        if os.path.isdir(dst):
            shutil.rmtree(dst)
        shutil.copytree(os.path.dirname(os.path.abspath(mp)), dst)
        members.append({"path": f"members/{len(members)}/{os.path.basename(mp)}", "coef": coef,
                        "source": os.path.abspath(mp), "type": meta.get("type")})
    out = os.path.join(a.out, "model.json")
    with open(out, "w") as f:
        json.dump({"type": "ensemble", "members": members, "weights_file": os.path.abspath(a.weights)}, f, indent=1)
    print(f"ансамбль из {len(members)} моделей: {out}  (для сдачи: predict.py --model {out})")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--preds-dir", required=True)
    f.add_argument("--members", nargs="+", required=True)
    f.add_argument("--results", default="results")
    f.add_argument("--step", type=float, default=0.05)
    f.add_argument("--out", required=True)
    e = sub.add_parser("eval")
    e.add_argument("--preds-dir", required=True)
    e.add_argument("--weights", required=True)
    e.add_argument("--members", nargs="+", required=True)
    e.add_argument("--results", default="results")
    b = sub.add_parser("build")
    b.add_argument("--weights", required=True)
    b.add_argument("--models", nargs="+", required=True)
    b.add_argument("--out", required=True)
    a = ap.parse_args()
    {"fit": cmd_fit, "eval": cmd_eval, "build": cmd_build}[a.cmd](a)


if __name__ == "__main__":
    from textio import utf8_console
    utf8_console()
    main()
