"""Optuna: подбор параметров одного участника ансамбля (lgbm или lgbm_pool) по SCORE АНСАМБЛЯ.

Каждая попытка:
  1) experiment.py с параметрами попытки (отдельный процесс — память освобождается после каждой),
     прогнозы на днях внутренней проверки пишутся в --preds-dir (там же лежат прогнозы остальных участников);
  2) участник --replace заменяется прогнозами попытки, остальные берутся готовыми;
  3) веса подбираются в режиме cell честно: подбор на чётных днях → счёт на нечётных и наоборот, среднее.
     Это и есть целевая функция (как «честно» в ensemble.py fit).
Первой идёт попытка с текущими параметрами — точка отсчёта при том же --max-rows.
Прогнозы попыток удаляются, кроме лучшей. Состояние — в results/optuna_<модель>.db: после падения
повторный запуск продолжает с того же места.

Запуск (обычно через run_tune.ps1):
  py -3.13 tune.py --model lgbm --members int14_lgbm int14_ridge int14_cb int14_pool int14_mlp ^
      --replace int14_lgbm --preds-dir D:\\hft\\preds\\internal --hours 8 ^
      --cache-dir D:\\hft\\cache --data-dir D:\\hft\\data --block base+v2+v3 --threads 10 --internal --drop ...
Все незнакомые tune.py аргументы передаются в experiment.py как есть.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASE_PARAMS = {"leaves": 31, "min_leaf": 2000, "lr": 0.05, "ff": 0.8, "bagging": 0.8, "l2": 10.0,
               "huber_delta": 1.0}
FLAGS = {"leaves": "--leaves", "min_leaf": "--min-leaf", "lr": "--lr", "ff": "--ff", "bagging": "--bagging",
         "l2": "--l2", "huber_delta": "--huber-delta"}
TAGS = {"lgbm": "lgbm", "lgbm_pool": "pool"}


def suggest(trial):
    return {"leaves": trial.suggest_int("leaves", 15, 255, log=True),
            "min_leaf": trial.suggest_int("min_leaf", 300, 8000, log=True),
            "lr": trial.suggest_float("lr", 0.02, 0.1, log=True),
            "ff": trial.suggest_float("ff", 0.4, 1.0),
            "bagging": trial.suggest_float("bagging", 0.5, 1.0),
            "l2": trial.suggest_float("l2", 0.1, 100.0, log=True),
            "huber_delta": trial.suggest_float("huber_delta", 0.3, 3.0, log=True)}


def param_args(p):
    out = []
    for k, flag in FLAGS.items():
        v = p[k]
        out += [flag, str(int(v)) if k in ("leaves", "min_leaf") else f"{v:.6g}"]
    return out


class Tuner:
    def __init__(self, a, fwd):
        import ensemble as en
        self.en, self.a, self.fwd = en, a, fwd
        self.k = a.members.index(a.replace)
        self.grid = en.simplex(len(a.members), a.step)
        self.log_dir = os.path.join(a.results, f"optuna_{TAGS[a.model]}_logs")
        os.makedirs(self.log_dir, exist_ok=True)
        self.best_path = os.path.join(a.results, f"optuna_{TAGS[a.model]}_best.json")
        self.fails = 0

    def say(self, s):
        print(s, flush=True)
        with open(self.a.log, "a", encoding="utf-8") as f:
            f.write(s + "\n")

    def ens_score(self, members):
        """→ (честный SCORE ансамбля cell, SCORE участника k в одиночку, SCORE ансамбля на всех днях)."""
        en = self.en
        stats = en.load_stats(self.a.preds_dir, members, self.a.results)
        days = sorted(stats[en.CELLS[0]][0])
        odd = {d: i % 2 for i, d in enumerate(days)}
        cv = []
        for f in (0, 1):
            W = en.fit_weights(stats, self.grid, "cell", lambda d, f=f: odd.get(d, 0) == f)
            cv.append(en.score_of(stats, W, lambda d, f=f: odd.get(d, 0) != f)[0])
        M = len(members)
        single = en.score_of(stats, {c: np.eye(M)[self.k] for c in en.CELLS})[0]
        full = en.score_of(stats, en.fit_weights(stats, self.grid, "cell"))[0]
        return float(np.mean(cv)), float(single), float(full), len(days)

    def run_exp(self, name, p):
        a = self.a
        cmd = [sys.executable, a.experiment, "--name", name, "--model", a.model, "--rounds", str(a.rounds),
               "--max-rows", str(a.max_rows), "--save-preds", a.preds_dir, "--results", a.results] \
            + param_args(p) + self.fwd
        lp = os.path.join(self.log_dir, f"{name}.txt")
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        with open(lp, "w", encoding="utf-8") as f:
            r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
        if r.returncode != 0:
            with open(lp, encoding="utf-8", errors="replace") as f:
                tail = f.read()[-1500:]
            raise RuntimeError(f"experiment.py упал (код {r.returncode}), лог {lp}:\n{tail}")

    def drop_preds(self, name):
        shutil.rmtree(os.path.join(self.a.preds_dir, name), ignore_errors=True)

    def objective(self, trial):
        a = self.a
        p = suggest(trial)
        name = f"opt_{TAGS[a.model]}_{trial.number:03d}"
        t0 = time.time()
        self.say(f"--- попытка {trial.number} ({name}) {time.strftime('%H:%M')}: "
                 + ", ".join(f"{k}={v:.4g}" for k, v in p.items()))
        try:
            self.run_exp(name, p)
            members = list(a.members)
            members[self.k] = name
            cv, single, full, nd = self.ens_score(members)
        except Exception:
            self.drop_preds(name)
            self.fails += 1
            if self.fails >= 2:
                trial.study.stop()
                self.say("ОСТАНОВ: две неудачные попытки подряд (память? см. лог попытки)")
            raise
        self.fails = 0
        trial.set_user_attr("name", name)
        trial.set_user_attr("single", single)
        trial.set_user_attr("full", full)
        trial.set_user_attr("minutes", (time.time() - t0) / 60)
        try:
            prev = trial.study.best_trial
        except ValueError:
            prev = None
        better = prev is None or cv > prev.value
        if better:
            if prev is not None and prev.user_attrs.get("name") != name:
                self.drop_preds(prev.user_attrs.get("name", ""))
            self.write_best(name, p, cv, single, full)
        else:
            self.drop_preds(name)
        best = cv if better else prev.value
        self.say(f"    {(time.time() - t0) / 60:.0f} мин; ансамбль честно {cv:.4f} (на всех днях {full:.4f}), "
                 f"сам по себе {single:.4f}; дней {nd}; лучший {best:.4f} (отсчёт {self.ref:.4f})"
                 + ("  ← НОВЫЙ ЛУЧШИЙ" if better else ""))
        return cv

    def write_best(self, name, p, cv, single, full):
        """Лучшие параметры + готовые команды следующих шагов."""
        a = self.a
        tag = TAGS[a.model]
        fwd = " ".join(x for x in self.fwd if x != "--internal")
        # для полного прогона: lgbm — 3 млн строк (как int14/val14), min_leaf растёт вместе с числом строк;
        # lgbm_pool — столько же строк, сколько при подборе (больше не влезает в память)
        full_rows = 3_000_000 if a.model == "lgbm" else a.max_rows
        pf = dict(p, min_leaf=int(round(p["min_leaf"] * full_rows / a.max_rows)))
        pa, paf = " ".join(param_args(p)), " ".join(param_args(pf))
        mem_now = list(a.members); mem_now[self.k] = name
        mem_int = list(a.members); mem_int[self.k] = f"int15_{tag}"
        vm = [m.replace("int", "val", 1) if m.startswith("int") else m for m in a.members]
        vm[self.k] = f"val15_{tag}"
        pval = os.path.join(os.path.dirname(os.path.abspath(a.preds_dir)), "valid")
        with open(self.best_path, "w", encoding="utf-8") as f:
            json.dump({"trial": name, "params": p, "cv_ensemble": cv, "single": single, "full_days": full,
                       "ref_cv_ensemble": self.ref, "max_rows_in_tuning": a.max_rows,
                       "params_full_rows": pf, "full_rows": full_rows,
                       "1_fit_now (прогнозы лучшей попытки уже лежат в preds)":
                           f"py -3.13 ensemble.py fit --preds-dir {a.preds_dir} --members {' '.join(mem_now)} "
                           f"--out results\\ens15_weights.json",
                       "2_internal_full (по желанию, ~2 ч)":
                           f"py -3.13 experiment.py --name int15_{tag} --model {a.model} --rounds {a.rounds} "
                           f"--max-rows {full_rows} {paf} --internal --save-preds {a.preds_dir} {fwd}",
                       "2b_fit_after_full":
                           f"py -3.13 ensemble.py fit --preds-dir {a.preds_dir} --members {' '.join(mem_int)} "
                           f"--out results\\ens15_weights.json",
                       "3_valid":
                           f"py -3.13 experiment.py --name val15_{tag} --model {a.model} --rounds {a.rounds} "
                           f"--max-rows {full_rows} {paf} --save-preds {pval} --save models\\val15_{tag} {fwd}",
                       "valid_members_order": vm, "tuning_params_args": pa},
                      f, ensure_ascii=False, indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                 allow_abbrev=False)   # чужие флаги идут в experiment.py — без сокращений
    ap.add_argument("--model", choices=list(TAGS), required=True)
    ap.add_argument("--members", nargs="+", required=True, help="участники ансамбля в порядке ensemble.py fit")
    ap.add_argument("--replace", required=True, help="кого из --members заменяют попытки")
    ap.add_argument("--preds-dir", required=True, help="прогнозы участников на внутренней проверке")
    ap.add_argument("--results", default="results")
    ap.add_argument("--trials", type=int, default=30, help="всего попыток (с учётом уже сделанных)")
    ap.add_argument("--hours", type=float, default=None, help="не начинать новых попыток после N часов")
    ap.add_argument("--max-rows", type=int, default=1_500_000)
    ap.add_argument("--rounds", type=int, default=3000, help="потолок деревьев; останавливает ранняя остановка")
    ap.add_argument("--step", type=float, default=0.05, help="шаг сетки весов")
    ap.add_argument("--log", default=None, help="лог (UTF-8); по умолчанию results/optuna_<модель>_log.txt")
    ap.add_argument("--experiment", default=os.path.join(HERE, "experiment.py"), help=argparse.SUPPRESS)
    a, fwd = ap.parse_known_args()
    if a.replace not in a.members:
        raise SystemExit(f"--replace {a.replace} нет среди --members")
    if "--save" in fwd:
        raise SystemExit("--save в подборе не нужен")
    os.makedirs(a.results, exist_ok=True)
    a.log = a.log or os.path.join(a.results, f"optuna_{TAGS[a.model]}_log.txt")
    try:
        import optuna
    except ImportError:
        raise SystemExit("нет optuna: py -3.13 -m pip install optuna")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    t = Tuner(a, fwd)
    # проверка до старта: остальные участники на месте, совпадают дни и точки (иначе load_stats остановит)
    t.ref, single0, full0, nd = t.ens_score(a.members)
    t.say(f"=== tune {a.model} {time.strftime('%Y-%m-%d %H:%M')}: участники {' '.join(a.members)}, "
          f"меняем {a.replace}; дней {nd}")
    t.say(f"    отсчёт (текущий ансамбль): честно {t.ref:.4f}, на всех днях {full0:.4f}; {a.replace} сам {single0:.4f}")

    db = os.path.abspath(os.path.join(a.results, f"optuna_{TAGS[a.model]}.db"))
    storage = f"sqlite:///{db}"
    try:                                          # зерно = число уже сделанных попыток: после перезапуска
        n0 = len(optuna.load_study(study_name=f"opt_{TAGS[a.model]}", storage=storage).trials)
    except KeyError:                              # случайные стартовые точки не повторяются
        n0 = 0
    study = optuna.create_study(study_name=f"opt_{TAGS[a.model]}", storage=storage,
                                direction="maximize", load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=n0, multivariate=True, n_startup_trials=6))
    done = [x for x in study.trials if x.state == optuna.trial.TrialState.COMPLETE]
    if not study.trials:
        study.enqueue_trial(BASE_PARAMS)          # точка отсчёта: текущие параметры при том же --max-rows
    left = a.trials - len(done)
    if done:
        t.say(f"    продолжаем: готово {len(done)}, лучший {study.best_value:.4f}")
    if left <= 0:
        t.say("    все попытки уже сделаны")
    else:
        study.optimize(t.objective, n_trials=left, timeout=None if a.hours is None else a.hours * 3600,
                       catch=(RuntimeError,))
    done = [x for x in study.trials if x.state == optuna.trial.TrialState.COMPLETE]
    if not done:
        raise SystemExit("нет ни одной успешной попытки")
    b = study.best_trial
    base = next((x for x in done if x.number == 0), None)
    t.say(f"\n=== ИТОГ: попыток {len(done)}; лучшая {b.user_attrs['name']}: ансамбль честно {b.value:.4f}"
          + (f" (текущие параметры при том же числе строк: {base.value:.4f}, прирост {b.value - base.value:+.4f})"
             if base else ""))
    t.say("    параметры: " + ", ".join(f"{k}={v:.4g}" for k, v in b.params.items()))
    t.say(f"    дальше — команды в {t.best_path}")
    top = sorted(done, key=lambda x: -x.value)[:5]
    t.say("    пятёрка: " + "; ".join(f"{x.number}: {x.value:.4f}" for x in top))


if __name__ == "__main__":
    from textio import utf8_console
    utf8_console()
    main()
