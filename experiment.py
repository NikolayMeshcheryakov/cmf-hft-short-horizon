"""Эксперимент на кэше признаков: обучение → оценка на валидации как в метрике → сравнение с прошлым прогоном.

Примеры:
  # точка отсчёта (baseline организаторов)
  py -3.13 experiment.py --cache-dir D:\\hft\\cache --data-dir D:\\hft\\data --name baseline --model ridge --features imb1

  # все признаки, линейная модель, сравнить с baseline
  py -3.13 experiment.py --cache-dir D:\\hft\\cache --data-dir D:\\hft\\data --name ridge_all --model ridge --ref baseline

  # LightGBM с Huber-лоссом, сравнить с ridge_all, сохранить модель для predict.py
  py -3.13 experiment.py --cache-dir D:\\hft\\cache --data-dir D:\\hft\\data --name lgbm_v1 --model lgbm --ref ridge_all --save models\\lgbm_v1

  # внутренняя проверка на трейне (для выбора признаков и параметров; валидация не трогается)
  py -3.13 experiment.py --cache-dir D:\\hft\\cache --data-dir D:\\hft\\data --name int_lgbm --model lgbm --internal

Результаты:
  results/journal.csv          — журнал: одна строка на эксперимент (SCORE и 12 клеток)
  results/<name>_daily.csv     — корреляции по дням (нужны для парного сравнения)

Модели:
  ridge — Ridge на стандартизованных признаках, каждый день весит одинаково (вес 1/дисперсия таргета дня)
  lgbm  — LightGBM, лосс Huber; таргет делится на std своего дня (дни весят одинаково, выбросы
          не тянут модель). Ранняя остановка — по последним 10% дней трейна (валидация не трогается).
  catboost — CatBoost с тем же рецептом, что lgbm (для ансамбля; pip install catboost)
  lgbm_pool — ОДИН LightGBM на горизонт для всех 4 инструментов (номер инструмента — категориальный
          признак); --max-rows считается на инструмент
  mlp   — нейросеть на numpy (mlp.py): одна на все инструменты и горизонты; --max-rows на инструмент

Для ансамбля (ensemble.py): --save-preds ПАПКА сохраняет прогнозы на днях проверки, а в
results/<name>_predstd.json пишется масштаб прогнозов модели (std на последних 10% дней трейна).
  hgb   — запасной вариант из scikit-learn (если LightGBM не установлен), без сохранения
"""
import argparse
import datetime as dt
import json
import os
import time

import numpy as np

from cache import NO_CLIP, TRAIN, VALID, cached_names, check_cache_version, day_available, load_day
from common import HORIZONS, TARGET_INSTRUMENTS, dates, load_clip
from fversion import FEATURES_HASH, FEATURES_V2_HASH, FEATURES_V3_HASH, FHASH
from metric import day_corr, load_targets
from train import MIN_DAY_VAR, Accumulator, purge_mask

RNG = np.random.default_rng(0)

# Внутренняя проверка: выбор признаков и параметров делаем на трейне (учим на ранних месяцах,
# проверяем на поздних), а настоящую валидацию трогаем один раз — для итога.
INTERNAL_TRAIN = ("2025-09-15", "2026-04-30")
INTERNAL_VALID = ("2026-05-01", "2026-07-14")


# ============================================================ данные
def clip_y(Y, inst, clip):
    out = np.empty_like(Y, dtype=np.float64)
    for j, h in enumerate(HORIZONS):
        out[:, j] = np.clip(Y[:, j], *clip[(inst, h)])
    return out


def nan0(X):
    return np.nan_to_num(X.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)


def load_train(a, inst, feats, clip, max_rows=None, whole_days=False):
    """Список (день, X, Y_clipped) трейна; при max_rows — случайное подмножество строк каждого дня.
    whole_days (gru): дни берутся целиком, а max_rows ограничивает число дней (равномерно по трейну)."""
    days = dates(*a.train)[::a.train_day_step]
    if whole_days and max_rows:
        n_keep = max(2, int(max_rows // (172_800 // 5)))
        if n_keep < len(days):
            days = [days[i] for i in np.unique(np.linspace(0, len(days) - 1, n_keep).round().astype(int))]
        max_rows = None
    per_day = None if not max_rows else int(np.ceil(max_rows / len(days)))
    out = []
    for d in days:
        r = load_day(a.cache_dir, a.block, inst, d, feats)
        if r is None:
            continue
        X, Y, rws = r
        if len(rws) > 1 and int(rws[1] - rws[0]) == 1:   # день из периода валидации (в кэше все точки):
            th = np.asarray(rws) % 5 == 0                 # в обучение — каждая 5-я, как у дней трейна
            X, Y, rws = X[th], Y[th], np.asarray(rws)[th]
        if whole_days and len(rws) > 1 and int(rws[1] - rws[0]) != 5:
            raise SystemExit(f"gru: строки трейна в кэше идут через {int(rws[1] - rws[0])}, а сеть учится "
                             f"с шагом 5 (cache.py --row-step 5)")
        Y = clip_y(Y, inst, clip)
        keep = purge_mask(d, rws, a.valid[0])              # таргеты не должны залезать в валидацию
        X, Y = X[keep], Y[keep]
        if per_day and len(X) > per_day:
            idx = np.sort(RNG.choice(len(X), per_day, replace=False))
            X, Y = X[idx], Y[idx]
        out.append((d, X, Y))
    return out


def check_train_coverage(a):
    """--final: в кэше должны быть все дни трейна, для которых есть таргеты. → {} (оценки нет)."""
    missing = [(inst, d) for inst in TARGET_INSTRUMENTS for d in dates(*a.train)[::a.train_day_step]
               if not day_available(a.cache_dir, a.block, inst, d)
               and load_targets(a.data_dir, inst, d, NO_CLIP) is not None]
    if missing:
        raise SystemExit(f"кэш не достроен: нет {len(missing)} инструмент-дней (например {missing[:3]})")
    return {}


def check_coverage(a):
    """Перед обучением: в кэше должны быть ВСЕ выбранные дни трейна и валидации.
    День без файла кэша допустим, только если в данных за него нет таргетов (метрика его тоже
    не считает). Иначе кэш не достроен — считать SCORE на части дней нельзя.
    → {инструмент: [дни валидации, которые войдут в оценку]}."""
    expected, missing = {}, []
    for part, (d0, d1), step in (("трейн", a.train, a.train_day_step), ("валидация", a.valid, a.valid_day_step)):
        for inst in TARGET_INSTRUMENTS:
            for d in dates(d0, d1)[::step]:
                if day_available(a.cache_dir, a.block, inst, d):
                    if part == "валидация":
                        expected.setdefault(inst, []).append(d)
                elif load_targets(a.data_dir, inst, d, NO_CLIP) is not None:
                    missing.append((part, inst, d))
    if missing:
        ex = ", ".join(f"{p} {i} {d}" for p, i, d in missing[:5])
        raise SystemExit(f"кэш не достроен: нет {len(missing)} инструмент-дней, для которых есть таргеты "
                         f"(например: {ex}). Дождитесь конца cache.py или запустите его ещё раз — "
                         f"он докачает недостающее.")
    for inst in TARGET_INSTRUMENTS:
        if not expected.get(inst):
            raise SystemExit(f"{inst}: в кэше нет ни одного дня валидации {a.valid[0]}..{a.valid[1]}")
    return expected


# ============================================================ модели
def fit_ridge(train, lam, rw=None):
    k = train[0][1].shape[1]
    acc = [Accumulator(k) for _ in HORIZONS]
    for d, X, Y in train:
        X = nan0(X)
        wd = 1.0 if rw is None else rw(d)
        for j in range(len(HORIZONS)):
            m = np.isfinite(Y[:, j])
            if m.sum() < 100:
                continue
            y = Y[m, j]
            v = float(y.var())
            if not np.isfinite(v) or v < MIN_DAY_VAR:     # вырожденный день (цена стояла) — пропускаем
                continue
            acc[j].add(X[m], y, wd / v)
    models = []
    for j, h in enumerate(HORIZONS):
        if acc[j].n <= 0:
            print(f"     {h}: нет данных для обучения — прогноз будет 0")
            models.append(None)
            continue
        a_, b = acc[j].solve(lam)
        models.append({"a": a_, "b": b.tolist()})
    return models


def predict_ridge(m, X):
    return m["a"] + nan0(X) @ np.array(m["b"])


def recency_weight(train_end, halflife):
    """Вес дня по свежести: 0.5 ** (дней до конца трейна / halflife). halflife=None → None (все дни по 1)."""
    if not halflife:
        return None
    end = dt.date.fromisoformat(train_end)
    return lambda d: 0.5 ** ((end - dt.date.fromisoformat(d)).days / halflife)


def stack_norm(train, j, rw=None):
    """Склеить дни; таргет каждого дня делим на его std (дни весят одинаково).
    rw — вес дня по свежести (recency_weight): → X, y, w (w = None без весов)."""
    xs, ys, ws = [], [], []
    for d, X, Y in train:
        y = Y[:, j]
        m = np.isfinite(y)
        if m.sum() < 100:
            continue
        v = float(y[m].var())
        if not np.isfinite(v) or v < MIN_DAY_VAR:
            continue
        xs.append(X[m])
        ys.append(y[m] / np.sqrt(v))
        if rw is not None:
            ws.append(np.full(int(m.sum()), rw(d), np.float32))
    if not xs:
        return None, None, None
    return (np.vstack(xs).astype(np.float32), np.concatenate(ys).astype(np.float32),
            np.concatenate(ws) if rw is not None else None)


def fit_lgbm(train, a):
    import lightgbm as lgb
    n_es = max(1, len(train) // 10) if len(train) >= 2 else 0
    models = []
    for j, h in enumerate(HORIZONS):
        Xtr, ytr, wtr = stack_norm(train[:len(train) - n_es], j, a.rw)
        Xes, yes, _ = stack_norm(train[len(train) - n_es:], j) if n_es else (None, None, None)
        if Xtr is None:
            print(f"     {h}: нет данных для обучения — прогноз будет 0")
            models.append(None)
            continue
        params = dict(objective="huber", alpha=a.huber_delta, learning_rate=a.lr, num_leaves=a.leaves,
                      min_data_in_leaf=a.min_leaf, feature_fraction=a.ff, bagging_fraction=a.bagging,
                      bagging_freq=1, lambda_l2=a.l2, verbose=-1, num_threads=a.threads,
                      seed=0, deterministic=True, force_col_wise=True)
        dtr = lgb.Dataset(Xtr, ytr, weight=wtr)
        if Xes is not None:
            des = lgb.Dataset(Xes, yes, reference=dtr)
            bst = lgb.train(params, dtr, num_boost_round=a.rounds, valid_sets=[des],
                            callbacks=[lgb.early_stopping(50, verbose=False)])
        else:   # нет отдельных дней для ранней остановки — фиксированное число деревьев
            print(f"     {h}: ВНИМАНИЕ: нет отдельных дней для ранней остановки, деревьев {a.rounds // 3}")
            bst = lgb.train(params, dtr, num_boost_round=max(a.rounds // 3, 1))
            bst.best_iteration = bst.current_iteration()
        print(f"     {h}: деревьев {bst.best_iteration}, строк {len(ytr):,}")
        models.append(bst)
    return models


def fit_catboost(train, a):
    """CatBoost — тот же рецепт, что у LightGBM: Huber, таргет делится на std дня, ранняя остановка
    по последним 10% дней трейна. Деревья симметричные глубины --cb-depth; обычно им нужно больше
    итераций, чем LightGBM (--rounds 1500)."""
    from catboost import CatBoostRegressor
    n_es = max(1, len(train) // 10) if len(train) >= 2 else 0
    models = []
    for j, h in enumerate(HORIZONS):
        Xtr, ytr, wtr = stack_norm(train[:len(train) - n_es], j, a.rw)
        Xes, yes, _ = stack_norm(train[len(train) - n_es:], j) if n_es else (None, None, None)
        if Xtr is None:
            print(f"     {h}: нет данных для обучения — прогноз будет 0")
            models.append(None)
            continue
        m = CatBoostRegressor(loss_function=f"Huber:delta={a.huber_delta}", iterations=a.rounds,
                              learning_rate=a.lr, depth=a.cb_depth, l2_leaf_reg=10.0, random_seed=0,
                              thread_count=a.threads, bootstrap_type="Bernoulli", subsample=0.8,
                              border_count=128, verbose=False, allow_writing_files=False)
        if Xes is not None:
            m.fit(Xtr.astype(np.float32), ytr, sample_weight=wtr, eval_set=(Xes.astype(np.float32), yes),
                  early_stopping_rounds=50, use_best_model=True)
        else:
            print(f"     {h}: ВНИМАНИЕ: нет отдельных дней для ранней остановки, итераций {a.rounds // 3}")
            m.set_params(iterations=max(a.rounds // 3, 1))
            m.fit(Xtr.astype(np.float32), ytr, sample_weight=wtr)
        print(f"     {h}: деревьев {m.tree_count_}, строк {len(ytr):,}")
        models.append(m)
    return models


def fit_hgb(train, a):
    from sklearn.ensemble import HistGradientBoostingRegressor
    models = []
    for j, h in enumerate(HORIZONS):
        X, y, w = stack_norm(train, j, a.rw)
        if X is None:
            models.append(None)
            continue
        m = HistGradientBoostingRegressor(learning_rate=a.lr, max_iter=a.rounds, max_leaf_nodes=a.leaves,
                                          min_samples_leaf=a.min_leaf, l2_regularization=10.0,
                                          early_stopping=True, validation_fraction=0.1, random_state=0)
        m.fit(X, y, sample_weight=w)
        print(f"     {h}: итераций {m.n_iter_}, строк {len(y):,}")
        models.append(m)
    return models


# ------------------------------------------------------------ общие модели (все инструменты сразу)
POOL_KINDS = ("lgbm_pool", "mlp", "gru")


def stack_multi(train):
    """Все 3 горизонта сразу: X, Y (таргет дня / std дня, пропуск = 0), M (маска 0/1)."""
    xs, ys, ms = [], [], []
    for _, X, Y in train:
        Yn = np.zeros(Y.shape, np.float32)
        M = np.zeros(Y.shape, np.float32)
        for j in range(Y.shape[1]):
            y = Y[:, j]
            m = np.isfinite(y)
            if m.sum() < 100:
                continue
            v = float(y[m].var())
            if not np.isfinite(v) or v < MIN_DAY_VAR:
                continue
            Yn[m, j] = y[m] / np.sqrt(v)
            M[m, j] = 1.0
        keep = M.sum(1) > 0
        xs.append(X[keep]); ys.append(Yn[keep]); ms.append(M[keep])
    if not xs:
        return None, None, None
    return np.vstack(xs).astype(np.float32), np.vstack(ys), np.vstack(ms)


def split_es(train):
    n_es = max(1, len(train) // 10) if len(train) >= 2 else 0
    return train[:len(train) - n_es], train[len(train) - n_es:]


class PoolHead:
    """Общий LightGBM: прогноз для одного инструмента (номер инструмента — последний столбец)."""

    def __init__(self, bst, k):
        self.bst, self.k = bst, k

    def predict(self, X):
        X = np.hstack([X.astype(np.float32), np.full((len(X), 1), self.k, np.float32)])
        return self.bst.predict(X, num_iteration=self.bst.best_iteration)


class GRUHead:
    """GRU: один горизонт одного инструмента. X — строки одного дня подряд (сеть сама разбирает потоки)."""

    def __init__(self, net, k, j, memo):
        self.net, self.k, self.j, self.memo = net, k, j, memo

    def predict(self, X):
        if self.memo.get("X") is not X or self.memo.get("k") != self.k:
            self.memo.update(X=X, k=self.k, p=self.net.predict(X, self.k))
        return self.memo["p"][:, self.j]


class MLPHead:
    """Нейросеть: один горизонт одного инструмента. Сеть считает сразу 3 горизонта — запоминаем."""

    def __init__(self, net, k, j, memo):
        self.net, self.k, self.j, self.memo = net, k, j, memo

    def predict(self, X):
        if self.memo.get("X") is not X or self.memo.get("k") != self.k:     # тот же объект, не id()
            self.memo.update(X=X, k=self.k, p=self.net.predict(X.astype(np.float32), self.k))
        return self.memo["p"][:, self.j]


def fit_lgbm_pool(trains, a):
    import lightgbm as lgb
    insts = list(trains)
    boosters = []
    for j, h in enumerate(HORIZONS):
        parts = {"tr": [], "es": []}
        for k, inst in enumerate(insts):
            tr, es = split_es(trains[inst])
            for nm, part in (("tr", tr), ("es", es)):
                X, y, w = stack_norm(part, j, a.rw if nm == "tr" else None) if part else (None, None, None)
                if X is not None:
                    parts[nm].append((np.hstack([X, np.full((len(X), 1), k, np.float32)]), y, w))
        if not parts["tr"]:
            print(f"     {h}: нет данных для обучения — прогноз будет 0")
            boosters.append(None)
            continue
        Xtr = np.vstack([x for x, _, _ in parts["tr"]]); ytr = np.concatenate([y for _, y, _ in parts["tr"]])
        wtr = None if a.rw is None else np.concatenate([w for _, _, w in parts["tr"]])
        params = dict(objective="huber", alpha=a.huber_delta, learning_rate=a.lr, num_leaves=a.leaves,
                      min_data_in_leaf=a.min_leaf, feature_fraction=a.ff, bagging_fraction=a.bagging,
                      bagging_freq=1, lambda_l2=a.l2, verbose=-1, num_threads=a.threads,
                      seed=0, deterministic=True, force_col_wise=True)
        cat = [Xtr.shape[1] - 1]
        dtr = lgb.Dataset(Xtr, ytr, weight=wtr, categorical_feature=cat)
        if parts["es"]:
            Xes = np.vstack([x for x, _, _ in parts["es"]]); yes = np.concatenate([y for _, y, _ in parts["es"]])
            des = lgb.Dataset(Xes, yes, reference=dtr, categorical_feature=cat)
            bst = lgb.train(params, dtr, num_boost_round=a.rounds, valid_sets=[des],
                            callbacks=[lgb.early_stopping(50, verbose=False)])
        else:
            print(f"     {h}: ВНИМАНИЕ: нет отдельных дней для ранней остановки, деревьев {a.rounds // 3}")
            bst = lgb.train(params, dtr, num_boost_round=max(a.rounds // 3, 1))
            bst.best_iteration = bst.current_iteration()
        print(f"     {h}: деревьев {bst.best_iteration}, строк {len(ytr):,} (все инструменты)")
        boosters.append(bst)
        del Xtr, ytr, dtr
    return {inst: [None if b is None else PoolHead(b, k) for b in boosters] for k, inst in enumerate(insts)}


def fit_mlp(trains, a):
    from mlp import train_mlp
    insts = list(trains)
    tr_parts, es_days = [], []
    for k, inst in enumerate(insts):
        tr, es = split_es(trains[inst])
        X, Y, M = stack_multi(tr)
        if X is not None:
            tr_parts.append((X, Y, M, np.full(len(X), k, np.int64)))
        es_days += [(k, d, X_, Y_) for d, X_, Y_ in es]
    if not tr_parts:
        raise SystemExit("mlp: нет данных для обучения")
    Xtr = np.vstack([p[0] for p in tr_parts]); Ytr = np.vstack([p[1] for p in tr_parts])
    Mtr = np.vstack([p[2] for p in tr_parts]); Itr = np.concatenate([p[3] for p in tr_parts])
    del tr_parts

    def es_score(net):
        cs = []
        for k, d, X_, Y_ in es_days:
            P = net.predict(X_, k)
            for j in range(Y_.shape[1]):
                cs.append(day_corr(P[:, j], Y_[:, j]))
        return float(np.mean(cs))

    print(f"     нейросеть: строк {len(Xtr):,} (все инструменты), вход {Xtr.shape[1]} + {len(insts)}, "
          f"слои {a.mlp_hidden}")
    net, best = train_mlp(Xtr, Ytr, Mtr, Itr, es_score if es_days else None, len(insts),
                          hidden=tuple(a.mlp_hidden), lr=a.mlp_lr, batch=a.mlp_batch,
                          epochs=a.mlp_epochs, seed=0)
    print(f"     лучшая корреляция на днях остановки: {best:.4f}")
    memo = {}
    return {inst: [MLPHead(net, k, j, memo) for j in range(len(HORIZONS))] for k, inst in enumerate(insts)}, net


def fit_gru(trains, a):
    from gru import train_gru
    insts = list(trains)
    days, es_days = [], []
    for k, inst in enumerate(insts):
        tr, es = split_es(trains[inst])
        for d, X, Y in tr:                                # как stack_multi, но строки дня не выкидываем:
            Yf = np.zeros(Y.shape, np.float32)            # последовательность должна идти без дыр
            Mf = np.zeros(Y.shape, np.float32)
            for j in range(Y.shape[1]):
                y = Y[:, j]
                m = np.isfinite(y)
                if m.sum() < 100:
                    continue
                v = float(y[m].var())
                if not np.isfinite(v) or v < MIN_DAY_VAR:
                    continue
                Yf[m, j] = y[m] / np.sqrt(v)
                Mf[m, j] = 1.0
            if Mf.any():
                days.append((k, X, Yf, Mf))
        es_days += [(k, d, X_, Y_) for d, X_, Y_ in es]

    def es_score(net):
        cs = []
        for k, d, X_, Y_ in es_days:
            P = net.predict(X_, k)
            for j in range(Y_.shape[1]):
                cs.append(day_corr(P[:, j], Y_[:, j]))
        return float(np.mean(cs))

    net, best = train_gru(days, es_score if es_days else None, len(insts), hidden=a.gru_hidden,
                          burn=a.gru_burn, seq=a.gru_seq, batch=a.gru_batch, steps=a.gru_steps,
                          epochs=a.gru_epochs, lr=a.gru_lr, seed=0)
    print(f"     лучшая корреляция на днях остановки: {best:.4f}")
    memo = {}
    return {inst: [GRUHead(net, k, j, memo) for j in range(len(HORIZONS))] for k, inst in enumerate(insts)}, net


def pred_std_tail(kind, models, train):
    """Масштаб прогнозов модели: std на последних 10% дней трейна (нужен ансамблю; не зависит
    от валидации). → [std на каждый горизонт] (None, если модели/данных нет)."""
    _, tail = split_es(train) if len(train) >= 2 else ([], train)
    if not tail:
        return [None] * len(HORIZONS)
    X = np.vstack([x for _, x, _ in tail])
    out = []
    for j in range(len(HORIZONS)):
        if models[j] is None:
            out.append(None)
            continue
        if kind == "gru":                             # последовательность — по дням, не через границу дня
            p = np.concatenate([np.asarray(models[j].predict(x), np.float64) for _, x, _ in tail])
        else:
            p = np.asarray(predict_any(kind, models[j], X), dtype=np.float64)
        s = float(np.std(p))
        out.append(s if np.isfinite(s) and s > 0 else None)
    return out


def predict_any(kind, m, X):
    if m is None:                     # модели нет (не было данных) — прогноз 0, как «пропуск» в метрике
        return np.zeros(len(X))
    if kind in POOL_KINDS:
        return m.predict(X)
    if kind == "ridge":
        return predict_ridge(m, X)
    if kind == "lgbm":
        return m.predict(X.astype(np.float32), num_iteration=m.best_iteration)
    return m.predict(X.astype(np.float32))           # catboost, hgb


# ============================================================ оценка и сравнение
FULL_ROWS = np.arange(172_800)


def evaluate(a, inst, kind, models, feats, clip, days):
    rows = []
    for d in days:
        r = load_day(a.cache_dir, a.block, inst, d, feats)
        if r is None:
            raise SystemExit(f"{inst} {d}: файл кэша пропал во время эксперимента")
        X, Y, rws = r
        if getattr(a, "internal", False):
            # дни трейна в кэше прорежены (каждая k-я точка) — для внутренней проверки это допустимо:
            # корреляция по каждой k-й точке — приближённая оценка, не точная метрика
            step = int(rws[1] - rws[0]) if len(rws) > 1 else 1
            want = np.arange(0, len(FULL_ROWS), max(step, 1))
            if step < 1 or not np.array_equal(rws, want) or len(X) != len(want):
                raise SystemExit(f"{inst} {d}: строки дня в кэше не идут ровной сеткой — кэш повреждён")
        elif len(rws) != len(FULL_ROWS) or not np.array_equal(rws, FULL_ROWS) or len(X) != len(FULL_ROWS):
            raise SystemExit(f"{inst} {d}: в кэше день валидации прорежен — метрика была бы неверной. "
                             f"Пересоберите кэш так, чтобы --split был не позже {a.valid[0]}.")
        Y = clip_y(Y, inst, clip)
        P = np.column_stack([np.asarray(predict_any(kind, models[j], X), dtype=np.float64)
                             for j in range(len(HORIZONS))])
        for j, h in enumerate(HORIZONS):
            rows.append({"day": d, "instrument": inst, "horizon": h, "corr": day_corr(P[:, j], Y[:, j])})
        if getattr(a, "save_preds", None):
            save_day_preds(a.save_preds, a.name, inst, d, P, Y)
    return rows


def save_day_preds(root, name, inst, d, P, Y):
    """Прогнозы модели (float32) и таргет (один на папку) для ensemble.py."""
    pd_ = os.path.join(root, name, inst)
    yd = os.path.join(root, "_y", inst)
    os.makedirs(pd_, exist_ok=True); os.makedirs(yd, exist_ok=True)
    yp = os.path.join(yd, f"{d}.npy")
    if os.path.isfile(yp):
        y0 = np.load(yp)
        if y0.shape != Y.shape or not np.array_equal(y0, Y.astype(np.float32), equal_nan=True):
            raise SystemExit(f"{yp}: таргет в папке прогнозов другой (другой режим/кэш) — "
                             f"используйте отдельную --save-preds для --internal и для валидации")
    else:
        np.save(yp, Y.astype(np.float32))
    np.save(os.path.join(pd_, f"{d}.npy"), P.astype(np.float32))


def read_daily(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        cols = f.readline().strip().split(",")
        for line in f:
            r = dict(zip(cols, line.strip().split(",")))
            out[(r["day"], r["instrument"], r["horizon"])] = float(r["corr"]) if r["corr"] else np.nan
    return out


def write_daily(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        f.write("day,instrument,horizon,corr\n")
        for r in rows:
            f.write(f"{r['day']},{r['instrument']},{r['horizon']},{r['corr']:.8f}\n")


CELLS = [(i, h) for i in TARGET_INSTRUMENTS for h in HORIZONS]


def cell_table(rows, expected=None):
    """Как метрика: среднее по дням внутри каждой из 12 пар; SCORE = среднее по 12 парам.
    expected — {инструмент: дни, которые должны быть оценены}. Полнота проверяется по САМИМ
    ДАТАМ (не по их числу): у каждой пары ровно ожидаемый набор дней, без повторов.
    Неполное покрытие → SCORE = NaN, чтобы его нельзя было выдать за настоящий.
    → (cells, score, n_days_per_cell, полное ли покрытие)."""
    cells, seen = {}, {}
    for r in rows:
        k = (r["instrument"], r["horizon"])
        cells.setdefault(k, []).append(r["corr"])
        seen.setdefault(k, []).append(r["day"])
    means = {k: float(np.mean(v)) for k, v in cells.items()}
    ndays = {k: len(v) for k, v in cells.items()}
    complete = set(means) == set(CELLS) and all(len(v) == len(set(v)) for v in seen.values())
    if complete and expected is not None:
        complete = all(set(seen[(i, h)]) == set(expected.get(i, ())) for (i, h) in CELLS)
    elif complete:                                   # без ожиданий: хотя бы одинаковые даты у всех пар
        complete = len({frozenset(v) for v in seen.values()}) == 1
    score = float(np.mean([means[c] for c in CELLS])) if complete else np.nan
    return means, score, ndays, complete


def paired(cur, ref):
    """Сравнение двух экспериментов по структуре метрики: SCORE = среднее по 12 парам
    от среднего по дням внутри пары. Поэтому и разница считается так же: внутри каждой пары —
    средняя дневная разница по дням, где эта пара есть у ОБОИХ экспериментов, потом среднее
    по 12 парам. Дни, где у инструмента нет таргетов, честно выпадают только из его пар
    (как в метрике), а не целиком.

    Неопределённость — бутстрэп по календарным дням и sign-flip (знак разницы меняется сразу
    у всего дня); внутри каждой повторной выборки статистика считается той же формулой.
    """
    if set(cur) != set(ref):
        return None                       # разное покрытие (дни или пары) — сравнивать нельзя (см. main)
    common = {k for k in cur if k in ref and np.isfinite(cur[k]) and np.isfinite(ref[k])}
    only_one = len((set(cur) | set(ref)) - common)
    days = sorted({k[0] for k in common})
    if not days or {(i, h) for (_, i, h) in common} != set(CELLS):
        return None
    di = {d: n for n, d in enumerate(days)}
    D = np.full((len(days), len(CELLS)), np.nan)              # дни × 12, NaN = пары нет в этот день
    for (d, i, h) in common:
        D[di[d], CELLS.index((i, h))] = cur[(d, i, h)] - ref[(d, i, h)]
    have = np.isfinite(D)
    D0 = np.where(have, D, 0.0)

    def stat(w):
        """w — вес каждого дня (сколько раз он попал в выборку, со знаком для sign-flip)."""
        num = (w[:, None] * D0).sum(0)
        den = (np.abs(w)[:, None] * have).sum(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            cell = num / den
        return np.nanmean(cell, axis=-1) if cell.ndim > 1 else float(np.nanmean(cell))

    n = len(days)
    delta = stat(np.ones(n))
    W = np.stack([np.bincount(RNG.integers(0, n, n), minlength=n) for _ in range(4000)]).astype(float)
    num = W @ D0
    den = W @ have.astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        bm = np.nanmean(num / den, axis=1)
    S = RNG.choice([-1.0, 1.0], size=(20000, n))
    with np.errstate(invalid="ignore", divide="ignore"):
        null = np.nanmean((S @ D0) / have.sum(0), axis=1)
    hits = int(np.count_nonzero(np.abs(null) >= abs(delta) - 1e-15))
    p = (hits + 1) / (len(S) + 1)                        # Монте-Карло: p не может быть 0
    dd = np.nanmean(D, axis=1)                           # описательно: средняя разница дня по его парам
    with np.errstate(invalid="ignore"):
        cells = {c: float(np.nanmean(D[:, k])) for k, c in enumerate(CELLS)}
    return {"days": n, "dropped": only_one,
            "delta": float(delta), "lo": float(np.nanpercentile(bm, 2.5)),
            "hi": float(np.nanpercentile(bm, 97.5)), "p": p, "share_days_better": float((dd > 0).mean()),
            "cells": cells}


def feats_id(feats):
    """Короткий отпечаток набора и порядка признаков (сверяет ensemble.py)."""
    import hashlib
    return hashlib.sha1("|".join(feats).encode()).hexdigest()[:12]


# ============================================================ сохранение модели для predict.py
def save_model(a, kind, feats, all_models, pred_std=None, net=None):
    os.makedirs(a.save, exist_ok=True)
    meta = {"type": kind, "features": feats, "name": a.name, "train": a.train, "precision": "float32",
            "fhash": FHASH, "features_hash": FEATURES_HASH}   # predict.py сверит версию признаков
    if pred_std is not None:
        meta["pred_std"] = pred_std
    from features_v2 import V2_NAMES
    from features_v3 import V3_NAMES
    if any(f in V2_NAMES + V3_NAMES for f in feats):          # v3 зависит от помощников v2
        meta["features_v2_hash"] = FEATURES_V2_HASH
    if any(f in V3_NAMES for f in feats):
        meta["features_v3_hash"] = FEATURES_V3_HASH
    if kind == "ridge":
        meta["coef"] = {inst: {h: all_models[inst][j] for j, h in enumerate(HORIZONS)
                               if all_models[inst][j] is not None} for inst in all_models}
        meta["lam"] = a.lam
    elif kind == "lgbm":
        meta["files"] = {}
        for inst, ms in all_models.items():
            meta["files"][inst] = {}
            for j, h in enumerate(HORIZONS):
                if ms[j] is None:
                    continue
                fn = f"{inst}_{h}.txt"
                ms[j].save_model(os.path.join(a.save, fn), num_iteration=ms[j].best_iteration)
                meta["files"][inst][h] = fn
    elif kind == "catboost":
        meta["files"] = {}
        for inst, ms in all_models.items():
            meta["files"][inst] = {}
            for j, h in enumerate(HORIZONS):
                if ms[j] is None:
                    continue
                fn = f"{inst}_{h}.cbm"
                ms[j].save_model(os.path.join(a.save, fn))
                meta["files"][inst][h] = fn
    elif kind == "lgbm_pool":
        insts = list(all_models)
        meta["inst_index"] = {inst: k for k, inst in enumerate(insts)}
        meta["files"] = {}
        for j, h in enumerate(HORIZONS):
            head = all_models[insts[0]][j]
            if head is None:
                continue
            fn = f"pool_{h}.txt"
            head.bst.save_model(os.path.join(a.save, fn), num_iteration=head.bst.best_iteration)
            meta["files"][h] = fn
    elif kind == "mlp":
        meta["inst_index"] = {inst: k for k, inst in enumerate(all_models)}
        net.save(os.path.join(a.save, "mlp.npz"))
        meta["file"] = "mlp.npz"
    elif kind == "gru":
        meta["inst_index"] = {inst: k for k, inst in enumerate(all_models)}
        meta["precision"] = "float32"
        net.save(os.path.join(a.save, "gru.npz"))
        meta["file"] = "gru.npz"
    else:
        print("  (модель hgb не сохраняется — для сдачи используйте ridge, lgbm, catboost, lgbm_pool, mlp)")
        return
    with open(os.path.join(a.save, "model.json"), "w") as f:
        json.dump(meta, f, indent=1)
    mp = os.path.join(a.save, "model.json")
    print(f"модель сохранена: {mp}  (для сдачи: predict.py --model {mp})")


# ============================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--data-dir", required=True, help="нужна только meta/target_clip.parquet")
    ap.add_argument("--block", default="base")
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", choices=["ridge", "lgbm", "catboost", "hgb", "lgbm_pool", "mlp", "gru"], default="ridge")
    ap.add_argument("--cb-depth", type=int, default=6, help="catboost: глубина симметричных деревьев")
    ap.add_argument("--features", nargs="*", default=None, help="подмножество признаков (по умолчанию все из кэша)")
    ap.add_argument("--drop", nargs="*", default=[], help="убрать эти признаки")
    ap.add_argument("--ref", default=None, help="имя прошлого эксперимента для сравнения")
    ap.add_argument("--final", action="store_true",
                    help="финальная модель для сдачи: учим на трейне + валидации (по умолчанию "
                         "2025-09-15..2026-09-14), без оценки; нужен --save")
    ap.add_argument("--internal", action="store_true",
                    help=f"внутренняя проверка на трейне: учим {INTERNAL_TRAIN[0]}..{INTERNAL_TRAIN[1]}, "
                         f"проверяем {INTERNAL_VALID[0]}..{INTERNAL_VALID[1]} (валидация не трогается)")
    ap.add_argument("--train", nargs=2, default=None)
    ap.add_argument("--valid", nargs=2, default=None)
    ap.add_argument("--train-day-step", type=int, default=1)
    ap.add_argument("--valid-day-step", type=int, default=1)
    ap.add_argument("--lam", type=float, default=0.01, help="ridge")
    ap.add_argument("--max-rows", type=int, default=3_000_000, help="lgbm/hgb: строк на модель")
    ap.add_argument("--rounds", type=int, default=600)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--leaves", type=int, default=31)
    ap.add_argument("--min-leaf", type=int, default=2000)
    ap.add_argument("--huber-delta", type=float, default=1.0, help="в единицах std таргета дня")
    ap.add_argument("--ff", type=float, default=0.8, help="lgbm/lgbm_pool: feature_fraction")
    ap.add_argument("--bagging", type=float, default=0.8, help="lgbm/lgbm_pool: bagging_fraction")
    ap.add_argument("--l2", type=float, default=10.0, help="lgbm/lgbm_pool: lambda_l2")
    ap.add_argument("--recency-halflife", type=float, default=None,
                    help="вес дня трейна = 0.5 ** (дней до конца трейна / N); без флага все дни весят одинаково "
                         "(ridge, lgbm, catboost, hgb, lgbm_pool; не mlp)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--mlp-hidden", type=int, nargs="+", default=[256, 128, 64])
    ap.add_argument("--mlp-epochs", type=int, default=15)
    ap.add_argument("--mlp-lr", type=float, default=1e-3)
    ap.add_argument("--mlp-batch", type=int, default=4096)
    ap.add_argument("--gru-hidden", type=int, default=64)
    ap.add_argument("--gru-burn", type=int, default=64, help="шагов разогрева состояния (шаг 2.5 с)")
    ap.add_argument("--gru-seq", type=int, default=64, help="шагов, на которых считается лосс")
    ap.add_argument("--gru-batch", type=int, default=256)
    ap.add_argument("--gru-steps", type=int, default=1500, help="батчей в эпохе")
    ap.add_argument("--gru-epochs", type=int, default=20)
    ap.add_argument("--gru-lr", type=float, default=2e-3)
    ap.add_argument("--strict-cache", action="store_true",
                    help="не принимать файлы кэша без отпечатка версии (финальное обучение)")
    ap.add_argument("--save-preds", default=None,
                    help="папка: сохранить прогнозы на днях проверки (для ensemble.py)")
    ap.add_argument("--save", default=None, help="папка для модели (для predict.py)")
    ap.add_argument("--results", default="results")
    a = ap.parse_args()
    if a.final:
        if a.internal or a.save_preds or a.ref or a.valid:
            raise SystemExit("--final несовместим с --internal / --save-preds / --ref / --valid")
        if not a.save:
            raise SystemExit("--final: укажите --save <папка>")
        a.train = a.train or [TRAIN[0], VALID[1]]
        nxt = (dt.date.fromisoformat(a.train[1]) + dt.timedelta(days=1)).isoformat()
        a.valid = [nxt, nxt]                          # только для отсечки строк, чей таргет уходит за конец данных
    a.train = a.train or list(INTERNAL_TRAIN if a.internal else TRAIN)
    a.valid = a.valid or list(INTERNAL_VALID if a.internal else VALID)
    if a.internal:
        if a.valid[1] >= VALID[0]:
            raise SystemExit(f"--internal: проверочный период должен закончиться до настоящей валидации {VALID[0]}")
        if a.save:
            raise SystemExit("--internal: модель учится на укороченном трейне — для сдачи её не сохраняем")
    if a.recency_halflife is not None and a.recency_halflife <= 0:
        raise SystemExit("--recency-halflife должен быть > 0")
    if a.recency_halflife and a.model in ("mlp", "gru"):
        raise SystemExit(f"--recency-halflife для {a.model} не сделан")
    a.rw = recency_weight(a.train[1], a.recency_halflife)
    os.makedirs(a.results, exist_ok=True)

    names = cached_names(a.cache_dir, a.block)
    if names is None:
        raise SystemExit("кэш пуст — сначала запустите cache.py")
    unknown = [f for f in list(a.drop) + list(a.features or []) if f not in names]
    if unknown:
        raise SystemExit(f"в кэше '{a.block}' нет признаков {unknown} — опечатка в --drop / --features?")
    feats = [f for f in (a.features or names) if f not in a.drop]
    if a.train[1] >= a.valid[0]:
        raise SystemExit(f"трейн ({a.train[0]}..{a.train[1]}) должен заканчиваться до начала валидации {a.valid[0]}")
    check_cache_version(a.cache_dir, a.block, strict=a.strict_cache)
    expected = check_coverage(a) if not a.final else check_train_coverage(a)
    clip = load_clip(a.data_dir)
    print(f"[{a.name}] модель {a.model}, признаков {len(feats)}"
          + (f", вес дня по свежести: полураспад {a.recency_halflife:g} дн." if a.rw else ""))

    t0 = time.time()
    rows, all_models, pstd, trains, net = [], {}, {}, {}, None
    for inst in TARGET_INSTRUMENTS:
        tt = time.time()
        train = load_train(a, inst, feats, clip, None if a.model == "ridge" else a.max_rows,
                           whole_days=a.model == "gru")
        if not train:
            print(f"  {inst}: нет дней трейна в кэше")
            continue
        if a.model in POOL_KINDS:                    # общая модель: сначала собираем все инструменты
            trains[inst] = train
            print(f"  {inst}: загружено {sum(len(x) for _, x, _ in train):,} строк")
            continue
        if a.model == "ridge":
            models = fit_ridge(train, a.lam, a.rw)
        elif a.model == "lgbm":
            models = fit_lgbm(train, a)
        elif a.model == "catboost":
            models = fit_catboost(train, a)
        else:
            models = fit_hgb(train, a)
        pstd[inst] = pred_std_tail(a.model, models, train)
        del train
        all_models[inst] = models
        if not a.final:
            rows += evaluate(a, inst, a.model, models, feats, clip, expected[inst])
        print(f"  {inst}: {time.time() - tt:.0f} с")
    if a.model in POOL_KINDS and trains:
        tt = time.time()
        if a.model == "lgbm_pool":
            all_models = fit_lgbm_pool(trains, a)
        elif a.model == "gru":
            all_models, net = fit_gru(trains, a)
        else:
            all_models, net = fit_mlp(trains, a)
        for inst in all_models:
            pstd[inst] = pred_std_tail(a.model, all_models[inst], trains[inst])
        trains.clear()
        print(f"  обучение: {time.time() - tt:.0f} с")
        for inst in all_models:
            if not a.final:
                rows += evaluate(a, inst, a.model, all_models[inst], feats, clip, expected[inst])
    pstd = {inst: {h: s[j] for j, h in enumerate(HORIZONS) if s[j] is not None} for inst, s in pstd.items()}
    with open(os.path.join(a.results, f"{a.name}_predstd.json"), "w") as f:   # + тип и набор признаков:
        json.dump({**pstd, "_kind": a.model, "_feat": feats_id(feats)}, f, indent=1)  # ensemble.py сверит
    if a.final:
        save_model(a, a.model, feats, all_models, pstd, net)
        print(f"финальная модель (трейн {a.train[0]}..{a.train[1]}, без оценки): {time.time() - t0:.0f} с")
        return

    write_daily(os.path.join(a.results, f"{a.name}_daily.csv"), rows)
    cells, score, ndays, complete = cell_table(rows, expected)
    if not complete:
        print(f"ВНИМАНИЕ: покрытие неполное (пар: {len(cells)} из 12, дней по парам: "
              f"{sorted(set(ndays.values()))}) — SCORE не считается")

    ref = None
    if a.ref:
        rp = os.path.join(a.results, f"{a.ref}_daily.csv")
        if os.path.isfile(rp):
            cur = {(r["day"], r["instrument"], r["horizon"]): r["corr"] for r in rows}
            ref = paired(cur, read_daily(rp))
        else:
            print(f"нет {rp} — сравнение пропущено")
        if os.path.isfile(rp) and ref is None:
            print(f"у {a.ref} другое покрытие оценки (дни или пары: другой режим --internal / обычный, "
                  f"другой --valid или кэш) — сравнение пропущено")

    print(f"\n{'instrument':20} {'horizon':8} {'corr':>8}" + (f" {'vs ' + a.ref:>14}" if ref else ""))
    for inst in TARGET_INSTRUMENTS:
        for h in HORIZONS:
            if (inst, h) in cells:
                extra = f" {ref['cells'].get((inst, h), np.nan):+14.4f}" if ref else ""
                print(f"{inst:20} {h:8} {cells[(inst, h)]:8.4f}{extra}")
    if a.internal:
        label = f"SCORE внутренней проверки ({a.valid[0]}..{a.valid[1]}, прореженные точки — не настоящая валидация)"
    elif a.valid_day_step == 1:
        label = "SCORE"
    else:
        label = f"SCORE по выборке дней (каждый {a.valid_day_step}-й, не полная валидация)"
    print(f"\n{label}: {score:.4f}   (дней валидации: {max(ndays.values()) if ndays else 0}, {time.time() - t0:.0f} с)")
    if ref:
        sig = ref["p"] < 0.05
        verdict = ("лучше" if sig and ref["lo"] > 0 else "хуже" if sig and ref["hi"] < 0
                   else "разница в пределах шума")
        print(f"против {a.ref}: {ref['delta']:+.4f} (95% по дням: {ref['lo']:+.4f} … {ref['hi']:+.4f}), "
              f"p = {ref['p']:.4f}, лучше в {ref['share_days_better']:.0%} дней из {ref['days']} → {verdict}")
        if ref["dropped"]:
            print(f"  (в сравнении пропущено {ref['dropped']} значений «день × пара»: они есть только у одного "
                  f"из двух экспериментов)")

    jp = os.path.join(a.results, "journal.csv")
    new = not os.path.isfile(jp)
    with open(jp, "a", encoding="utf-8") as f:
        cols = [f"{i}|{h}" for i in TARGET_INSTRUMENTS for h in HORIZONS]
        if new:
            f.write("time,name,model,n_features,features,ref,score,delta_vs_ref,delta_lo,delta_hi," + ",".join(cols) + "\n")
        f.write(",".join([dt.datetime.now().strftime("%Y-%m-%d %H:%M"), a.name,
                          a.model + ("@internal" if a.internal else ""), str(len(feats)),
                          "|".join(feats) if len(feats) < 40 else "many", a.ref or "", f"{score:.5f}",
                          f"{ref['delta']:.5f}" if ref else "", f"{ref['lo']:.5f}" if ref else "",
                          f"{ref['hi']:.5f}" if ref else ""] +
                         [f"{cells.get(tuple(c.split('|')), np.nan):.5f}" for c in cols]) + "\n")
    print(f"записано в {jp}")

    if a.save:
        save_model(a, a.model, feats, all_models, pstd, net)


if __name__ == "__main__":
    from textio import utf8_console
    utf8_console()
    main()
