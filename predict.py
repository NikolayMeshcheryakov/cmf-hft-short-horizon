"""Прогноз на один день в формате сдачи.

    python predict.py --data-dir ./data --date 2026-09-20 --out preds/2026-09-20.parquet

Берёт модель из model.json (лежит рядом со скриптом, или укажите --model).
В папке данных должен быть нужный день и предыдущий (для прогрева окон).
Выход: ts, instrument, pred_500ms, pred_2s, pred_8s — 4 × 172 800 строк.
"""
import argparse
import json
import os
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from common import HORIZONS, PREDICTION_COLUMNS, TARGET_INSTRUMENTS, grid_points
from features import FEATURE_NAMES, compute_features
from features_v2 import V2_NAMES, compute_v2
from features_v3 import V3_NAMES, compute_v3
from fversion import FEATURES_HASH, FEATURES_V2_HASH, FEATURES_V3_HASH, FHASH

DEFAULT_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.json")


def check_model_version(m, path):
    """Модель обучена на признаках той же версии features.py? Иначе прогноз — мусор.
    features.py (наш файл) — строго; common.py (файл организаторов) — только предупреждение."""
    fh = m.get("features_hash")
    if fh is None:
        print(f"ВНИМАНИЕ: {path} без отпечатка версии признаков (старая модель) — версия не проверена",
              file=sys.stderr)
    elif fh != FEATURES_HASH:
        raise SystemExit(f"{path}: модель обучена на другой версии features.py ({fh} ≠ {FEATURES_HASH}). "
                         f"Переобучите модель.")
    elif m.get("fhash") not in (None, FHASH):
        print("ВНИМАНИЕ: common.py отличается от того, с которым обучалась модель — "
              "проверьте, что формулы котировок не менялись", file=sys.stderr)
    missing = [f for f in m["features"] if f not in FEATURE_NAMES + V2_NAMES + V3_NAMES]
    if missing:
        raise SystemExit(f"{path}: в features.py / features_v2.py / features_v3.py нет признаков {missing}")
    if any(f in V3_NAMES for f in m["features"]):
        f3 = m.get("features_v3_hash")
        if f3 != FEATURES_V3_HASH:
            raise SystemExit(f"{path}: модель обучена на другой версии features_v3.py ({f3} ≠ {FEATURES_V3_HASH}). "
                             f"Переобучите модель.")
    if any(f in V2_NAMES + V3_NAMES for f in m["features"]):   # v3 использует помощники из v2
        f2 = m.get("features_v2_hash")
        if f2 is None:
            raise SystemExit(f"{path}: модель использует признаки v2/v3, но без отпечатка features_v2.py")
        if f2 != FEATURES_V2_HASH:
            raise SystemExit(f"{path}: модель обучена на другой версии features_v2.py ({f2} ≠ {FEATURES_V2_HASH}). "
                             f"Переобучите модель.")


def load_model(path=DEFAULT_MODEL):
    """model.json от train.py / experiment.py / ensemble.py. Рядом лежат файлы деревьев / сети;
    у ансамбля — папки участников (members/<k>/model.json)."""
    with open(path) as f:
        m = json.load(f)
    base = os.path.dirname(os.path.abspath(path))
    if m.get("type") == "ensemble":
        m["loaded"] = [load_model(os.path.join(base, mb["path"])) for mb in m["members"]]
        feats = []
        for sub in m["loaded"]:
            feats += [f for f in sub["features"] if f not in feats]
        m["features"] = feats
        return m
    check_model_version(m, path)
    if m.get("type") == "lgbm_pool":
        import lightgbm as lgb
        m["boosters"] = {h: lgb.Booster(model_file=os.path.join(base, fn)) for h, fn in m["files"].items()}
    elif m.get("type") == "mlp":
        from mlp import MLPNumpy
        m["net"] = MLPNumpy.load(os.path.join(base, m["file"]))
    elif m.get("type") == "gru":
        from gru import GRUNumpy
        m["net"] = GRUNumpy.load(os.path.join(base, m["file"]))
    elif m.get("type") == "lgbm":
        import lightgbm as lgb
        base = os.path.dirname(os.path.abspath(path))
        m["boosters"] = {inst: {h: lgb.Booster(model_file=os.path.join(base, fn)) for h, fn in hs.items()}
                         for inst, hs in m["files"].items()}
    elif m.get("type") == "catboost":
        from catboost import CatBoostRegressor
        base = os.path.dirname(os.path.abspath(path))
        m["boosters"] = {}
        for inst, hs in m["files"].items():
            m["boosters"][inst] = {}
            for h, fn in hs.items():
                cb = CatBoostRegressor()
                cb.load_model(os.path.join(base, fn))
                m["boosters"][inst][h] = cb
    return m


def model_predict(model, inst, h, X):
    """Прогноз одной модели (инструмент, горизонт); None, если такой модели нет."""
    if model.get("type") == "lgbm_pool":
        b, k = model["boosters"].get(h), model["inst_index"].get(inst)
        if b is None or k is None:
            return None
        return b.predict(np.hstack([X.astype(np.float32), np.full((len(X), 1), k, np.float32)]))
    if model.get("type") in ("mlp", "gru"):
        k = model["inst_index"].get(inst)
        if k is None:
            return None
        memo = model.setdefault("_memo", {})
        if memo.get("X") is not X or memo.get("k") != k:    # сеть считает 3 горизонта разом
            if model["type"] == "gru":                        # полный день: строки сетки подряд
                memo.update(X=X, k=k, p=model["net"].predict(X.astype(np.float32), k, row_gap=1))
            else:
                memo.update(X=X, k=k, p=model["net"].predict(X.astype(np.float32), k))
        return memo["p"][:, HORIZONS.index(h)]
    if model.get("type") in ("lgbm", "catboost"):
        b = model["boosters"].get(inst, {}).get(h)
        return None if b is None else b.predict(X.astype(np.float32))
    c = model["coef"].get(inst, {}).get(h)
    return None if c is None else c["a"] + X @ np.array(c["b"])


def single_predict(model, inst, Xall, names, n):
    """Одна (не ансамбль) модель: {горизонт: массив n}."""
    X = Xall[:, [names.index(f) for f in model["features"]]]
    if model.get("precision", "float32" if model.get("type") != "ridge" else "float64") == "float32":
        X = X.astype(np.float32).astype(np.float64)            # модель училась на кэше (float32)
    out = {}
    for h in HORIZONS:
        p = model_predict(model, inst, h, X)
        p = np.zeros(n) if p is None else np.asarray(p, dtype=np.float64)
        out[h] = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
    model.pop("_memo", None)
    return out


def predict_day(data_dir, date, model):
    """{инструмент: {горизонт: массив 172_800}} — прогноз на полной сетке дня.
    Ансамбль: признаки дня считаются один раз, прогноз = Σ коэф. участника × его прогноз
    (коэффициент = вес / масштаб прогнозов участника на трейне, задаётся ensemble.py)."""
    feats = model["features"]
    use_v2 = any(f in V2_NAMES for f in feats)
    use_v3 = any(f in V3_NAMES for f in feats)
    names = FEATURE_NAMES + (V2_NAMES if use_v2 else []) + (V3_NAMES if use_v3 else [])
    cache = {}                                   # котировки соседей и спота читаем один раз
    n = len(grid_points(date))
    out = {}
    for inst in TARGET_INSTRUMENTS:
        Xall = compute_features(data_dir, inst, date, cache)
        if use_v2:
            Xall = np.hstack([Xall, compute_v2(data_dir, inst, date, cache)])
        if use_v3:
            Xall = np.hstack([Xall, compute_v3(data_dir, inst, date, cache)])
        if model.get("type") != "ensemble":
            out[inst] = single_predict(model, inst, Xall, names, n)
            continue
        out[inst] = {h: np.zeros(n) for h in HORIZONS}
        for mb, sub in zip(model["members"], model["loaded"]):
            ps = single_predict(sub, inst, Xall, names, n)
            for h in HORIZONS:
                c = mb["coef"].get(inst, {}).get(h, 0.0)
                if c:
                    out[inst][h] = out[inst][h] + c * ps[h]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--date", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    a = ap.parse_args()

    preds = predict_day(a.data_dir, a.date, load_model(a.model))
    grid = grid_points(a.date)
    table = pa.table({
        "ts": np.concatenate([grid] * len(TARGET_INSTRUMENTS)),
        "instrument": np.concatenate([np.full(len(grid), i) for i in TARGET_INSTRUMENTS]),
        **{c: np.concatenate([preds[i][h] for i in TARGET_INSTRUMENTS]).astype("float64")
           for c, h in zip(PREDICTION_COLUMNS, HORIZONS)},
    })
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    pq.write_table(table, a.out, compression="zstd")
    print(f"{table.num_rows:,} строк -> {a.out}")


if __name__ == "__main__":
    from textio import utf8_console
    utf8_console()
    main()
