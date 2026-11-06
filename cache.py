"""Кэш признаков: считаем один раз, дальше эксперименты читают готовое с диска.

    py -3.13 cache.py --data-dir D:\\hft\\data --cache-dir D:\\hft\\cache --workers 4

Что сохраняется (по файлу на инструмент-день, cache/<block>/<инструмент>/<день>.npz):
  X     — признаки (float32), имена в names
  Y     — СЫРЫЕ таргеты y_500ms, y_2s, y_8s (float32, обрезка делается при загрузке)
  rows  — номера строк 500-мс сетки дня, которые сохранены
Трейн (до --split) хранится с прореживанием --row-step (по умолчанию каждая 5-я точка = раз в 2.5 с:
соседние таргеты всё равно почти одинаковые). Валидация хранится целиком — чтобы метрика была точной.

Повторный запуск докачивает только недостающие файлы. Если признаки в features.py поменялись,
задайте новое --block (например v2), старый кэш останется.

Место на диске: трейн ≈ 3 МБ, валидация ≈ 15 МБ на инструмент-день (при 21 признаке)
→ вся история ≈ 7–8 ГБ.
"""
import argparse
import os
import time
from multiprocessing import Pool

import numpy as np

from common import HORIZONS, TARGET_INSTRUMENTS, dates
from features import FEATURE_NAMES, compute_features
from fversion import FHASH, V2HASH, V3HASH, files_hash
from metric import load_targets

TRAIN = ("2025-09-15", "2026-07-14")
VALID = ("2026-07-15", "2026-09-14")
NO_CLIP = {(i, h): (-np.inf, np.inf) for i in TARGET_INSTRUMENTS for h in HORIZONS}


def features_hash():
    return files_hash(["features.py", "common.py"])


# ---------------------------------------------------------------- блоки признаков
# Блок — папка кэша с одним набором признаков. «base» — 21 признак из features.py,
# «v2…» — дополнительные признаки из features_v2.py (без повтора базовых).
# В экспериментах блоки складываются через «+»: --block base+v2.
def parts(block):
    return [b for b in block.split("+") if b]


def is_v2(block):
    return block.startswith("v2")


def is_v3(block):
    return block.startswith("v3")


def block_hash(block):
    return V3HASH if is_v3(block) else V2HASH if is_v2(block) else FHASH


def block_names(block):
    if is_v3(block):
        from features_v3 import V3_NAMES
        return V3_NAMES
    if is_v2(block):
        from features_v2 import V2_NAMES
        return V2_NAMES
    return FEATURE_NAMES


def block_compute(block):
    if is_v3(block):
        from features_v3 import compute_v3
        return compute_v3
    if is_v2(block):
        from features_v2 import compute_v2
        return compute_v2
    return compute_features


def first_file(cache_dir, block):
    base = os.path.join(cache_dir, block)
    for inst in TARGET_INSTRUMENTS:
        d = os.path.join(base, inst)
        if os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                if f.endswith(".npz"):
                    return os.path.join(d, f)
    return None


def all_files(cache_dir, block):
    base = os.path.join(cache_dir, block)
    for inst in TARGET_INSTRUMENTS:
        d = os.path.join(base, inst)
        if os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                if f.endswith(".npz"):
                    yield os.path.join(d, f)


_NO_HASH_WARNED = [False]


def check_cache_version(cache_dir, block, strict=False):
    """Весь кэш посчитан тем же кодом признаков, что сейчас? Проверяется КАЖДЫЙ файл каждого
    блока (читается только отпечаток, это быстро). Чужая версия хоть в одном файле — стоп.
    strict — файлы без отпечатка тоже стоп (для финального обучения; подписать: cache.py --sign)."""
    for blk in parts(block):
        want = block_hash(blk)
        no_hash, bad = 0, []
        for p in all_files(cache_dir, blk):
            with open(p, "rb") as f:
                z = np.load(f, allow_pickle=False)
                fh = str(z["fhash"]) if "fhash" in z.files else None
            if fh is None:
                no_hash += 1
            elif fh != want:
                bad.append((p, fh))
        if bad:
            raise SystemExit(f"в кэше '{blk}' {len(bad)} файл(ов) посчитаны другой версией признаков "
                             f"(например {bad[0][0]}: {bad[0][1]} ≠ {want}). "
                             f"Соберите кэш заново с другим --block.")
        if no_hash and strict:
            raise SystemExit(f"в кэше '{blk}' {no_hash} файл(ов) без отпечатка версии — для финального обучения "
                             f"нельзя. Подпишите после проверки: cache.py --sign --block {blk}")
        if no_hash:
            print(f"ВНИМАНИЕ: {no_hash} файл(ов) кэша '{blk}' без отпечатка версии (собраны старым cache.py). "
                  "Если features.py с тех пор не менялся — всё в порядке.")
            _NO_HASH_WARNED[0] = True


def path_for(cache_dir, block, inst, day):
    return os.path.join(cache_dir, block, inst, f"{day}.npz")


def day_available(cache_dir, block, inst, day):
    """Есть ли день во ВСЕХ блоках (base+v2 → оба файла)."""
    return all(os.path.isfile(path_for(cache_dir, b, inst, day)) for b in parts(block))


def build_one(job):
    data_dir, cache_dir, block, inst, day, step = job
    p = path_for(cache_dir, block, inst, day)
    if os.path.isfile(p):
        return inst, day, "есть"
    ys = load_targets(data_dir, inst, day, NO_CLIP)
    if ys is None:
        return inst, day, "нет таргетов"
    X = block_compute(block)(data_dir, inst, day)
    rows = np.arange(0, X.shape[0], step, dtype=np.int32)
    Y = np.column_stack([ys[h] for h in HORIZONS])
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".tmp"
    with open(tmp, "wb") as f:                    # сначала во временный файл: обрыв не оставит битый
        np.savez(f, X=X[rows].astype(np.float32), Y=Y[rows].astype(np.float32), rows=rows,
                 names=np.array(block_names(block)), fhash=np.array(block_hash(block)))
    os.replace(tmp, p)
    return inst, day, "ok"


def _check_file_version(path, fh, want=FHASH):
    """Каждый файл кэша сверяется с текущей версией признаков, а не только первый."""
    if fh is None:
        if not _NO_HASH_WARNED[0]:
            print("ВНИМАНИЕ: в кэше есть файлы без отпечатка версии (собраны старым cache.py). "
                  "Если features.py с тех пор не менялся — всё в порядке.")
            _NO_HASH_WARNED[0] = True
    elif fh != want:
        raise SystemExit(f"{path}: посчитан другой версией признаков ({fh} ≠ {want}). "
                         f"В одном кэше смешаны версии признаков — соберите кэш заново с новым --block.")


def _load_one(cache_dir, blk, inst, day):
    p = path_for(cache_dir, blk, inst, day)
    if not os.path.isfile(p):
        return None
    with open(p, "rb") as f:
        z = np.load(f, allow_pickle=False)
        names = [str(n) for n in z["names"]]
        X, Y, rows = z["X"], z["Y"], z["rows"]
        fh = str(z["fhash"]) if "fhash" in z.files else None
    _check_file_version(p, fh, block_hash(blk))
    return names, X, Y, rows


def load_day(cache_dir, block, inst, day, features=None):
    """→ (X, Y, rows) или None. block может быть суммой блоков (base+v2): признаки склеиваются,
    строки обязаны совпадать. features — список имён (подмножество сохранённых)."""
    got = [_load_one(cache_dir, b, inst, day) for b in parts(block)]
    if any(g is None for g in got):
        return None
    names, Xs = [], []
    rows, Y = got[0][3], got[0][2]
    for (nm, X, yb, r), b in zip(got, parts(block)):
        if not np.array_equal(yb, Y, equal_nan=True):
            raise SystemExit(f"{inst} {day}: в блоке '{b}' другие таргеты, чем в '{parts(block)[0]}' — "
                             f"блоки собраны по разным данным, пересоберите кэш")
        if not np.array_equal(r, rows):
            raise SystemExit(f"{inst} {day}: в блоках {block} разные строки (разный --row-step или --split "
                             f"при сборке '{b}') — соберите блоки с одинаковыми параметрами")
        names += nm
        Xs.append(X)
    if len(set(names)) != len(names):
        raise SystemExit(f"в блоках {block} повторяются имена признаков")
    X = Xs[0] if len(Xs) == 1 else np.hstack(Xs)
    if features is not None:
        missing = [f for f in features if f not in names]
        if missing:
            raise KeyError(f"в кэше нет признаков {missing}; пересоберите кэш с новым --block")
        X = X[:, [names.index(f) for f in features]]
    return X, Y, rows


def cached_names(cache_dir, block):
    out = []
    for b in parts(block):
        p = first_file(cache_dir, b)
        if p is None:
            return None
        with open(p, "rb") as fh:
            out += [str(n) for n in np.load(fh)["names"]]
    return out


def sign_block(data_dir, cache_dir, blk, per_inst=3):
    """Подписать файлы кэша без отпечатка версии — только если они совпадают с тем, что считает
    ТЕКУЩИЙ код. Проверка: per_inst дней на инструмент (равномерно по истории, включая первый и
    последний) пересчитываются заново и сравниваются с кэшем побитово (X, Y, имена). Любое
    расхождение — стоп, ничего не подписывается."""
    want = block_hash(blk)
    unsigned = []
    for p in all_files(cache_dir, blk):
        with open(p, "rb") as f:
            if "fhash" not in np.load(f, allow_pickle=False).files:
                unsigned.append(p)
    if not unsigned:
        print(f"'{blk}': все файлы уже подписаны")
        return
    print(f"'{blk}': без отпечатка {len(unsigned)} файл(ов); проверяю {per_inst} дн. на инструмент пересчётом")
    by_inst = {}
    for p in unsigned:
        by_inst.setdefault(os.path.basename(os.path.dirname(p)), []).append(p)
    for inst, ps in sorted(by_inst.items()):
        ps = sorted(ps)
        for i in sorted({int(round(x)) for x in np.linspace(0, len(ps) - 1, min(per_inst, len(ps)))}):
            p = ps[i]
            day = os.path.basename(p)[:-4]
            with open(p, "rb") as f:
                z = np.load(f, allow_pickle=False)
                X0, Y0, rows, names = z["X"], z["Y"], z["rows"], [str(n) for n in z["names"]]
            X = block_compute(blk)(data_dir, inst, day)[rows].astype(np.float32)
            ys = load_targets(data_dir, inst, day, NO_CLIP)
            Y = np.column_stack([ys[h] for h in HORIZONS])[rows].astype(np.float32)
            same = (names == list(block_names(blk)) and np.array_equal(X, X0, equal_nan=True)
                    and np.array_equal(Y, Y0, equal_nan=True))
            print(f"  {inst} {day}: {'совпадает' if same else 'НЕ СОВПАДАЕТ'}")
            if not same:
                raise SystemExit(f"{p}: кэш не совпадает с текущим кодом признаков — НЕ подписан. "
                                 f"Соберите блок '{blk}' заново.")
    for k, p in enumerate(unsigned):
        with open(p, "rb") as f:
            z = np.load(f, allow_pickle=False)
            d = {n: z[n] for n in z.files}
        d["fhash"] = np.array(want)
        tmp = p + ".tmp"
        with open(tmp, "wb") as f:
            np.savez(f, **d)
        os.replace(tmp, p)
        if (k + 1) % 200 == 0 or k + 1 == len(unsigned):
            print(f"  подписано {k + 1}/{len(unsigned)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--block", default="base")
    ap.add_argument("--start", default=TRAIN[0])
    ap.add_argument("--end", default=VALID[1])
    ap.add_argument("--split", default=VALID[0], help="с этой даты — валидация, хранится целиком")
    ap.add_argument("--row-step", type=int, default=5, help="прореживание строк трейна")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--sign", action="store_true",
                    help="подписать старые файлы без отпечатка (после проверки пересчётом, см. sign_block)")
    ap.add_argument("--sign-days", type=int, default=3, help="--sign: сколько дней на инструмент пересчитать")
    a = ap.parse_args()
    if a.sign:
        for b in parts(a.block):
            sign_block(a.data_dir, a.cache_dir, b, a.sign_days)
        return
    if "+" in a.block:
        raise SystemExit("cache.py собирает один блок за раз (например --block v2); складывать блоки "
                         "можно в experiment.py: --block base+v2")

    check_cache_version(a.cache_dir, a.block)        # не смешивать в одном кэше разные версии признаков
    days = dates(a.start, a.end)
    jobs = [(a.data_dir, a.cache_dir, a.block, inst, d, 1 if d >= a.split else a.row_step)
            for d in days for inst in TARGET_INSTRUMENTS]
    todo = [j for j in jobs if not os.path.isfile(path_for(a.cache_dir, a.block, j[3], j[4]))]
    n_tr = sum(1 for j in todo if j[5] > 1)
    est_gb = (n_tr * 172_800 / a.row_step + (len(todo) - n_tr) * 172_800) * (len(block_names(a.block)) + 3) * 4 / 1e9
    print(f"инструмент-дней всего: {len(jobs)}, осталось посчитать: {len(todo)}, "
          f"нужно места ≈ {est_gb:.1f} ГБ, процессов: {a.workers}")
    t0, done = time.time(), 0
    with Pool(a.workers) as pool:
        for inst, day, st in pool.imap_unordered(build_one, todo):
            done += 1
            if st != "ok" or done % 20 == 0 or done == len(todo):
                el = time.time() - t0
                eta = el / done * (len(todo) - done)
                print(f"  {done}/{len(todo)}  {inst} {day}: {st}   прошло {el / 60:.0f} мин, осталось ≈ {eta / 60:.0f} мин")
    print("кэш готов")


if __name__ == "__main__":
    from textio import utf8_console
    utf8_console()
    main()
