"""Отпечатки версии кода признаков. Нужны кэшу и сохранённым моделям, чтобы не смешать
признаки, посчитанные разными версиями features.py.

FHASH          — features.py + common.py (так помечается кэш; значение то же, что было в cache.py)
FEATURES_HASH  — только features.py (наш файл). predict.py сверяет его строго, а common.py —
                 только с предупреждением: это файл организаторов, у проверяющих он может
                 отличаться переводами строк или комментариями, и падать из-за этого нельзя.
"""
import hashlib
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def files_hash(names):
    h = hashlib.sha256()
    for f in names:
        with open(os.path.join(HERE, f), "rb") as fh:
            h.update(fh.read().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:16]


FHASH = files_hash(["features.py", "common.py"])
FEATURES_HASH = files_hash(["features.py"])
# блок v2: features_v2.py опирается на помощники features.py и common.py
V2HASH = files_hash(["features.py", "features_v2.py", "common.py"])
FEATURES_V2_HASH = files_hash(["features_v2.py"])
# блок v3: features_v3.py опирается на помощники features.py и features_v2.py
V3HASH = files_hash(["features.py", "features_v2.py", "features_v3.py", "common.py"])
FEATURES_V3_HASH = files_hash(["features_v3.py"])
