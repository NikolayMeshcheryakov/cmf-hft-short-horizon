"""Автотест всего пайплайна. Запускать перед каждой отправкой изменений.

    python selftest.py --work ./selftest_work            # полный (≈ 5–10 минут, ≈ 5 ГБ места)
    python selftest.py --work ./selftest_work --quick    # только юнит-тесты (секунды)

1. Юнит-тесты: краевые случаи, найденные ревью (NaN, пустые файлы, постоянный признак,
   одинаковая волатильность, тик через шаг, p-value, дубликаты сделок…).
2. Интеграция на «грязных» данных (make_fake_data.py → make_dirty_data.py):
   eda.py, cache.py, experiment.py, train.py, predict.py (каждый день), causality_check.py
   на днях со сбоями. Всё должно отработать без падений, прогнозы — полные и конечные.
Выход: список PASS/FAIL; код возврата 1, если что-то упало.
"""
import argparse
import os
import subprocess
import sys
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RESULTS = []


def check(name, fn):
    try:
        msg = fn()
        RESULTS.append((True, name, msg or ""))
        print(f"  PASS  {name}" + (f" — {msg}" if msg else ""))
    except Exception as e:                                      # noqa: BLE001
        RESULTS.append((False, name, repr(e)))
        print(f"  FAIL  {name}: {e!r}")
        traceback.print_exc(limit=2)


def run(args, name, env=None):
    """Запуск скрипта пайплайна тем же Python; падение = исключение с хвостом вывода."""
    r = subprocess.run([sys.executable] + args, cwd=HERE, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", env=env)
    if r.returncode != 0:
        raise RuntimeError(f"{name}: код {r.returncode}\n{(r.stdout + r.stderr)[-1500:]}")
    return r.stdout


# ============================================================ 1. юнит-тесты
def unit_tests():
    import eda
    from metric import day_corr
    g = np.arange(0, 86_400_000, 100, dtype=np.int64)
    n = len(g)
    nan = np.full(n, np.nan)
    one = np.ones(n)
    bid, ask = np.full(n, 1.0), np.full(n, 1.01)

    def t_pvalue():
        p3 = eda.boot_mean([0.01, 0.02, 0.03])[3]
        p17 = eda.boot_mean([0.02] * 17)[3]
        assert abs(p3 - 0.25) < 1e-12, p3
        assert 0 < p17 < 1e-3, p17
        pn = eda.boot_mean(np.random.default_rng(1).normal(0, 1, 12))[3]
        assert pn > 0.05, pn
        return f"3 дня → {p3}, 17 дней → {p17:.1e}, шум → {pn:.2f}"

    def t_all_nan():
        r = eda.quality_from_arrays(nan, nan, nan, nan, None, None, g)
        assert r["missing_share"] == 1 and np.isnan(r["price_change_share_100ms"]), r
        assert r["longest_no_price_change_s"] == 0

    def t_nan_breaks_run():
        b = bid.copy()
        b[1000:2000] = np.nan
        r = eda.quality_from_arrays(b, ask, one, one, g[::5], None, g)
        assert abs(r["longest_no_price_change_s"] - (n - 2000 - 1) * 0.1) < 1e-6, r["longest_no_price_change_s"]

    def t_book_gap():
        bt = g[::5]
        bt = bt[(bt < 3_600_000) | (bt > 3_610_000)]
        r = eda.quality_from_arrays(bid, ask, one, one, bt, None, g)
        assert abs(r["longest_book_gap_s"] - 10.5) < 1e-9 and r["book_rows_missing"] == 21, r

    def t_trades_dup_offgrid():
        r = eda.quality_from_arrays(bid, ask, one, one, g[::5], np.r_[g[:5], g[:3], g[5] + 50], g)
        assert r["trade_duplicate_rows"] == 3 and r["trade_offgrid_rows"] == 1, r
        assert r["trade_bar_share_100ms"] * n == 5
        r2 = eda.quality_from_arrays(bid, ask, one, one, g[::5], np.array([], dtype=np.int64), g)
        assert r2["trades_file_present"] and r2["trade_bar_share_100ms"] == 0

    def t_invalid_quotes():
        b, a = bid.copy(), ask.copy()
        b[:10] = 0
        a[10:20] = b[10:20] - 0.01
        a[20:25] = b[20:25]
        r = eda.quality_from_arrays(b, a, one, one, g[::5], None, g)
        assert r["nonpositive_share"] * n == 10 and r["crossed_share"] * n == 10 and r["locked_share"] * n == 5, r

    def t_tick():
        b = np.full(n, 1.00)
        b[::3] += 0.02
        a = b + 0.01
        assert abs(eda.tick_estimate(b, a) - 0.01) < 1e-9
        b2 = b.copy()
        b2[100] = 1.005
        assert abs(eda.tick_estimate(b2, b2 + 0.01) - 0.01) < 1e-9
        assert np.isnan(eda.tick_estimate(nan, nan))

    def t_micro_no_tick():
        m = eda.micro_from_arrays(bid, ask, np.nan)
        for k in ("tick_bps", "spread_1tick_share", "spread_mean_ticks", "spread_nonint_share"):
            assert k in m and np.isnan(m[k])
        assert np.isnan(eda.micro_from_arrays(nan, nan, 0.01)["mid_change_share_100ms"])

    def t_day_corr_const():
        assert day_corr(np.ones(1000), np.random.randn(1000)) == 0.0
        assert day_corr(np.full(10, np.nan), np.random.randn(10)) == 0.0

    def t_regimes():
        m500 = 172_800
        vol = np.zeros(m500)
        obs = np.ones(m500, bool)
        qb, qa = np.full(m500, 1.0), np.full(m500, 1.01)
        r = eda.regime_masks(vol, obs, obs, qb, qa, 0.01)
        assert r["vol_low"].sum() == m500 and r["vol_high"].sum() == 0, "нулевая волатильность ушла не туда"
        assert r["spread_1tick"].sum() == m500 and r["spread_wider"].sum() == 0
        r2 = eda.regime_masks(vol, obs, obs, qb, qa, np.nan)
        assert r2["spread_1tick"].sum() == 0 and r2["spread_wider"].sum() == 0, "без тика — ни один режим спреда"
        obs2 = obs.copy()
        obs2[:1000] = False
        r3 = eda.regime_masks(np.r_[np.zeros(1000), np.arange(m500 - 1000.0)], obs2, obs2, qb, qa, 0.01)
        assert not (r3["vol_low"][:1000].any() or r3["spread_1tick"][:1000].any()), "ненаблюдаемые точки попали в режим"

    def t_targets_empty_hour():
        y = np.random.randn(172_800).astype(np.float32)
        y[:7200] = np.nan
        clip = {("BTCUSDT_PERP", h): (-1, 1) for h in ("y_500ms", "y_2s", "y_8s")}
        rows, hourly = eda.target_stats({"BTCUSDT_PERP": [{"y_500ms": y, "y_2s": y, "y_8s": y}]}, clip)
        assert hourly[0]["coverage"] == 0 and np.isnan(hourly[0]["std_y500ms_bps"])
        assert rows[0]["clipped_share"] > 0

    def t_bh():
        q = eda.bh_qvalues([0.01, 0.02, 0.03, 0.5, np.nan])
        assert np.allclose(q[:4], [0.04, 0.04, 0.04, 0.5]) and np.isnan(q[4]), q

    import experiment as ex
    import train as tr
    from types import SimpleNamespace

    def t_ridge_degenerate_day():
        rng = np.random.default_rng(0)
        x = rng.normal(size=(5000, 1))
        y = 0.02 * x[:, 0] + rng.normal(0, 0.01, 5000)
        Y = np.column_stack([y, y, y])
        good = ex.fit_ridge([("d1", x, Y)], 0.0)[0]["b"][0]
        bad = ex.fit_ridge([("d1", x, Y), ("d2", x, np.zeros_like(Y))], 0.0)[0]["b"][0]
        assert abs(good - 0.02) < 1e-3 and abs(bad - good) < 1e-9, (good, bad)
        return f"коэффициент {good:.4f} → {bad:.4f} (день с нулевым таргетом пропущен)"

    def t_ridge_empty():
        try:
            tr.Accumulator(2).solve(0.01)
            raise AssertionError("пустой аккумулятор не вызвал ошибку")
        except ValueError:
            pass
        m = ex.fit_ridge([("d1", np.ones((500, 1)), np.zeros((500, 3)))], 0.01)
        assert all(x is None for x in m)
        assert (ex.predict_any("ridge", None, np.ones((7, 1))) == 0).all()

    def t_purge():
        rows = np.arange(172_800)
        k = tr.purge_mask("2026-07-14", rows, "2026-07-15")
        assert (~k).sum() == 16 and not k[-16:].any() and k[:-16].all(), (~k).sum()
        assert tr.purge_mask("2026-07-13", rows, "2026-07-15").all()

    def t_paired_coverage():
        days = ["d1", "d2", "d3"]
        ref = {(d, i, h): 0.10 for d in days for (i, h) in ex.CELLS}
        cur = {(d, i, h): 0.12 for d in days for (i, h) in ex.CELLS}
        cur[("d3", "BTCUSDT_PERP", "y_8s")] = -5.0
        del cur[("d2", "ETHUSDT_PERP", "y_2s")]                # у cur в d2 не все пары
        assert ex.paired(cur, ref) is None, "разное покрытие пар — сравнение должно быть отклонено"
        del ref[("d2", "ETHUSDT_PERP", "y_2s")]                # одинаковое покрытие у обоих — сравниваем
        r = ex.paired(cur, ref)
        assert r["days"] == 3 and r["dropped"] == 0, r
        want = (11 * 0.02 + np.mean([0.02, 0.02, -5.1])) / 12      # как метрика: внутри пары по дням, потом по парам
        assert abs(r["delta"] - want) < 1e-12, (r["delta"], want)

        # случай из ревью: у BTC есть только первый день, у остальных — два; модель хуже в 1-й день,
        # лучше во 2-й. Разница обязана совпасть с разницей двух SCORE, посчитанных как метрика.
        def build(v1, v2):
            rows = []
            for (i, h) in ex.CELLS:
                rows.append({"day": "d1", "instrument": i, "horizon": h, "corr": v1})
                if i != "BTCUSDT_PERP":
                    rows.append({"day": "d2", "instrument": i, "horizon": h, "corr": v2})
            return rows
        exp = {i: (["d1"] if i == "BTCUSDT_PERP" else ["d1", "d2"]) for i in ex.TARGET_INSTRUMENTS}
        ra, rb = build(0.0, 0.3), build(0.2, 0.0)
        sa, sb = ex.cell_table(ra, exp)[1], ex.cell_table(rb, exp)[1]
        r = ex.paired({(x["day"], x["instrument"], x["horizon"]): x["corr"] for x in ra},
                      {(x["day"], x["instrument"], x["horizon"]): x["corr"] for x in rb})
        assert abs(r["delta"] - (sa - sb)) < 1e-12, (r["delta"], sa - sb)
        assert r["lo"] <= r["delta"] <= r["hi"], r
        cells, score, nd, complete = ex.cell_table([{"day": d, "instrument": i, "horizon": h, "corr": 0.1}
                                                    for d in days for (i, h) in ex.CELLS[:-1]])
        assert not complete and np.isnan(score), "неполное покрытие выдано за SCORE"

    def t_cache_version():
        import cache as ca
        tmp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_st_cache")
        d = os.path.join(tmp, "blk", "BTCUSDT_PERP")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "2026-01-01.npz"), "wb") as f:
            np.savez(f, X=np.zeros((1, 1)), names=np.array(["a"]), fhash=np.array("deadbeef"))
        try:
            ca.check_cache_version(tmp, "blk")
            raise AssertionError("чужая версия кэша не остановила запуск")
        except SystemExit:
            pass
        finally:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)

    def t_cell_table_dates():
        # одинаковое ЧИСЛО дней, но разные даты у разных пар — это не полное покрытие
        rows = [{"day": d, "instrument": i, "horizon": h, "corr": 0.1}
                for (i, h) in ex.CELLS for d in (["d1", "d2"] if i != "ETHUSDT_PERP" else ["d2", "d3"])]
        _, score, _, complete = ex.cell_table(rows)
        assert not complete and np.isnan(score), "разные даты при одинаковом числе дней выданы за полный SCORE"
        exp = {i: ["d1", "d2"] for i in ex.TARGET_INSTRUMENTS}
        rows = [{"day": d, "instrument": i, "horizon": h, "corr": 0.1} for (i, h) in ex.CELLS for d in ("d1", "d2")]
        assert ex.cell_table(rows, exp)[3], "полное покрытие не признано"
        exp["SOLUSDT_PERP"] = ["d1", "d2", "d3"]                # ожидался ещё день
        assert not ex.cell_table(rows, exp)[3], "пропущенный ожидаемый день не замечен"
        rows.append(dict(rows[0]))                              # повтор дня
        assert not ex.cell_table(rows)[3], "повтор дня не замечен"

    def t_cache_mixed_versions():
        import shutil
        import cache as ca
        tmp = os.path.join(HERE, "_st_cache2")
        d = os.path.join(tmp, "blk", "ETHUSDT_PERP")
        os.makedirs(d, exist_ok=True)
        try:
            for day, fh in (("2026-01-01", ca.FHASH), ("2026-01-02", "deadbeef")):   # первый файл — «правильный»
                with open(os.path.join(d, f"{day}.npz"), "wb") as f:
                    np.savez(f, X=np.zeros((1, 1), np.float32), Y=np.zeros((1, 3), np.float32),
                             rows=np.zeros(1, np.int32), names=np.array(["a"]), fhash=np.array(fh))
            try:
                ca.check_cache_version(tmp, "blk")
                raise AssertionError("смешанные версии в кэше не остановили запуск")
            except SystemExit:
                pass
            assert ca.load_day(tmp, "blk", "ETHUSDT_PERP", "2026-01-01") is not None
            try:
                ca.load_day(tmp, "blk", "ETHUSDT_PERP", "2026-01-02")
                raise AssertionError("load_day прочитал файл чужой версии")
            except SystemExit:
                pass
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def t_valid_rows_order():
        import shutil
        import cache as ca
        tmp = os.path.join(HERE, "_st_cache3")
        d = os.path.join(tmp, "blk", "BTCUSDT_PERP")
        os.makedirs(d, exist_ok=True)
        n = 172_800
        try:
            with open(os.path.join(d, "2026-01-01.npz"), "wb") as f:
                np.savez(f, X=np.zeros((n, 1), np.float32), Y=np.ones((n, 3), np.float32),
                         rows=np.arange(n, dtype=np.int32)[::-1], names=np.array(["a"]), fhash=np.array(ca.FHASH))
            a = SimpleNamespace(cache_dir=tmp, block="blk", valid=("2026-01-01", "2026-01-01"))
            clip = {(i, h): (-1.0, 1.0) for (i, h) in ex.CELLS}
            try:
                ex.evaluate(a, "BTCUSDT_PERP", "ridge", [None] * 3, ["a"], clip, ["2026-01-01"])
                raise AssertionError("перепутанный порядок строк валидации не замечен")
            except SystemExit:
                pass
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def t_model_version():
        import json
        import predict as pr
        tmp = os.path.join(HERE, "_st_model.json")
        try:
            json.dump({"type": "ridge", "features": ["imb1"], "coef": {}, "features_hash": "deadbeef"}, open(tmp, "w"))
            try:
                pr.load_model(tmp)
                raise AssertionError("модель другой версии признаков загрузилась")
            except SystemExit:
                pass
            json.dump({"type": "ridge", "features": ["imb1"], "coef": {}, "features_hash": pr.FEATURES_HASH,
                       "fhash": pr.FHASH}, open(tmp, "w"))
            pr.load_model(tmp)
        finally:
            os.remove(tmp)

    def t_feat_analysis_math():
        import feat_analysis as fa
        rng = np.random.default_rng(1)
        b = rng.normal(size=5000)
        x = 2.0 * b + 0.5 + rng.normal(0, 0.1, 5000)
        r = fa.resid_on(x, b)
        assert abs(np.corrcoef(r, b)[0, 1]) < 1e-10 and abs(r.mean()) < 1e-10, "остаток коррелирует с imb1"
        assert np.allclose(fa.resid_on(x, np.ones(5000)), x - x.mean()), "постоянный imb1"
        X = np.column_stack([b, x, np.ones(5000), -b])
        C = fa.day_corr_matrix(X)
        assert np.allclose(np.diag(C), 1) and np.allclose(C[2, [0, 1, 3]], 0), "постоянный признак"
        assert abs(C[0, 3] + 1) < 1e-12, C[0, 3]
        g = fa.clusters(["b", "x", "c", "nb"], C, 0.9)
        assert sorted(map(sorted, g)) == [[0, 1, 3]], g

    def t_v2_helpers():
        import features_v2 as f2
        ev = np.array([False, True, False, False, True, False])
        a = f2.bars_since(ev)
        assert np.isnan(a[0]) and list(a[1:]) == [0, 1, 2, 0, 1], a
        assert list(f2.last_nonzero(np.array([0, 1, 0, -1, 0, 0.]))) == [0, 1, 1, -1, -1, -1]
        assert list(f2.rmax(np.array([1, 5, 2, 0, 0, 0.]), 3)) == [1, 5, 5, 5, 2, 0]
        m = f2.rmean_valid(np.array([1, np.nan, 3, np.nan, np.nan, np.nan]), 2, min_frac=0.5)
        assert m[0] == 1 and m[1] == 1 and m[2] == 3 and m[3] == 3 and np.isnan(m[5]), m
        # причинность помощников: значение в i не зависит от того, что после i
        x = np.random.default_rng(3).normal(size=500)
        for fn in (lambda v: f2.rmax(v, 7), lambda v: f2.rmean_valid(v, 9), lambda v: f2.last_nonzero(np.round(v)),
                   lambda v: f2.bars_since(v > 1)):
            full, cut = fn(x), fn(x[:300])
            assert np.allclose(full[:300], cut, equal_nan=True), "помощник v2 смотрит вперёд"
        assert len(set(f2.V2_NAMES)) == len(f2.V2_NAMES)
        from features import FEATURE_NAMES
        assert not set(f2.V2_NAMES) & set(FEATURE_NAMES), "имена v2 пересекаются с базовыми"

    def t_fx_align():
        import shutil
        import pyarrow as pa
        import pyarrow.parquet as pq
        import features_v3 as f3
        from common import grid_points
        root = os.path.join(HERE, "_st_fx")
        try:
            os.makedirs(os.path.join(root, "stable"), exist_ok=True)
            day = "2026-01-02"
            t0 = int(grid_points(day)[0])
            ts = t0 + np.arange(0, 86_400_000, 1000)
            k = len(ts)
            bid = np.full(k, 1.0); bid[3600:] = 1.001                  # курс меняется ровно в 01:00:00
            stale = np.zeros(k, "int64"); stale[7200:7210] = np.arange(10) * 1000 + 5000
            pq.write_table(pa.table({"ts": ts, "pair": np.full(k, "USDCUSDT"), "bid_px": bid, "ask_px": bid + 1e-4,
                                     "buy_qty": np.ones(k), "sell_qty": np.zeros(k), "stale_ms": stale}),
                           os.path.join(root, "stable", f"{day}.parquet"))
            fx = f3.fx_two_days(root, day, "USDCUSDT")
            g = np.concatenate([grid_points("2026-01-01", 100), grid_points(day, 100)])
            i = int(np.searchsorted(g, t0 + 3_600_000))                # ровно 01:00:00.000 — новая строка уже видна
            assert abs(fx["bid"][i] - 1.001) < 1e-12 and abs(fx["bid"][i - 1] - 1.0) < 1e-12, "курс не по ts <= t"
            assert abs(fx["age_s"][i + 3] - 0.3) < 1e-9, fx["age_s"][i + 3]          # возраст растёт внутри секунды
            j = int(np.searchsorted(g, t0 + 7_205_000))
            assert abs(fx["age_s"][j] - 10.0) < 1e-9, fx["age_s"][j]                 # stale_ms + время с строки
            assert np.isnan(fx["mid"][:864_000]).all(), "вчерашнего stable нет — должно быть NaN"
            assert abs(fx["flow5"][i] - 1.0) < 1e-9
        finally:
            shutil.rmtree(root, ignore_errors=True)

    # ---- мини-данные для сценариев из ревью v3 (2 дня: вчера + сегодня)
    D0, D1 = "2026-01-01", "2026-01-02"

    def mini(root, markets, stable):
        """markets: {инструмент: {день: (bid, ask) массивы на сетке 500 мс или None}};
        stable: {день: dict колонок}."""
        import pyarrow as pa
        import pyarrow.parquet as pq
        from common import grid_points
        for inst, days_ in markets.items():
            for d, v in days_.items():
                if v is None:
                    continue
                bid, ask = v
                g = grid_points(d)
                os.makedirs(os.path.join(root, "book", inst), exist_ok=True)
                pq.write_table(pa.table({"ts": g, "bid_px": bid, "ask_px": ask,
                                         "bid_qty": np.full(len(g), 5.0), "ask_qty": np.full(len(g), 5.0)}),
                               os.path.join(root, "book", inst, f"{d}.parquet"))
        os.makedirs(os.path.join(root, "stable"), exist_ok=True)
        for d, cols in stable.items():
            pq.write_table(pa.table(cols), os.path.join(root, "stable", f"{d}.parquet"))

    def fx_rows(d, secs=None, stale=None, mid=1.0, buy=None, sell=None):
        from common import grid_points
        t0 = int(grid_points(d)[0])
        sec = np.arange(86_400) if secs is None else np.asarray(secs)
        k = len(sec)
        return {"ts": t0 + sec * 1000, "pair": np.full(k, "USDCUSDT"),
                "bid_px": np.full(k, mid - 5e-5), "ask_px": np.full(k, mid + 5e-5),
                "buy_qty": np.zeros(k) if buy is None else buy, "sell_qty": np.zeros(k) if sell is None else sell,
                "stale_ms": np.zeros(k, "int64") if stale is None else stale}

    def t_v3_stale_ref():
        import shutil
        import features_v3 as f3
        root = os.path.join(HERE, "_st_v3a")
        n = 172_800
        try:
            mini(root, {"BTCUSDT_SPOT": {D0: (np.full(n, 99.995), np.full(n, 100.005)), D1: None},   # сегодня нет
                        "BTCUSDC_SPOT": {D0: (np.full(n, 100.995), np.full(n, 101.005)),
                                         D1: (np.full(n, 100.995), np.full(n, 101.005))}},
                 {D0: fx_rows(D0), D1: fx_rows(D1)})
            X = f3.compute_v3(root, "BTCUSDT_PERP", D1)
            # первые точки дня законны: вчерашняя котировка моложе 1 с; дальше — никакой протянутой цены
            av, gp = X[3:, f3.V3_NAMES.index("avail_usdc")], X[3:, f3.V3_NAMES.index("gap_usdc")]
            assert av.max() == 0 and np.abs(gp).max() == 0, \
                f"опорного спота сегодня нет, а разрыв выдан как актуальный: avail={av.max()}, gap={np.abs(gp).max():.1f}"
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def t_v3_gap_recovery():
        import shutil
        import features_v3 as f3
        from common import grid_points
        root = os.path.join(HERE, "_st_v3b")
        n = 172_800
        g = grid_points(D1)
        rec = int(g[0]) + 3_603_000                                  # курс снова свежий с 01:00:03.000
        try:
            ab = np.where(g >= rec, 101.995, 99.995)                 # и ровно в этот момент цена сдвинулась
            stale = np.zeros(86_400, "int64"); stale[3600:3603] = 10_000
            mini(root, {"BTCUSDT_SPOT": {D0: (np.full(n, 99.995), np.full(n, 100.005)),
                                         D1: (np.full(n, 99.995), np.full(n, 100.005))},
                        "BTCUSDC_SPOT": {D0: (np.full(n, 99.995), np.full(n, 100.005)), D1: (ab, ab + 0.01)}},
                 {D0: fx_rows(D0), D1: fx_rows(D1, stale=stale)})
            X = f3.compute_v3(root, "BTCUSDT_PERP", D1)
            i = int(np.searchsorted(g, rec))
            av = X[:, f3.V3_NAMES.index("avail_usdc")]
            assert av[i - 2] == 0 and av[i] == 1, (av[i - 2], av[i])
            for nm in ("alt_ofi_5", "dgap_usdc_5", "dgap_usdc_20"):
                v = X[i, f3.V3_NAMES.index(nm)]
                assert v == 0, f"{nm} сразу после разрыва доступности = {v} (сравнение через разрыв)"
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def t_v3_flow_by_time():
        import shutil
        import features_v3 as f3
        from common import grid_points
        root = os.path.join(HERE, "_st_v3c")
        try:
            secs = np.array([0, 1, 60])
            mini(root, {}, {D1: fx_rows(D1, secs=secs, buy=np.array([100.0, 0, 0]), sell=np.array([0, 0, 100.0]))})
            fx = f3.fx_two_days(root, D1, "USDCUSDT")
            g = np.concatenate([grid_points(D0, 100), grid_points(D1, 100)])
            i = int(np.searchsorted(g, int(grid_points(D1)[0]) + 60_000))
            assert abs(fx["flow5"][i] + 1.0) < 1e-9, f"поток за 5 с на 60-й секунде = {fx['flow5'][i]} (ожидалось −1)"
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def t_v3_silent_midday():
        """Рынок замолчал посреди дня (нет строк book/bbo 10 минут): протянутая цена — не живая."""
        import shutil
        import pyarrow as pa
        import pyarrow.parquet as pq
        import features_v3 as f3
        from common import grid_points
        root = os.path.join(HERE, "_st_v3d")
        n = 172_800
        g = grid_points(D1)
        t_off, t_on = int(g[0]) + 3_600_000, int(g[0]) + 4_200_000      # 01:00–01:10 строк нет
        try:
            mini(root, {"BTCUSDT_SPOT": {D0: (np.full(n, 99.995), np.full(n, 100.005)),
                                         D1: (np.full(n, 99.995), np.full(n, 100.005))},
                        "BTCUSDC_SPOT": {D0: (np.full(n, 100.995), np.full(n, 101.005)),
                                         D1: (np.full(n, 100.995), np.full(n, 101.005))}},
                 {D0: fx_rows(D0), D1: fx_rows(D1)})
            for inst in ("BTCUSDT_SPOT", "BTCUSDC_SPOT"):          # вырезаем строки у одного из рынков по очереди
                p = os.path.join(root, "book", inst, f"{D1}.parquet")
                t = pq.read_table(p)
                ts = t["ts"].to_numpy()
                keep = ~((ts >= t_off) & (ts < t_on))
                full = {c: t[c].to_numpy() for c in t.column_names}
                pq.write_table(pa.table({c: v[keep] for c, v in full.items()}), p)
                X = f3.compute_v3(root, "BTCUSDT_PERP", D1)
                av = X[:, f3.V3_NAMES.index("avail_usdc")]
                i_off, i_on = int(np.searchsorted(g, t_off)), int(np.searchsorted(g, t_on))
                assert av[i_off + 3:i_on].max() == 0, f"{inst} молчит 10 минут, а пара считается доступной"
                assert av[i_on + 2] == 1 and av[i_off - 1] == 1, "до и после тишины рынок должен быть доступен"
                pq.write_table(pa.table(full), p)                   # вернуть строки
            # курс без stale_ms (NaN) — не «свежий»
            fx = fx_rows(D1)
            fx["stale_ms"] = np.full(86_400, np.nan)
            pq.write_table(pa.table(fx), os.path.join(root, "stable", f"{D1}.parquet"))
            X = f3.compute_v3(root, "BTCUSDT_PERP", D1)
            assert X[:, f3.V3_NAMES.index("avail_usdc")].max() == 0, "неизвестный stale_ms принят за свежий курс"
            assert np.abs(X[:, f3.V3_NAMES.index("fx_dev_usdc")]).max() == 0, "отклонение по курсу неизвестного возраста"
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def t_v3_gapdev_after_break():
        """После разрыва доступности gapdev = 0, пока не наберётся непрерывная минута."""
        import shutil
        import features_v3 as f3
        from common import grid_points
        root = os.path.join(HERE, "_st_v3e")
        n = 172_800
        g = grid_points(D1)
        try:
            stale = np.zeros(86_400, "int64"); stale[3600:3603] = 10_000      # курс устарел 01:00:00–01:00:03
            mini(root, {"BTCUSDT_SPOT": {D0: (np.full(n, 99.995), np.full(n, 100.005)),
                                         D1: (np.full(n, 99.995), np.full(n, 100.005))},
                        "BTCUSDC_SPOT": {D0: (np.full(n, 100.995), np.full(n, 101.005)),
                                         D1: (np.where(np.arange(n) % 2 == 0, 100.995, 100.985),
                                              np.where(np.arange(n) % 2 == 0, 101.005, 100.995))}},
                 {D0: fx_rows(D0), D1: fx_rows(D1, stale=stale)})
            X = f3.compute_v3(root, "BTCUSDT_PERP", D1)
            gd = X[:, f3.V3_NAMES.index("gapdev_usdc")]
            rec = int(np.searchsorted(g, int(g[0]) + 3_603_000))
            assert np.abs(gd[rec:rec + 119]).max() == 0, "gapdev считается по среднему через разрыв"
            assert np.abs(gd[rec + 125:rec + 200]).max() > 0, "после непрерывной минуты gapdev должен вернуться"
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def t_v3_flow_between_rows():
        """Поток за 5 с — от момента прогноза: через 2.5 с после строки старая покупка уже вне окна."""
        import shutil
        import features_v3 as f3
        from common import grid_points
        root = os.path.join(HERE, "_st_v3f")
        try:
            secs = np.array([0, 4])
            fx = fx_rows(D1, secs=secs, buy=np.array([100.0, 0]), sell=np.array([0, 100.0]))
            mini(root, {}, {D1: fx})
            f = f3.fx_two_days(root, D1, "USDCUSDT")
            g = np.concatenate([grid_points(D0, 100), grid_points(D1, 100)])
            i = int(np.searchsorted(g, int(grid_points(D1)[0]) + 6_500))
            assert abs(f["flow5"][i] + 1.0) < 1e-9, f"поток в 6.5 с = {f['flow5'][i]} (ожидалось −1)"
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def t_v3_spot_fresh():
        """Новая группа спота v3: спот замолчал посреди дня / нет за день → savail=0, sbasis=0."""
        import shutil
        import pyarrow as pa
        import pyarrow.parquet as pq
        import features_v3 as f3
        from common import grid_points
        root = os.path.join(HERE, "_st_v3g")
        n = 172_800
        g = grid_points(D1)
        try:
            mini(root, {"BTCUSDT_PERP": {D0: (np.full(n, 100.995), np.full(n, 101.005)),
                                         D1: (np.full(n, 100.995), np.full(n, 101.005))},
                        "BTCUSDT_SPOT": {D0: (np.full(n, 99.995), np.full(n, 100.005)),
                                         D1: (np.full(n, 99.995), np.full(n, 100.005))}},
                 {D0: fx_rows(D0), D1: fx_rows(D1)})
            p = os.path.join(root, "book", "BTCUSDT_SPOT", f"{D1}.parquet")
            t = pq.read_table(p)
            full = {c: t[c].to_numpy() for c in t.column_names}
            off, on = int(g[0]) + 3_600_000, int(g[0]) + 4_200_000
            keep = ~((full["ts"] >= off) & (full["ts"] < on))
            pq.write_table(pa.table({c: v[keep] for c, v in full.items()}), p)
            X = f3.compute_v3(root, "BTCUSDT_PERP", D1)
            i0, i1 = int(np.searchsorted(g, off)), int(np.searchsorted(g, on))
            sa, sb = X[:, f3.V3_NAMES.index("savail")], X[:, f3.V3_NAMES.index("sbasis")]
            assert sa[i0 - 1] == 1 and abs(sb[i0 - 1] - 99.5) < 1, (sa[i0 - 1], sb[i0 - 1])
            assert sa[i0 + 3:i1].max() == 0 and np.abs(sb[i0 + 3:i1]).max() == 0, "спот молчит, а базис выдан"
            for nm in ("dsbasis_5", "dsbasis_20", "sret_5", "sret_20"):
                assert X[i1, f3.V3_NAMES.index(nm)] == 0, f"{nm} сразу после тишины спота считается через разрыв"
            os.remove(p)                                            # спота нет за весь день
            X = f3.compute_v3(root, "BTCUSDT_PERP", D1)
            assert X[3:, f3.V3_NAMES.index("savail")].max() == 0, "спота за день нет, а базис доступен"
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def t_review_misc():
        import json
        import shutil
        import cache as ca
        import predict as pr
        a = {(d, i, h): 0.1 for d in ("d1", "d2") for (i, h) in ex.CELLS}
        b = {(d, i, h): 0.0 for d in ("d2", "d3") for (i, h) in ex.CELLS}
        assert ex.paired(a, b) is None, "сравнение экспериментов с разными днями выдано"
        c2 = {(d, i, h): 0.2 for d in ("d1", "d2") for (i, h) in ex.CELLS}
        del c2[("d1", "BTCUSDT_PERP", "y_500ms")]
        assert ex.paired(a, c2) is None, "сравнение при разном составе пар внутри одинаковых дней выдано"
        assert ex.paired(a, {k: v + 0.01 for k, v in a.items()}) is not None
        tmp = os.path.join(HERE, "_st_cache4")
        try:
            for blk, y in (("base", 3.0), ("v3", 987.0)):
                d = os.path.join(tmp, blk, "BTCUSDT_PERP")
                os.makedirs(d, exist_ok=True)
                with open(os.path.join(d, "2026-01-01.npz"), "wb") as f:
                    np.savez(f, X=np.zeros((2, 1), np.float32), Y=np.full((2, 3), y, np.float32),
                             rows=np.arange(2, dtype=np.int32), names=np.array([f"x_{blk}"]),
                             fhash=np.array(ca.block_hash(blk)))
            try:
                ca.load_day(tmp, "base+v3", "BTCUSDT_PERP", "2026-01-01")
                raise AssertionError("разные таргеты в блоках не замечены")
            except SystemExit:
                pass
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        mp = os.path.join(HERE, "_st_model3.json")
        try:
            json.dump({"type": "ridge", "features": ["gapdev_usdc"], "coef": {}, "features_hash": pr.FEATURES_HASH,
                       "fhash": pr.FHASH, "features_v3_hash": pr.FEATURES_V3_HASH, "features_v2_hash": "deadbeef"},
                      open(mp, "w"))
            try:
                pr.load_model(mp)
                raise AssertionError("модель v3 при другой версии features_v2.py загрузилась")
            except SystemExit:
                pass
            # точность: модель из train.py (float64) не приводится к float32, модель из кэша — приводится
            # (проверяется САМО место приведения — single_predict, а не уже приведённый вход)
            Xq = np.array([[0.5, 1.00000007]])
            names_q = ["micro_bps", "imb1"]
            for prec, want in (("float64", 1.00000007 * 1000), ("float32", float(np.float32(1.00000007)) * 1000),
                               (None, 1.00000007 * 1000)):             # без поля: ridge → float64
                mdl = {"type": "ridge", "features": ["imb1"],
                       "coef": {"BTCUSDT_PERP": {"y_500ms": {"a": 0.0, "b": [1000.0]}}}}
                if prec:
                    mdl["precision"] = prec
                got = pr.single_predict(mdl, "BTCUSDT_PERP", Xq, names_q, 1)["y_500ms"][0]
                assert abs(got - want) < 1e-9, (prec, got, want)
            assert pr.single_predict(mdl, "ETHUSDT_PERP", Xq, names_q, 1)["y_2s"][0] == 0.0   # нет модели → 0
        finally:
            os.remove(mp)

    def t_lgbm_one_day():
        try:
            import lightgbm  # noqa: F401
        except ImportError:
            return "lightgbm не установлен — пропущено"
        rng = np.random.default_rng(0)
        x = rng.normal(size=(3000, 2)).astype(np.float32)
        Y = np.column_stack([x[:, 0] * 0.01 + rng.normal(0, 0.01, 3000)] * 3)
        a = SimpleNamespace(huber_delta=1.0, lr=0.1, leaves=7, min_leaf=50, threads=1, rounds=30,
                            ff=0.8, bagging=0.8, l2=10.0, rw=None)
        m = ex.fit_lgbm([("d1", x, Y)], a)
        assert all(mm is not None for mm in m)
        m2 = ex.fit_lgbm([("d1", x, np.zeros_like(Y)), ("d2", x, np.zeros_like(Y))], a)
        assert all(mm is None for mm in m2)
        return "1 день и вырожденные дни — без падения"

    def t_recency():
        rw = ex.recency_weight("2026-01-11", 5)
        assert ex.recency_weight("2026-01-11", None) is None
        assert abs(rw("2026-01-11") - 1) < 1e-12 and abs(rw("2026-01-06") - 0.5) < 1e-12
        assert abs(rw("2026-01-01") - 0.25) < 1e-12
        rng = np.random.default_rng(1)
        days = [("2026-01-01", rng.normal(size=(300, 2)), rng.normal(size=(300, 3))),
                ("2026-01-11", rng.normal(size=(200, 2)), rng.normal(size=(200, 3)))]
        X, y, w = ex.stack_norm(days, 0, rw)
        assert len(w) == len(y) == 500 and np.allclose(w[:300], 0.25) and np.allclose(w[300:], 1.0)
        X0, y0, w0 = ex.stack_norm(days, 0)
        assert w0 is None and np.array_equal(X0, X) and np.array_equal(y0, y)
        # ridge: вес дня меняет решение; без весов — как раньше (rw=None ≡ веса 1)
        x = rng.normal(size=(500, 1))
        Y1 = np.column_stack([x[:, 0] * 1.0 + rng.normal(0, 0.1, 500)] * 3)
        Y2 = np.column_stack([x[:, 0] * -1.0 + rng.normal(0, 0.1, 500)] * 3)
        tr = [("2026-01-01", x, Y1), ("2026-01-11", x, Y2)]
        b_eq = ex.fit_ridge(tr, 0.0)[0]["b"][0]
        b_rec = ex.fit_ridge(tr, 0.0, ex.recency_weight("2026-01-11", 2))[0]["b"][0]
        assert abs(b_eq) < 0.2 and b_rec < -0.8, (b_eq, b_rec)
        return f"веса 1 / 0.5 / 0.25; ridge без весов b={b_eq:+.2f}, со свежестью b={b_rec:+.2f}"

    def t_mlp_grad():
        import mlp
        rng = np.random.default_rng(1)
        sizes = [5, 7, 4, 3]
        W = [rng.normal(0, 0.5, (a_, b_)) for a_, b_ in zip(sizes, sizes[1:])]
        B = [rng.normal(0, 0.1, b_) for b_ in sizes[1:]]
        x = rng.normal(size=(40, 5)); y = rng.normal(size=(40, 3)) * 2
        m = (rng.random((40, 3)) > 0.2).astype(float)
        loss, gW, gB, den = mlp.loss_grads(W, B, x, y, m, 1.0, 0.0)
        worst = 0.0
        for j in range(len(W)):
            for P, G in ((W[j], gW[j]), (B[j], gB[j])):
                for idx in [tuple(rng.integers(0, s) for s in P.shape) for _ in range(6)]:
                    old = P[idx]
                    P[idx] = old + 1e-6; lp = mlp.loss_grads(W, B, x, y, m)[0] / den
                    P[idx] = old - 1e-6; lm = mlp.loss_grads(W, B, x, y, m)[0] / den
                    P[idx] = old
                    num = (lp - lm) / 2e-6
                    worst = max(worst, abs(num - G[idx]) / max(1e-6, abs(num) + abs(G[idx])))
        assert worst < 1e-5, f"градиент сети не совпал с численным: {worst:.2e}"
        # сеть учится: y = sin(x0) + x1*x2 на 3 выходах
        X = rng.normal(size=(20000, 4)).astype(np.float32)
        f = np.sin(X[:, 0]) + X[:, 1] * X[:, 2]
        Yt = np.column_stack([f, 0.5 * f, -f]).astype(np.float32)
        Yn = Yt + rng.normal(0, 0.3, Yt.shape).astype(np.float32)
        I = rng.integers(0, 2, len(X))
        net, _ = mlp.train_mlp(X, Yn, np.ones_like(Yn), I, None, 2, hidden=(32, 16), lr=3e-3, batch=512,
                               epochs=8, log=lambda *_: None)
        P = net.predict(X[:5000], I[:5000])
        r = min(np.corrcoef(P[:, j], Yt[:5000, j])[0, 1] for j in range(3))
        assert r > 0.9, f"сеть не выучила простую функцию: corr {r:.3f}"
        return f"градиент = численный (отн. ошибка {worst:.1e}); простая функция выучена, corr {r:.3f}"

    def t_mlp_chunks():
        import mlp
        rng = np.random.default_rng(2)
        k = 6
        net = mlp.MLPNumpy([rng.normal(size=(k + 4, 16)), rng.normal(size=(16, 3))],
                           [rng.normal(size=16), rng.normal(size=3)],
                           -np.ones(k) * 3, np.ones(k) * 3, np.zeros(k), np.ones(k), 4)
        X = rng.normal(size=(mlp.CHUNK * 2 + 777, k)) * 2
        X[5, 2] = np.nan
        full = net.predict(X, 1)
        for T in (1, 13, mlp.CHUNK - 1, mlp.CHUNK, mlp.CHUNK + 1, 2 * mlp.CHUNK + 100):
            cut = net.predict(X[:T], 1)
            assert np.array_equal(cut, full[:T]), f"прогноз сети до T={T} изменился после обрезки"
        assert np.isfinite(full).all()
        import tempfile
        p = os.path.join(tempfile.mkdtemp(), "m.npz")
        net.save(p)
        assert np.array_equal(mlp.MLPNumpy.load(p).predict(X, 1), full), "сохранение сети меняет прогноз"
        return "прогноз до T побитово тот же при любой обрезке; сохранение без потерь"

    def t_ensemble_math():
        import ensemble as en
        from metric import day_corr
        rng = np.random.default_rng(3)
        M, D, n = 3, 6, 500
        C, S, V, ref = [], [], [], []
        w = np.array([0.2, 0.5, 0.3])
        for _ in range(D):
            y = rng.normal(size=n)
            P = np.column_stack([y + rng.normal(0, s_, n) for s_ in (1.0, 2.0, 0.5)]) * [3.0, 0.1, 7.0]
            A = P - P.mean(0); yc = y - y.mean()
            C.append(A.T @ yc / n); S.append(A.T @ A / n); V.append(yc @ yc / n)
            ref.append(day_corr(P @ w, y))
        r = en.day_corrs(w[None, :], np.array(C), np.array(S), np.array(V))[0]
        assert np.allclose(r, ref, atol=1e-12), "корреляция по статистикам ≠ метрике"
        g = en.simplex(3, 0.25)
        assert len(g) == 15 and np.allclose(g.sum(1), 1) and (g >= 0).all()
        return "корреляция смеси по статистикам = metric.day_corr; сетка весов корректна"

    def t_ensemble_order():
        import json as _json
        import tempfile
        import ensemble as en
        tmp = tempfile.mkdtemp()
        insts = ("BTCUSDT_PERP", "ETHUSDT_PERP", "SOLUSDT_PERP", "1000PEPEUSDT_PERP")
        hs = ("y_500ms", "y_2s", "y_8s")
        paths = {}
        for nm, kind, feats in (("A", "lgbm", ["imb1", "ofi_5"]), ("B", "ridge", ["imb1", "ofi_5"])):
            d = os.path.join(tmp, nm); os.makedirs(d)
            with open(os.path.join(d, "model.json"), "w") as f:
                _json.dump({"type": kind, "features": feats, "pred_std": {i: {h: 1.0 for h in hs} for i in insts}}, f)
            paths[nm] = os.path.join(d, "model.json")
        fid = ex.feats_id(["imb1", "ofi_5"])
        wj = os.path.join(tmp, "w.json")
        with open(wj, "w") as f:
            _json.dump({"members": ["int_A", "int_B"], "kinds": ["lgbm", "ridge"], "feats": [fid, fid],
                        "weights": {i: {h: [1.0, 0.0] for h in hs} for i in insts}}, f)
        SN = SimpleNamespace
        en.cmd_build(SN(weights=wj, models=[paths["A"], paths["B"]], out=os.path.join(tmp, "ok")))
        try:
            en.cmd_build(SN(weights=wj, models=[paths["B"], paths["A"]], out=os.path.join(tmp, "bad")))
            raise AssertionError("ансамбль собрался с переставленными участниками")
        except SystemExit as e:
            assert "перепутан" in str(e), e
        with open(paths["B"]) as f:
            mb = _json.load(f)
        mb.update(type="lgbm", features=["imb1"])                      # тот же тип, другие признаки
        with open(paths["B"], "w") as f:
            _json.dump(mb, f)
        try:
            en.cmd_build(SN(weights=wj, models=[paths["B"], paths["A"]], out=os.path.join(tmp, "bad2")))
            raise AssertionError("ансамбль принял модель на другом наборе признаков")
        except SystemExit as e:
            assert "набор" in str(e), e
        return "перестановка участников и чужой набор признаков — остановлены"

    for name, fn in [("Ridge: день с нулевым таргетом", t_ridge_degenerate_day),
                     ("ансамбль: порядок и состав участников при сборке", t_ensemble_order),
                     ("нейросеть: градиенты и обучение", t_mlp_grad),
                     ("нейросеть: причинность прогноза (куски), сохранение", t_mlp_chunks),
                     ("ансамбль: формула корреляции смеси", t_ensemble_math),
                     ("веса дней по свежести (--recency-halflife)", t_recency),
                     ("Ridge: пустое обучение", t_ridge_empty),
                     ("purging на границе трейн/валидация", t_purge),
                     ("сравнение экспериментов: структура метрики и покрытие", t_paired_coverage),
                     ("версия кэша", t_cache_version),
                     ("версия кэша: проверяется каждый файл", t_cache_mixed_versions),
                     ("полнота валидации — по датам, а не по числу дней", t_cell_table_dates),
                     ("валидация: порядок и полнота строк дня", t_valid_rows_order),
                     ("predict.py: модель другой версии признаков", t_model_version),
                     ("feat_analysis: остаток, корреляции, группы", t_feat_analysis_math),
                     ("признаки v2: помощники и причинность", t_v2_helpers),
                     ("признаки v3: курс стейблкоина по ts <= t, возраст, поток", t_fx_align),
                     ("v3: опорного рынка сегодня нет → разрыв не выдаётся", t_v3_stale_ref),
                     ("v3: после разрыва доступности нет скачков OFI и dgap", t_v3_gap_recovery),
                     ("v3: поток курса за 5 с — по времени, а не по строкам", t_v3_flow_by_time),
                     ("v3: рынок замолчал посреди дня / stale_ms пустой → недоступен", t_v3_silent_midday),
                     ("v3: gapdev после разрыва — только по непрерывной минуте", t_v3_gapdev_after_break),
                     ("v3: поток курса — окно от момента прогноза, а не от строки", t_v3_flow_between_rows),
                     ("v3: спот к USDT со свежестью (тишина посреди дня, нет дня)", t_v3_spot_fresh),
                     ("ревью: покрытие в сравнении, таргеты блоков, версия v2 для v3, точность", t_review_misc),
                     ("LightGBM: один день, вырожденные дни", t_lgbm_one_day),
                     ("p-value (sign-flip, точный и Монте-Карло)", t_pvalue),
                     ("качество: всё NaN", t_all_nan),
                     ("качество: NaN разрывает серию", t_nan_breaks_run),
                     ("качество: дыра в book", t_book_gap),
                     ("качество: дубликаты / вне сетки / пустой trades", t_trades_dup_offgrid),
                     ("качество: нулевые, скрещенные, залоченные", t_invalid_quotes),
                     ("тик: шаг через тик, выброс, пусто", t_tick),
                     ("микроструктура без тика", t_micro_no_tick),
                     ("day_corr: постоянный признак = 0", t_day_corr_const),
                     ("режимы: одинаковая волатильность, нет тика, ненаблюдаемые", t_regimes),
                     ("таргеты: пустой час", t_targets_empty_hour),
                     ("поправка BH", t_bh)]:
        check(name, fn)


# ============================================================ 2. интеграция
def integration(work):
    clean, dirty = os.path.join(work, "fake_clean"), os.path.join(work, "fake_dirty")
    cache, res, models = (os.path.join(work, x) for x in ("cache", "results", "models"))
    days = ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05"]

    # чистый фейк нужен только как исходник для грязного: генерируем, портим, удаляем (экономия места)
    import shutil
    check("make_fake_data", lambda: run(["make_fake_data.py", "--out", clean, "--start", days[0],
                                         "--days", "5", "--train-days", "3"], "make_fake_data") and None)
    check("make_dirty_data", lambda: run(["make_dirty_data.py", "--src", clean, "--out", dirty],
                                         "make_dirty_data") and None)
    shutil.rmtree(clean, ignore_errors=True)

    def t_eda():
        out = os.path.join(work, "eda")
        run(["eda.py", "--data-dir", dirty, "--days", "5", "--start", days[0], "--end", days[-1],
             "--out", out], "eda")
        for f in ("eda_report.md", "quality.csv", "ic.csv", "ic_regimes.csv", "alpha_decay.csv",
                  "leadlag.csv", "autocorr.csv", "targets.csv", "inventory.csv"):
            assert os.path.isfile(os.path.join(out, f)), f"нет {f}"
        q = open(os.path.join(out, "quality.csv"), encoding="utf-8").read()
        assert f"SOLUSDT_PERP,{days[2]},False" in q, "день без book не отмечен"
        rep = open(os.path.join(out, "eda_report.md"), encoding="utf-8").read()
        assert "надёжно" not in rep.replace("ненадёжно", ""), "осталась формулировка «надёжно»"
        return "отчёт и все таблицы на месте"
    check("eda.py на грязных данных", t_eda)

    def t_cache_exp():
        run(["cache.py", "--data-dir", dirty, "--cache-dir", cache, "--start", days[0], "--end", days[-1],
             "--split", days[3], "--workers", "2"], "cache")
        base = ["experiment.py", "--cache-dir", cache, "--data-dir", dirty, "--train", days[0], days[2],
                "--valid", days[3], days[4], "--results", res]
        victim = os.path.join(cache, "base", "ETHUSDT_PERP", f"{days[4]}.npz")   # «недостроенный» кэш
        os.remove(victim)
        try:
            run(base + ["--name", "st_partial", "--model", "ridge"], "exp partial")
            raise AssertionError("эксперимент на недостроенном кэше выдал SCORE")
        except RuntimeError as e:
            assert "кэш не достроен" in str(e), str(e)[-400:]
        run(["cache.py", "--data-dir", dirty, "--cache-dir", cache, "--start", days[0], "--end", days[-1],
             "--split", days[3], "--workers", "2"], "cache докачка")
        assert os.path.isfile(victim), "cache.py не докачал пропавший день"
        run(base + ["--name", "st_base", "--model", "ridge", "--features", "imb1"], "exp baseline")
        # внутренняя проверка: проверочный день из трейна (прорежен) — должна работать и честно подписываться
        inner = ["experiment.py", "--cache-dir", cache, "--data-dir", dirty, "--internal", "--results", res,
                 "--train", days[0], days[1], "--valid", days[2], days[2]]
        o = run(inner + ["--name", "st_int_base", "--model", "ridge", "--features", "imb1"], "exp internal")
        assert "внутренней проверки" in o and np.isfinite(float(o.split("):")[-1].split()[0])), o[-400:]
        o = run(inner + ["--name", "st_int_all", "--model", "ridge", "--ref", "st_int_base"], "exp internal ref")
        assert "против st_int_base" in o, o[-400:]
        o = run(inner + ["--name", "st_int_x", "--model", "ridge", "--ref", "st_base"], "exp internal чужой ref")
        assert "другое покрытие" in o, o[-400:]
        try:                                   # без --internal прореженный проверочный день — ошибка
            run(["experiment.py", "--cache-dir", cache, "--data-dir", dirty, "--results", res,
                 "--train", days[0], days[1], "--valid", days[2], days[2],
                 "--name", "st_bad", "--model", "ridge"], "exp thinned")
            raise AssertionError("прореженная валидация без --internal не остановила запуск")
        except RuntimeError as e:
            assert "прорежен" in str(e), str(e)[-300:]
        fo = os.path.join(work, "feat")
        run(["feat_analysis.py", "--cache-dir", cache, "--data-dir", dirty, "--start", days[0], "--end", days[2],
             "--day-step", "1", "--out", fo], "feat_analysis", env=dict(os.environ, PYTHONIOENCODING="cp1251"))
        rep = open(os.path.join(fo, "feat_report.md"), encoding="utf-8").read()
        assert os.path.isfile(os.path.join(fo, "feat_late.csv")), "нет feat_late.csv"
        for need in ("Самые похожие пары", "ic_beyond_imb1_mean_12", "drop_ofi", "drop_micro",
                     "## 4. Горизонт 8 с", "late_share_mean"):
            assert need in rep, f"в feat_report.md нет «{need}»"
        for inst in ("BTCUSDT_PERP", "1000PEPEUSDT_PERP"):
            assert os.path.isfile(os.path.join(fo, f"feat_corr_{inst}.csv")), inst
        win = dict(os.environ, PYTHONIOENCODING="cp1251")       # как вывод в трубу на русской Windows
        out = run(base + ["--name", "st_all", "--model", "ridge", "--ref", "st_base",
                          "--save", os.path.join(models, "st_all")], "exp ridge", env=win)
        assert "против st_base" in out, "строка сравнения не напечаталась"
        score = float(out.split("SCORE:")[1].split()[0])
        assert np.isfinite(score) and 0 < score < 1, score
        msg = f"ridge SCORE {score:.4f}"
        try:
            import lightgbm  # noqa: F401
            out2 = run(base + ["--name", "st_lgbm", "--model", "lgbm", "--ref", "st_all", "--rounds", "60",
                               "--min-leaf", "200", "--max-rows", "200000",
                               "--save", os.path.join(models, "st_lgbm")], "exp lgbm")
            msg += f", lgbm SCORE {float(out2.split('SCORE:')[1].split()[0]):.4f}"
        except ImportError:
            msg += ", lgbm пропущен (не установлен)"
        try:
            import catboost  # noqa: F401
            o3 = run(base + ["--name", "st_cb", "--model", "catboost", "--ref", "st_all", "--rounds", "60",
                             "--max-rows", "200000", "--save", os.path.join(models, "st_cb")], "exp catboost")
            msg += f", catboost SCORE {float(o3.split('SCORE:')[1].split()[0]):.4f}"
            import pyarrow.parquet as pq
            pp = os.path.join(work, "preds_cb", f"{days[3]}.parquet")
            run(["predict.py", "--data-dir", dirty, "--date", days[3], "--out", pp,
                 "--model", os.path.join(models, "st_cb", "model.json")], "predict catboost")
            t = pq.read_table(pp)
            assert t.num_rows == 4 * 172_800 and np.isfinite(t["pred_2s"].to_numpy()).all()
        except ImportError:
            msg += ", catboost пропущен (не установлен)"
        return msg
    check("cache.py + experiment.py", t_cache_exp)

    def t_v2():
        run(["cache.py", "--data-dir", dirty, "--cache-dir", cache, "--block", "v2", "--start", days[0],
             "--end", days[-1], "--split", days[3], "--workers", "2"], "cache v2")
        base = ["experiment.py", "--cache-dir", cache, "--data-dir", dirty, "--train", days[0], days[2],
                "--valid", days[3], days[4], "--results", res, "--block", "base+v2"]
        out = run(base + ["--name", "st_v2", "--model", "ridge", "--ref", "st_all",
                          "--save", os.path.join(models, "st_v2")], "exp v2 ridge")
        assert "признаков 60" in out, out[:300]
        msg = f"ridge base+v2 SCORE {float(out.split('SCORE:')[1].split()[0]):.4f}"
        try:
            import lightgbm  # noqa: F401
            o2 = run(base + ["--name", "st_v2_lgbm", "--model", "lgbm", "--rounds", "40", "--min-leaf", "200",
                             "--max-rows", "200000", "--save", os.path.join(models, "st_v2_lgbm")], "exp v2 lgbm")
            msg += f", lgbm {float(o2.split('SCORE:')[1].split()[0]):.4f}"
        except ImportError:
            pass
        import pyarrow.parquet as pq
        mp = os.path.join(models, "st_v2", "model.json")
        for d in days[1:]:
            p = os.path.join(work, "preds_v2", f"{d}.parquet")
            run(["predict.py", "--data-dir", dirty, "--date", d, "--out", p, "--model", mp], f"predict v2 {d}")
            t = pq.read_table(p)
            assert t.num_rows == 4 * 172_800
            for c in ("pred_500ms", "pred_2s", "pred_8s"):
                assert np.isfinite(t[c].to_numpy()).all(), f"{d} {c}: NaN/inf"
        for d in (days[1], days[2], days[3]):          # сбои перпов и спота (BTC-спот d2 нет, PEPE-спот d1 дыра)
            o = run(["causality_check.py", "--data-dir", dirty, "--date", d, "--trials", "3", "--model", mp],
                    f"causality v2 {d}")
            assert "утечек не найдено" in o, o[-600:]
        for d in (days[1], days[2], days[3]):          # причинность каждого признака (base+v2+v3), опасные T
            o = run(["causality_check.py", "--data-dir", dirty, "--date", d, "--trials", "2", "--features"],
                    f"causality features {d}")
            assert "утечек не найдено" in o, o[-900:]
        # контроль: признак, нарочно заглядывающий на 1 точку вперёд, проверка обязана поймать.
        # Отдельным процессом — чтобы память освобождалась (иначе тест упирается в лимит памяти).
        ctl = (
            "import sys, numpy as np\n"
            "import causality_check as cc\n"
            "from types import SimpleNamespace as NS\n"
            "real = cc.compute_v2\n"
            "def leaky(root, inst, date, cache=None):\n"
            "    X = real(root, inst, date, cache); X[:-1, 0] = X[1:, 0]; return X\n"
            "cc.compute_v2 = leaky\n"
            f"ok = cc.check_features(NS(data_dir={dirty!r}, date={days[2]!r}, trials=0), "
            f"cc.grid_points({days[2]!r}), np.random.default_rng(0))\n"
            "print('CAUGHT' if not ok else 'MISSED')\n")
        o = run(["-c", ctl], "контроль подглядывающего признака")
        assert "CAUGHT" in o, "проверка признаков не поймала нарочно подглядывающий признак"
        o = run(["causality_check.py", "--data-dir", dirty, "--date", days[0], "--trials", "1", "--features"],
                "causality features first day")       # первый день: вчерашнего нет — запасной шаг цены
        assert "утечек не найдено" in o, o[-900:]
        return msg + "; predict 4 дня; причинность прогнозов и всех признаков base+v2+v3 на 4 днях — OK"
    check("признаки v2: кэш, эксперимент, predict, причинность", t_v2)

    def t_v3():
        run(["cache.py", "--data-dir", dirty, "--cache-dir", cache, "--block", "v3", "--start", days[0],
             "--end", days[-1], "--split", days[3], "--workers", "2"], "cache v3")
        base = ["experiment.py", "--cache-dir", cache, "--data-dir", dirty, "--train", days[0], days[2],
                "--valid", days[3], days[4], "--results", res, "--block", "base+v2+v3"]
        out = run(base + ["--name", "st_v3", "--model", "ridge", "--ref", "st_v2",
                          "--save", os.path.join(models, "st_v3")], "exp v3 ridge")
        assert "признаков 90" in out, out[:300]
        msg = f"ridge base+v2+v3 SCORE {float(out.split('SCORE:')[1].split()[0]):.4f}"
        # на дне отклонения USDC (d3) пары к USDC недоступны ровно в окне маскировки
        import features_v3 as f3
        X = f3.compute_v3(dirty, "BTCUSDT_PERP", days[3])
        av = X[:, f3.V3_NAMES.index("avail_usdc")]
        win = slice(6 * 7200, 6 * 7200 + 3600)
        assert av[win].max() == 0 and av[:6 * 7200].mean() > 0.99, "маскировка USDC не учтена"
        assert (X[win, f3.V3_NAMES.index("gap_usdc")] == 0).all()
        X1 = f3.compute_v3(dirty, "BTCUSDT_PERP", days[1])
        assert X1[:, f3.V3_NAMES.index("avail_usd1")].max() == 0, "USD1 до листинга помечена доступной"
        X4 = f3.compute_v3(dirty, "BTCUSDT_PERP", days[4])
        assert X4[9 * 7200 + 100:10 * 7200, f3.V3_NAMES.index("avail_fdusd")].max() == 0, "устаревший курс FDUSD"
        import pyarrow.parquet as pq
        mp = os.path.join(models, "st_v3", "model.json")
        for d in days[1:]:
            p = os.path.join(work, "preds_v3", f"{d}.parquet")
            run(["predict.py", "--data-dir", dirty, "--date", d, "--out", p, "--model", mp], f"predict v3 {d}")
            t = pq.read_table(p)
            assert t.num_rows == 4 * 172_800
            for c in ("pred_500ms", "pred_2s", "pred_8s"):
                assert np.isfinite(t[c].to_numpy()).all(), f"{d} {c}: NaN/inf"
        for d in (days[3], days[4]):                   # маскировка USDC, устаревший курс FDUSD
            o = run(["causality_check.py", "--data-dir", dirty, "--date", d, "--trials", "2", "--model", mp],
                    f"causality v3 {d}")
            assert "утечек не найдено" in o, o[-600:]
            o = run(["causality_check.py", "--data-dir", dirty, "--date", d, "--trials", "1", "--features"],
                    f"causality features v3 {d}")
            assert "утечек не найдено" in o, o[-900:]
        o = run(["stale_check.py", "--data-dir", dirty, "--days", "3", "--start", days[1], "--end", days[3],
                 "--out", os.path.join(work, "stale.csv")], "stale_check")
        for m, want in (("BTCUSDT_SPOT", "86399.0"), ("BTCUSDT_PERP", "599.0"), ("PEPEUSDT_SPOT", "3599.0")):
            line = next(l for l in o.splitlines() if l.startswith(m))
            assert line.split()[-1] == want, f"stale_check: {m} — {line}"
        # общий LightGBM, нейросеть, ансамбль: прогнозы → веса → model.json → predict → причинность
        pdir = os.path.join(work, "ens_preds")
        shutil.rmtree(pdir, ignore_errors=True)
        sp = ["--save-preds", pdir]
        run(base + ["--name", "st_e_ridge", "--model", "ridge", "--save", os.path.join(models, "st_e_ridge")] + sp,
            "exp ens ridge")
        members = ["st_e_ridge"]
        try:
            import lightgbm  # noqa: F401
            for nm, kind in (("st_e_lgbm", "lgbm"), ("st_e_pool", "lgbm_pool")):
                o = run(base + ["--name", nm, "--model", kind, "--rounds", "40", "--min-leaf", "200",
                                "--max-rows", "100000", "--save", os.path.join(models, nm)] + sp, f"exp ens {kind}")
                assert "SCORE:" in o, o[-400:]
                members.append(nm)
        except ImportError:
            pass
        o = run(base + ["--name", "st_e_mlp", "--model", "mlp", "--mlp-hidden", "16", "8", "--mlp-epochs", "2",
                        "--max-rows", "60000", "--save", os.path.join(models, "st_e_mlp")] + sp, "exp ens mlp")
        assert "эпоха 1" in o and "SCORE:" in o, o[-600:]
        members.append("st_e_mlp")
        msg += f"; нейросеть SCORE {float(o.split('SCORE:')[1].split()[0]):.4f}"
        wj = os.path.join(res, "st_ens_w.json")
        o = run(["ensemble.py", "fit", "--preds-dir", pdir, "--members", *members, "--results", res,
                 "--step", "0.25", "--out", wj], "ensemble fit")
        assert "в ансамбль вошли" in o, o[-600:]
        o = run(["ensemble.py", "eval", "--preds-dir", pdir, "--weights", wj, "--members", *members,
                 "--results", res], "ensemble eval")
        assert "АНСАМБЛЬ" in o, o[-400:]
        try:
            run(["ensemble.py", "eval", "--preds-dir", pdir, "--weights", wj, "--members", members[0],
                 "--results", res], "ensemble eval не тот состав")
            raise AssertionError("eval с другим составом участников не остановился")
        except RuntimeError as e:
            assert "тот же состав" in str(e), str(e)[-300:]
        try:                                           # tune.py: 2 попытки, цель — честный SCORE ансамбля
            import optuna  # noqa: F401
            have_opt = "st_e_lgbm" in members
        except ImportError:
            have_opt = False
        if have_opt:
            import json
            for x in ("optuna_lgbm.db", "optuna_lgbm_best.json", "optuna_lgbm_log.txt"):   # прошлый автотест
                if os.path.isfile(os.path.join(res, x)):
                    os.remove(os.path.join(res, x))
            o = run(["tune.py", "--model", "lgbm", "--members", *members, "--replace", "st_e_lgbm",
                     "--preds-dir", pdir, "--results", res, "--trials", "2", "--step", "0.25", "--rounds", "40",
                     "--max-rows", "100000"] + base[1:], "tune.py")
            assert "=== ИТОГ" in o and "отсчёт" in o, o[-800:]
            with open(os.path.join(res, "optuna_lgbm_best.json"), encoding="utf-8") as f:
                bj = json.load(f)
            assert os.path.isdir(os.path.join(pdir, bj["trial"])), "прогнозы лучшей попытки удалены"
            left = [x for x in os.listdir(pdir) if x.startswith("opt_lgbm_")]
            assert left == [bj["trial"]], f"прогнозы неудачных попыток не удалены: {left}"
            msg += f"; tune.py: 2 попытки, ансамбль честно {bj['cv_ensemble']:.4f}"
        else:
            msg += "; tune.py пропущен (нет optuna)"
        if "st_e_lgbm" in members:                     # веса по свежести: lgbm и общий lgbm проходят целиком
            for nm, kind in (("st_rec_lgbm", "lgbm"), ("st_rec_pool", "lgbm_pool")):
                o = run(base + ["--name", nm, "--model", kind, "--rounds", "40", "--min-leaf", "200",
                                "--max-rows", "100000", "--recency-halflife", "1"], f"exp {kind} --recency-halflife")
                assert "полураспад 1 дн." in o and "SCORE:" in o, o[-400:]
            msg += "; --recency-halflife (lgbm, lgbm_pool)"
        try:                                           # GRU: обучение (torch) → model.json → predict → причинность
            import torch  # noqa: F401
            have_torch = True
        except ImportError:
            have_torch = False
        if have_torch:
            gdir = os.path.join(models, "st_gru")
            o = run(base + ["--name", "st_gru", "--model", "gru", "--gru-hidden", "8", "--gru-burn", "8",
                            "--gru-seq", "8", "--gru-batch", "16", "--gru-steps", "20", "--gru-epochs", "2",
                            "--max-rows", "100000", "--save", gdir], "exp gru")
            assert "эпоха 1" in o and "SCORE:" in o, o[-600:]
            gscore = float(o.split("SCORE:")[1].split()[0])
            gm = os.path.join(gdir, "model.json")
            p = os.path.join(work, "preds_gru.parquet")
            run(["predict.py", "--data-dir", dirty, "--date", days[3], "--out", p, "--model", gm], "predict gru")
            import pyarrow.parquet as pq
            t = pq.read_table(p)
            assert t.num_rows == 4 * 172_800 and np.isfinite(t["pred_2s"].to_numpy()).all()
            o = run(["causality_check.py", "--data-dir", dirty, "--date", days[3], "--trials", "2", "--special",
                     "--special-markets", "2", "--model", gm], "causality gru")
            assert "утечек не найдено" in o, o[-600:]
            msg += f"; GRU SCORE {gscore:.4f}, predict, причинность"
        else:
            msg += "; GRU пропущен (нет torch)"
        import json
        fdir = os.path.join(models, "st_final")       # --final: трейн + дни валидации (все точки → каждая 5-я)
        o = run(["experiment.py", "--cache-dir", cache, "--data-dir", dirty, "--block", "base+v2+v3",
                 "--results", res, "--name", "st_final", "--model", "ridge", "--final",
                 "--train", days[0], days[4], "--save", fdir], "exp --final")
        assert "финальная модель" in o and "SCORE" not in o, o[-500:]
        with open(os.path.join(fdir, "model.json"), encoding="utf-8") as f:
            fm = json.load(f)
        assert fm["train"] == [days[0], days[4]] and fm.get("pred_std"), fm.get("train")
        p = os.path.join(work, "preds_final.parquet")
        run(["predict.py", "--data-dir", dirty, "--date", days[4], "--out", p, "--model",
             os.path.join(fdir, "model.json")], "predict --final")
        msg += "; --final (трейн + валидация) → predict"
        import json as _json
        import predict as pr
        wq = os.path.join(res, "st_ens_eq.json")           # все участники с равными весами — проверка суммы
        with open(wq, "w") as f:
            _json.dump({"members": members, "weights": {i: {h: [1 / len(members)] * len(members)
                                                             for h in ("y_500ms", "y_2s", "y_8s")}
                                                         for i in ("BTCUSDT_PERP", "ETHUSDT_PERP", "SOLUSDT_PERP",
                                                                   "1000PEPEUSDT_PERP")}}, f)
        for nm, wf in (("st_ens", wj), ("st_ens_eq", wq)):
            ens = os.path.join(models, nm)
            run(["ensemble.py", "build", "--weights", wf, "--models",
                 *[os.path.join(models, m, "model.json") for m in members], "--out", ens], f"ensemble build {nm}")
            # прогноз ансамбля = Σ коэф × прогноз участника (по их собственным model.json)
            em = pr.load_model(os.path.join(ens, "model.json"))
            pe = pr.predict_day(dirty, days[3], em)
            ps = [pr.predict_day(dirty, days[3], pr.load_model(os.path.join(ens, mb["path"]))) for mb in em["members"]]
            for inst in pe:
                for h in pe[inst]:
                    want = np.zeros(172_800)
                    for mb, p_ in zip(em["members"], ps):
                        c = mb["coef"].get(inst, {}).get(h, 0.0)
                        if c:
                            want = want + c * p_[inst][h]
                    assert np.array_equal(pe[inst][h], want), f"{nm}: ансамбль {inst} {h} ≠ сумме участников"
                    assert np.isfinite(pe[inst][h]).all()
            if nm == "st_ens":
                em_fit = em
        assert len(em["members"]) == len(members), "ансамбль с равными весами потерял участника"
        for kind in ("lgbm_pool", "mlp"):                # общие модели по отдельности тоже через predict.py
            mp_ = os.path.join(models, "st_e_pool" if kind == "lgbm_pool" else "st_e_mlp", "model.json")
            if os.path.isfile(mp_):
                pp = os.path.join(work, f"preds_{kind}", f"{days[4]}.parquet")
                run(["predict.py", "--data-dir", dirty, "--date", days[4], "--out", pp, "--model", mp_],
                    f"predict {kind}")
                t = pq.read_table(pp)
                assert t.num_rows == 4 * 172_800 and np.isfinite(t["pred_8s"].to_numpy()).all()
                o = run(["causality_check.py", "--data-dir", dirty, "--date", days[3], "--trials", "2",
                         "--model", mp_], f"causality {kind}")
                assert "утечек не найдено" in o, o[-600:]
        for d in (days[2], days[3]):                     # ens = ансамбль со всеми участниками
            o = run(["causality_check.py", "--data-dir", dirty, "--date", d, "--trials", "2",
                     "--model", os.path.join(ens, "model.json")], f"causality ensemble {d}")
            assert "утечек не найдено" in o, o[-600:]
        o = run(["causality_check.py", "--data-dir", dirty, "--date", days[3], "--trials", "1", "--special",
                 "--special-markets", "3", "--model", os.path.join(ens, "model.json")], "causality ensemble --special")
        assert "утечек не найдено" in o and o.count("проба") >= 8 + 1 + 2 * 3, o[-600:]
        import causality_check as cc
        sp = cc.special_times(dirty, days[3], cc.grid_points(days[3]), np.random.default_rng(1))
        assert len(sp) >= 8 + 2 * 6, f"опасных моментов по рынкам мало: {len(sp)}"

        # подпись старого кэша: файлы без отпечатка → strict отказывает → --sign проверяет пересчётом и подписывает
        bdir = os.path.join(cache, "base", "BTCUSDT_PERP")

        def strip(p, tamper=False):
            with open(p, "rb") as f:
                z = np.load(f, allow_pickle=False)
                d = {n: z[n] for n in z.files if n != "fhash"}
            if tamper:
                d["X"] = d["X"].copy(); d["X"][5, 0] += 1e-3
            with open(p, "wb") as f:
                np.savez(f, **d)
        for d in days[:3]:
            strip(os.path.join(bdir, f"{d}.npz"))
        strict = ["experiment.py", "--cache-dir", cache, "--data-dir", dirty, "--train", days[0], days[2],
                  "--valid", days[3], days[4], "--results", res, "--name", "st_strict", "--model", "ridge",
                  "--strict-cache"]
        try:
            run(strict, "strict на неподписанном кэше")
            raise AssertionError("--strict-cache принял кэш без отпечатка")
        except RuntimeError as e:
            assert "без отпечатка" in str(e), str(e)[-300:]
        o = run(["cache.py", "--data-dir", dirty, "--cache-dir", cache, "--block", "base", "--sign"], "cache --sign")
        assert "совпадает" in o and "подписано 3/3" in o, o[-500:]
        run(strict, "strict после подписи")
        victim = os.path.join(bdir, f"{days[1]}.npz")
        strip(victim, tamper=True)
        try:
            run(["cache.py", "--data-dir", dirty, "--cache-dir", cache, "--block", "base", "--sign"], "sign подменённого")
            raise AssertionError("--sign подписал кэш, не совпадающий с кодом")
        except RuntimeError as e:
            assert "НЕ подписан" in str(e), str(e)[-300:]
        os.remove(victim)                              # восстановить: cache.py досчитает пропавший файл
        run(["cache.py", "--data-dir", dirty, "--cache-dir", cache, "--start", days[0], "--end", days[-1],
             "--split", days[3], "--workers", "1"], "cache восстановление")
        msg += "; подпись старого кэша (совпал → подписан, подмена → отказ), --strict-cache, опасные T по всем рынкам"
        msg += (f"; подобранный ансамбль: {', '.join(m['type'] for m in em_fit['members'])}; ансамбль из всех "
                f"{len(em['members'])} ({', '.join(m['type'] for m in em['members'])}) = сумма участников, причинность OK")
        return msg + "; маскировка USDC, нет USD1, устаревший FDUSD — учтены; predict; причинность; stale_check — OK"
    check("признаки v3: кэш, эксперимент, predict, маскировка, причинность", t_v3)

    def t_train():
        out = run(["train.py", "--data-dir", dirty, "--train", days[0], days[2], "--valid", days[3], days[4],
                   "--out", os.path.join(work, "train_model.json")], "train")
        assert "SCORE" in out
    check("train.py на грязных данных", t_train)

    def t_predict():
        import pyarrow.parquet as pq
        mp = os.path.join(models, "st_all", "model.json")
        for d in days[1:]:
            p = os.path.join(work, "preds", f"{d}.parquet")
            run(["predict.py", "--data-dir", dirty, "--date", d, "--out", p, "--model", mp], f"predict {d}")
            t = pq.read_table(p)
            assert t.num_rows == 4 * 172_800, t.num_rows
            for c in ("pred_500ms", "pred_2s", "pred_8s"):
                assert np.isfinite(t[c].to_numpy()).all(), f"{d} {c}: NaN/inf"
        return "4 дня × 691 200 строк, всё конечное"
    check("predict.py по всем дням (включая сбойные)", t_predict)

    def t_metric():
        out = run(["metric.py", "--data-dir", dirty, "--pred-dir", os.path.join(work, "preds"),
                   "--dates", days[0], days[4]], "metric")
        assert "нет файлов прогноза за 1 дн." in out, out[-600:]
        return "день без файла прогноза учтён как 0"
    check("metric.py: пропущенный файл прогноза", t_metric)

    def t_causal():
        mp = os.path.join(models, "st_all", "model.json")
        for d in (days[1], days[2], days[3]):          # сбой BTC, нет book SOL, пустой trades PEPE
            out = run(["causality_check.py", "--data-dir", dirty, "--date", d, "--trials", "2", "--model", mp],
                      f"causality {d}")
            assert "утечек не найдено" in out, out[-500:]
        return "3 сбойных дня — OK"
    check("causality_check.py на сбойных днях", t_causal)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=os.path.join(HERE, "selftest_work"))
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)
    print("== 1. юнит-тесты")
    unit_tests()
    if not a.quick:
        print("== 2. интеграция на грязных данных")
        integration(os.path.abspath(a.work))
    bad = [r for r in RESULTS if not r[0]]
    print(f"\nИТОГ: {len(RESULTS) - len(bad)} PASS, {len(bad)} FAIL")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    from textio import utf8_console
    utf8_console()
    main()
