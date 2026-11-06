"""Shared constants and loading helpers for the dataset.

All paths are relative to the data root you were given; point `--data-dir` at it.
"""
import datetime as dt
import os

import numpy as np
import pyarrow.parquet as pq

# ----------------------------------------------------------------- instruments

TARGET_INSTRUMENTS = [
    "BTCUSDT_PERP",
    "ETHUSDT_PERP",
    "SOLUSDT_PERP",
    "1000PEPEUSDT_PERP",
]

FEATURE_INSTRUMENTS = TARGET_INSTRUMENTS + [
    "1000PEPEUSDC_PERP",
    "BTCUSDT_SPOT", "BTCUSDC_SPOT", "BTCFDUSD_SPOT", "BTCUSD1_SPOT",
    "ETHUSDT_SPOT", "ETHUSDC_SPOT", "ETHFDUSD_SPOT", "ETHUSD1_SPOT",
    "SOLUSDT_SPOT", "SOLUSDC_SPOT", "SOLFDUSD_SPOT",
    "PEPEUSDT_SPOT",
]

# quoted in the `stable` table rather than as instruments of their own
STABLE_PAIRS = ["USDCUSDT", "FDUSDUSDT", "USD1USDT"]

HORIZONS = ["y_500ms", "y_2s", "y_8s"]
PREDICTION_COLUMNS = ["pred_500ms", "pred_2s", "pred_8s"]

# ------------------------------------------------------------------- the grids

GRID_MS = 500       # prediction grid; `book` and `targets` live here
BBO_MS = 100        # quote grid
TRADE_MS = 100      # trade bars, interval (ts - 100, ts]
STABLE_MS = 1000    # stablecoin grid
DELAY_MS = 10       # order-send delay before the target window opens

POINTS_PER_DAY = 86_400_000 // GRID_MS   # 172_800

# nominal bucket edges in bps from mid; widened per instrument-day when the
# tick is coarser — read the real ones from meta/daily.parquet
NOMINAL_BUCKET_EDGES = [0.5, 1, 1.5, 2, 10, 20, 30, 40, 50, 75, 100]
N_BUCKETS = len(NOMINAL_BUCKET_EDGES)

BID_BUCKETS = [f"bid_v{i:02d}" for i in range(1, N_BUCKETS + 1)]
ASK_BUCKETS = [f"ask_v{i:02d}" for i in range(1, N_BUCKETS + 1)]

# --------------------------------------------------------------------- loading


def day_path(root, table, date, instrument=None):
    if instrument is None:           # `stable` is one file per day
        return f"{root}/{table}/{date}.parquet"
    return f"{root}/{table}/{instrument}/{date}.parquet"


def read_day(root, table, date, instrument=None, columns=None):
    """One day of one table, or None when that instrument has no data that day.

    Missing files are normal: a pair that was not listed yet simply has none.
    """
    p = day_path(root, table, date, instrument)
    if not os.path.isfile(p):
        return None
    return pq.read_table(p, columns=columns)


def dates(start, end):
    d0, d1 = dt.date.fromisoformat(start), dt.date.fromisoformat(end)
    return [(d0 + dt.timedelta(days=i)).isoformat() for i in range((d1 - d0).days + 1)]


def grid_points(date, step_ms=GRID_MS):
    """Grid timestamps for a UTC day, in milliseconds."""
    t0 = int(dt.datetime.fromisoformat(date)
             .replace(tzinfo=dt.timezone.utc).timestamp()) * 1000
    return np.arange(t0, t0 + 86_400_000, step_ms, dtype="int64")


def quote_grid(root, instrument, date):
    """Full 100 ms quote grid: `bbo` holds only changes, `book` holds the rest.

    Returns (ts, bid_px, ask_px, bid_qty, ask_qty) forward-filled onto the grid,
    or None when the instrument has no data that day.
    """
    cols = ["ts", "bid_px", "ask_px", "bid_qty", "ask_qty"]
    book = read_day(root, "book", date, instrument, columns=cols)
    if book is None:
        return None
    bbo = read_day(root, "bbo", date, instrument, columns=cols)

    ts = book["ts"].to_numpy()
    arrs = [book[c].to_numpy().astype("float64") for c in cols[1:]]
    if bbo is not None and bbo.num_rows:
        ts = np.concatenate([ts, bbo["ts"].to_numpy()])
        arrs = [np.concatenate([a, bbo[c].to_numpy().astype("float64")])
                for a, c in zip(arrs, cols[1:])]
    order = np.argsort(ts, kind="stable")
    ts, arrs = ts[order], [a[order] for a in arrs]

    grid = grid_points(date, BBO_MS)
    i = np.searchsorted(ts, grid, "right") - 1
    ok = i >= 0
    i = np.maximum(i, 0)
    out = [np.where(ok, a[i], np.nan) for a in arrs]
    return (grid, *out)


def align_to_grid(table, grid_ms):
    """Index of each grid point inside `table`, or -1 where the row is absent.

    `trades` skips bars with no trades, so a plain join would silently drop them.
    """
    if table is None:
        return np.full(len(grid_ms), -1)
    ts = table["ts"].to_numpy()
    i = np.searchsorted(ts, grid_ms)
    hit = (i < len(ts)) & (ts[np.minimum(i, max(len(ts) - 1, 0))] == grid_ms)
    return np.where(hit, i, -1)


def load_clip(root):
    """Clip thresholds used by the metric, keyed by (instrument, horizon)."""
    t = pq.read_table(f"{root}/meta/target_clip.parquet").to_pydict()
    return {(i, h): (lo, hi) for i, h, lo, hi
            in zip(t["instrument"], t["horizon"], t["lo"], t["hi"])}
