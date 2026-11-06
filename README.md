# Short-Term Return Prediction for Binance Perpetual Futures

Every 500 ms, the model predicts log returns of BTCUSDT, ETHUSDT, SOLUSDT and 1000PEPEUSDT perpetuals over 500 ms, 2 s and 8 s horizons.

## Running

Requires Python 3.12+ (uv will fetch it if missing).

```
uv sync
uv run python predict.py --data-dir <DIR> --date <YYYY-MM-DD> --out <PATH>.parquet
```

The default model is `model.json` next to `predict.py` (an ensemble; its members are in `members/`).
Only the requested day and the preceding day are needed (warm-up of rolling windows). One day takes about 1.5 minutes on 10 CPU cores.
No GPU or torch is needed for prediction: both neural networks run in numpy.

## Approach

**Features (76, all strictly causal: a prediction at time t uses only data with ts ≤ t):**
- the perpetual's own order book: best-quote and depth-bucket imbalances, microprice, spread, and their dynamics;
- trade flow (buy − sell volume) and order flow imbalance over windows from 0.5 s to several minutes and longer, trading intensity;
- returns of the instrument itself and of leading markets (BTC/ETH for the others) over several windows;
- spot markets and the second perpetual (1000PEPEUSDC): price gaps in a common currency via the stablecoin rate, their deviation from a moving average, quote freshness; volatility and feature interactions with it.

Of 90 engineered features, 14 were removed after group-wise ablation (day-level comparison with bootstrap).

**Models (6 ensemble members):**

| member | description |
|---|---|
| LightGBM | one model per instrument × horizon; hyperparameters tuned with Optuna on the score of the whole ensemble |
| pooled LightGBM | one model for all 4 instruments (instrument id as a categorical feature) |
| CatBoost | one model per instrument × horizon |
| Ridge | linear regression |
| MLP | 256-128-64, SiLU, one network for all instruments and horizons |
| GRU | recurrent network over the feature sequence with a 2.5 s step; state starts at the beginning of the day |

The target of each day is divided by its standard deviation, so all days carry equal weight. Tree models and neural networks use the Huber loss. Early stopping uses the last 10% of training days.

**Ensemble:** each member's prediction is divided by its standard deviation. Weights (≥ 0, summing to 1) are fitted separately for each instrument × horizon pair by grid search with step 0.05 on the exact metric (mean daily correlation). The weighting scheme (global / per horizon / per pair) was chosen honestly: fit on even days, evaluate on odd days, and vice versa.

## Results

Models were trained on the training period (2025-09-15 to 2026-07-14) and evaluated on the validation period (2026-07-15 to 2026-09-14) with the competition metric.
"Train" is measured on the same days the models were trained on (every 3rd day, every 5th grid point).

| model | train | validation |
|---|---|---|
| baseline (OLS on best bid/ask queue imbalance) | 0.1991 | 0.1838 |
| Ridge, 76 features | 0.2248 | 0.2130 |
| MLP | 0.2398 | 0.2245 |
| GRU | 0.2398 | 0.2253 |
| CatBoost | 0.2442 | 0.2285 |
| pooled LightGBM | 0.2436 | 0.2304 |
| LightGBM | 0.2475 | 0.2308 |
| **ensemble (6 models)** | **0.2488** | **0.2346** |

Validation by pair (ensemble):

| | 500 ms | 2 s | 8 s |
|---|---|---|---|
| BTC | 0.3050 | 0.2728 | 0.2199 |
| ETH | 0.2868 | 0.2435 | 0.1789 |
| SOL | 0.2732 | 0.2599 | 0.2091 |
| PEPE | 0.2196 | 0.2017 | 0.1445 |

- The train − validation gap is the same for the ensemble (0.014) and for the linear baseline (0.015). This points to a harder validation period rather than overfitting; in particular, during the week of Aug 19–25 the longer horizons dropped for all instruments.
- Models, hyperparameters and ensemble weights were selected on a held-out part of the training period (2026-05-01 to 2026-07-14); the validation period was not used for selection.
- **The submitted model** is the same ensemble retrained on training + validation data (2025-09-15 to 2026-09-14), as allowed by the rules.
- **Causality:** checked with the same procedure the organizers use (truncate the data at time T; predictions for t ≤ T must match), bit-for-bit, on real days, including "dangerous" T values: exactly at, and 1 ms before, a quote change in any market. Every feature was also checked individually. No leaks were found.

## Files

- `predict.py`: prediction; `common.py`: the organizers' file, unchanged;
- `features.py`, `features_v2.py`, `features_v3.py`: features; `fversion.py`: feature-code fingerprints (a model will not load with a different version of the feature code);
- `mlp.py`, `gru.py`: numpy inference for the neural networks (and their training);
- `model.json`, `members/`: ensemble weights;
- the rest: training and validation code: `cache.py` (feature cache), `experiment.py` (training and evaluation), `ensemble.py` (weight fitting and assembly), `tune.py` (Optuna), `report.py` (train scores), `causality_check.py`, `selftest.py`.
