# Ethiopian Blood Supply Chain — Predictive Intelligence (simulation + baseline)

This repo is the **runnable skeleton** for the analysis proposed in the grant
proposal *National Blood Supply Chain Predictive Intelligence* (the proposal
PDF itself is **not** redistributed in this repo; the technical blueprint it
maps to is `ANALYSIS_PLAN.md`). It implements:

1. a **calibrated synthetic data generator** for the Ethiopian blood supply chain
   (because the real BSIS/DHIS2 data is behind ethics/partnership), and
2. a **first baseline model** — XGBoost with engineered temporal features —
   evaluated at 7 / 30 / 90-day horizons with MAE, RMSE, MAPE.

The design keeps the model layer **decoupled from the data source** so you can
run everything on simulation now and swap in real BSIS/DHIS2 data later without
rewriting the models.

---

## 1. Prerequisites

- Python 3.10+ (tested on 3.12)
- A working `python3` + `pip`
- Internet access for the one-time package install

All commands below are run from the **project root** — the root of this repo,
i.e. the folder containing `src/`, `data/` and `output/`.

> If the project root sits inside a parent folder whose name contains a space,
> remember to **quote** the full path in shell commands.

## 2. Install

```bash
cd blood_forecast      # or the path to your clone

# Create and activate an isolated environment
python3 -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows (PowerShell):
# .venv\Scripts\Activate.ps1

# Install the dependencies
pip install -r requirements.txt
```

That installs `pandas`, `numpy`, `scikit-learn`, `xgboost`, `matplotlib`,
`convertdate` (for the Hijri/Ramadan calendar). Nothing else is required for
this stage.

## 3. Run the whole thing (one command)

```bash
python -m src.run_pipeline
```

That does two steps:

- **Step 1** writes the synthetic CSVs into `data/`
- **Step 2** trains & evaluates the XGBoost baseline and writes
  `output/baseline_metrics.json` and `output/baseline_plot.png`

### Run just one step

```bash
# Only regenerate the data
python -m src.synthetic_data

# Only train the baseline (reuse existing data)
python -m src.train_baseline

# Change the target series (e.g. forecast clinical demand instead of collection)
python -m src.train_baseline --target issuance
```

You can also pass `--target issuance --region national`, or train on a single
region (e.g. `--region "Afar"`) so long as that region exists in the data.

---

## 4. What each file does (read in this order)

| File | Responsibility | Why it matters |
|---|---|---|
| `src/config.py` | **Every** tunable constant: window, national rates, ABO/Rh shares, component shares, calendar multipliers, train/val/test split, horizon list, seeds. | Change calibration here only. |
| `src/calendar_features.py` | Computes **Orthodox Easter** (Julian computus), the **Great Lent** window, **Ramadan** (Hijri via `convertdate`), public holidays, and cyclic (sin/cos) encodings of day-of-week and month. | The two moving calendars are the whole point — they are *not* fixed Gregorian dates. |
| `src/synthetic_data.py` | Generates a 5-year daily simulation per region, component, and ABO/Rh group, calibrated to the documented 87k-vs-200k gap and ABO/Rh base rates. | Lets you develop before the real data is released. |
| `src/data_source.py` | A `DataSource` interface: `SyntheticSource` (implemented), `BSISSource` / `DHIS2Source` (stubs with the field mapping). | This is your "simulation first, real data later" seam. |
| `src/feature_engineering.py` | Builds the supervised `(X, y)` matrix: calendar covariates + lagged targets + rolling means/stds, **without leakage**, and makes a **direct** target per horizon. | Where the forecast problem is actually defined. |
| `src/train_baseline.py` | Trains **one XGBoost model per horizon** (7/30/90), evaluates on a strict chronological test set, compares against a naive *persistence* baseline, plots the 30-day forecast vs actual. | The first measurable result. |
| `src/run_pipeline.py` | One-command orchestrator for steps 1 and 2. | Convenience. |
| `requirements.txt` | Pinned (commentable) dependency list. | Reproducibility. |

---

## 5. How the pipeline is designed (and why)

### 5.1 Calendar (the "moving holiday" problem)

Great Lent and Ramadan do **not** sit on fixed Gregorian dates. So the code
computes:

- **Orthodox Easter** via the Julian-calculus (Meeus) algorithm, then converts
  the Julian date to Gregorian (+13 days in 1900–2099). Great Lent is the
  55-day window ending on that Easter.
- **Ramadan** by converting each Gregorian date to the Hijri calendar with
  `convertdate` and testing month == 9. Because it's lunar, it shifts ~11 days
  earlier every Gregorian year.

You can verify with the self-test:

```bash
python -m src.calendar_features
```

### 5.2 Synthetic data (the "build now, swap later" path)

For each region the generator builds a daily volume from

`baseline × trend × season × weekday × holiday × fasting × noise`

and then allocates each day's total into **4 components × 8 blood groups** using
Dirichlet-perturbed shares that sum exactly to the daily total. Regions flagged
as conflict-affected (Afar, Somali, Tigray, Amhara) get random **disruption
windows** where collection collapses to 15% of normal. Rare groups (O−, A−, B−,
AB−) therefore become sparse — exactly the real data pattern.

The two target series behave differently by construction:

- **collection** — dips during Lent/Ramadan and on holidays/weekends,
- **issuance (demand)** — held at ~constant or slightly higher clinical need
  during fasts/holidays (emergencies don't stop), reproducing the supply gap.

**Realized vs nominal calibration.** `config.py` sets the *nominal* baselines
(87,000 collected / 200,000 demanded per year), but those are multiplied by the
weekday, fasting and holiday factors, so the **realized** series comes out
~5% below on collection and ~5% above on demand — i.e. in the current run
≈ **82,600 units/yr collected vs ≈ 211,500 units/yr demanded**. That widens,
rather than narrows, the documented supply gap, so the qualitative story holds;
but if you need the *realized* totals to land exactly on 87k/200k, rescale
`NATIONAL_YEARLY_COLLECTION` / `NATIONAL_YEARLY_DEMAND` (or normalise the factor
weights) — the check is the `collection/yr` and `issuance/yr` lines that
`run_pipeline` prints in step 1.

### 5.3 No target leakage (important!)

All lag/rolling features are produced with `.shift()` so they only use values
**strictly before** time *t*. The target for horizon *h* is `y[t+h]`. The
train/val/test split is **chronological** (48 / 6 / 6 months), never shuffled, so
the test set is a genuine out-of-sample hold-out.

### 5.4 Direct multi-horizon forecasting

A separate model is trained per horizon rather than one model predicting one day
ahead and being run recursively. This matches the proposal (which reports each
horizon independently) and avoids accumulating recursive error — at the cost of
three models.

---

## 6. Interpreting the output

`output/baseline_metrics.json`:
- each horizon gives `n_train/n_val/n_test`, `MAE`, `RMSE`, `MAPE` for XGBoost,
- plus a `persistence` MAE — the naive "repeat the last observed value"
  benchmark, so you can see the model actually *adds* value.

`output/baseline_plot.png` shows the **30-day** XGBoost forecast over the actual
values on the test set (the last ~6 months of the window).

**Reading the numbers:** a lower MAE/RMSE/MAPE is better. If the model's MAE is
close to the naive persistence MAE, the engineered features aren't yet capturing
real signal — a signal that the next stage (CEEMDAN decomposition / FECAM
attention) should improve.

### Worked example — national total collection (this dataset, 5-year window)

Measured with **real XGBoost** (v3.4.1, `n_estimators=800`, `max_depth=6`,
`learning_rate=0.05`, `seed=42`) on a fresh `python -m src.run_pipeline`:

| Horizon | XGBoost MAE | RMSE | MAPE | Naive (persistence) MAE | Improvement |
|---|---|---|---|---|---|
| 7-day | 18.6 | 23.6 | 8.2% | 23.0 | 1.24× |
| 30-day | 22.5 | 27.8 | 9.6% | 54.8 | 2.44× |
| 90-day | 20.5 | 25.9 | 8.4% | 47.0 | 2.29× |

> Reproduce with `python -m src.run_pipeline`. Exact values shift slightly with
> the installed `numpy`/`pandas` versions (the synthetic generator is seeded but
> the numeric stack is not byte-stable across releases), so treat the
> *ordering and ratios* as the result, not the third decimal.

Interpretation:
- **At 30 and 90 days the model is ~2.3–2.4× better than just repeating the last
  observed value** — the calendar/holiday/lag features are clearly adding signal.
- **At 7 days it beats naive too** (18.6 vs 23.0), by a smaller margin — daily
  collection is strongly persistent over one week, so "repeat yesterday" is hard
  to beat.
- The 7-day MAPE (8.2%) is the *lowest*, with 90-day close behind (8.4%); the
  30-day horizon is the hardest (9.6%), where neither short persistence nor
  long-run smoothing fully pays off.

> **Engine selection / optional deps:** the model prefers `xgboost` and falls
> back to scikit-learn's `HistGradientBoostingRegressor` (same tree family) if
> xgboost isn't importable. Install it with `pip install xgboost`
> (or `pip install --no-deps xgboost` if the GPU-only `nvidia-nccl-cu13`
> download is an issue — numpy/scipy are already present). Matplotlib is optional
> (skips the PNG); `convertdate` adds the Hijri/Ramadan calendar — without it the
> Ramadan flags stay 0, so the synthetic data only shows the Great-Lent
> contraction. On your machine `pip install -r requirements.txt` gives the full
> xgboost + matplotlib + convertdate stack.

---

## 7. Swapping in real BSIS / DHIS2 data

Do **not** touch the models. Implement one method in `src/data_source.py`:

- `BSISSource.load_frame(...)` → BSIS Web API → return a tidy frame with columns
  `date, region, component, blood_group, units`.
- `DHIS2Source.load_frame(...)` → DHIS2 REST `/api/analytics` → same schema.

De-identify and aggregate at source, per proposal SOP 1. Then run
`python -m src.train_baseline` — it just works because it only talks to
`DataSource`.

---

## 8. Where this is heading (next stages)

See `ANALYSIS_PLAN.md` section 6 for the roadmap: **CEEMDAN-LSTM** decomposition
(`PyEMD`), **FECAM** frequency-attention module, re-validation on **ETTh**
benchmarks, **TimePro** for the 90-day strategic horizon (needs CUDA for its
`selective_scan` extension), and **SHAP** explainability (Obj 5).

### Publishing forecasts through the MoH analytics stack

If this project is to be surfaced in the **MoH Superset** deployment — reusing
its **HABETL** ETL — read the source-of-truth integration document first
(kept outside this repo, in the MoH ETL project's `docs/` folder):

`docs/12-blood-bank-forecast-integration.md`

Headline findings that shape this project's output contract:

- **Superset only reads a warehouse** — it is not an ETL and **cannot write back**
  to DHIS2, so it cannot by itself deliver the proposal's forecast push-back.
- **HABETL extracts only from a JSON *array* over HTTP and writes PostgreSQL only**,
  truncating the target table and storing every column as text.
- **The fork's org-unit access control is ClickHouse-specific and dashboard-level**
  (a yes/no gate), *not* per-region row filtering — per-region restriction needs
  standard Superset RLS.

Practical consequence for this repo: keep emitting forecasts as a tidy table with a
**frozen grain** — `date × region × component × blood_group` plus `measure`,
`horizon_days` (7/30/90), `model_name`, `model_version`, `issued_at` — and expose it
as an HTTP JSON **array** endpoint. That single contract satisfies every loading
option described in doc 12.

---

## 9. Troubleshooting

- **`ModuleNotFoundError: xgboost`** → you didn't activate the venv or install
  `requirements.txt`.
- **The plot file is empty/not created** → you need `matplotlib`; the code uses a
  headless `Agg` backend so it works without a display.
- **`convertdate` import fails** → it's optional for Ramadan; the code tolerates
  it (Ramadan flags become 0), but install it for the full effect.
- **Real data looks nothing like the synthetic** → that's expected; synthetic is
  only for development. Recalibrate the constants in `src/config.py`.
