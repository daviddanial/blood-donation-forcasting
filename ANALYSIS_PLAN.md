# ANALYSIS_PLAN.md — Ethiopian National Blood Supply Chain Predictive Intelligence

**Purpose.** This file is the blueprint for the technical analysis. It is the
"what to build and why" companion to the grant proposal. It was written while
inspecting four GitHub references (CEEMDAN_LSTM, FECAM, TimePro, SMETimes) and
mapping their techniques onto the proposal's objectives.

**Status.** The simulation + XGBoost baseline are implemented and runnable in
this repo. The CEEMDAN / FECAM / TimePro / SMETimes stages are *planned* next
steps (Section 6).

---

## 1. Objective recap (from the proposal)

Predict daily/weekly national blood **collection** and **demand**, disaggregated
by product (whole blood, PRBC, platelets, FFP) and blood group (ABO/Rh), across
the 43-regional-bank EBTBS network, over three plan horizons — **7 (tactical),
30 (medium), 90 (strategic) days** — with SHAP-based explainability for
operational decisions.

The two hard problems the data exposes:
- **Structural supply gap:** ~200,000 units/yr demanded vs ~87,000 collected.
- **Fasting shocks:** Great Lent (55–56 days) and Ramadan cut voluntary donation
  20–30%, while clinical demand stays constant → the gap widens when it hurts most.
- **Component/group scarcity:** platelets have a 5–7 day shelf life; O− is only
  2.76% of the donor base → a national healthy aggregate can mask a regional or
  component-level shortfall.

---

## 2. Where the four GitHub references fit

The proposal's core engine (TFT, BiLSTM, XGBoost) stays. The references add
**decomposition, frequency-domain attention, efficient long-horizon models, and
a benchmark harness.**

| Reference | What it gives us | Proposal objectives it serves |
|---|---|---|
| **[CEEMDAN_LSTM](https://github.com/bhaskatripathi/CEEMDAN_LSTM)** | CEEMDAN decomposes a series into IMFs + residue (re-decomposed iteratively); each IMF is forecast by an LSTM and summed. Separates trend / weekly / seasonal / shock structure → a cleaner signal than raw noise. | **Obj 1** (temporal patterns), **Obj 7** (platelets — the shortest shelf life, most noise-sensitive). Good at 7/30-day and for disrupted regions. |
| **[FECAM](https://github.com/likegopher/FECAM)** | Frequency-Enhanced **Channel** Attention — a plug-in FFT-based attention module for multivariate models. | **Obj 2 & 6** (multi-component + multi-region): learns dependencies across components and ABO/Rh channels. |
| **[TimePro](https://github.com/jingmouren/xwmaxwma-TimePro)** | ICML 2025 Mamba/state-space multivariate **long-term** forecaster (variable- & time-aware hyper-state). | **Obj 4** 90-day strategic horizon; and the proposal's "proactive rather than reactive" goal. ⚠ requires `selective_scan` (CUDA compile). |
| **[SMETimes](https://github.com/xiyan1234567/SMETimes)** | Long-term forecasting **benchmark suite** + preprocessing harness. | Use as an orchestration/baselines template and to validate reimplementations on the ETTh benchmarks before touching blood data. |

**Recommended engine (this repo):** TFT + BiLSTM + XGBoost (as required),
**plus** a 4th CEEMDAN-LSTM decomposition model, with **FECAM** used as a
drop-in attention module inside the BiLSTM/TFT. TimePro and SMETimes serve as
the *long-horizon baselines and validation harness*.

---

## 3. Data sources

### 3a. Real data (highest fidelity — behind ethics/partnership, proposal SOP 1)

| Source | What it holds | How to access |
|---|---|---|
| **BSIS** (Blood Safety Information System) | donor registries, collection volumes, testing outcomes, component-preparation logs | BSIS **Web API**; needs the EBTBS/HABTech agreement (already in the proposal; v1.3 live Feb 2025) |
| **DHIS2 national aggregate** | transfusion requests, bed occupancy, emergency obstetric admissions | DHIS2 **REST API** (`/api/analytics`); prototype on a public demo instance first |
| **Published constants** (for calibration / sanity checks) | ABO/Rh base rates, annual demand vs collection, wastage rates | Ref [24] (ABO/Rh), refs [2][37] (200k vs 87k), refs [3][40] (16.6% platelet wastage) — see Section 4 |
| **WHO GDBS** | country-level blood collection/utilization | Public WHO database (Ethiopia coverage is spotty) |
| **Open benchmark TS datasets** | ETTh1/2, ETTm1/2, Electricity, Traffic, Weather, Exchange, Solar | Verify your model reimplementation before injecting blood data |
| **Blood-specific open data** (limited) | UCI Blood Transfusion set; Kaggle donation sets | Tiny, mostly classification; use only for a first sanity pass |

### 3b. Simulation (implemented — build now, swap later)

Because BSIS/DHIS2 access waits on IRB/site agreements, the repo generates a
**domain-aware synthetic series** calibrated to the proposal's factual
constants. A `DataSource` interface (`synthetic_data` + `data_source.py`) means
you swap in `BSISSource`/`DHIS2Source` **without touching the models**.

The simulation reproduces:
- the ~87k/yr collection vs ~200k/yr demand gap,
- **Great Lent** (Orthodox Easter-computus → Gregorian window) and **Ramadan**
  (Hijri calendar, shifts ≈11 days earlier each year) collection drops,
- weekly rhythm, annual seasonality, public holidays,
- ABO/Rh proportions (incl. rare O−, A−, B−, AB−),
- random **conflict-disruption windows** for Afar/Somali/Tigray/Amhara,
- Poisson count noise and 5 years of daily data (~1,826 points).

---

## 4. Calibration constants used

| Constant | Value | Grounding |
|---|---|---|
| National collection | 87,000 units/yr | refs [2][37] |
| National demand | 200,000 units/yr | refs [2][37] |
| Lent collection drop | ×0.80 | refs [5][6] (20% contraction) |
| Ramadan collection drop | ×0.78 | refs [7–10] (20–30% drop) |
| ABO/Rh shares | O+ .4189, O− .0276, A+ .2747, A− .0094, B+ .2006, B− .0118, AB+ .0541, AB− .0030 | ref [24] |
| Platelet shelf life | 5–7 days | ref [3] |
| Train / val / test | 48 mo / 6 mo / 6 mo | proposal §3.4 |

---

## 5. Pipeline already implemented

```
synthetic_data.py ──> data/collection_long.csv, issuance_long.csv, national_daily.csv
        │
calendar_features.py ──> Lent / Ramadan / holidays / cyclic encodings / lags / rolling
        │
feature_engineering.py ──> (X, y) supervised frame, no leakage
        │
train_baseline.py ──> XGBoost per horizon (7/30/90) → MAE / RMSE / MAPE + persistence baseline
        │
output/baseline_metrics.json + baseline_plot.png
```

**Evaluation is strict and chronological** (never shuffled): train < val < test,
and metrics are reported per horizon. A naive *persistence* baseline is computed
so you can see whether the model beats "just repeat the last value."

---

## 6. Next stages (mapped to the references)

1. **CEEMDAN-LSTM** — `PyEMD.EMD` → CEEMDAN IMFs → per-IMF LSTM → sum. Uses the
   long tables for disaggregation (Obj 1, 7).
2. **FECAM module** — wrap the FFT channel-attention around BiLSTM/TFT for
   multivariate, multi-component forecasting (Obj 2, 6).
3. **Re-validate on ETTh** — confirm your FECAM/TimePro reimplementation matches
   the paper on ETTh1/2 before applying to blood (uses SMETimes harness).
4. **TimePro** for 90-day strategic forecasting — requires a CUDA box for the
   `selective_scan` extension; otherwise use TFT for 90-day.
5. **SHAP** (Obj 5) — explain the XGBoost/CEEMDAN forecasts to feed the tiered
   recommendation protocol in proposal §5.1.1.

---

## 7. Stack & repo layout

- Python 3.12 venv; deps in `requirements.txt` (pandas, numpy, scikit-learn,
  xgboost, matplotlib, convertdate). Deep-learning deps are optional/commented.
- Files:
  - `src/config.py` — every tunable constant
  - `src/calendar_features.py` — Ethiopian/Hijri calendar + cyclic encoding
  - `src/synthetic_data.py` — calibrated generator
  - `src/data_source.py` — `DataSource` interface (synthetic/BSIS/DHIS2)
  - `src/feature_engineering.py` — supervised feature builder
  - `src/train_baseline.py` — XGBoost baseline + metrics
  - `src/run_pipeline.py` — one-command end-to-end
