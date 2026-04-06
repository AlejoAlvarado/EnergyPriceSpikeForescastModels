# Alberta Electricity Price Spike Forecasting

Predicting short-term electricity price spikes on the Alberta Electric System Operator (AESO) pool market using machine learning. The task is a binary classification problem: **will the pool price exceed $200/MWh two hours from now?**

---

## Project Structure

```
EnergyPriceSpikeForescastModels/
├── FINAL IPYNBS/                         # Submission notebooks
│   ├── eda.ipynb                         # Exploratory data analysis
│   ├── mlp_aeso_local_no_optuna.ipynb    # Multi-layer perceptron model
│   ├── LSTM_F1_THRESHOLD TUNNING_MAX_50.ipynb  # LSTM model
│   └── cnn_v2_colab_postrun.ipynb        # 1D CNN model
├── Data/
│   ├── CSVs/
│   │   ├── aeso_merged_2020_2025.csv              # Final merged dataset (pipeline output)
│   │   ├── Hourly_Metered_Volumes_and_Pool_Price_and_AIL_2020-Jul2025.csv
│   │   └── CSD Generation (Hourly)*.csv           # One or more generation files
│   └── aeso_merge_pipeline.py            # Script that builds aeso_merged_2020_2025.csv
├── shared/
│   └── data_prep.py                      # Shared data loading & utilities (used by EDA and CNN)
└── requirements.txt
```

---

## Data

### Source Files

Two raw data sources, both downloaded from the [AESO website](https://www.aeso.ca/), are merged to produce the final dataset.

| File | Description |
|---|---|
| `Hourly_Metered_Volumes_and_Pool_Price_and_AIL_2020-Jul2025.csv` | Hourly pool price ($/MWh), Alberta Internal Load (MW), and intertie import/export flows to/from BC, MT, and SK |
| `CSD Generation (Hourly)*.csv` | Hourly generation output (MW) and system capability (MW) by fuel type for all AESO-connected assets |

### Building the Merged Dataset

Run the pipeline script once to produce `aeso_merged_2020_2025.csv`:

```bash
python Data/aeso_merge_pipeline.py
```

This script:
1. Loads pool price and demand data, converts timestamps from GMT to MST
2. Loads all CSD generation files, aggregates to hourly totals by fuel type
3. Inner-joins the two datasets on datetime
4. Engineers all features (see below)
5. Saves the result to `Data/CSVs/aeso_merged_2020_2025.csv`

### Final Dataset

| Property | Value |
|---|---|
| Period | January 2020 – July 2025 |
| Granularity | Hourly |
| Rows | ~48,800 (Feb 29 removed for consistent yearly Fourier encoding) |
| Columns | 117 |
| Spike rate | ~10% |
| Spike threshold | $200 CAD/MWh |

#### Train / Validation / Test Split (time-based, no shuffling)

| Split | Period | Rows |
|---|---|---|
| Train | 2020-01-02 → 2023-11-05 | ~33,700 |
| Validation | 2023-11-06 → 2024-12-11 | ~9,600 |
| Test | 2024-12-12 → 2025-07-30 | ~5,500 |

### Engineered Features

**Target**
- `spike_lead_2` — 1 if pool price at t+2 > $200/MWh, else 0

**Generation & capacity (by fuel type)**
Coal, gas, dual fuel, hydro, wind, solar, energy storage, other — both total generation (MW) and system capability (MW)

**Demand & intertie**
- `ACTUAL_AIL` — Alberta Internal Load (MW)
- `IMPORT_BC/MT/SK`, `EXPORT_BC/MT/SK` — intertie flows
- `net_export` — total exports minus total imports

**Derived system ratios**
- `total_generation`, `total_system_capability`, `dispatchable_generation`
- `renewable_generation`, `renewable_capability`
- `renewables_share`, `dispatchable_ratio`, `gas_ratio`, `intertie_support`
- `reserve_margin`, `resilience_buffer`
- `net_load`, `net_load_3h_change`
- Per-fuel generation mix shares (`coal_mix`, `gas_mix`, etc.)
- Capacity utilisation ratios

**Lag features**
- `price_lag_1h`, `price_lag_6h`, `price_lag_24h`, `price_rolling_mean_6h`
- `spike_lag_1` … `spike_lag_24` — was there a spike N hours ago?
- `spike_lead_1` … `spike_lead_24` — future spike indicators (targets only, never inputs)

**Calendar & cyclical features**
- `hour_of_day`, `day_of_week`, `month`, `is_weekend`
- Fourier encodings: `sin_day/cos_day`, `sin_week/cos_week`, `sin_year_1/cos_year_1`, `sin_year_2/cos_year_2`
- `is_stampede` — Calgary Stampede (July, 2021–2025)

---

## Notebooks

All four notebooks are in `FINAL IPYNBS/` and are designed to run locally. They all read from `Data/CSVs/aeso_merged_2020_2025.csv` (one level up from the notebook folder).

### 1. `EDA.ipynb` — Exploratory Data Analysis
Explores the dataset before any modelling:
- Pool price time series and monthly median trend
- Price distribution and spike tail analysis
- Average price heatmap by hour × month
- Generation mix over time by fuel type
- Spike rate heatmap by hour × month
- Correlation matrix across key features
- System condition comparisons: spike vs non-spike hours
- Alberta Internal Load seasonality

### 2. `MLP.ipynb` — Multi-Layer Perceptron
Flat tabular model treating each hour independently:
- Time-series cross-validation (5 folds) for hyperparameter selection via random search
- Threshold tuning on validation set to maximise F1
- Final evaluation on held-out test set with classification report and confusion matrix

### 3. `LSTM.ipynb` — LSTM
Sequence model using a sliding window of past hours:
- LSTM captures temporal dependencies without manual lag features
- Same random search + cross-validation structure as MLP
- F2 score used for threshold tuning (emphasises recall over precision)
- Windows-safe (`num_workers=0`)

### 4. `CNN.ipynb` — 1D CNN
Convolutional model over sliding windows:
- `CNN1D` architecture with `AdaptiveMaxPool1d` — lookback-length independent
- `kernel_size` values of 7 and 12 capture multi-hour ramp patterns
- CosineAnnealingLR scheduler with early stopping on validation F1
- Threshold calibration on validation set before final test evaluation

---

## Requirements

```
numpy
pandas
torch
matplotlib
scikit-learn
seaborn
joblib
```

Install with:

```bash
pip install -r requirements.txt
```

---

## How to Run

1. Ensure `Data/CSVs/aeso_merged_2020_2025.csv` exists. If not, run the pipeline:
   ```bash
   python Data/aeso_merge_pipeline.py
   ```
2. Open any notebook in `FINAL IPYNBS/` using Jupyter and run all cells top to bottom.
