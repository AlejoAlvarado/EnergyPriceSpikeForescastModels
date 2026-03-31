from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
JORGE_ROOT = PROJECT_ROOT / "JorgeFolder"
DATA_ROOT = PROJECT_ROOT / "Data"
SOURCE_DATA = DATA_ROOT / "CSVs" / "aeso_merged_2020_2025.csv"

OUTPUT_DATA_DIR = JORGE_ROOT / "outputs" / "data"
OUTPUT_FIGURES_DIR = JORGE_ROOT / "outputs" / "figures"
OUTPUT_METRICS_DIR = JORGE_ROOT / "outputs" / "metrics"
OUTPUT_REPORT_DIR = JORGE_ROOT / "outputs" / "report"

MODELS_ROOT = JORGE_ROOT / "models"
MODEL_DIRS = {
    "mlp": MODELS_ROOT / "mlp",
    "cnn": MODELS_ROOT / "cnn",
    "lstm": MODELS_ROOT / "lstm",
}

SPIKE_THRESHOLD = 200.0
RANDOM_SEED = 607
DEVICE = "cpu"

TRAIN_START = pd.Timestamp("2020-01-01 00:00:00")
TRAIN_END_EXCLUSIVE = pd.Timestamp("2023-11-06 00:00:00")
VAL_START = pd.Timestamp("2023-11-06 00:00:00")
VAL_END_EXCLUSIVE = pd.Timestamp("2024-12-12 00:00:00")
TEST_START = pd.Timestamp("2024-12-12 00:00:00")
TEST_END_INCLUSIVE = pd.Timestamp("2025-07-01 23:00:00")

TS_CV_SPLITS = 5

CURRENT_FEATURES = [
    "ACTUAL_POOL_PRICE",
    "ACTUAL_AIL",
    "coal_total",
    "dual_fuel_total",
    "energy_storage_total",
    "gas_total",
    "hydro_total",
    "other_total",
    "solar_total",
    "wind_total",
    "coal_system_capability",
    "dual_fuel_system_capability",
    "energy_storage_system_capability",
    "gas_system_capability",
    "hydro_system_capability",
    "other_system_capability",
    "solar_system_capability",
    "wind_system_capability",
    "IMPORT_BC",
    "IMPORT_MT",
    "IMPORT_SK",
    "EXPORT_BC",
    "EXPORT_MT",
    "EXPORT_SK",
    "net_export",
    "total_generation",
    "total_system_capability",
    "dispatchable_generation",
    "dispatchable_capability",
    "renewable_generation",
    "renewable_capability",
    "renewables_share",
    "dispatchable_ratio",
    "gas_ratio",
    "intertie_support",
    "capacity_renewable",
    "capacity_dispatchable",
    "capacity_gas",
    "net_load",
    "net_load_3h_change",
    "resilience_buffer",
    "reserve_margin",
]

LAG_FEATURE_SOURCES = [
    "ACTUAL_POOL_PRICE",
    "ACTUAL_AIL",
    "wind_total",
    "solar_total",
    "gas_total",
    "renewable_generation",
    "net_load",
    "reserve_margin",
]
LAG_HOURS = [1, 6, 24]

CHANGE_FEATURE_SOURCES = [
    "ACTUAL_AIL",
    "wind_total",
    "solar_total",
    "gas_total",
    "net_load",
    "reserve_margin",
]
CHANGE_HOURS = [1, 24]

BINARY_FEATURES = ["is_weekend", "is_stampede"]
CYCLICAL_FEATURES = [
    "sin_day",
    "cos_day",
    "sin_week",
    "cos_week",
    "sin_year_1",
    "cos_year_1",
    "sin_year_2",
    "cos_year_2",
]

SEQUENCE_FEATURES = [
    "ACTUAL_POOL_PRICE",
    "ACTUAL_AIL",
    "gas_total",
    "wind_total",
    "solar_total",
    "hydro_total",
    "coal_total",
    "net_export",
    "renewable_generation",
    "total_generation",
    "capacity_dispatchable",
    "capacity_renewable",
    "net_load",
    "resilience_buffer",
    "reserve_margin",
    "IMPORT_BC",
    "IMPORT_MT",
    "IMPORT_SK",
    "EXPORT_BC",
    "EXPORT_MT",
    "EXPORT_SK",
    "ACTUAL_POOL_PRICE_lag_1h",
    "ACTUAL_POOL_PRICE_lag_6h",
    "ACTUAL_POOL_PRICE_lag_24h",
    *CYCLICAL_FEATURES,
    *BINARY_FEATURES,
]

MLP_TUNING_TRIALS = 3
CNN_TUNING_TRIALS = 3
LSTM_TUNING_TRIALS = 3
MAX_EPOCHS = 10
EARLY_STOPPING_PATIENCE = 3

MODEL_DEFAULTS = {
    "mlp": {"trials": MLP_TUNING_TRIALS},
    "cnn": {"trials": CNN_TUNING_TRIALS},
    "lstm": {"trials": LSTM_TUNING_TRIALS},
}
