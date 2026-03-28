"""
AESO Data Merge Pipeline
Merges pool-price/demand data with CSD hourly generation (all fuel types).
Outputs: aeso_merged_2020_2025.csv
"""

import pandas as pd
import glob
import os

# ── Paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR   = os.path.join(SCRIPT_DIR, 'CSVs')
POOL_PRICE_CSV = os.path.join(DATASETS_DIR, 'Hourly_Metered_Volumes_and_Pool_Price_and_AIL_2020-Jul2025.csv')
CSD_PATTERN    = os.path.join(DATASETS_DIR, 'CSD Generation (Hourly)*.csv')
OUTPUT_CSV     = os.path.join(DATASETS_DIR, 'aeso_merged_2020_2025.csv')

# Valid fuel types and their output column names
FUEL_TYPE_COLS = {
    'COAL':           'coal_total',
    'DUAL FUEL':      'dual_fuel_total',
    'ENERGY STORAGE': 'energy_storage_total',
    'GAS':            'gas_total',
    'HYDRO':          'hydro_total',
    'OTHER':          'other_total',
    'SOLAR':          'solar_total',
    'WIND':           'wind_total',
}
# System capability column names — same keys, derived from volume names
FUEL_CAP_COLS = {ft: name.replace('_total', '_system_capability') for ft, name in FUEL_TYPE_COLS.items()}

# ── 1. Load Dataset 1: Pool Price & Demand ─────────────────────────────────
print("Loading pool price & demand data...")
PRICE_COLS = [
    'Date_Begin_GMT',
    'ACTUAL_POOL_PRICE',
    'ACTUAL_AIL',
    'IMPORT_BC', 'IMPORT_MT', 'IMPORT_SK',
    'EXPORT_BC', 'EXPORT_MT', 'EXPORT_SK',
]

df_price = pd.read_csv(
    POOL_PRICE_CSV,
    usecols=PRICE_COLS,
    parse_dates=['Date_Begin_GMT'],
)
# Convert GMT → MST (constant UTC-7) to get an unambiguous, DST-free datetime axis.
# Date_Begin_Local (MPT) has 23-hour spring-forward days and 25-hour fall-back days,
# which cause duplicate/missing keys when merging. GMT - 7h is always unique and complete.
df_price.rename(columns={'Date_Begin_GMT': 'datetime'}, inplace=True)
df_price['datetime'] = df_price['datetime'] - pd.Timedelta(hours=7)
print(f"  Pool price rows: {len(df_price):,}")

# ── 2. Load & Aggregate Dataset 2: CSD Generation (all fuel types) ─────────
print("\nLoading CSD generation files...")
csd_files = sorted(glob.glob(CSD_PATTERN))
print(f"  Found {len(csd_files)} CSD files")

chunks = []
for fpath in csd_files:
    df = pd.read_csv(
        fpath,
        usecols=['Date (MST)', 'Fuel Type', 'Volume', 'System Capability'],
        parse_dates=['Date (MST)'],
    )
    df = df[df['Fuel Type'].isin(FUEL_TYPE_COLS)]
    chunks.append(df)
    print(f"  {os.path.basename(fpath):50s}  rows: {len(df):>8,}")

df_csd_raw = pd.concat(chunks, ignore_index=True)
print(f"\n  Total rows (all fuel types): {len(df_csd_raw):,}")

# Rename, floor to hour
df_csd_raw.rename(columns={'Date (MST)': 'datetime'}, inplace=True)
df_csd_raw['datetime'] = df_csd_raw['datetime'].dt.floor('h')

# Single groupby pass — aggregate Volume and System Capability together
df_agg = (
    df_csd_raw
    .groupby(['datetime', 'Fuel Type'])[['Volume', 'System Capability']]
    .sum()
    .unstack('Fuel Type')
)

# Split the two metric levels, rename columns, fill NaN → 0, then join into one flat DataFrame
# NaN appears after unstack for hours where a fuel type had no generating assets reporting
vol_df = df_agg['Volume'].rename(columns=FUEL_TYPE_COLS).fillna(0.0)
cap_df = df_agg['System Capability'].rename(columns=FUEL_CAP_COLS).fillna(0.0)
df_csd = vol_df.join(cap_df).reset_index()

# Ensure every expected column exists (for fuel types completely absent across all files)
for col in list(FUEL_TYPE_COLS.values()) + list(FUEL_CAP_COLS.values()):
    if col not in df_csd.columns:
        df_csd[col] = 0.0

# Clip solar volume values below 0.5 MW to 0 (nighttime SCADA artifact)
df_csd['solar_total'] = df_csd['solar_total'].clip(lower=0)
df_csd.loc[df_csd['solar_total'] < 0.5, 'solar_total'] = 0.0

print(f"\n  CSD hourly rows after aggregation: {len(df_csd):,}")

# ── 3. Merge ───────────────────────────────────────────────────────────────
print("\nMerging datasets (inner join on datetime)...")
df_merged = df_price.merge(df_csd, on='datetime', how='inner')

FINAL_COLS = [
    'datetime',
    'ACTUAL_POOL_PRICE',
    'ACTUAL_AIL',
    *sorted(FUEL_TYPE_COLS.values()),
    *sorted(FUEL_CAP_COLS.values()),
    'IMPORT_BC', 'IMPORT_MT', 'IMPORT_SK',
    'EXPORT_BC', 'EXPORT_MT', 'EXPORT_SK',
]
df = df_merged[FINAL_COLS].copy()

# ── 4. Feature Engineering ─────────────────────────────────────────────────
print("\nEngineering features...")

SPIKE_THRESHOLD = 200  # CAD/MWh

# 1. Target variable and spike flags
df["spike"] = (df["ACTUAL_POOL_PRICE"] > SPIKE_THRESHOLD).astype(int)

# Lead targets: spike at t+N — target variables for multi-hour-ahead models, never model inputs
for n in range(1, 25):
    df[f"spike_lead_{n}"] = df["spike"].shift(-n).astype("Int64")

# Lag features: was there a spike N hours ago — legitimate model inputs
for n in range(1, 25):
    df[f"spike_lag_{n}"] = df["spike"].shift(n).astype("Int64")

# 2. Net intertie flow
df["net_export"] = (
    (df["EXPORT_BC"] + df["EXPORT_MT"] + df["EXPORT_SK"])
    - (df["IMPORT_BC"] + df["IMPORT_MT"] + df["IMPORT_SK"])
)

# 3. Intermediate totals
df["total_generation"] = (
    df["coal_total"] + df["dual_fuel_total"] + df["energy_storage_total"]
    + df["gas_total"] + df["hydro_total"] + df["other_total"]
    + df["solar_total"] + df["wind_total"]
)
df["total_system_capability"] = (
    df["coal_system_capability"] + df["dual_fuel_system_capability"]
    + df["energy_storage_system_capability"] + df["gas_system_capability"]
    + df["hydro_system_capability"] + df["other_system_capability"]
    + df["solar_system_capability"] + df["wind_system_capability"]
)
df["dispatchable_generation"] = (
    df["coal_total"] + df["dual_fuel_total"] + df["gas_total"]
    + df["hydro_total"] + df["other_total"]
    + df["IMPORT_BC"] + df["IMPORT_MT"] + df["IMPORT_SK"]
)
df["dispatchable_capability"] = (
    df["coal_system_capability"] + df["dual_fuel_system_capability"]
    + df["gas_system_capability"] + df["hydro_system_capability"]
    + df["other_system_capability"]
)
df["renewable_generation"] = df["wind_total"] + df["solar_total"]
df["renewable_capability"]  = df["wind_system_capability"] + df["solar_system_capability"]

# 4. Generation mix — share of total generation per fuel type
for fuel in ["coal", "dual_fuel", "energy_storage", "gas", "hydro", "other", "solar", "wind"]:
    df[f"{fuel}_mix"] = df[f"{fuel}_total"] / df["total_generation"]

# 5. Demand-based ratios
df["renewables_share"]   = df["renewable_generation"] / df["ACTUAL_AIL"]
df["dispatchable_ratio"] = df["dispatchable_generation"] / df["ACTUAL_AIL"]
df["gas_ratio"]          = df["gas_total"] / df["ACTUAL_AIL"]
df["intertie_support"]   = (df["IMPORT_BC"] + df["IMPORT_MT"] + df["IMPORT_SK"]) / df["ACTUAL_AIL"]

# 6. Capacity utilisation ratios
df["capacity_renewable"]    = df["renewable_generation"] / df["total_system_capability"]
df["capacity_dispatchable"] = df["dispatchable_generation"] / df["total_system_capability"]
df["capacity_gas"]          = df["gas_total"] / df["gas_system_capability"].replace(0, float("nan"))

# 7. Net load and rolling change
df["net_load"]           = df["ACTUAL_AIL"] - df["renewable_generation"]
df["net_load_3h_change"] = df["net_load"].diff(3)

# 8. Reserve margin and resilience buffer
df["resilience_buffer"] = df["total_system_capability"] - df["ACTUAL_AIL"]
df["reserve_margin"]    = df["resilience_buffer"] / df["ACTUAL_AIL"]

# 9. Price lag features — shift ensures only past values are used, no leakage
for n in [1, 6, 24]:
    df[f"price_lag_{n}h"] = df["ACTUAL_POOL_PRICE"].shift(n)

df["price_rolling_mean_6h"] = df["ACTUAL_POOL_PRICE"].shift(1).rolling(6).mean()

# 10. Seasonality and time features
df["datetime"]    = pd.to_datetime(df["datetime"])
df["hour_of_day"] = df["datetime"].dt.hour
df["day_of_week"] = df["datetime"].dt.dayofweek   # 0=Monday, 6=Sunday
df["month"]       = df["datetime"].dt.month
df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)

# Drop edge rows where lead/lag windows are incomplete
df = df.dropna(subset=["spike_lead_24", "spike_lag_24"])
print(f"  Rows after feature engineering + edge trim: {len(df):,}")

# ── 5. Validation ──────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("VALIDATION REPORT")
print("=" * 50)

print(f"\n[1] Shape: {df.shape}")
print(f"    Expected ~47,000 rows — within range: {40_000 <= df.shape[0] <= 55_000}")

print(f"\n[2] Date range:")
print(f"    Min datetime: {df['datetime'].min()}")
print(f"    Max datetime: {df['datetime'].max()}")

# Only check nulls on base columns — ratio/mix columns may have structural NaNs (e.g. div-by-zero)
base_cols = ['datetime', 'ACTUAL_POOL_PRICE', 'ACTUAL_AIL', 'spike',
             'total_generation', 'total_system_capability', 'net_load']
null_counts = df[base_cols].isnull().sum()
print(f"\n[3] Null values on base columns:")
print(null_counts.to_string())
print(f"    All zeros: {(null_counts == 0).all()}")

high_price_count = (df['ACTUAL_POOL_PRICE'] > 200).sum()
print(f"\n[4] Hours with ACTUAL_POOL_PRICE > $200: {high_price_count:,}")
print(f"    Expected 2,000–4,000: {2_000 <= high_price_count <= 4_000}")

print(f"\n[5] Spike rate: {df['spike'].mean():.1%}")
print(f"    Spike counts  0={( df['spike']==0).sum():,}  1={(df['spike']==1).sum():,}")

print("\n" + "=" * 50)

# ── 6. Save ────────────────────────────────────────────────────────────────
df.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved: {OUTPUT_CSV}")
print(f"File size: {os.path.getsize(OUTPUT_CSV) / 1e6:.1f} MB")