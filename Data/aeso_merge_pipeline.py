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
OUTPUT_CSV     = os.path.join(SCRIPT_DIR, 'aeso_merged_2020_2025.csv')

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
    'Date_Begin_Local',
    'ACTUAL_POOL_PRICE',
    'ACTUAL_AIL',
    'IMPORT_BC', 'IMPORT_MT', 'IMPORT_SK',
    'EXPORT_BC', 'EXPORT_MT', 'EXPORT_SK',
]

df_price = pd.read_csv(
    POOL_PRICE_CSV,
    usecols=PRICE_COLS,
    parse_dates=['Date_Begin_Local'],
)
df_price.rename(columns={'Date_Begin_Local': 'datetime'}, inplace=True)
df_price['datetime'] = df_price['datetime'].dt.floor('h')
print(f"  Pool price rows: {len(df_price):,}")

# ── 2. Load & Aggregate Dataset 2: CSD Generation (all fuel types) ─────────
print("\nLoading CSD generation files...")
csd_files = sorted(glob.glob(CSD_PATTERN))
print(f"  Found {len(csd_files)} CSD files")

chunks = []
for fpath in csd_files:
    df = pd.read_csv(
        fpath,
        usecols=['Date (MPT)', 'Fuel Type', 'Volume', 'System Capability'],
        parse_dates=['Date (MPT)'],
    )
    df = df[df['Fuel Type'].isin(FUEL_TYPE_COLS)]
    chunks.append(df)
    print(f"  {os.path.basename(fpath):50s}  rows: {len(df):>8,}")

df_csd_raw = pd.concat(chunks, ignore_index=True)
print(f"\n  Total rows (all fuel types): {len(df_csd_raw):,}")

# Rename, floor to hour
df_csd_raw.rename(columns={'Date (MPT)': 'datetime'}, inplace=True)
df_csd_raw['datetime'] = df_csd_raw['datetime'].dt.floor('h')

# Single groupby pass — aggregate Volume and System Capability together
df_agg = (
    df_csd_raw
    .groupby(['datetime', 'Fuel Type'])[['Volume', 'System Capability']]
    .sum()
    .unstack('Fuel Type')
)

# Split the two metric levels, rename columns, then join into one flat DataFrame
vol_df = df_agg['Volume'].rename(columns=FUEL_TYPE_COLS)
cap_df = df_agg['System Capability'].rename(columns=FUEL_CAP_COLS)
df_csd = vol_df.join(cap_df).reset_index()

# Ensure every expected column exists (fills 0 for types absent in early years)
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
df_merged = df_merged[FINAL_COLS]

# ── 4. Validation ──────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("VALIDATION REPORT")
print("=" * 50)

print(f"\n[1] Shape: {df_merged.shape}")
print(f"    Expected ~47,000 rows — within range: {40_000 <= df_merged.shape[0] <= 55_000}")

print(f"\n[2] Date range:")
print(f"    Min datetime: {df_merged['datetime'].min()}")
print(f"    Max datetime: {df_merged['datetime'].max()}")

null_counts = df_merged.isnull().sum()
print(f"\n[3] Null values per column:")
print(null_counts.to_string())
print(f"    All zeros: {(null_counts == 0).all()}")

high_price_count = (df_merged['ACTUAL_POOL_PRICE'] > 200).sum()
print(f"\n[4] Hours with ACTUAL_POOL_PRICE > $200: {high_price_count:,}")
print(f"    Expected 2,000–4,000: {2_000 <= high_price_count <= 4_000}")

print("\n" + "=" * 50)

# ── 5. Save ────────────────────────────────────────────────────────────────
df_merged.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved: {OUTPUT_CSV}")
print(f"File size: {os.path.getsize(OUTPUT_CSV) / 1e6:.1f} MB")