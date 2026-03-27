# Data Dictionary: Hourly Metered Volumes, Pool Price, and AIL (2020–Jul 2025)

**Source:** Alberta Electric System Operator (AESO)
**Granularity:** Hourly
**Coverage:** January 2020 – July 2025
**Total columns:** ~220+ (2 timestamp + ~210 generator assets + 9 market variables)

---

## Timestamp Columns

| Column | Type | Format | Description |
|---|---|---|---|
| `Date_Begin_GMT` | datetime | `YYYY-MM-DD HH:MM` | Start of the hour in **Greenwich Mean Time (UTC+0)** |
| `Date_Begin_Local` | datetime | `YYYY-MM-DD HH:MM` | Start of the hour in **Mountain Standard Time (UTC−7)**. Note: Alberta does not observe Daylight Saving Time, so this offset is constant year-round. |

---

## Generator / Facility Metered Volume Columns

**Column naming convention:** Each column uses a short alphanumeric code (e.g., `AFG1`, `AKE1`, `GN1`) that identifies a **specific generating unit or asset** registered with the AESO.

| Attribute | Detail |
|---|---|
| **Units** | Megawatts (MW) — hourly average output (equivalent to MWh for that hour) |
| **Values** | Non-negative numeric; `0` means the unit produced no generation; blank/empty cells indicate the asset was not yet registered or active during that period |
| **Coverage** | Columns span roughly columns 3 through ~215. Assets registered later in the dataset will have leading empty/null values |
| **Asset types covered** | Coal, natural gas (combined cycle, peakers, cogeneration), wind, solar, hydro, energy storage, and others |

**Example generator codes and approximate capacity (illustrative, not exhaustive):**

| Code | Approximate Capacity | Fuel Type (general) |
|---|---|---|
| `GN1`, `GN2`, `GN3` | ~400 MW each | Natural Gas |
| `BOW1` | ~40–80 MW | Hydro |
| `AKE1` | ~70 MW | Natural Gas / Cogen |
| Many others | Varies | Various |

> **Note:** A full registry of asset codes and their associated owner, fuel type, and nameplate capacity can be cross-referenced against the AESO's **Generating Units** registry or the companion CSD Generation files in this project.

---

## Market / Pricing Columns

| Column | Type | Units | Description |
|---|---|---|---|
| `ACTUAL_POOL_PRICE` | float | $/MWh | The **settled pool price** for the hour — the single clearing price paid to all dispatched generators in Alberta's energy-only market. Can range from the price floor (typically −$1,000/MWh in some periods) to the price cap ($1,000/MWh or higher for extreme spike hours). **This is the primary target variable for price spike forecasting.** |
| `ACTUAL_AIL` | integer | MW | **Alberta Internal Load** — the total electricity demand (load) consumed within Alberta during the hour. Equals total generation plus net imports minus exports. |
| `HOUR_AHEAD_POOL_PRICE_FORECAST` | float | $/MWh | The AESO's official **hour-ahead price forecast** published before the operating hour begins. Useful as a baseline predictor or a feature. |

---

## Interconnect / Interchange Columns

Alberta has electricity interties with three neighboring jurisdictions. Positive values indicate flow **away from Alberta**; positive imports indicate flow **into Alberta**.

| Column | Type | Units | Description |
|---|---|---|---|
| `EXPORT_BC` | numeric | MW | Power exported to **British Columbia** during the hour |
| `EXPORT_MT` | numeric | MW | Power exported to **Montana (USA)** during the hour |
| `EXPORT_SK` | numeric | MW | Power exported to **Saskatchewan** during the hour |
| `IMPORT_BC` | numeric | MW | Power imported from **British Columbia** during the hour |
| `IMPORT_MT` | numeric | MW | Power imported from **Montana (USA)** during the hour |
| `IMPORT_SK` | numeric | MW | Power imported from **Saskatchewan** during the hour |

> **Net interchange** for each intertie = Import − Export. A positive net import reduces upward price pressure; a negative net (net export) increases it.

---

## Key Relationships and Notes

- **Price spikes** in Alberta's market typically occur when `ACTUAL_AIL` is high relative to available generation capacity, particularly during cold winter evenings or hot summer afternoons.
- Generator columns with all-zero or all-blank values for long stretches indicate units that were offline, mothballed, or not yet commissioned.
- The `HOUR_AHEAD_POOL_PRICE_FORECAST` vs. `ACTUAL_POOL_PRICE` difference is a useful residual feature — large divergences often precede or coincide with spike events.
- Interconnect flows (`IMPORT_*` / `EXPORT_*`) reflect Alberta's limited import capacity (~1,000 MW from BC, ~300 MW from Montana, ~150 MW from Saskatchewan), which constrains how much neighboring markets can dampen a price spike.

---

## Data Quality Notes

| Issue | Description |
|---|---|
| Trailing empty cells in generator columns | Assets added to the market after 2020 will have blank values for earlier rows — treat as 0 MW or filter by asset registration date |
| Commas in numeric values | Values are plain floats; no thousands separators observed |
| GMT vs. Local alignment | Each row's GMT and Local timestamps refer to the **same operating hour**; use Local time for seasonal/diurnal analysis to avoid off-by-one errors across seasons |
