# Data Dictionary

| Column | Type | Units | Definition | Transformation | Source |
|---|---|---|---|---|---|
| ACTUAL_POOL_PRICE | float | CAD/MWh | Actual pool price at time t | Original or directly derived from AESO merge pipeline | AESO merged hourly dataset |
| ACTUAL_AIL | float | MW | Alberta Internal Load at time t | Original or directly derived from AESO merge pipeline | AESO merged hourly dataset |
| coal_total | float | MW or ratio | Current system feature: coal_total | Inherited from AESO merge pipeline | AESO merged hourly dataset |
| dual_fuel_total | float | MW or ratio | Current system feature: dual_fuel_total | Inherited from AESO merge pipeline | AESO merged hourly dataset |
| energy_storage_total | float | MW or ratio | Current system feature: energy_storage_total | Inherited from AESO merge pipeline | AESO merged hourly dataset |
| gas_total | float | MW or ratio | Current system feature: gas_total | Inherited from AESO merge pipeline | AESO merged hourly dataset |
| hydro_total | float | MW or ratio | Current system feature: hydro_total | Inherited from AESO merge pipeline | AESO merged hourly dataset |
| other_total | float | MW or ratio | Current system feature: other_total | Inherited from AESO merge pipeline | AESO merged hourly dataset |
| solar_total | float | MW or ratio | Current system feature: solar_total | Inherited from AESO merge pipeline | AESO merged hourly dataset |
| wind_total | float | MW or ratio | Current system feature: wind_total | Inherited from AESO merge pipeline | AESO merged hourly dataset |
| coal_system_capability | float | MW or ratio | Current system feature: coal_system_capability | Inherited from AESO merge pipeline | AESO merged hourly dataset |
| dual_fuel_system_capability | float | MW or ratio | Current system feature: dual_fuel_system_capability | Inherited from AESO merge pipeline | AESO merged hourly dataset |
| energy_storage_system_capability | float | MW or ratio | Current system feature: energy_storage_system_capability | Inherited from AESO merge pipeline | AESO merged hourly dataset |
| gas_system_capability | float | MW or ratio | Current system feature: gas_system_capability | Inherited from AESO merge pipeline | AESO merged hourly dataset |
| hydro_system_capability | float | MW or ratio | Current system feature: hydro_system_capability | Inherited from AESO merge pipeline | AESO merged hourly dataset |
| other_system_capability | float | MW or ratio | Current system feature: other_system_capability | Inherited from AESO merge pipeline | AESO merged hourly dataset |
| solar_system_capability | float | MW or ratio | Current system feature: solar_system_capability | Inherited from AESO merge pipeline | AESO merged hourly dataset |
| wind_system_capability | float | MW or ratio | Current system feature: wind_system_capability | Inherited from AESO merge pipeline | AESO merged hourly dataset |
| IMPORT_BC | float | MW or ratio | Current system feature: IMPORT_BC | Inherited from AESO merge pipeline | AESO merged hourly dataset |
| IMPORT_MT | float | MW or ratio | Current system feature: IMPORT_MT | Inherited from AESO merge pipeline | AESO merged hourly dataset |
| IMPORT_SK | float | MW or ratio | Current system feature: IMPORT_SK | Inherited from AESO merge pipeline | AESO merged hourly dataset |
| EXPORT_BC | float | MW or ratio | Current system feature: EXPORT_BC | Inherited from AESO merge pipeline | AESO merged hourly dataset |
| EXPORT_MT | float | MW or ratio | Current system feature: EXPORT_MT | Inherited from AESO merge pipeline | AESO merged hourly dataset |
| EXPORT_SK | float | MW or ratio | Current system feature: EXPORT_SK | Inherited from AESO merge pipeline | AESO merged hourly dataset |
| net_export | float | MW | Exports minus imports across interties at time t | Original or directly derived from AESO merge pipeline | AESO merged hourly dataset |
| total_generation | float | MW or ratio | Current system feature: total_generation | Inherited from AESO merge pipeline | AESO merged hourly dataset |
| total_system_capability | float | MW or ratio | Current system feature: total_system_capability | Inherited from AESO merge pipeline | AESO merged hourly dataset |
| dispatchable_generation | float | MW or ratio | Current system feature: dispatchable_generation | Inherited from AESO merge pipeline | AESO merged hourly dataset |
| dispatchable_capability | float | MW or ratio | Current system feature: dispatchable_capability | Inherited from AESO merge pipeline | AESO merged hourly dataset |
| renewable_generation | float | MW | Wind plus solar generation at time t | Original or directly derived from AESO merge pipeline | AESO merged hourly dataset |
| renewable_capability | float | MW or ratio | Current system feature: renewable_capability | Inherited from AESO merge pipeline | AESO merged hourly dataset |
| renewables_share | float | MW or ratio | Current system feature: renewables_share | Inherited from AESO merge pipeline | AESO merged hourly dataset |
| dispatchable_ratio | float | MW or ratio | Current system feature: dispatchable_ratio | Inherited from AESO merge pipeline | AESO merged hourly dataset |
| gas_ratio | float | MW or ratio | Current system feature: gas_ratio | Inherited from AESO merge pipeline | AESO merged hourly dataset |
| intertie_support | float | MW or ratio | Current system feature: intertie_support | Inherited from AESO merge pipeline | AESO merged hourly dataset |
| capacity_renewable | float | MW or ratio | Current system feature: capacity_renewable | Inherited from AESO merge pipeline | AESO merged hourly dataset |
| capacity_dispatchable | float | MW or ratio | Current system feature: capacity_dispatchable | Inherited from AESO merge pipeline | AESO merged hourly dataset |
| capacity_gas | float | MW or ratio | Current system feature: capacity_gas | Inherited from AESO merge pipeline | AESO merged hourly dataset |
| net_load | float | MW or ratio | Current system feature: net_load | Inherited from AESO merge pipeline | AESO merged hourly dataset |
| net_load_3h_change | float | MW or ratio | Current system feature: net_load_3h_change | Inherited from AESO merge pipeline | AESO merged hourly dataset |
| resilience_buffer | float | MW | Total system capability minus ACTUAL_AIL at time t | Original or directly derived from AESO merge pipeline | AESO merged hourly dataset |
| reserve_margin | float | ratio | Resilience buffer divided by ACTUAL_AIL at time t | Original or directly derived from AESO merge pipeline | AESO merged hourly dataset |
| ACTUAL_POOL_PRICE_lag_1h | float | same as source variable | Lagged feature: ACTUAL_POOL_PRICE_lag_1h | Backward time shift | Derived from AESO merged hourly dataset |
| ACTUAL_POOL_PRICE_lag_6h | float | same as source variable | Lagged feature: ACTUAL_POOL_PRICE_lag_6h | Backward time shift | Derived from AESO merged hourly dataset |
| ACTUAL_POOL_PRICE_lag_24h | float | same as source variable | Lagged feature: ACTUAL_POOL_PRICE_lag_24h | Backward time shift | Derived from AESO merged hourly dataset |
| ACTUAL_AIL_lag_1h | float | same as source variable | Lagged feature: ACTUAL_AIL_lag_1h | Backward time shift | Derived from AESO merged hourly dataset |
| ACTUAL_AIL_lag_6h | float | same as source variable | Lagged feature: ACTUAL_AIL_lag_6h | Backward time shift | Derived from AESO merged hourly dataset |
| ACTUAL_AIL_lag_24h | float | same as source variable | Lagged feature: ACTUAL_AIL_lag_24h | Backward time shift | Derived from AESO merged hourly dataset |
| wind_total_lag_1h | float | same as source variable | Lagged feature: wind_total_lag_1h | Backward time shift | Derived from AESO merged hourly dataset |
| wind_total_lag_6h | float | same as source variable | Lagged feature: wind_total_lag_6h | Backward time shift | Derived from AESO merged hourly dataset |
| wind_total_lag_24h | float | same as source variable | Lagged feature: wind_total_lag_24h | Backward time shift | Derived from AESO merged hourly dataset |
| solar_total_lag_1h | float | same as source variable | Lagged feature: solar_total_lag_1h | Backward time shift | Derived from AESO merged hourly dataset |
| solar_total_lag_6h | float | same as source variable | Lagged feature: solar_total_lag_6h | Backward time shift | Derived from AESO merged hourly dataset |
| solar_total_lag_24h | float | same as source variable | Lagged feature: solar_total_lag_24h | Backward time shift | Derived from AESO merged hourly dataset |
| gas_total_lag_1h | float | same as source variable | Lagged feature: gas_total_lag_1h | Backward time shift | Derived from AESO merged hourly dataset |
| gas_total_lag_6h | float | same as source variable | Lagged feature: gas_total_lag_6h | Backward time shift | Derived from AESO merged hourly dataset |
| gas_total_lag_24h | float | same as source variable | Lagged feature: gas_total_lag_24h | Backward time shift | Derived from AESO merged hourly dataset |
| renewable_generation_lag_1h | float | same as source variable | Lagged feature: renewable_generation_lag_1h | Backward time shift | Derived from AESO merged hourly dataset |
| renewable_generation_lag_6h | float | same as source variable | Lagged feature: renewable_generation_lag_6h | Backward time shift | Derived from AESO merged hourly dataset |
| renewable_generation_lag_24h | float | same as source variable | Lagged feature: renewable_generation_lag_24h | Backward time shift | Derived from AESO merged hourly dataset |
| net_load_lag_1h | float | same as source variable | Lagged feature: net_load_lag_1h | Backward time shift | Derived from AESO merged hourly dataset |
| net_load_lag_6h | float | same as source variable | Lagged feature: net_load_lag_6h | Backward time shift | Derived from AESO merged hourly dataset |
| net_load_lag_24h | float | same as source variable | Lagged feature: net_load_lag_24h | Backward time shift | Derived from AESO merged hourly dataset |
| reserve_margin_lag_1h | float | same as source variable | Lagged feature: reserve_margin_lag_1h | Backward time shift | Derived from AESO merged hourly dataset |
| reserve_margin_lag_6h | float | same as source variable | Lagged feature: reserve_margin_lag_6h | Backward time shift | Derived from AESO merged hourly dataset |
| reserve_margin_lag_24h | float | same as source variable | Lagged feature: reserve_margin_lag_24h | Backward time shift | Derived from AESO merged hourly dataset |
| ACTUAL_AIL_change_1h | float | same as source variable | Change feature: ACTUAL_AIL_change_1h | Difference between time t and time t-k | Derived from AESO merged hourly dataset |
| ACTUAL_AIL_change_24h | float | same as source variable | Change feature: ACTUAL_AIL_change_24h | Difference between time t and time t-k | Derived from AESO merged hourly dataset |
| wind_total_change_1h | float | same as source variable | Change feature: wind_total_change_1h | Difference between time t and time t-k | Derived from AESO merged hourly dataset |
| wind_total_change_24h | float | same as source variable | Change feature: wind_total_change_24h | Difference between time t and time t-k | Derived from AESO merged hourly dataset |
| solar_total_change_1h | float | same as source variable | Change feature: solar_total_change_1h | Difference between time t and time t-k | Derived from AESO merged hourly dataset |
| solar_total_change_24h | float | same as source variable | Change feature: solar_total_change_24h | Difference between time t and time t-k | Derived from AESO merged hourly dataset |
| gas_total_change_1h | float | same as source variable | Change feature: gas_total_change_1h | Difference between time t and time t-k | Derived from AESO merged hourly dataset |
| gas_total_change_24h | float | same as source variable | Change feature: gas_total_change_24h | Difference between time t and time t-k | Derived from AESO merged hourly dataset |
| net_load_change_1h | float | same as source variable | Change feature: net_load_change_1h | Difference between time t and time t-k | Derived from AESO merged hourly dataset |
| net_load_change_24h | float | same as source variable | Change feature: net_load_change_24h | Difference between time t and time t-k | Derived from AESO merged hourly dataset |
| reserve_margin_change_1h | float | same as source variable | Change feature: reserve_margin_change_1h | Difference between time t and time t-k | Derived from AESO merged hourly dataset |
| reserve_margin_change_24h | float | same as source variable | Change feature: reserve_margin_change_24h | Difference between time t and time t-k | Derived from AESO merged hourly dataset |
| sin_day | float | unitless | Cyclical seasonal encoding: sin_day | Sine/cosine transform of calendar position | Derived from datetime |
| cos_day | float | unitless | Cyclical seasonal encoding: cos_day | Sine/cosine transform of calendar position | Derived from datetime |
| sin_week | float | unitless | Cyclical seasonal encoding: sin_week | Sine/cosine transform of calendar position | Derived from datetime |
| cos_week | float | unitless | Cyclical seasonal encoding: cos_week | Sine/cosine transform of calendar position | Derived from datetime |
| sin_year_1 | float | unitless | Cyclical seasonal encoding: sin_year_1 | Sine/cosine transform of calendar position | Derived from datetime |
| cos_year_1 | float | unitless | Cyclical seasonal encoding: cos_year_1 | Sine/cosine transform of calendar position | Derived from datetime |
| sin_year_2 | float | unitless | Cyclical seasonal encoding: sin_year_2 | Sine/cosine transform of calendar position | Derived from datetime |
| cos_year_2 | float | unitless | Cyclical seasonal encoding: cos_year_2 | Sine/cosine transform of calendar position | Derived from datetime |
| is_weekend | binary | 0/1 | Calendar/event indicator: is_weekend | Existing binary flag | Derived from datetime and event ranges |
| is_stampede | binary | 0/1 | Calendar/event indicator: is_stampede | Existing binary flag | Derived from datetime and event ranges |
| dow_0 | binary | 0/1 | Dummy variable for day-of-week == 0 | One-hot encoding | Derived from datetime |
| dow_1 | binary | 0/1 | Dummy variable for day-of-week == 1 | One-hot encoding | Derived from datetime |
| dow_2 | binary | 0/1 | Dummy variable for day-of-week == 2 | One-hot encoding | Derived from datetime |
| dow_3 | binary | 0/1 | Dummy variable for day-of-week == 3 | One-hot encoding | Derived from datetime |
| dow_4 | binary | 0/1 | Dummy variable for day-of-week == 4 | One-hot encoding | Derived from datetime |
| dow_5 | binary | 0/1 | Dummy variable for day-of-week == 5 | One-hot encoding | Derived from datetime |
| dow_6 | binary | 0/1 | Dummy variable for day-of-week == 6 | One-hot encoding | Derived from datetime |
| hour_0 | binary | 0/1 | Dummy variable for hour-of-day == 0 | One-hot encoding | Derived from datetime |
| hour_1 | binary | 0/1 | Dummy variable for hour-of-day == 1 | One-hot encoding | Derived from datetime |
| hour_10 | binary | 0/1 | Dummy variable for hour-of-day == 10 | One-hot encoding | Derived from datetime |
| hour_11 | binary | 0/1 | Dummy variable for hour-of-day == 11 | One-hot encoding | Derived from datetime |
| hour_12 | binary | 0/1 | Dummy variable for hour-of-day == 12 | One-hot encoding | Derived from datetime |
| hour_13 | binary | 0/1 | Dummy variable for hour-of-day == 13 | One-hot encoding | Derived from datetime |
| hour_14 | binary | 0/1 | Dummy variable for hour-of-day == 14 | One-hot encoding | Derived from datetime |
| hour_15 | binary | 0/1 | Dummy variable for hour-of-day == 15 | One-hot encoding | Derived from datetime |
| hour_16 | binary | 0/1 | Dummy variable for hour-of-day == 16 | One-hot encoding | Derived from datetime |
| hour_17 | binary | 0/1 | Dummy variable for hour-of-day == 17 | One-hot encoding | Derived from datetime |
| hour_18 | binary | 0/1 | Dummy variable for hour-of-day == 18 | One-hot encoding | Derived from datetime |
| hour_19 | binary | 0/1 | Dummy variable for hour-of-day == 19 | One-hot encoding | Derived from datetime |
| hour_2 | binary | 0/1 | Dummy variable for hour-of-day == 2 | One-hot encoding | Derived from datetime |
| hour_20 | binary | 0/1 | Dummy variable for hour-of-day == 20 | One-hot encoding | Derived from datetime |
| hour_21 | binary | 0/1 | Dummy variable for hour-of-day == 21 | One-hot encoding | Derived from datetime |
| hour_22 | binary | 0/1 | Dummy variable for hour-of-day == 22 | One-hot encoding | Derived from datetime |
| hour_23 | binary | 0/1 | Dummy variable for hour-of-day == 23 | One-hot encoding | Derived from datetime |
| hour_3 | binary | 0/1 | Dummy variable for hour-of-day == 3 | One-hot encoding | Derived from datetime |
| hour_4 | binary | 0/1 | Dummy variable for hour-of-day == 4 | One-hot encoding | Derived from datetime |
| hour_5 | binary | 0/1 | Dummy variable for hour-of-day == 5 | One-hot encoding | Derived from datetime |
| hour_6 | binary | 0/1 | Dummy variable for hour-of-day == 6 | One-hot encoding | Derived from datetime |
| hour_7 | binary | 0/1 | Dummy variable for hour-of-day == 7 | One-hot encoding | Derived from datetime |
| hour_8 | binary | 0/1 | Dummy variable for hour-of-day == 8 | One-hot encoding | Derived from datetime |
| hour_9 | binary | 0/1 | Dummy variable for hour-of-day == 9 | One-hot encoding | Derived from datetime |
| hour_of_day | binary | 0/1 | Dummy variable for hour-of-day == of_day | One-hot encoding | Derived from datetime |
| month_1 | binary | 0/1 | Dummy variable for month == 1 | One-hot encoding | Derived from datetime |
| month_10 | binary | 0/1 | Dummy variable for month == 10 | One-hot encoding | Derived from datetime |
| month_11 | binary | 0/1 | Dummy variable for month == 11 | One-hot encoding | Derived from datetime |
| month_12 | binary | 0/1 | Dummy variable for month == 12 | One-hot encoding | Derived from datetime |
| month_2 | binary | 0/1 | Dummy variable for month == 2 | One-hot encoding | Derived from datetime |
| month_3 | binary | 0/1 | Dummy variable for month == 3 | One-hot encoding | Derived from datetime |
| month_4 | binary | 0/1 | Dummy variable for month == 4 | One-hot encoding | Derived from datetime |
| month_5 | binary | 0/1 | Dummy variable for month == 5 | One-hot encoding | Derived from datetime |
| month_6 | binary | 0/1 | Dummy variable for month == 6 | One-hot encoding | Derived from datetime |
| month_7 | binary | 0/1 | Dummy variable for month == 7 | One-hot encoding | Derived from datetime |
| month_8 | binary | 0/1 | Dummy variable for month == 8 | One-hot encoding | Derived from datetime |
| month_9 | binary | 0/1 | Dummy variable for month == 9 | One-hot encoding | Derived from datetime |
| spike_lead_2 | binary | 0/1 | Target indicator for pool_price_lead_2 > 200.0 CAD/MWh | Original or directly derived from AESO merge pipeline | AESO merged hourly dataset |
| pool_price_lead_2 | float | CAD/MWh | Actual pool price at time t+2; used for evaluation only | Original or directly derived from AESO merge pipeline | AESO merged hourly dataset |
| datetime | timestamp | MPT (UTC-7) | Operating hour aligned to Mountain Prevailing Time | Original or directly derived from AESO merge pipeline | AESO merged hourly dataset |
| split | category | label | Fixed split assignment | Rule-based assignment from datetime cutoffs | Project preprocessing |
