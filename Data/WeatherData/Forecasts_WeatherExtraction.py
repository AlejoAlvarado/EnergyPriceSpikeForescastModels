from __future__ import annotations

import argparse
import math
import os
import tempfile
from pathlib import Path

import openmeteo_requests
import pandas as pd
import requests
import requests_cache
import xarray as xr
from retry_requests import retry

try:
    import cfgrib  # noqa: F401
except ImportError:  # pragma: no cover - exercised only when dependency is missing
    cfgrib = None


"""
Backfill wind forecast data for Pincher Creek, Alberta.

Default behavior:
- Reads the target hourly range from Data/CSVs/aeso_merged_2020_2025.csv
- Writes an Open-Meteo day-ahead hourly forecast file aligned to that AESO range
- Writes a small coverage summary CSV

Optional NOAA behavior:
- `--run-noaa-backfill` downloads NOAA GFS 00Z +24h point forecasts for each
  available archive day and writes a daily CSV
- The AWS GFS archive used here starts on 2021-03-23 for this endpoint, so NOAA
  cannot fill the 2020 portion of the AESO range from this source

Outputs:
- open_meteo_pincher_creek_forecast.csv
- weather_backfill_coverage.csv
- noaa_gfs_pincher_creek_100m_daily_forecast.csv  (only when requested)
"""


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DATA_DIR = SCRIPT_DIR.parent
AESO_MERGED_CSV = PROJECT_DATA_DIR / "CSVs" / "aeso_merged_2020_2025.csv"
OUTPUT_DIR = PROJECT_DATA_DIR / "Wheather Forecast"

CACHE_DIR = SCRIPT_DIR / ".cache"
OPEN_METEO_CACHE_DIR = CACHE_DIR / "open_meteo"
NOAA_TMP_DIR = CACHE_DIR / "noaa_tmp"

LOCATION_NAME = "pincher_creek_ab"
LATITUDE = 49.4895
LONGITUDE = -113.9458
NOAA_LONGITUDE_360 = LONGITUDE % 360

FORECAST_HORIZON_HOURS = 24
OPEN_METEO_MODEL = "gfs_seamless"
OPEN_METEO_HOURLY_VARIABLE = "wind_speed_120m"
NOAA_CYCLE = "00"
NOAA_FORECAST_HOUR = 24
NOAA_ARCHIVE_AVAILABLE_FROM = pd.Timestamp("2021-03-23")

OPEN_METEO_OUTPUT_CSV = OUTPUT_DIR / "open_meteo_pincher_creek_forecast.csv"
NOAA_OUTPUT_CSV = OUTPUT_DIR / "noaa_gfs_pincher_creek_100m_daily_forecast.csv"
COVERAGE_OUTPUT_CSV = OUTPUT_DIR / "weather_backfill_coverage.csv"


def setup_directory() -> Path:
    """Ensure the output and cache directories exist."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OPEN_METEO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    NOAA_TMP_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def load_target_datetimes(aeso_csv: Path = AESO_MERGED_CSV) -> pd.DatetimeIndex:
    """Load the exact hourly timestamps used by the merged AESO dataset."""
    df = pd.read_csv(aeso_csv, usecols=["datetime"])
    datetimes = pd.to_datetime(df["datetime"]).sort_values().drop_duplicates()
    if datetimes.empty:
        raise RuntimeError(f"No datetime values were found in {aeso_csv}")
    return pd.DatetimeIndex(datetimes)


def build_open_meteo_client() -> openmeteo_requests.Client:
    """Create an Open-Meteo client with local caching and retries."""
    cache_session = requests_cache.CachedSession(str(OPEN_METEO_CACHE_DIR), expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    return openmeteo_requests.Client(session=retry_session)


def fetch_open_meteo_backfill(output_dir: Path, target_datetimes: pd.DatetimeIndex) -> tuple[Path, pd.DataFrame]:
    """
    Retrieve a day-ahead hourly Open-Meteo forecast and align it to AESO timestamps.

    Open-Meteo returns the requested local wall-clock axis as a fixed 24-hour/day
    series, which matches the MST-style hourly axis used by the project data.
    """
    print("Fetching Open-Meteo day-ahead backfill...")

    openmeteo = build_open_meteo_client()
    start_date = target_datetimes.min().date().isoformat()
    end_date = target_datetimes.max().date().isoformat()

    response = openmeteo.weather_api(
        "https://previous-runs-api.open-meteo.com/v1/forecast",
        params={
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": OPEN_METEO_HOURLY_VARIABLE,
            "previous_days": 1,
            "models": OPEN_METEO_MODEL,
            "timezone": "America/Edmonton",
        },
    )[0]

    hourly = response.Hourly()
    utc_offset_seconds = response.UtcOffsetSeconds()
    fetched_datetimes = pd.date_range(
        start=pd.to_datetime(hourly.Time() + utc_offset_seconds, unit="s"),
        end=pd.to_datetime(hourly.TimeEnd() + utc_offset_seconds, unit="s"),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left",
    )

    raw_df = pd.DataFrame(
        {
            "datetime": fetched_datetimes,
            "wind_speed_120m_kmh_forecast": hourly.Variables(0).ValuesAsNumpy(),
        }
    )

    aligned_df = pd.DataFrame({"datetime": target_datetimes}).merge(raw_df, on="datetime", how="left")
    aligned_df["source_model"] = OPEN_METEO_MODEL
    aligned_df["source_system"] = "open_meteo_previous_runs"
    aligned_df["location_name"] = LOCATION_NAME
    aligned_df["latitude"] = LATITUDE
    aligned_df["longitude"] = LONGITUDE
    aligned_df["forecast_horizon_hours"] = FORECAST_HORIZON_HOURS

    non_null_mask = aligned_df["wind_speed_120m_kmh_forecast"].notna()
    non_null_count = int(non_null_mask.sum())
    first_valid = aligned_df.loc[non_null_mask, "datetime"].min() if non_null_count else pd.NaT
    last_valid = aligned_df.loc[non_null_mask, "datetime"].max() if non_null_count else pd.NaT

    aligned_df.to_csv(output_dir / OPEN_METEO_OUTPUT_CSV.name, index=False)

    print(f"Open-Meteo CSV saved to: {output_dir / OPEN_METEO_OUTPUT_CSV.name}")
    print(f"Requested rows: {len(aligned_df)}")
    print(f"Non-null forecast rows: {non_null_count}")
    print(f"Coverage: {first_valid} to {last_valid}")
    return output_dir / OPEN_METEO_OUTPUT_CSV.name, aligned_df


def _parse_noaa_idx(idx_text: str) -> list[dict[str, object]]:
    """Parse NOAA .idx inventory lines into byte-range records."""
    lines = [line.strip() for line in idx_text.splitlines() if line.strip()]
    records: list[dict[str, object]] = []

    for i, line in enumerate(lines):
        parts = line.split(":")
        if len(parts) < 5:
            continue

        start_byte = int(parts[1])
        next_start_byte = int(lines[i + 1].split(":")[1]) if i + 1 < len(lines) else None
        records.append(
            {
                "line_number": int(parts[0]),
                "start_byte": start_byte,
                "end_byte": None if next_start_byte is None else next_start_byte - 1,
                "var_name": parts[3],
                "level": parts[4],
            }
        )

    return records


def _build_noaa_base_url(run_date: pd.Timestamp) -> str:
    file_name = f"gfs.t{NOAA_CYCLE}z.pgrb2.0p25.f{NOAA_FORECAST_HOUR:03d}"
    return (
        f"https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.{run_date.strftime('%Y%m%d')}/"
        f"{NOAA_CYCLE}/atmos/{file_name}"
    )


def _download_selected_noaa_messages(
    session: requests.Session,
    base_url: str,
    output_file: Path,
) -> Path:
    """Download only the UGRD/VGRD 100 m records from a NOAA GFS GRIB2 file."""
    idx_response = session.get(f"{base_url}.idx", timeout=30)
    if idx_response.status_code == 404:
        raise FileNotFoundError(f"NOAA index not found for {base_url}")
    idx_response.raise_for_status()

    records = _parse_noaa_idx(idx_response.text)
    selected_records = [
        record
        for record in records
        if record["level"] == "100 m above ground" and record["var_name"] in {"UGRD", "VGRD"}
    ]

    if len(selected_records) != 2:
        raise RuntimeError(f"Could not find 100 m UGRD/VGRD messages in {base_url}.idx")

    with output_file.open("wb") as file_handle:
        for record in selected_records:
            end_byte = record["end_byte"]
            if end_byte is None:
                range_header = f"bytes={record['start_byte']}-"
            else:
                range_header = f"bytes={record['start_byte']}-{end_byte}"

            response = session.get(base_url, headers={"Range": range_header}, timeout=60)
            response.raise_for_status()
            file_handle.write(response.content)

    return output_file


def _fetch_noaa_daily_point_forecast(
    run_date: pd.Timestamp,
    session: requests.Session,
) -> dict[str, object]:
    """Fetch a single NOAA daily +24h point forecast row."""
    if cfgrib is None:
        raise RuntimeError(
            "NOAA extraction requires cfgrib and eccodes. Install with "
            "'python -m pip install cfgrib eccodes'."
        )

    base_url = _build_noaa_base_url(run_date)
    fd, tmp_path_str = tempfile.mkstemp(
        suffix=f"_{run_date.strftime('%Y%m%d')}.grib2",
        dir=str(NOAA_TMP_DIR),
    )
    os.close(fd)
    tmp_path = Path(tmp_path_str)

    try:
        _download_selected_noaa_messages(session, base_url, tmp_path)

        dataset = xr.open_dataset(
            tmp_path,
            engine="cfgrib",
            backend_kwargs={"indexpath": ""},
        )
        try:
            point = dataset.sel(
                latitude=LATITUDE,
                longitude=NOAA_LONGITUDE_360,
                method="nearest",
            )
            u_component_ms = float(point["u100"].values)
            v_component_ms = float(point["v100"].values)
            wind_speed_ms = math.hypot(u_component_ms, v_component_ms)
            wind_speed_kmh = wind_speed_ms * 3.6

            valid_time_utc = pd.Timestamp(point["valid_time"].values)
            if valid_time_utc.tzinfo is None:
                valid_time_utc = valid_time_utc.tz_localize("UTC")
            else:
                valid_time_utc = valid_time_utc.tz_convert("UTC")

            valid_time_mst = valid_time_utc.tz_localize(None) - pd.Timedelta(hours=7)

            row = {
                "run_date_utc": run_date.date().isoformat(),
                "run_time_utc": pd.Timestamp(
                    f"{run_date.strftime('%Y-%m-%d')} {NOAA_CYCLE}:00:00",
                    tz="UTC",
                ).isoformat(),
                "valid_time_utc": valid_time_utc.isoformat(),
                "valid_time_mst": valid_time_mst.isoformat(sep=" "),
                "forecast_horizon_hours": NOAA_FORECAST_HOUR,
                "source_system": "noaa_gfs_aws",
                "location_name": LOCATION_NAME,
                "requested_latitude": LATITUDE,
                "requested_longitude": LONGITUDE,
                "grid_latitude": float(point.latitude.values),
                "grid_longitude": float(point.longitude.values),
                "grid_longitude_west": (
                    float(point.longitude.values) - 360
                    if float(point.longitude.values) > 180
                    else float(point.longitude.values)
                ),
                "u100_ms": u_component_ms,
                "v100_ms": v_component_ms,
                "wind_speed_100m_ms": wind_speed_ms,
                "wind_speed_100m_kmh": wind_speed_kmh,
            }
        finally:
            dataset.close()
    finally:
        tmp_path.unlink(missing_ok=True)

    return row


def backfill_noaa_daily_point_forecasts(
    output_dir: Path,
    target_datetimes: pd.DatetimeIndex,
    max_days: int | None = None,
) -> tuple[Path, pd.DataFrame]:
    """
    Backfill NOAA daily +24h point forecasts across the available archive range.

    This is a daily point series, not a full hourly 24-hour-ahead grid extraction.
    """
    print("Fetching NOAA daily +24h point forecast backfill...")

    requested_start = pd.Timestamp(target_datetimes.min().date())
    requested_end = pd.Timestamp(target_datetimes.max().date())
    available_start = max(requested_start, NOAA_ARCHIVE_AVAILABLE_FROM)
    output_file = output_dir / NOAA_OUTPUT_CSV.name

    existing_rows: list[dict[str, object]] = []
    completed_run_dates: set[str] = set()
    if output_file.exists():
        existing_df = pd.read_csv(output_file)
        if not existing_df.empty and "run_date_utc" in existing_df.columns:
            existing_rows = existing_df.to_dict("records")
            completed_run_dates = set(existing_df["run_date_utc"].astype(str))
            print(f"Resuming NOAA backfill from existing file with {len(existing_rows)} rows")

    pending_run_dates = [
        run_date
        for run_date in pd.date_range(start=available_start, end=requested_end, freq="D")
        if run_date.date().isoformat() not in completed_run_dates
    ]

    if max_days is not None:
        pending_run_dates = pending_run_dates[:max_days]

    if len(pending_run_dates) == 0 and not existing_rows:
        raise RuntimeError("No NOAA archive dates overlap the requested target range.")

    rows: list[dict[str, object]] = list(existing_rows)
    session = requests.Session()
    failures: list[str] = []

    for i, run_date in enumerate(pending_run_dates, start=1):
        try:
            rows.append(_fetch_noaa_daily_point_forecast(run_date, session))
        except FileNotFoundError:
            failures.append(run_date.date().isoformat())
        except Exception as exc:
            failures.append(f"{run_date.date().isoformat()} ({type(exc).__name__})")

        if i % 50 == 0 or i == len(pending_run_dates):
            print(f"NOAA progress: {i}/{len(pending_run_dates)} pending days processed")

    df = pd.DataFrame(rows).drop_duplicates(subset=["run_date_utc"]).sort_values("run_date_utc").reset_index(drop=True)
    df.to_csv(output_file, index=False)

    print(f"NOAA CSV saved to: {output_file}")
    print(f"Rows written: {len(df)}")
    if failures:
        print(f"Days skipped or failed: {len(failures)}")

    return output_file, df


def write_coverage_summary(
    output_dir: Path,
    target_datetimes: pd.DatetimeIndex,
    open_meteo_df: pd.DataFrame,
    noaa_df: pd.DataFrame | None,
) -> Path:
    """Write a compact coverage report for the generated weather datasets."""
    open_meteo_non_null = open_meteo_df["wind_speed_120m_kmh_forecast"].notna()
    open_meteo_first_valid = (
        open_meteo_df.loc[open_meteo_non_null, "datetime"].min()
        if open_meteo_non_null.any()
        else pd.NaT
    )
    open_meteo_last_valid = (
        open_meteo_df.loc[open_meteo_non_null, "datetime"].max()
        if open_meteo_non_null.any()
        else pd.NaT
    )

    rows = [
        {
            "dataset": "aeso_target_range",
            "requested_start": target_datetimes.min(),
            "requested_end": target_datetimes.max(),
            "rows_in_output": len(target_datetimes),
            "non_null_rows": len(target_datetimes),
            "available_start": target_datetimes.min(),
            "available_end": target_datetimes.max(),
            "notes": "Reference hourly index from AESO merged dataset",
        },
        {
            "dataset": "open_meteo_gfs_seamless_day_ahead",
            "requested_start": target_datetimes.min(),
            "requested_end": target_datetimes.max(),
            "rows_in_output": len(open_meteo_df),
            "non_null_rows": int(open_meteo_non_null.sum()),
            "available_start": open_meteo_first_valid,
            "available_end": open_meteo_last_valid,
            "notes": "Hourly 24-hour-ahead forecast aligned to AESO timestamps",
        },
        {
            "dataset": "noaa_gfs_00z_f024_daily_point",
            "requested_start": target_datetimes.min(),
            "requested_end": target_datetimes.max(),
            "rows_in_output": 0 if noaa_df is None else len(noaa_df),
            "non_null_rows": 0 if noaa_df is None else len(noaa_df),
            "available_start": NOAA_ARCHIVE_AVAILABLE_FROM,
            "available_end": (
                pd.NaT
                if noaa_df is None or noaa_df.empty
                else pd.to_datetime(noaa_df["run_date_utc"]).max()
            ),
            "notes": (
                "Daily point backfill from NOAA AWS archive. "
                "Archive availability begins 2021-03-23 for this endpoint."
            ),
        },
    ]

    coverage_df = pd.DataFrame(rows)
    output_file = output_dir / COVERAGE_OUTPUT_CSV.name
    coverage_df.to_csv(output_file, index=False)
    print(f"Coverage summary saved to: {output_file}")
    return output_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill Pincher Creek wind forecast data.")
    parser.add_argument(
        "--run-noaa-backfill",
        action="store_true",
        help="Download the NOAA daily +24h point backfill. This is much slower than Open-Meteo.",
    )
    parser.add_argument(
        "--noaa-max-days",
        type=int,
        default=None,
        help="Optional cap for NOAA backfill days, useful for smoke tests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("Starting weather backfill process...\n")
    output_directory = setup_directory()
    target_datetimes = load_target_datetimes()

    print(f"Target AESO range: {target_datetimes.min()} to {target_datetimes.max()}")
    print(f"Target hourly rows: {len(target_datetimes)}")

    open_meteo_file, open_meteo_df = fetch_open_meteo_backfill(output_directory, target_datetimes)

    noaa_file = None
    noaa_df = None
    if args.run_noaa_backfill:
        noaa_file, noaa_df = backfill_noaa_daily_point_forecasts(
            output_directory,
            target_datetimes,
            max_days=args.noaa_max_days,
        )
    else:
        print("Skipping NOAA backfill. Use --run-noaa-backfill to generate the daily NOAA file.")

    coverage_file = write_coverage_summary(output_directory, target_datetimes, open_meteo_df, noaa_df)

    print("\nProcess completed.")
    print(f"Open-Meteo CSV: {open_meteo_file}")
    if noaa_file is not None:
        print(f"NOAA CSV: {noaa_file}")
    print(f"Coverage summary: {coverage_file}")


if __name__ == "__main__":
    main()
