from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .config import (
    BINARY_FEATURES,
    CHANGE_FEATURE_SOURCES,
    CHANGE_HOURS,
    CURRENT_FEATURES,
    CYCLICAL_FEATURES,
    DEVICE,
    EARLY_STOPPING_PATIENCE,
    LAG_FEATURE_SOURCES,
    LAG_HOURS,
    MAX_EPOCHS,
    MODEL_DEFAULTS,
    MODEL_DIRS,
    OUTPUT_DATA_DIR,
    OUTPUT_FIGURES_DIR,
    OUTPUT_METRICS_DIR,
    RANDOM_SEED,
    SEQUENCE_FEATURES,
    SOURCE_DATA,
    SPIKE_THRESHOLD,
    TEST_END_INCLUSIVE,
    TEST_START,
    TRAIN_END_EXCLUSIVE,
    TRAIN_START,
    TS_CV_SPLITS,
    VAL_END_EXCLUSIVE,
    VAL_START,
)

sns.set_theme(style="whitegrid")


@dataclass
class ModelArtifacts:
    name: str
    features: list[str]
    continuous_features: list[str]
    best_params: dict[str, Any]
    cv_metrics: dict[str, Any]
    validation_metrics: dict[str, float]
    test_metrics: dict[str, float]
    threshold: float
    predictions_test_path: Path
    predictions_val_path: Path
    metrics_path: Path


def set_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_output_dirs() -> None:
    for path in [OUTPUT_DATA_DIR, OUTPUT_FIGURES_DIR, OUTPUT_METRICS_DIR, *MODEL_DIRS.values()]:
        path.mkdir(parents=True, exist_ok=True)
    for path in MODEL_DIRS.values():
        (path / "checkpoints").mkdir(parents=True, exist_ok=True)


def _assign_split(ts: pd.Timestamp) -> str | None:
    if TRAIN_START <= ts < TRAIN_END_EXCLUSIVE:
        return "train"
    if VAL_START <= ts < VAL_END_EXCLUSIVE:
        return "validation"
    if TEST_START <= ts <= TEST_END_INCLUSIVE:
        return "test"
    return None


def _lag_feature_name(source: str, lag: int) -> str:
    return f"{source}_lag_{lag}h"


def _change_feature_name(source: str, lag: int) -> str:
    return f"{source}_change_{lag}h"


def get_dummy_columns(df: pd.DataFrame) -> list[str]:
    prefixes = ("hour_", "dow_", "month_")
    return sorted([column for column in df.columns if column.startswith(prefixes)])


def get_mlp_feature_columns(df: pd.DataFrame) -> list[str]:
    lag_columns = [_lag_feature_name(source, lag) for source in LAG_FEATURE_SOURCES for lag in LAG_HOURS]
    change_columns = [
        _change_feature_name(source, lag) for source in CHANGE_FEATURE_SOURCES for lag in CHANGE_HOURS
    ]
    return CURRENT_FEATURES + lag_columns + change_columns + CYCLICAL_FEATURES + BINARY_FEATURES + get_dummy_columns(df)


def load_source_data() -> pd.DataFrame:
    df = pd.read_csv(SOURCE_DATA, parse_dates=["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    return df


def build_modeling_dataset(source_df: pd.DataFrame) -> pd.DataFrame:
    df = source_df.copy()
    if "is_stampede" not in df.columns:
        df["is_stampede"] = 0
        stampede_ranges = {
            2021: ("2021-07-09", "2021-07-18"),
            2022: ("2022-07-08", "2022-07-17"),
            2023: ("2023-07-07", "2023-07-16"),
            2024: ("2024-07-05", "2024-07-14"),
            2025: ("2025-07-04", "2025-07-13"),
        }
        for start, end in stampede_ranges.values():
            df.loc[df["datetime"].between(start, end), "is_stampede"] = 1

    for column in ["sin_day", "cos_day", "sin_week", "cos_week", "sin_year_1", "cos_year_1", "sin_year_2", "cos_year_2"]:
        if column not in df.columns:
            df[column] = np.nan

    day_of_year = df["datetime"].dt.day_of_year.astype(float)
    leap_year_after_feb = df["datetime"].dt.is_leap_year & (df["datetime"].dt.month >= 3)
    adjusted_day_of_year = day_of_year - leap_year_after_feb.astype(int)
    df["sin_day"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
    df["cos_day"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)
    df["sin_week"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["cos_week"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["sin_year_1"] = np.sin(2 * np.pi * adjusted_day_of_year / 365)
    df["cos_year_1"] = np.cos(2 * np.pi * adjusted_day_of_year / 365)
    df["sin_year_2"] = np.sin(4 * np.pi * adjusted_day_of_year / 365)
    df["cos_year_2"] = np.cos(4 * np.pi * adjusted_day_of_year / 365)

    df["pool_price_lead_2"] = df["ACTUAL_POOL_PRICE"].shift(-2)
    df["spike_lead_2"] = (df["pool_price_lead_2"] > SPIKE_THRESHOLD).astype("Int64")

    for source in LAG_FEATURE_SOURCES:
        for lag in LAG_HOURS:
            df[_lag_feature_name(source, lag)] = df[source].shift(lag)

    for source in CHANGE_FEATURE_SOURCES:
        for lag in CHANGE_HOURS:
            df[_change_feature_name(source, lag)] = df[source].diff(lag)

    hour_dummies = pd.get_dummies(df["hour_of_day"], prefix="hour", dtype=int)
    dow_dummies = pd.get_dummies(df["day_of_week"], prefix="dow", dtype=int)
    month_dummies = pd.get_dummies(df["month"], prefix="month", dtype=int)
    df = pd.concat([df, hour_dummies, dow_dummies, month_dummies], axis=1)

    df["split"] = df["datetime"].apply(_assign_split)
    df = df[df["split"].notna()].copy()

    required_columns = (
        get_mlp_feature_columns(df)
        + SEQUENCE_FEATURES
        + ["spike_lead_2", "pool_price_lead_2", "ACTUAL_POOL_PRICE", "datetime", "split"]
    )
    required_columns = list(dict.fromkeys(required_columns))
    df = df.dropna(subset=required_columns).reset_index(drop=True)
    df = df.loc[:, required_columns].copy()
    df["spike_lead_2"] = df["spike_lead_2"].astype(int)

    return df


def build_split_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split in ["train", "validation", "test"]:
        part = df[df["split"] == split]
        rows.append(
            {
                "split": split,
                "rows": len(part),
                "start": part["datetime"].min(),
                "end": part["datetime"].max(),
                "spike_rate_lead_2": part["spike_lead_2"].mean(),
            }
        )
    return pd.DataFrame(rows)


def build_data_dictionary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    base_map = {
        "datetime": ("timestamp", "MPT (UTC-7)", "Operating hour aligned to Mountain Prevailing Time"),
        "ACTUAL_POOL_PRICE": ("float", "CAD/MWh", "Actual pool price at time t"),
        "pool_price_lead_2": ("float", "CAD/MWh", "Actual pool price at time t+2; used for evaluation only"),
        "spike_lead_2": ("binary", "0/1", f"Target indicator for pool_price_lead_2 > {SPIKE_THRESHOLD} CAD/MWh"),
        "ACTUAL_AIL": ("float", "MW", "Alberta Internal Load at time t"),
        "net_export": ("float", "MW", "Exports minus imports across interties at time t"),
        "renewable_generation": ("float", "MW", "Wind plus solar generation at time t"),
        "reserve_margin": ("float", "ratio", "Resilience buffer divided by ACTUAL_AIL at time t"),
        "resilience_buffer": ("float", "MW", "Total system capability minus ACTUAL_AIL at time t"),
    }

    for column in df.columns:
        if column in base_map:
            dtype, units, definition = base_map[column]
            transform = "Original or directly derived from AESO merge pipeline"
            source = "AESO merged hourly dataset"
        elif column in CURRENT_FEATURES:
            dtype, units, definition = ("float", "MW or ratio", f"Current system feature: {column}")
            transform = "Inherited from AESO merge pipeline"
            source = "AESO merged hourly dataset"
        elif column in CYCLICAL_FEATURES:
            dtype, units, definition = ("float", "unitless", f"Cyclical seasonal encoding: {column}")
            transform = "Sine/cosine transform of calendar position"
            source = "Derived from datetime"
        elif column in BINARY_FEATURES:
            dtype, units, definition = ("binary", "0/1", f"Calendar/event indicator: {column}")
            transform = "Existing binary flag"
            source = "Derived from datetime and event ranges"
        elif column.startswith("hour_"):
            hour = column.split("_", maxsplit=1)[1]
            dtype, units, definition = ("binary", "0/1", f"Dummy variable for hour-of-day == {hour}")
            transform = "One-hot encoding"
            source = "Derived from datetime"
        elif column.startswith("dow_"):
            dow = column.split("_", maxsplit=1)[1]
            dtype, units, definition = ("binary", "0/1", f"Dummy variable for day-of-week == {dow}")
            transform = "One-hot encoding"
            source = "Derived from datetime"
        elif column.startswith("month_"):
            month = column.split("_", maxsplit=1)[1]
            dtype, units, definition = ("binary", "0/1", f"Dummy variable for month == {month}")
            transform = "One-hot encoding"
            source = "Derived from datetime"
        elif "_lag_" in column:
            dtype, units, definition = ("float", "same as source variable", f"Lagged feature: {column}")
            transform = "Backward time shift"
            source = "Derived from AESO merged hourly dataset"
        elif "_change_" in column:
            dtype, units, definition = ("float", "same as source variable", f"Change feature: {column}")
            transform = "Difference between time t and time t-k"
            source = "Derived from AESO merged hourly dataset"
        elif column == "split":
            dtype, units, definition = ("category", "label", "Fixed split assignment")
            transform = "Rule-based assignment from datetime cutoffs"
            source = "Project preprocessing"
        else:
            dtype, units, definition = ("float", "mixed", f"Feature from AESO merge pipeline: {column}")
            transform = "Inherited"
            source = "AESO merged hourly dataset"

        rows.append(
            {
                "column": column,
                "dtype": dtype,
                "units": units,
                "definition": definition,
                "transformation": transform,
                "source": source,
            }
        )

    return pd.DataFrame(rows)


def save_data_assets(df: pd.DataFrame) -> dict[str, Path]:
    modeling_data_path = OUTPUT_DATA_DIR / "modeling_dataset.csv"
    split_summary_path = OUTPUT_DATA_DIR / "split_summary.csv"
    data_dictionary_csv_path = OUTPUT_DATA_DIR / "data_dictionary.csv"
    data_dictionary_md_path = OUTPUT_DATA_DIR / "data_dictionary.md"

    df.to_csv(modeling_data_path, index=False)
    split_summary = build_split_summary(df)
    split_summary.to_csv(split_summary_path, index=False)

    data_dictionary = build_data_dictionary(df)
    data_dictionary.to_csv(data_dictionary_csv_path, index=False)

    with data_dictionary_md_path.open("w", encoding="utf-8") as handle:
        handle.write("# Data Dictionary\n\n")
        handle.write(
            "| Column | Type | Units | Definition | Transformation | Source |\n"
            "|---|---|---|---|---|---|\n"
        )
        for row in data_dictionary.itertuples(index=False):
            handle.write(
                f"| {row.column} | {row.dtype} | {row.units} | {row.definition} | "
                f"{row.transformation} | {row.source} |\n"
            )

    return {
        "modeling_data": modeling_data_path,
        "split_summary": split_summary_path,
        "data_dictionary_csv": data_dictionary_csv_path,
        "data_dictionary_md": data_dictionary_md_path,
    }


def generate_eda(df: pd.DataFrame) -> dict[str, Path]:
    figure_paths: dict[str, Path] = {}

    monthly = df.set_index("datetime").resample("MS").agg(
        {
            "ACTUAL_POOL_PRICE": "mean",
            "ACTUAL_AIL": "mean",
            "gas_total": "mean",
            "wind_total": "mean",
            "solar_total": "mean",
        }
    )
    fig, axes = plt.subplots(5, 1, figsize=(14, 14), sharex=True)
    for ax, column, title in [
        (axes[0], "ACTUAL_POOL_PRICE", "Monthly Average Pool Price"),
        (axes[1], "ACTUAL_AIL", "Monthly Average Alberta Internal Load"),
        (axes[2], "gas_total", "Monthly Average Gas Generation"),
        (axes[3], "wind_total", "Monthly Average Wind Generation"),
        (axes[4], "solar_total", "Monthly Average Solar Generation"),
    ]:
        monthly[column].plot(ax=ax, color="#174A7E")
        ax.set_title(title)
        ax.set_xlabel("")
    plt.tight_layout()
    figure_paths["time_series"] = OUTPUT_FIGURES_DIR / "time_series_overview.png"
    plt.savefig(figure_paths["time_series"], dpi=200, bbox_inches="tight")
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(df["ACTUAL_POOL_PRICE"], bins=120, ax=axes[0], color="#174A7E")
    axes[0].set_title("Distribution of Hourly Pool Prices")
    axes[0].set_xlabel("CAD/MWh")
    sns.boxplot(x=df["ACTUAL_POOL_PRICE"], ax=axes[1], color="#F4B400")
    axes[1].set_title("Boxplot of Pool Prices")
    axes[1].set_xlabel("CAD/MWh")
    plt.tight_layout()
    figure_paths["distribution"] = OUTPUT_FIGURES_DIR / "price_distribution.png"
    plt.savefig(figure_paths["distribution"], dpi=200, bbox_inches="tight")
    plt.close()

    corr_features = [
        "ACTUAL_POOL_PRICE",
        "ACTUAL_AIL",
        "gas_total",
        "wind_total",
        "solar_total",
        "reserve_margin",
        "net_load",
        "renewable_generation",
        "net_export",
        "resilience_buffer",
    ]
    corr = df[corr_features].corr(numeric_only=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap="RdBu_r", center=0, annot=True, fmt=".2f")
    plt.title("Correlation Matrix of Key Explanatory Variables")
    plt.tight_layout()
    figure_paths["correlation"] = OUTPUT_FIGURES_DIR / "correlation_matrix.png"
    plt.savefig(figure_paths["correlation"], dpi=200, bbox_inches="tight")
    plt.close()

    heatmap_df = (
        df.assign(hour=df["datetime"].dt.hour, month_num=df["datetime"].dt.month)
        .pivot_table(values="ACTUAL_POOL_PRICE", index="month_num", columns="hour", aggfunc="mean")
    )
    plt.figure(figsize=(14, 6))
    sns.heatmap(heatmap_df, cmap="YlOrRd")
    plt.title("Average Pool Price by Month and Hour of Day")
    plt.xlabel("Hour of Day")
    plt.ylabel("Month")
    plt.tight_layout()
    figure_paths["heatmap"] = OUTPUT_FIGURES_DIR / "price_heatmap_hour_month.png"
    plt.savefig(figure_paths["heatmap"], dpi=200, bbox_inches="tight")
    plt.close()

    spike_compare = df.copy()
    spike_compare["lead_2_spike"] = spike_compare["spike_lead_2"].map({0: "No Spike", 1: "Spike"})
    long_compare = spike_compare.melt(
        id_vars=["lead_2_spike"],
        value_vars=["ACTUAL_AIL", "reserve_margin", "wind_total", "gas_total"],
        var_name="feature",
        value_name="value",
    )
    plt.figure(figsize=(14, 7))
    sns.boxplot(data=long_compare, x="feature", y="value", hue="lead_2_spike")
    plt.title("System Conditions at Time t by Two-Hour-Ahead Spike Outcome")
    plt.xticks(rotation=20)
    plt.tight_layout()
    figure_paths["spike_compare"] = OUTPUT_FIGURES_DIR / "spike_vs_non_spike.png"
    plt.savefig(figure_paths["spike_compare"], dpi=200, bbox_inches="tight")
    plt.close()

    return figure_paths


def fit_scaler(df: pd.DataFrame, continuous_features: list[str]) -> dict[str, pd.Series]:
    means = df[continuous_features].mean()
    stds = df[continuous_features].std().replace(0, 1).fillna(1)
    return {"means": means, "stds": stds}


def transform_features(
    df: pd.DataFrame,
    feature_columns: list[str],
    continuous_features: list[str],
    scaler: dict[str, pd.Series],
) -> pd.DataFrame:
    transformed = df[feature_columns].copy()
    if continuous_features:
        transformed = transformed.astype({column: float for column in continuous_features}, copy=False)
    transformed.loc[:, continuous_features] = (
        transformed[continuous_features] - scaler["means"]
    ) / scaler["stds"]
    return transformed


def save_scaler(path: Path, scaler: dict[str, pd.Series], continuous_features: list[str]) -> None:
    payload = {
        "means": scaler["means"].to_dict(),
        "stds": scaler["stds"].to_dict(),
        "continuous_features": continuous_features,
    }
    joblib.dump(payload, path)


class TabularDataset(Dataset):
    def __init__(self, features: np.ndarray, target: np.ndarray):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.target = torch.tensor(target, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.target)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.target[index]


class SequenceDataset(Dataset):
    def __init__(self, sequences: np.ndarray, target: np.ndarray):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.target = torch.tensor(target, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.target)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[index], self.target[index]


class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        last_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(last_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)])
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs).squeeze(-1)


class CNNClassifier(nn.Module):
    def __init__(self, input_channels: int, conv_channels: int, kernel_size: int, dropout: float):
        super().__init__()
        padding = kernel_size // 2
        self.network = nn.Sequential(
            nn.Conv1d(input_channels, conv_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv1d(conv_channels, conv_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(conv_channels, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = inputs.transpose(1, 2)
        return self.network(inputs).squeeze(-1)


class LSTMClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(inputs)
        last_hidden = output[:, -1, :]
        last_hidden = self.dropout(last_hidden)
        return self.output(last_hidden).squeeze(-1)


def make_model(model_name: str, params: dict[str, Any], input_dim: int) -> nn.Module:
    if model_name == "mlp":
        hidden_dims = [params["hidden_dim_1"], params["hidden_dim_2"]]
        return MLPClassifier(input_dim=input_dim, hidden_dims=hidden_dims, dropout=params["dropout"])
    if model_name == "cnn":
        return CNNClassifier(
            input_channels=input_dim,
            conv_channels=params["conv_channels"],
            kernel_size=params["kernel_size"],
            dropout=params["dropout"],
        )
    if model_name == "lstm":
        return LSTMClassifier(
            input_size=input_dim,
            hidden_size=params["hidden_size"],
            num_layers=params["num_layers"],
            dropout=params["dropout"],
        )
    raise ValueError(f"Unsupported model_name: {model_name}")


def predict_probabilities(model: nn.Module, loader: DataLoader, device: str = DEVICE) -> np.ndarray:
    model.eval()
    probabilities: list[np.ndarray] = []
    with torch.no_grad():
        for features, _ in loader:
            features = features.to(device)
            logits = model(features)
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probabilities)


def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.linspace(0.1, 0.9, 81):
        preds = (y_prob >= threshold).astype(int)
        score = f1_score(y_true, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)
    return best_threshold


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, float]:
    preds = (y_prob >= threshold).astype(int)
    return {
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, preds)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    y_val: np.ndarray,
    params: dict[str, Any],
    pos_weight: float,
    checkpoint_dir: Path | None = None,
) -> tuple[nn.Module, list[dict[str, float]], float, dict[str, float]]:
    device = DEVICE
    model = model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=params["lr"],
        weight_decay=params["weight_decay"],
    )
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=device))

    best_state = None
    best_f1 = -1.0
    best_threshold = 0.5
    best_metrics: dict[str, float] = {}
    history: list[dict[str, float]] = []
    epochs_without_improvement = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_losses: list[float] = []
        for features, target in train_loader:
            features = features.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            logits = model(features)
            loss = loss_fn(logits, target)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        val_prob = predict_probabilities(model, val_loader, device=device)
        val_threshold = find_best_threshold(y_val, val_prob)
        val_metrics = compute_metrics(y_val, val_prob, val_threshold)
        history.append({"epoch": epoch, "train_loss": float(np.mean(train_losses)), **val_metrics})

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_threshold = val_threshold
            best_metrics = val_metrics
            best_state = {key: value.cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
            if checkpoint_dir is not None:
                torch.save(best_state, checkpoint_dir / "best.pt")
        else:
            epochs_without_improvement += 1

        if checkpoint_dir is not None:
            torch.save(model.state_dict(), checkpoint_dir / "latest.pt")

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            break

    if best_state is None:
        raise RuntimeError("Training failed to produce a valid model state.")

    model.load_state_dict(best_state)
    return model, history, best_threshold, best_metrics


def create_tabular_loaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_columns: list[str],
    continuous_features: list[str],
    batch_size: int,
) -> tuple[DataLoader, DataLoader, dict[str, pd.Series], np.ndarray, np.ndarray]:
    scaler = fit_scaler(train_df, continuous_features)
    x_train = transform_features(train_df, feature_columns, continuous_features, scaler).to_numpy(dtype=np.float32)
    x_val = transform_features(val_df, feature_columns, continuous_features, scaler).to_numpy(dtype=np.float32)
    y_train = train_df["spike_lead_2"].to_numpy(dtype=np.float32)
    y_val = val_df["spike_lead_2"].to_numpy(dtype=np.float32)

    train_loader = DataLoader(TabularDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TabularDataset(x_val, y_val), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, scaler, y_train, y_val


def create_sequence_arrays(
    full_df: pd.DataFrame,
    feature_columns: list[str],
    continuous_features: list[str],
    scaler: dict[str, pd.Series],
    target_indices: np.ndarray,
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    transformed_full = transform_features(full_df, feature_columns, continuous_features, scaler).to_numpy(dtype=np.float32)
    sequences: list[np.ndarray] = []
    targets: list[float] = []
    usable_indices: list[int] = []

    for idx in target_indices:
        start_idx = idx - sequence_length + 1
        if start_idx < 0:
            continue
        sequences.append(transformed_full[start_idx : idx + 1])
        targets.append(float(full_df.iloc[idx]["spike_lead_2"]))
        usable_indices.append(int(idx))

    return np.asarray(sequences, dtype=np.float32), np.asarray(targets, dtype=np.float32), np.asarray(usable_indices)


def create_sequence_loaders(
    full_df: pd.DataFrame,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    feature_columns: list[str],
    continuous_features: list[str],
    batch_size: int,
    sequence_length: int,
) -> tuple[DataLoader, DataLoader, dict[str, pd.Series], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_rows = full_df.iloc[train_indices]
    scaler = fit_scaler(train_rows, continuous_features)

    x_train, y_train, train_target_indices = create_sequence_arrays(
        full_df, feature_columns, continuous_features, scaler, train_indices, sequence_length
    )
    x_val, y_val, val_target_indices = create_sequence_arrays(
        full_df, feature_columns, continuous_features, scaler, val_indices, sequence_length
    )

    train_loader = DataLoader(SequenceDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(SequenceDataset(x_val, y_val), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, scaler, y_train, y_val, train_target_indices, val_target_indices


def sample_params(trial: optuna.Trial, model_name: str) -> dict[str, Any]:
    common = {
        "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
        "dropout": trial.suggest_float("dropout", 0.1, 0.4),
    }
    if model_name == "mlp":
        common["hidden_dim_1"] = trial.suggest_categorical("hidden_dim_1", [64, 128, 256])
        common["hidden_dim_2"] = trial.suggest_categorical("hidden_dim_2", [32, 64, 128])
        return common
    if model_name == "cnn":
        common["conv_channels"] = trial.suggest_categorical("conv_channels", [32, 64, 96])
        common["kernel_size"] = trial.suggest_categorical("kernel_size", [3, 5])
        common["sequence_length"] = trial.suggest_categorical("sequence_length", [24, 48])
        return common
    if model_name == "lstm":
        common["hidden_size"] = trial.suggest_categorical("hidden_size", [32, 64, 96])
        common["num_layers"] = trial.suggest_categorical("num_layers", [1, 2])
        common["sequence_length"] = trial.suggest_categorical("sequence_length", [24, 48])
        return common
    raise ValueError(f"Unknown model_name: {model_name}")


def run_time_series_cv(df: pd.DataFrame, model_name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    feature_columns = get_mlp_feature_columns(df) if model_name == "mlp" else SEQUENCE_FEATURES
    dummy_columns = get_dummy_columns(df)
    continuous_features = [column for column in feature_columns if column not in set(dummy_columns + BINARY_FEATURES)]

    trainval_df = df[df["split"].isin(["train", "validation"])].reset_index(drop=True)
    tscv = TimeSeriesSplit(n_splits=TS_CV_SPLITS)

    def objective(trial: optuna.Trial) -> float:
        params = sample_params(trial, model_name)
        fold_scores: list[float] = []

        for fold_train_idx, fold_val_idx in tscv.split(trainval_df):
            if model_name == "mlp":
                fold_train = trainval_df.iloc[fold_train_idx].reset_index(drop=True)
                fold_val = trainval_df.iloc[fold_val_idx].reset_index(drop=True)
                train_loader, val_loader, _, y_train, y_val = create_tabular_loaders(
                    fold_train,
                    fold_val,
                    feature_columns,
                    continuous_features,
                    batch_size=params["batch_size"],
                )
                pos_weight = max(float((len(y_train) - y_train.sum()) / max(y_train.sum(), 1.0)), 1.0)
                model = make_model(model_name, params, input_dim=len(feature_columns))
                _, _, _, metrics = train_model(model, train_loader, val_loader, y_val, params, pos_weight)
            else:
                train_loader, val_loader, _, y_train, y_val, _, _ = create_sequence_loaders(
                    trainval_df,
                    fold_train_idx,
                    fold_val_idx,
                    feature_columns,
                    continuous_features,
                    batch_size=params["batch_size"],
                    sequence_length=params["sequence_length"],
                )
                pos_weight = max(float((len(y_train) - y_train.sum()) / max(y_train.sum(), 1.0)), 1.0)
                model = make_model(model_name, params, input_dim=len(feature_columns))
                _, _, _, metrics = train_model(model, train_loader, val_loader, y_val, params, pos_weight)

            fold_scores.append(metrics["f1"])

        return float(np.mean(fold_scores))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
    study.optimize(objective, n_trials=MODEL_DEFAULTS[model_name]["trials"], show_progress_bar=False)

    return study.best_params, {
        "best_value": study.best_value,
        "best_trial_number": study.best_trial.number,
        "n_trials": len(study.trials),
        "trials": [
            {
                "trial_number": trial.number,
                "value": trial.value,
                "params": trial.params,
            }
            for trial in study.trials
        ],
    }


def build_prediction_frame(
    indices: np.ndarray,
    base_df: pd.DataFrame,
    y_prob: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    frame = base_df.iloc[indices].copy().reset_index(drop=True)
    frame["predicted_probability"] = y_prob
    frame["predicted_label"] = (y_prob >= threshold).astype(int)
    frame["actual_label"] = frame["spike_lead_2"]
    return frame


def error_analysis(predictions: pd.DataFrame, model_name: str) -> Path:
    labelled = predictions.copy()
    labelled["error_group"] = np.select(
        [
            (labelled["actual_label"] == 1) & (labelled["predicted_label"] == 1),
            (labelled["actual_label"] == 1) & (labelled["predicted_label"] == 0),
            (labelled["actual_label"] == 0) & (labelled["predicted_label"] == 1),
        ],
        ["true_positive", "false_negative", "false_positive"],
        default="true_negative",
    )
    summary = (
        labelled.groupby("error_group")[
            ["pool_price_lead_2", "ACTUAL_AIL", "reserve_margin", "renewable_generation", "wind_total", "gas_total"]
        ]
        .mean()
        .round(3)
        .reset_index()
    )
    path = OUTPUT_METRICS_DIR / f"{model_name}_error_analysis.csv"
    summary.to_csv(path, index=False)
    return path


def train_and_evaluate_final_model(df: pd.DataFrame, model_name: str, best_params: dict[str, Any], cv_info: dict[str, Any]) -> ModelArtifacts:
    feature_columns = get_mlp_feature_columns(df) if model_name == "mlp" else SEQUENCE_FEATURES
    dummy_columns = get_dummy_columns(df)
    continuous_features = [column for column in feature_columns if column not in set(dummy_columns + BINARY_FEATURES)]

    model_dir = MODEL_DIRS[model_name]
    checkpoint_dir = model_dir / "checkpoints"

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "validation"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    if model_name == "mlp":
        train_loader, val_loader, scaler, y_train, y_val = create_tabular_loaders(
            train_df,
            val_df,
            feature_columns,
            continuous_features,
            batch_size=best_params["batch_size"],
        )
        save_scaler(model_dir / "scaler.joblib", scaler, continuous_features)
        pos_weight = max(float((len(y_train) - y_train.sum()) / max(y_train.sum(), 1.0)), 1.0)
        model = make_model(model_name, best_params, input_dim=len(feature_columns))
        model, history, threshold, validation_metrics = train_model(
            model,
            train_loader,
            val_loader,
            y_val,
            best_params,
            pos_weight,
            checkpoint_dir=checkpoint_dir,
        )

        test_features = transform_features(test_df, feature_columns, continuous_features, scaler).to_numpy(dtype=np.float32)
        test_loader = DataLoader(
            TabularDataset(test_features, test_df["spike_lead_2"].to_numpy(dtype=np.float32)),
            batch_size=best_params["batch_size"],
            shuffle=False,
        )
        val_prob = predict_probabilities(model, val_loader)
        test_prob = predict_probabilities(model, test_loader)
        val_indices = np.arange(len(val_df))
        test_indices = np.arange(len(test_df))
        val_base = val_df
        test_base = test_df
    else:
        full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
        split_labels = full_df["split"].to_numpy()
        train_idx = np.flatnonzero(split_labels == "train")
        val_idx = np.flatnonzero(split_labels == "validation")
        test_idx = np.flatnonzero(split_labels == "test")

        train_loader, val_loader, scaler, y_train, y_val, _, val_target_indices = create_sequence_loaders(
            full_df,
            train_idx,
            val_idx,
            feature_columns,
            continuous_features,
            batch_size=best_params["batch_size"],
            sequence_length=best_params["sequence_length"],
        )
        save_scaler(model_dir / "scaler.joblib", scaler, continuous_features)
        pos_weight = max(float((len(y_train) - y_train.sum()) / max(y_train.sum(), 1.0)), 1.0)
        model = make_model(model_name, best_params, input_dim=len(feature_columns))
        model, history, threshold, validation_metrics = train_model(
            model,
            train_loader,
            val_loader,
            y_val,
            best_params,
            pos_weight,
            checkpoint_dir=checkpoint_dir,
        )

        x_test, y_test, test_target_indices = create_sequence_arrays(
            full_df,
            feature_columns,
            continuous_features,
            scaler,
            test_idx,
            best_params["sequence_length"],
        )
        test_loader = DataLoader(SequenceDataset(x_test, y_test), batch_size=best_params["batch_size"], shuffle=False)
        val_prob = predict_probabilities(model, val_loader)
        test_prob = predict_probabilities(model, test_loader)
        val_indices = val_target_indices
        test_indices = test_target_indices
        val_base = full_df
        test_base = full_df

    test_truth = test_base.iloc[test_indices]["spike_lead_2"].to_numpy()
    test_metrics = compute_metrics(test_truth, test_prob, threshold)

    val_predictions = build_prediction_frame(val_indices, val_base, val_prob, threshold)
    test_predictions = build_prediction_frame(test_indices, test_base, test_prob, threshold)

    history_path = model_dir / "training_history.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)

    val_predictions_path = model_dir / "validation_predictions.csv"
    test_predictions_path = model_dir / "test_predictions.csv"
    val_predictions.to_csv(val_predictions_path, index=False)
    test_predictions.to_csv(test_predictions_path, index=False)

    metrics_path = model_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "model_name": model_name,
                "feature_count": len(feature_columns),
                "threshold": threshold,
                "best_params": best_params,
                "cv": cv_info,
                "validation_metrics": validation_metrics,
                "test_metrics": test_metrics,
            },
            handle,
            indent=2,
        )

    error_analysis(test_predictions, model_name)

    return ModelArtifacts(
        name=model_name,
        features=feature_columns,
        continuous_features=continuous_features,
        best_params=best_params,
        cv_metrics=cv_info,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        threshold=threshold,
        predictions_test_path=test_predictions_path,
        predictions_val_path=val_predictions_path,
        metrics_path=metrics_path,
    )


def run_model_suite(df: pd.DataFrame) -> list[ModelArtifacts]:
    artifacts: list[ModelArtifacts] = []
    for model_name in ["mlp", "cnn", "lstm"]:
        print(f"\nTuning {model_name.upper()}...")
        best_params, cv_info = run_time_series_cv(df, model_name)
        print(f"Best params for {model_name}: {best_params}")

        print(f"Training final {model_name.upper()} on fixed train/validation split...")
        model_artifacts = train_and_evaluate_final_model(df, model_name, best_params, cv_info)
        artifacts.append(model_artifacts)
    return artifacts


def save_model_comparison(artifacts: list[ModelArtifacts]) -> Path:
    rows = []
    for artifact in artifacts:
        rows.append(
            {
                "model": artifact.name,
                "cv_best_f1": artifact.cv_metrics["best_value"],
                "validation_f1": artifact.validation_metrics["f1"],
                "validation_precision": artifact.validation_metrics["precision"],
                "validation_recall": artifact.validation_metrics["recall"],
                "validation_roc_auc": artifact.validation_metrics["roc_auc"],
                "test_f1": artifact.test_metrics["f1"],
                "test_precision": artifact.test_metrics["precision"],
                "test_recall": artifact.test_metrics["recall"],
                "test_roc_auc": artifact.test_metrics["roc_auc"],
                "threshold": artifact.threshold,
            }
        )
    comparison = pd.DataFrame(rows).sort_values("test_f1", ascending=False)
    comparison_path = OUTPUT_METRICS_DIR / "model_comparison.csv"
    comparison.to_csv(comparison_path, index=False)
    return comparison_path


def plot_test_roc_curves(artifacts: list[ModelArtifacts]) -> Path:
    from sklearn.metrics import RocCurveDisplay

    plt.figure(figsize=(8, 6))
    for artifact in artifacts:
        predictions = pd.read_csv(artifact.predictions_test_path)
        RocCurveDisplay.from_predictions(
            predictions["actual_label"],
            predictions["predicted_probability"],
            name=artifact.name.upper(),
            ax=plt.gca(),
        )
    plt.title("ROC Curves on Untouched Test Set")
    plt.tight_layout()
    output_path = OUTPUT_FIGURES_DIR / "test_roc_curves.png"
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    return output_path


def run_project_pipeline() -> dict[str, Any]:
    set_seed()
    ensure_output_dirs()

    source_df = load_source_data()
    modeling_df = build_modeling_dataset(source_df)
    data_assets = save_data_assets(modeling_df)
    figure_paths = generate_eda(modeling_df)
    model_artifacts = run_model_suite(modeling_df)
    comparison_path = save_model_comparison(model_artifacts)
    roc_path = plot_test_roc_curves(model_artifacts)

    return {
        "modeling_df": modeling_df,
        "data_assets": data_assets,
        "eda_figures": figure_paths,
        "model_artifacts": model_artifacts,
        "comparison_path": comparison_path,
        "roc_curve_path": roc_path,
    }
