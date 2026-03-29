"""
shared/data_prep.py
-------------------
Centralised data loading, feature definitions, train/val/test splitting,
StandardScaler fitting, and PyTorch Dataset classes shared by all model
notebooks (MLP, CNN, LSTM).
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from torch.utils.data import Dataset, DataLoader
import torch

# ── Paths ──────────────────────────────────────────────────────────────────
_MODULE_DIR  = Path(__file__).parent
PROJECT_ROOT = _MODULE_DIR.parent
DATA_PATH    = PROJECT_ROOT / "Data" / "CSVs" / "aeso_merged_2020_2025.csv"

# ── Target ─────────────────────────────────────────────────────────────────
TARGET = "spike_lead_1"

# ── Time-based split boundaries (exclusive upper bounds) ───────────────────
TRAIN_END = "2023-11-06"   # train:  [start, TRAIN_END)
VAL_END   = "2024-12-12"   # val:    [TRAIN_END, VAL_END)
                            # test:   [VAL_END, end]

# ── Feature column groups ──────────────────────────────────────────────────
ALL_LEADS  = [f"spike_lead_{n}" for n in range(1, 25)]
SPIKE_LAGS = [f"spike_lag_{n}"  for n in range(1, 25)]
PRICE_LAGS = ["price_lag_1h", "price_lag_6h", "price_lag_24h",
              "price_rolling_mean_6h"]

# Columns that are never model inputs regardless of model type
_EXCLUDE_ALWAYS = {"datetime"} | set(ALL_LEADS)

# Binary/dummy columns — standardised separately (kept as-is)
DUMMY_COLS = frozenset({"spike", "is_weekend", "is_stampede"} | set(SPIKE_LAGS))

# Manual lag features dropped for LSTM (the recurrent sequence captures them)
_LSTM_DROP = frozenset(set(SPIKE_LAGS) | set(PRICE_LAGS))


# ── Feature selection ──────────────────────────────────────────────────────

def get_feature_cols(model_type: str, all_cols: list) -> list:
    """
    Return ordered feature column list for the given model type.

    Parameters
    ----------
    model_type : "MLP", "CNN", or "LSTM"
    all_cols   : list of column names from the loaded DataFrame

    Returns
    -------
    list of str — columns to use as model inputs (excludes target)
    """
    base = [c for c in all_cols
            if c not in _EXCLUDE_ALWAYS and c != TARGET]
    if model_type.upper() == "LSTM":
        base = [c for c in base if c not in _LSTM_DROP]
    return base


def get_continuous_cols(feature_cols: list) -> list:
    """Return the subset of feature_cols that should be standardised."""
    return [c for c in feature_cols if c not in DUMMY_COLS]


# ── Data loading and splitting ─────────────────────────────────────────────

def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """
    Load the merged CSV, parse datetime, cast spike-lag columns to float,
    and sort chronologically.
    """
    df = pd.read_csv(path, parse_dates=["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    # Spike lag cols are stored as Int64 (nullable int); cast to float for tensors
    lag_cols = [c for c in df.columns if c.startswith("spike_lag_")]
    df[lag_cols] = df[lag_cols].astype(float)
    return df


def split_data(df: pd.DataFrame):
    """
    Split df into (train, val, test) using fixed time boundaries.
    No data from val or test is ever used to fit scalers or models.
    """
    train = df[df["datetime"] <  TRAIN_END].copy()
    val   = df[(df["datetime"] >= TRAIN_END) & (df["datetime"] < VAL_END)].copy()
    test  = df[df["datetime"] >= VAL_END].copy()
    return train, val, test


# ── Scaling ────────────────────────────────────────────────────────────────

def fit_scaler(train_df: pd.DataFrame, feature_cols: list):
    """
    Fit a StandardScaler on continuous features of the training set only.

    Returns
    -------
    scaler        : fitted StandardScaler
    continuous_cols : list of column names that were scaled
    """
    continuous_cols = get_continuous_cols(feature_cols)
    scaler = StandardScaler()
    scaler.fit(train_df[continuous_cols].fillna(0))
    return scaler, continuous_cols


def apply_scaler(df: pd.DataFrame,
                 scaler: StandardScaler,
                 continuous_cols: list) -> pd.DataFrame:
    """Apply a fitted scaler to df. Returns a copy; does not mutate df."""
    df = df.copy()
    df[continuous_cols] = scaler.transform(df[continuous_cols].fillna(0))
    return df


# ── Class-imbalance weight ─────────────────────────────────────────────────

def compute_pos_weight(train_target: pd.Series):
    """
    Compute pos_weight for BCEWithLogitsLoss from the training target.
    pos_weight = n_negative / n_positive so the loss is balanced.
    """
    import torch
    n_pos = float(train_target.sum())
    n_neg = float(len(train_target) - n_pos)
    return torch.tensor([n_neg / n_pos], dtype=torch.float32)


# ── Cross-validation splits ────────────────────────────────────────────────

def get_cv_splits(df: pd.DataFrame, n_splits: int = 5) -> list:
    """
    Return a list of (train_idx, val_idx) index arrays from TimeSeriesSplit,
    applied to the rows of df (typically the combined train+val period).
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    return list(tscv.split(np.arange(len(df))))


# ── PyTorch Datasets ───────────────────────────────────────────────────────

class TabularDataset(Dataset):
    """
    Flat feature-vector dataset for the MLP.
    Each sample is (feature_vector, label).
    """
    def __init__(self, df: pd.DataFrame, feature_cols: list,
                 target_col: str = TARGET):
        X = df[feature_cols].fillna(0).values.astype(np.float32)
        y = df[target_col].fillna(0).values.astype(np.float32)
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class SequenceDataset(Dataset):
    """
    Sliding-window sequence dataset for CNN and LSTM.
    Each sample is (window_tensor, label) where window_tensor has shape
    (lookback, n_features) and label is the spike indicator at the next step.
    The first `lookback` rows of df are consumed to build the first window,
    so this dataset has len(df) - lookback samples.
    """
    def __init__(self, df: pd.DataFrame, feature_cols: list,
                 lookback: int, target_col: str = TARGET):
        X_all = df[feature_cols].fillna(0).values.astype(np.float32)
        y_all = df[target_col].fillna(0).values.astype(np.float32)

        n = len(X_all)
        sequences = np.lib.stride_tricks.sliding_window_view(
            X_all, (lookback, X_all.shape[1])
        ).reshape(n - lookback + 1, lookback, X_all.shape[1])
        # Label at position i corresponds to features ending at row i+lookback-1;
        # target is the spike AT row i+lookback-1 (i.e. spike_lead_1 of that row)
        labels = y_all[lookback - 1:]

        self.X = torch.from_numpy(sequences.copy())
        self.y = torch.from_numpy(labels.copy())

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ── Training utilities ─────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device, threshold: float = 0.5) -> dict:
    """Return loss, F1, precision, recall, and ROC-AUC on the given loader."""
    from sklearn.metrics import (f1_score, precision_score,
                                 recall_score, roc_auc_score)
    model.eval()
    all_logits, all_labels = [], []
    total_loss = 0.0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            total_loss += criterion(logits, y).item() * len(y)
            all_logits.append(logits.cpu())
            all_labels.append(y.cpu())

    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy().astype(int)
    probs  = 1 / (1 + np.exp(-logits))        # sigmoid
    preds  = (probs >= threshold).astype(int)

    return {
        "loss":      total_loss / sum(len(b) for b in all_labels),
        "f1":        f1_score(labels, preds,      zero_division=0),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall":    recall_score(labels, preds,   zero_division=0),
        "auc":       roc_auc_score(labels, probs),
    }


def train_model(model, train_loader, val_loader, optimizer, criterion, device,
                max_epochs: int = 50, patience: int = 10,
                checkpoint_dir: Path = None, verbose: bool = True) -> tuple:
    """
    Full training loop with early stopping on validation F1.

    Returns
    -------
    history  : list of dicts (one per epoch)
    best_f1  : float — best validation F1 achieved
    """
    best_f1, patience_ctr = 0.0, 0
    history = []

    for epoch in range(1, max_epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_m   = evaluate(model, val_loader, criterion, device)
        row = {"epoch": epoch, "train_loss": tr_loss, **val_m}
        history.append(row)

        if verbose:
            print(f"  Ep {epoch:3d} | tr_loss={tr_loss:.4f} | "
                  f"val_f1={val_m['f1']:.4f} | val_auc={val_m['auc']:.4f} | "
                  f"val_loss={val_m['loss']:.4f}")

        # Save latest checkpoint
        if checkpoint_dir is not None:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), checkpoint_dir / "latest_model.pt")

        # Best model checkpoint and early stopping
        if val_m["f1"] > best_f1:
            best_f1 = val_m["f1"]
            patience_ctr = 0
            if checkpoint_dir is not None:
                torch.save(model.state_dict(), checkpoint_dir / "best_model.pt")
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch} (patience={patience})")
                break

    return history, best_f1


# ── Random hyperparameter search ───────────────────────────────────────────

def random_search(param_grid: dict, n_trials: int,
                  build_fn, train_val_df: pd.DataFrame,
                  feature_cols: list, device,
                  use_sequences: bool = False,
                  n_cv_splits: int = 5,
                  max_epochs: int = 30,
                  patience: int = 5) -> list:
    """
    Random search over param_grid using TimeSeriesSplit CV.

    Parameters
    ----------
    param_grid    : dict mapping param name → list of candidate values
    n_trials      : number of random configurations to evaluate
    build_fn      : callable(params) → (model, optimizer, criterion)
    train_val_df  : combined train+val DataFrame (scaled)
    feature_cols  : feature columns for this model type
    device        : torch device
    use_sequences : True for CNN/LSTM (builds SequenceDataset);
                    False for MLP (builds TabularDataset)
    n_cv_splits   : number of TimeSeriesSplit folds
    max_epochs    : max training epochs per fold
    patience      : early stopping patience per fold

    Returns
    -------
    list of dicts sorted by mean_cv_f1 descending, each containing
    {'params', 'mean_cv_f1', 'fold_f1s'}
    """
    import random as _random
    results = []

    for trial in range(1, n_trials + 1):
        params = {k: _random.choice(v) for k, v in param_grid.items()}
        print(f"\n── Trial {trial}/{n_trials} ──  {params}")

        fold_f1s = []
        for fold_idx, (tr_idx, va_idx) in enumerate(get_cv_splits(train_val_df, n_cv_splits)):
            fold_train = train_val_df.iloc[tr_idx]
            fold_val   = train_val_df.iloc[va_idx]

            # Fit scaler on fold training data only
            scaler, cont_cols = fit_scaler(fold_train, feature_cols)
            fold_train_s = apply_scaler(fold_train, scaler, cont_cols)
            fold_val_s   = apply_scaler(fold_val,   scaler, cont_cols)

            lb = params.get("lookback", None)
            bs = params.get("batch_size", 256)

            if use_sequences:
                tr_ds = SequenceDataset(fold_train_s, feature_cols, lb)
                va_ds = SequenceDataset(fold_val_s,   feature_cols, lb)
            else:
                tr_ds = TabularDataset(fold_train_s, feature_cols)
                va_ds = TabularDataset(fold_val_s,   feature_cols)

            tr_loader = DataLoader(tr_ds, batch_size=bs, shuffle=True,  drop_last=True)
            va_loader = DataLoader(va_ds, batch_size=bs * 2, shuffle=False)

            model, optimizer, criterion = build_fn(params, fold_train)
            model = model.to(device)

            _, best_f1 = train_model(model, tr_loader, va_loader, optimizer,
                                     criterion, device,
                                     max_epochs=max_epochs,
                                     patience=patience,
                                     verbose=False)
            fold_f1s.append(best_f1)
            print(f"  Fold {fold_idx + 1}/{n_cv_splits}  F1={best_f1:.4f}")

        mean_f1 = float(np.mean(fold_f1s))
        results.append({"params": params, "mean_cv_f1": mean_f1, "fold_f1s": fold_f1s})
        print(f"  → Mean CV F1 = {mean_f1:.4f}")

    results.sort(key=lambda r: r["mean_cv_f1"], reverse=True)
    return results
