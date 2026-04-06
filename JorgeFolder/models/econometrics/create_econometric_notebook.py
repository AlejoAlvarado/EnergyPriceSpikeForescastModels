from pathlib import Path

import nbformat as nbf


NOTEBOOK_PATH = Path(__file__).resolve().parent / "econometric_price_models_t_plus_2.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(text)


def code(text: str):
    return nbf.v4.new_code_cell(text)


nb = nbf.v4.new_notebook()
nb["cells"] = [
    md(
        """# Econometric Experiment: Price Forecasting at t+2

**Goal:** switch from spike classification to direct price forecasting.

- The target is `price_t_plus_2`
- A derived spike is `1(price_t_plus_2 > 200)`
- The date windows match the previous ML models for comparability

This notebook keeps an econometric perspective:

- check missing values
- test stationarity
- reduce multicollinearity with VIF
- forecast **price at t+2**
- then evaluate whether the predicted price is above `200`

Practical note:

- `arch` and `statsmodels` are available locally, so univariate GARCH-type models and VAR models are feasible
- `mvarch` is available locally, so we can also try a full unconstrained BEKK-type covariance model
- a full turnkey **MS-VAR** / full multivariate **VAR-GARCH** implementation is not available in the installed stack
- the notebook therefore uses:
  - direct regression + GARCH-type volatility models
  - a **Markov-switching regime model** as the regime-switching block
  - a **VAR** mean model as the multivariate benchmark
  - a **real bivariate BEKK / GARCH block** with small validation tuning, centered on price and load

The notebook is kept intentionally simple and readable, closer in spirit to the LSTM notebook than to a research codebase.
"""
    ),
    code(
        """# 1. Imports and config
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    f1_score,
    precision_score,
    recall_score,
)

from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from statsmodels.tools.sm_exceptions import InterpolationWarning, ConvergenceWarning as SMConvergenceWarning

from arch import arch_model
import torch
import mvarch
from arch.utility.exceptions import ConvergenceWarning as ArchConvergenceWarning

warnings.simplefilter("ignore", InterpolationWarning)
warnings.simplefilter("ignore", SMConvergenceWarning)
warnings.simplefilter("ignore", ArchConvergenceWarning)

SOURCE_DATA = Path("../../outputs/data/modeling_dataset.csv")
OUTPUT_DIR = Path("notebook_run")
FIG_DIR = OUTPUT_DIR / "figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_END_EXCLUSIVE = pd.Timestamp("2023-11-06 00:00:00")
VAL_END_EXCLUSIVE = pd.Timestamp("2024-12-12 00:00:00")
TEST_START = VAL_END_EXCLUSIVE
SPIKE_THRESHOLD = 200.0
ARCH_SCALE = 10.0
BEKK_SCALE = 10.0
BEKK_FIT_WINDOW = 1000
BEKK_PRED_HISTORY = 200

print(f"Source data : {SOURCE_DATA}")
print(f"Output dir  : {OUTPUT_DIR}")
"""
    ),
    code(
        """# 2. Load data and create the direct t+2 target
df = pd.read_csv(SOURCE_DATA, parse_dates=["datetime"]).sort_values("datetime").reset_index(drop=True)

# Direct target for this notebook:
# predict the actual price 2 hours ahead using information available at time t.
df["price_t_plus_2"] = df["ACTUAL_POOL_PRICE"].shift(-2)
df["spike_t_plus_2"] = (df["price_t_plus_2"] > SPIKE_THRESHOLD).astype("Int64")
df["target_change_2h"] = df["price_t_plus_2"] - df["ACTUAL_POOL_PRICE"]

# We use stationary transformations instead of raw levels whenever possible.
df["price_change_1h"] = df["ACTUAL_POOL_PRICE"] - df["ACTUAL_POOL_PRICE_lag_1h"]
df["price_change_6h"] = df["ACTUAL_POOL_PRICE"] - df["ACTUAL_POOL_PRICE_lag_6h"]
df["price_change_24h"] = df["ACTUAL_POOL_PRICE"] - df["ACTUAL_POOL_PRICE_lag_24h"]
df["net_export_change_1h"] = df["net_export"].diff()
df["renewables_share_change_1h"] = df["renewables_share"].diff()

candidate_continuous = [
    "price_change_1h",
    "price_change_6h",
    "price_change_24h",
    "ACTUAL_AIL_change_24h",
    "wind_total_change_24h",
    "solar_total_change_1h",
    "gas_total_change_1h",
    "net_load_change_1h",
    "reserve_margin_change_1h",
    "net_export_change_1h",
    "renewables_share_change_1h",
    "net_load_3h_change",
]

calendar_and_dummies = [
    "sin_day",
    "cos_day",
    "sin_week",
    "cos_week",
    "is_weekend",
    "is_stampede",
]

required = ["datetime", "ACTUAL_POOL_PRICE", "price_t_plus_2", "spike_t_plus_2", "target_change_2h"] + candidate_continuous + calendar_and_dummies
df = df.dropna(subset=required).reset_index(drop=True)

print(f"Rows after target creation and NA drop: {len(df):,}")
print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
print(f"Direct spike rate at t+2: {df['spike_t_plus_2'].mean():.4f}")
"""
    ),
    code(
        """# 3. Missing values and split summary
missing_summary = df[required].isna().sum().sort_values(ascending=False).to_frame("missing_count")
missing_summary.to_csv(OUTPUT_DIR / "missing_summary.csv")

train = df[df["datetime"] < TRAIN_END_EXCLUSIVE].copy()
validation = df[(df["datetime"] >= TRAIN_END_EXCLUSIVE) & (df["datetime"] < VAL_END_EXCLUSIVE)].copy()
test = df[df["datetime"] >= TEST_START].copy()
pretest = pd.concat([train, validation], ignore_index=False)

split_summary = pd.DataFrame(
    [
        {
            "split": "train",
            "rows": len(train),
            "start": train["datetime"].min(),
            "end": train["datetime"].max(),
            "spike_rate_t_plus_2": train["spike_t_plus_2"].mean(),
        },
        {
            "split": "validation",
            "rows": len(validation),
            "start": validation["datetime"].min(),
            "end": validation["datetime"].max(),
            "spike_rate_t_plus_2": validation["spike_t_plus_2"].mean(),
        },
        {
            "split": "test",
            "rows": len(test),
            "start": test["datetime"].min(),
            "end": test["datetime"].max(),
            "spike_rate_t_plus_2": test["spike_t_plus_2"].mean(),
        },
    ]
)
split_summary.to_csv(OUTPUT_DIR / "split_summary.csv", index=False)
split_summary
"""
    ),
    code(
        """# 4. Stationarity checks
# ADF: reject unit root when p < 0.05
# KPSS: fail to reject non-stationarity when p > 0.05

def adf_kpss_summary(series):
    series = pd.Series(series).dropna().astype(float)
    adf_p = adfuller(series, autolag="AIC")[1]
    try:
        kpss_p = kpss(series, regression="c", nlags="auto")[1]
    except Exception:
        kpss_p = np.nan
    stationary = (adf_p < 0.05) and (np.isnan(kpss_p) or kpss_p > 0.05)
    return adf_p, kpss_p, stationary


stationarity_rows = []
for col in ["target_change_2h"] + candidate_continuous:
    adf_p, kpss_p, stationary = adf_kpss_summary(df[col])
    stationarity_rows.append(
        {
            "series": col,
            "adf_pvalue": adf_p,
            "kpss_pvalue": kpss_p,
            "stationary": stationary,
        }
    )

stationarity_summary = pd.DataFrame(stationarity_rows).sort_values(["stationary", "series"], ascending=[False, True])
stationarity_summary.to_csv(OUTPUT_DIR / "stationarity_summary.csv", index=False)
stationarity_summary
"""
    ),
    code(
        """# 5. Reduce multicollinearity with VIF
# We only apply VIF to the continuous predictors.
# Calendar terms and dummies are added back afterward.

def vif_reduce(frame, features, threshold=10.0):
    kept = features.copy()
    history = []
    while True:
        X = frame[kept].astype(float)
        vif_df = pd.DataFrame(
            {
                "feature": kept,
                "vif": [variance_inflation_factor(X.values, i) for i in range(len(kept))],
            }
        ).sort_values("vif", ascending=False)
        history.append(vif_df.copy())
        if vif_df["vif"].max() <= threshold or len(kept) <= 5:
            return kept, vif_df, history
        kept.remove(vif_df.iloc[0]["feature"])


stationary_continuous = stationarity_summary.loc[stationarity_summary["stationary"], "series"].tolist()
stationary_continuous = [c for c in stationary_continuous if c != "target_change_2h"]

selected_continuous, vif_summary, vif_history = vif_reduce(df, stationary_continuous, threshold=10.0)
vif_summary.to_csv(OUTPUT_DIR / "vif_summary.csv", index=False)

FINAL_FEATURES = selected_continuous + calendar_and_dummies

print("Selected continuous features after stationarity + VIF:")
print(selected_continuous)
print("\\nFinal feature count:", len(FINAL_FEATURES))
vif_summary
"""
    ),
    code(
        """# 6. Small helper functions for metrics and model outputs
def regression_and_spike_metrics(actual_price, predicted_price):
    actual_price = np.asarray(actual_price, dtype=float)
    predicted_price = np.asarray(predicted_price, dtype=float)
    actual_spike = (actual_price > SPIKE_THRESHOLD).astype(int)
    predicted_spike = (predicted_price > SPIKE_THRESHOLD).astype(int)

    return {
        "rmse": mean_squared_error(actual_price, predicted_price) ** 0.5,
        "mae": mean_absolute_error(actual_price, predicted_price),
        "f1_spike": f1_score(actual_spike, predicted_spike, zero_division=0),
        "precision_spike": precision_score(actual_spike, predicted_spike, zero_division=0),
        "recall_spike": recall_score(actual_spike, predicted_spike, zero_division=0),
        "predicted_spike_rate": predicted_spike.mean(),
    }


def summarize_model(name, split_name, actual_price, predicted_price):
    row = {"model": name, "split": split_name}
    row.update(regression_and_spike_metrics(actual_price, predicted_price))
    return row


def make_arch_design(frame, features):
    X = frame[features].copy()
    continuous_in_arch = [c for c in features if c not in ["is_weekend", "is_stampede"]]
    X[continuous_in_arch] = X[continuous_in_arch] / ARCH_SCALE
    return X, continuous_in_arch
"""
    ),
    code(
        """# 7. Direct OLS benchmark
# The mean equation directly predicts the 2-hour price change using variables known at time t.

ols = LinearRegression()
ols.fit(train[FINAL_FEATURES], train["target_change_2h"])

validation_pred_ols = validation["ACTUAL_POOL_PRICE"] + ols.predict(validation[FINAL_FEATURES])
test_pred_ols = test["ACTUAL_POOL_PRICE"] + ols.predict(test[FINAL_FEATURES])

results_rows = []
results_rows.append(summarize_model("OLS direct", "validation", validation["price_t_plus_2"], validation_pred_ols))
results_rows.append(summarize_model("OLS direct", "test", test["price_t_plus_2"], test_pred_ols))

pd.DataFrame(results_rows).tail(2)
"""
    ),
    code(
        """# 8. Direct LS-GARCH and LS-GJR-GARCH
# We use the same direct mean equation, but estimate a conditional-volatility model for the residuals.
# This keeps the regressors limited to information available at time t.

def fit_arch_ls(sample, features, vol="GARCH", o=0):
    X, _ = make_arch_design(sample, features)
    y = sample["target_change_2h"] / ARCH_SCALE
    model = arch_model(
        y,
        x=X,
        mean="LS",
        vol=vol,
        p=1,
        o=o,
        q=1,
        dist="t",
        rescale=False,
    )
    return model.fit(disp="off", show_warning=False, options={"maxiter": 300})


def predict_arch_mean(result, sample, features):
    X, _ = make_arch_design(sample, features)
    params = result.params
    mean_scaled = np.full(len(X), params["Const"])
    for col in features:
        mean_scaled += params[col] * X[col].to_numpy()
    return sample["ACTUAL_POOL_PRICE"].to_numpy() + ARCH_SCALE * mean_scaled


garch_train = fit_arch_ls(train, FINAL_FEATURES, vol="GARCH", o=0)
validation_pred_garch = predict_arch_mean(garch_train, validation, FINAL_FEATURES)
results_rows.append(summarize_model("LS-GARCH", "validation", validation["price_t_plus_2"], validation_pred_garch))

garch_pretest = fit_arch_ls(pretest, FINAL_FEATURES, vol="GARCH", o=0)
test_pred_garch = predict_arch_mean(garch_pretest, test, FINAL_FEATURES)
results_rows.append(summarize_model("LS-GARCH", "test", test["price_t_plus_2"], test_pred_garch))

gjr_train = fit_arch_ls(train, FINAL_FEATURES, vol="GARCH", o=1)
validation_pred_gjr = predict_arch_mean(gjr_train, validation, FINAL_FEATURES)
results_rows.append(summarize_model("LS-GJR-GARCH", "validation", validation["price_t_plus_2"], validation_pred_gjr))

gjr_pretest = fit_arch_ls(pretest, FINAL_FEATURES, vol="GARCH", o=1)
test_pred_gjr = predict_arch_mean(gjr_pretest, test, FINAL_FEATURES)
results_rows.append(summarize_model("LS-GJR-GARCH", "test", test["price_t_plus_2"], test_pred_gjr))

pd.DataFrame(results_rows).tail(4)
"""
    ),
    code(
        """# 9. Markov-switching regime model (exploratory block)
# A full multivariate MS-VAR is not available as a clean local estimator in this environment.
# So this section fits a univariate Markov-switching model to the direct 2-hour price-change target.
# We treat it as a regime-identification block, not as the main out-of-sample benchmark.

markov_model = MarkovRegression(
    train["target_change_2h"],
    k_regimes=2,
    trend="c",
    switching_variance=True,
)
markov_result = markov_model.fit(disp=False)

regime_summary = pd.DataFrame(
    {
        "parameter": markov_result.params.index,
        "value": markov_result.params.values,
    }
)
regime_summary.to_csv(OUTPUT_DIR / "markov_regime_summary.csv", index=False)
regime_summary
"""
    ),
    code(
        """# 10. Visualize the estimated regimes on the training sample
regime_probs = markov_result.smoothed_marginal_probabilities

fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
axes[0].plot(train["datetime"], train["ACTUAL_POOL_PRICE"], color="tab:blue", linewidth=1.0)
axes[0].set_title("Training sample price series")
axes[0].set_ylabel("Pool price")

axes[1].plot(train["datetime"], regime_probs[1], color="tab:red", linewidth=1.0)
axes[1].set_title("Smoothed probability of regime 1")
axes[1].set_ylabel("Probability")
axes[1].set_xlabel("Datetime")

plt.tight_layout()
plt.savefig(FIG_DIR / "markov_regimes_train.png", dpi=150, bbox_inches="tight")
plt.show()
"""
    ),
    code(
        """# 11. VAR on a stationary multivariate system
# We forecast price_t_plus_2 indirectly by forecasting the next two hourly price changes.

VAR_SYSTEM = [
    "price_change_1h",
    "ACTUAL_AIL_change_1h",
    "wind_total_change_1h",
    "gas_total_change_1h",
]


def fit_var(sample):
    model = VAR(sample[VAR_SYSTEM])
    return model.fit(maxlags=4, ic="aic")


def predict_var_price(var_result, history_frame, eval_frame):
    predictions = []
    combined = pd.concat([history_frame, eval_frame], axis=0)
    k_ar = var_result.k_ar

    for idx in eval_frame.index:
        history = combined.loc[:idx, VAR_SYSTEM]
        two_step_forecast = var_result.forecast(history.values[-k_ar:], steps=2)
        predicted_change_2h = two_step_forecast[:, 0].sum()
        predicted_price = combined.loc[idx, "ACTUAL_POOL_PRICE"] + predicted_change_2h
        predictions.append(predicted_price)

    return np.array(predictions)


var_train = fit_var(train)
validation_pred_var = predict_var_price(var_train, train, validation)
results_rows.append(summarize_model("VAR", "validation", validation["price_t_plus_2"], validation_pred_var))

var_pretest = fit_var(pretest)
test_pred_var = predict_var_price(var_pretest, pretest, test)
results_rows.append(summarize_model("VAR", "test", test["price_t_plus_2"], test_pred_var))

pd.DataFrame(results_rows).tail(2)
"""
    ),
    code(
        """# 12. Bivariate BEKK / GARCH tuning
# This is the real multivariate volatility block.
# The main system is price + load, but we also try a small number of nearby candidates.
# Candidates are ranked by validation F1 after converting predicted price into a spike flag.

BEKK_CANDIDATES = [
    {
        "name": "BEKK price+load studentt arma 1000",
        "system": ["price_change_1h", "ACTUAL_AIL_change_1h"],
        "distribution": "studentt",
        "mean": "arma",
        "fit_window": 1000,
    },
    {
        "name": "BEKK price+load studentt zero 1000",
        "system": ["price_change_1h", "ACTUAL_AIL_change_1h"],
        "distribution": "studentt",
        "mean": "zero",
        "fit_window": 1000,
    },
    {
        "name": "BEKK price+load normal arma 1000",
        "system": ["price_change_1h", "ACTUAL_AIL_change_1h"],
        "distribution": "normal",
        "mean": "arma",
        "fit_window": 1000,
    },
    {
        "name": "BEKK price+load studentt arma 750",
        "system": ["price_change_1h", "ACTUAL_AIL_change_1h"],
        "distribution": "studentt",
        "mean": "arma",
        "fit_window": 750,
    },
    {
        "name": "BEKK price+wind studentt arma 1000",
        "system": ["price_change_1h", "wind_total_change_1h"],
        "distribution": "studentt",
        "mean": "arma",
        "fit_window": 1000,
    },
]


def fit_bekk(sample, candidate):
    system = candidate["system"]
    fit_window = candidate["fit_window"]
    fit_history = sample[system].tail(fit_window).to_numpy(dtype=np.float32) / BEKK_SCALE
    model = mvarch.model_factory(
        distribution=candidate["distribution"],
        mean=candidate["mean"],
        univariate="none",
        constraint="none",
        multivariate="mvarch",
    )
    model.tune_all = True
    model.fit(torch.tensor(fit_history))
    return model


def predict_bekk_price(model, history_frame, eval_frame, system):
    combined = pd.concat([history_frame.tail(BEKK_PRED_HISTORY), eval_frame], axis=0)
    predictions = []

    for idx in eval_frame.index:
        history = combined.loc[: idx - 1, system].tail(BEKK_PRED_HISTORY).to_numpy(dtype=np.float32) / BEKK_SCALE
        _, _, mean_1, _, _, _ = model.predict(torch.tensor(history))
        history_plus_1 = np.vstack([history, mean_1.detach().numpy()])
        _, _, mean_2, _, _, _ = model.predict(torch.tensor(history_plus_1))

        predicted_price = combined.loc[idx, "ACTUAL_POOL_PRICE"] + BEKK_SCALE * (
            float(mean_1.detach().numpy()[0]) + float(mean_2.detach().numpy()[0])
        )
        predictions.append(predicted_price)

    return np.array(predictions)


bekk_tuning_rows = []
best_bekk_candidate = None
best_bekk_model = None
best_bekk_validation_pred = None
best_bekk_f1 = -1.0

for candidate in BEKK_CANDIDATES:
    model = fit_bekk(train, candidate)
    validation_pred = predict_bekk_price(model, train, validation, candidate["system"])
    metrics_row = summarize_model(candidate["name"], "validation", validation["price_t_plus_2"], validation_pred)
    metrics_row["system"] = " + ".join(candidate["system"])
    metrics_row["distribution"] = candidate["distribution"]
    metrics_row["mean_model"] = candidate["mean"]
    metrics_row["fit_window"] = candidate["fit_window"]
    bekk_tuning_rows.append(metrics_row)

    if metrics_row["f1_spike"] > best_bekk_f1:
        best_bekk_f1 = metrics_row["f1_spike"]
        best_bekk_candidate = candidate
        best_bekk_model = model
        best_bekk_validation_pred = validation_pred


bekk_tuning_results = pd.DataFrame(bekk_tuning_rows).sort_values("f1_spike", ascending=False).reset_index(drop=True)
bekk_tuning_results.to_csv(OUTPUT_DIR / "bekk_tuning_results.csv", index=False)

results_rows.append(
    summarize_model("Bivariate BEKK", "validation", validation["price_t_plus_2"], best_bekk_validation_pred)
)

best_bekk_pretest = fit_bekk(pretest, best_bekk_candidate)
test_pred_bekk = predict_bekk_price(best_bekk_pretest, pretest, test, best_bekk_candidate["system"])
results_rows.append(summarize_model("Bivariate BEKK", "test", test["price_t_plus_2"], test_pred_bekk))

bekk_tuning_results
"""
    ),
    code(
        """# 13. Best BEKK specification
pd.DataFrame([best_bekk_candidate])
"""
    ),
    code(
        """# 14. Final comparison table
comparison = pd.DataFrame(results_rows).sort_values(["split", "f1_spike"], ascending=[True, False]).reset_index(drop=True)
comparison.to_csv(OUTPUT_DIR / "econometric_model_comparison.csv", index=False)
comparison
"""
    ),
    code(
        """# 15. Quick comparison plot on the test set
plot_df = comparison[comparison["split"] == "test"].copy()

plt.figure(figsize=(9, 4))
sns.barplot(data=plot_df, x="model", y="f1_spike", palette="crest")
plt.title("Derived spike F1 on the test window")
plt.ylabel("F1-score from predicted price > 200")
plt.xlabel("")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(FIG_DIR / "test_f1_from_price_predictions.png", dpi=150, bbox_inches="tight")
plt.show()
"""
    ),
    code(
        """# 16. Save prediction files for the main models
prediction_export = pd.DataFrame(
    {
        "datetime": test["datetime"].values,
        "actual_price_t_plus_2": test["price_t_plus_2"].values,
        "actual_spike_t_plus_2": test["spike_t_plus_2"].astype(int).values,
        "pred_ols": test_pred_ols,
        "pred_ls_garch": test_pred_garch,
        "pred_ls_gjr_garch": test_pred_gjr,
        "pred_var": test_pred_var,
        "pred_bivariate_bekk": test_pred_bekk,
    }
)
prediction_export.to_csv(OUTPUT_DIR / "test_price_predictions.csv", index=False)

print("Saved files:")
print("-", OUTPUT_DIR / "econometric_model_comparison.csv")
print("-", OUTPUT_DIR / "bekk_tuning_results.csv")
print("-", OUTPUT_DIR / "test_price_predictions.csv")
print("-", OUTPUT_DIR / "stationarity_summary.csv")
print("-", OUTPUT_DIR / "vif_summary.csv")
print("-", FIG_DIR / "markov_regimes_train.png")
print("-", FIG_DIR / "test_f1_from_price_predictions.png")
"""
    ),
]

nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12"},
}

NOTEBOOK_PATH.write_text(nbf.writes(nb), encoding="utf-8")
print(f"Wrote {NOTEBOOK_PATH}")
