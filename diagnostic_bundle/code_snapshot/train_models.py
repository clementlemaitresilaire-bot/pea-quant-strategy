from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.ml.config import get_ml_config, metrics_path, model_path
from src.ml.datasets import (
    COMMON_SIGNAL_FEATURES,
    DEPLOYMENT_FEATURES,
    REBALANCE_FEATURES,
    build_deployment_dataset,
    build_rebalance_dataset,
    build_trade_quality_dataset,
)

CATEGORICAL_SIGNAL_FEATURES = ["bucket", "sector"]
NUMERIC_SIGNAL_FEATURES = [c for c in COMMON_SIGNAL_FEATURES if c not in CATEGORICAL_SIGNAL_FEATURES]

CATEGORICAL_REBALANCE_FEATURES = ["bucket", "sector", "action_side"]
NUMERIC_REBALANCE_FEATURES = [c for c in REBALANCE_FEATURES if c not in CATEGORICAL_REBALANCE_FEATURES]


def _make_regression_pipeline(
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                numeric_cols,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            ),
        ]
    )

    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.04,
        max_iter=250,
        max_depth=3,
        min_samples_leaf=30,
        l2_regularization=0.15,
        random_state=42,
    )

    return Pipeline([("preprocessor", pre), ("model", model)])


def _clean_feature_frame(
    df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> pd.DataFrame:
    out = df.copy()

    for col in numeric_cols:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce")

    for col in categorical_cols:
        if col not in out.columns:
            out[col] = ""
        out[col] = out[col].fillna("").astype(str)

    return out[numeric_cols + categorical_cols].copy()


def _information_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 3:
        return float("nan")

    y_true = pd.Series(y_true).astype(float)
    y_pred = pd.Series(y_pred).astype(float)

    if y_true.nunique(dropna=True) < 2 or y_pred.nunique(dropna=True) < 2:
        return float("nan")

    return float(y_true.corr(y_pred, method="spearman"))


def _walk_forward_oof_regression(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> tuple[np.ndarray, float, float]:
    cfg = get_ml_config()

    data = df.copy()
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values("date").reset_index(drop=True)
    months = pd.Index(sorted(data["date"].dt.to_period("M").unique()))

    if len(months) < (cfg.min_train_months + cfg.validation_block_months):
        raise ValueError("Not enough monthly history for ML walk-forward training.")

    oof = np.full(len(data), np.nan)

    start = cfg.min_train_months
    step = cfg.validation_block_months

    for i in range(start, len(months) - step + 1, step):
        train_months = months[:i]
        test_months = months[i : i + step]

        train_mask = data["date"].dt.to_period("M").isin(train_months)
        test_mask = data["date"].dt.to_period("M").isin(test_months)

        train_df = data.loc[train_mask].copy()
        test_df = data.loc[test_mask].copy()

        if train_df.empty or test_df.empty:
            continue
        if train_df[target_col].notna().sum() < 100:
            continue

        X_train = _clean_feature_frame(train_df[feature_cols], numeric_cols, categorical_cols)
        y_train = pd.to_numeric(train_df[target_col], errors="coerce").astype(float)
        X_test = _clean_feature_frame(test_df[feature_cols], numeric_cols, categorical_cols)

        pipe = _make_regression_pipeline(numeric_cols, categorical_cols)
        pipe.fit(X_train, y_train)
        oof[test_df.index] = pipe.predict(X_test)

    valid = ~np.isnan(oof)
    ic = float("nan")
    rmse = float("nan")

    if valid.sum() > 10:
        y_valid = pd.to_numeric(data.loc[valid, target_col], errors="coerce").astype(float).to_numpy()
        pred_valid = oof[valid]
        ic = _information_coefficient(y_valid, pred_valid)
        rmse = float(np.sqrt(mean_squared_error(y_valid, pred_valid)))

    return oof, ic, rmse


def _fit_and_save_regression(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    name: str,
    ic_floor: float,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> None:
    data = df.dropna(subset=["date", target_col]).copy()
    data[target_col] = pd.to_numeric(data[target_col], errors="coerce")
    data = data.dropna(subset=[target_col]).copy()

    oof, ic, rmse = _walk_forward_oof_regression(
        data,
        feature_cols,
        target_col,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
    )

    enabled = bool(np.isfinite(ic) and ic >= ic_floor)

    X = _clean_feature_frame(data[feature_cols], numeric_cols, categorical_cols)
    y = pd.to_numeric(data[target_col], errors="coerce").astype(float)

    pipe = _make_regression_pipeline(numeric_cols, categorical_cols)
    pipe.fit(X, y)

    artifact = {
        "name": name,
        "enabled": enabled,
        "oof_ic": ic,
        "oof_rmse": rmse,
        "target_type": "regression",
        "feature_columns": feature_cols,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "model": pipe,
    }
    joblib.dump(artifact, model_path(name))

    metrics_df = pd.DataFrame(
        [
            {
                "model_name": name,
                "enabled": enabled,
                "oof_ic": ic,
                "oof_rmse": rmse,
                "n_samples": len(data),
                "target_mean": float(y.mean()) if len(y) > 0 else np.nan,
                "target_std": float(y.std()) if len(y) > 0 else np.nan,
            }
        ]
    )
    metrics_df.to_csv(metrics_path(name), index=False)


def train_all_models() -> None:
    cfg = get_ml_config()

    trade_df = build_trade_quality_dataset()
    deployment_df = build_deployment_dataset()
    rebalance_df = build_rebalance_dataset()

    _fit_and_save_regression(
        df=trade_df,
        feature_cols=COMMON_SIGNAL_FEATURES,
        target_col="target_value",
        name="trade_quality_model",
        ic_floor=cfg.trade_information_coeff_floor,
        numeric_cols=NUMERIC_SIGNAL_FEATURES,
        categorical_cols=CATEGORICAL_SIGNAL_FEATURES,
    )

    _fit_and_save_regression(
        df=deployment_df,
        feature_cols=DEPLOYMENT_FEATURES,
        target_col="target_value",
        name="deployment_model",
        ic_floor=cfg.deployment_information_coeff_floor,
        numeric_cols=DEPLOYMENT_FEATURES,
        categorical_cols=[],
    )

    _fit_and_save_regression(
        df=rebalance_df,
        feature_cols=REBALANCE_FEATURES,
        target_col="target_value",
        name="rebalance_model",
        ic_floor=cfg.rebalance_information_coeff_floor,
        numeric_cols=NUMERIC_REBALANCE_FEATURES,
        categorical_cols=CATEGORICAL_REBALANCE_FEATURES,
    )

    print("ML models trained.")
    print(model_path("trade_quality_model"))
    print(model_path("deployment_model"))
    print(model_path("rebalance_model"))


if __name__ == "__main__":
    train_all_models()