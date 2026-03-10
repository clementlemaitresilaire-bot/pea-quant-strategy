from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.ml.config import get_ml_config, metrics_path, model_path
from src.ml.datasets import (
    COMMON_SIGNAL_FEATURES,
    REBALANCE_FEATURES,
    build_deployment_dataset,
    build_rebalance_dataset,
    build_trade_quality_dataset,
)

CATEGORICAL_SIGNAL_FEATURES = ["bucket", "sector"]
NUMERIC_SIGNAL_FEATURES = [c for c in COMMON_SIGNAL_FEATURES if c not in CATEGORICAL_SIGNAL_FEATURES]

CATEGORICAL_REBALANCE_FEATURES = ["bucket", "sector", "action_side"]
NUMERIC_REBALANCE_FEATURES = [c for c in REBALANCE_FEATURES if c not in CATEGORICAL_REBALANCE_FEATURES]

DEPLOYMENT_FEATURES = [
    "breadth_ma200",
    "breadth_slope",
    "benchmark_ret_3m",
    "benchmark_ret_6m",
    "benchmark_mom12_ex1",
    "avg_ret_3m",
    "avg_ret_6m",
    "avg_mom12_ex1",
    "median_vol20",
    "cross_sectional_dispersion",
]


def _make_logit_pipeline(
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
                        ("scaler", StandardScaler()),
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

    model = LogisticRegression(
        max_iter=2000,
        C=1.4,
        class_weight="balanced",
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
        out[col] = out[col].astype(str)

    return out[numeric_cols + categorical_cols].copy()


def _walk_forward_oof(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> tuple[np.ndarray, float]:
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

        if train_df[target_col].nunique() < 2 or test_df.empty:
            continue

        X_train = _clean_feature_frame(train_df[feature_cols], numeric_cols, categorical_cols)
        y_train = train_df[target_col].astype(int)
        X_test = _clean_feature_frame(test_df[feature_cols], numeric_cols, categorical_cols)

        pipe = _make_logit_pipeline(numeric_cols, categorical_cols)
        pipe.fit(X_train, y_train)

        oof[test_df.index] = pipe.predict_proba(X_test)[:, 1]

    valid = ~np.isnan(oof)
    auc = np.nan
    if valid.sum() > 0 and data.loc[valid, target_col].nunique() >= 2:
        auc = roc_auc_score(data.loc[valid, target_col], oof[valid])

    return oof, float(auc) if not np.isnan(auc) else float("nan")


def _fit_and_save(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    name: str,
    auc_floor: float,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> None:
    data = df.dropna(subset=["date", target_col]).copy()
    data[target_col] = pd.to_numeric(data[target_col], errors="coerce").fillna(0).astype(int)

    oof, auc = _walk_forward_oof(
        data,
        feature_cols,
        target_col,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
    )

    enabled = bool(np.isfinite(auc) and auc >= auc_floor)

    X = _clean_feature_frame(data[feature_cols], numeric_cols, categorical_cols)
    y = data[target_col].astype(int)

    pipe = _make_logit_pipeline(numeric_cols, categorical_cols)
    if y.nunique() >= 2:
        pipe.fit(X, y)

    artifact = {
        "name": name,
        "enabled": enabled,
        "oof_auc": auc,
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
                "oof_auc": auc,
                "n_samples": len(data),
                "positive_rate": float(y.mean()) if len(y) > 0 else np.nan,
            }
        ]
    )
    metrics_df.to_csv(metrics_path(name), index=False)


def train_all_models() -> None:
    cfg = get_ml_config()

    trade_df = build_trade_quality_dataset()
    deployment_df = build_deployment_dataset()
    rebalance_df = build_rebalance_dataset()

    _fit_and_save(
        df=trade_df,
        feature_cols=COMMON_SIGNAL_FEATURES,
        target_col="target",
        name="trade_quality_model",
        auc_floor=cfg.trade_auc_floor,
        numeric_cols=NUMERIC_SIGNAL_FEATURES,
        categorical_cols=CATEGORICAL_SIGNAL_FEATURES,
    )

    _fit_and_save(
        df=deployment_df,
        feature_cols=DEPLOYMENT_FEATURES,
        target_col="target",
        name="deployment_model",
        auc_floor=cfg.deployment_auc_floor,
        numeric_cols=DEPLOYMENT_FEATURES,
        categorical_cols=[],
    )

    _fit_and_save(
        df=rebalance_df,
        feature_cols=REBALANCE_FEATURES,
        target_col="target",
        name="rebalance_model",
        auc_floor=cfg.rebalance_auc_floor,
        numeric_cols=NUMERIC_REBALANCE_FEATURES,
        categorical_cols=CATEGORICAL_REBALANCE_FEATURES,
    )

    print("ML models trained.")
    print(model_path("trade_quality_model"))
    print(model_path("deployment_model"))
    print(model_path("rebalance_model"))


if __name__ == "__main__":
    train_all_models()