from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.ml.config import is_ml_enabled, model_path


def _ensure_overlay_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    default_columns: dict[str, Any] = {
        "trade_quality_score": pd.NA,
        "trade_quality_signal": pd.NA,
        "trade_quality_proba": pd.NA,   # legacy compatibility
        "trade_quality_label": pd.NA,
        "trade_quality_overlay_applied": False,
        "rebalance_overlay_applied": False,
        "rebalance_score": pd.NA,
        "rebalance_signal": pd.NA,
        "rebalance_proba": pd.NA,       # legacy compatibility
        "rebalance_label": pd.NA,
        "ml_trade_quality_edge": pd.NA,
        "ml_rebalance_edge": pd.NA,
        "deployment_ml_signal": 0.0,
        "deployment_ml_proba": 0.5,     # legacy compatibility
        "core_deployment_multiplier_ml": 1.0,
        "opp_deployment_multiplier_ml": 1.0,
        "deployment_regime_ml": "neutral",
    }

    for col, default_value in default_columns.items():
        if col not in out.columns:
            out[col] = default_value

    return out


def _load_model_artifact(name: str) -> dict | None:
    path = model_path(name)
    if not Path(path).exists():
        return None

    try:
        artifact = joblib.load(path)
    except Exception:
        return None

    if not isinstance(artifact, dict):
        return None

    if not artifact.get("enabled", False):
        return None

    if "model" not in artifact:
        return None

    return artifact


def _prepare_features_for_model(
    df: pd.DataFrame,
    artifact: dict,
) -> pd.DataFrame:
    feature_columns = artifact.get("feature_columns", [])
    numeric_columns = artifact.get("numeric_columns", [])
    categorical_columns = artifact.get("categorical_columns", [])

    work = df.copy()

    for col in numeric_columns:
        if col not in work.columns:
            work[col] = 0.0
        work[col] = pd.to_numeric(work[col], errors="coerce")

    for col in categorical_columns:
        if col not in work.columns:
            work[col] = ""
        work[col] = work[col].astype(str)

    for col in feature_columns:
        if col not in work.columns:
            work[col] = 0.0 if col in numeric_columns else ""

    return work[feature_columns].copy()


def _edge_to_signal(edge: pd.Series, clip_low: float, clip_high: float) -> pd.Series:
    clipped = pd.to_numeric(edge, errors="coerce").clip(lower=clip_low, upper=clip_high)
    signal = 0.5 + 10.0 * clipped
    return signal.clip(lower=0.0, upper=1.0)


def apply_trade_quality_overlay(signal_df: pd.DataFrame, bucket: str) -> pd.DataFrame:
    if signal_df is None:
        return signal_df
    if signal_df.empty:
        return _ensure_overlay_columns(signal_df)

    out = _ensure_overlay_columns(signal_df)
    artifact = _load_model_artifact("trade_quality_model")
    if artifact is None:
        return out

    try:
        X = _prepare_features_for_model(out, artifact)
        raw_pred = artifact["model"].predict(X)
        edge = pd.Series(raw_pred, index=out.index, dtype=float).clip(-0.05, 0.05)
        signal = _edge_to_signal(edge, -0.05, 0.05)

        out["ml_trade_quality_edge"] = edge
        out["trade_quality_signal"] = signal
        out["trade_quality_proba"] = signal  # legacy compatibility
        out["trade_quality_score"] = edge
        out["trade_quality_label"] = (edge > 0.0).astype(int)
        out["trade_quality_overlay_applied"] = True

        score_bump = edge

        if "core_score" in out.columns:
            out["core_score"] = pd.to_numeric(out["core_score"], errors="coerce").fillna(0.0) + score_bump

        if "opp_score_net" in out.columns:
            out["opp_score_net"] = pd.to_numeric(out["opp_score_net"], errors="coerce").fillna(0.0) + score_bump

        if "opp_score_raw" in out.columns:
            out["opp_score_raw"] = pd.to_numeric(out["opp_score_raw"], errors="coerce").fillna(0.0) + 0.5 * score_bump

        if "score" in out.columns:
            out["score"] = pd.to_numeric(out["score"], errors="coerce").fillna(0.0) + score_bump

        if "selected_core_final" in out.columns:
            out.loc[edge < -0.01, "selected_core_final"] = False

        if "selected_opp_final" in out.columns:
            out.loc[edge < -0.015, "selected_opp_final"] = False

        return out

    except Exception:
        return out


def maybe_apply_trade_quality_overlay(
    signal_df: pd.DataFrame,
    bucket: str,
    enabled: bool = True,
) -> pd.DataFrame:
    if not enabled or not is_ml_enabled():
        return _ensure_overlay_columns(signal_df)
    return apply_trade_quality_overlay(signal_df=signal_df, bucket=bucket)


def apply_rebalance_overlay(df: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
    if df is None:
        return df
    if df.empty:
        return _ensure_overlay_columns(df)

    out = _ensure_overlay_columns(df)
    artifact = _load_model_artifact("rebalance_model")
    if artifact is None:
        return out

    try:
        X = _prepare_features_for_model(out, artifact)
        raw_pred = artifact["model"].predict(X)
        edge = pd.Series(raw_pred, index=out.index, dtype=float).clip(-0.05, 0.05)
        signal = _edge_to_signal(edge, -0.05, 0.05)

        out["ml_rebalance_edge"] = edge
        out["rebalance_signal"] = signal
        out["rebalance_proba"] = signal  # legacy compatibility
        out["rebalance_score"] = edge
        out["rebalance_label"] = (edge > 0.0).astype(int)
        out["rebalance_overlay_applied"] = True

        if "execute" in out.columns and "alpha_surplus" in out.columns:
            alpha_surplus = pd.to_numeric(out["alpha_surplus"], errors="coerce").fillna(0.0)

            reject_mask = (
                (out["execute"] == True)
                & (alpha_surplus < 0.02)
                & (edge < 0.0)
            )

            out.loc[reject_mask, "execute"] = False
            out.loc[reject_mask, "decision_reason"] = "ML_REJECT_NEGATIVE_NET_EDGE"

        return out

    except Exception:
        return out


def maybe_apply_rebalance_overlay(
    df: pd.DataFrame,
    enabled: bool = True,
    *args,
    **kwargs,
) -> pd.DataFrame:
    if not enabled or not is_ml_enabled():
        return _ensure_overlay_columns(df)
    return apply_rebalance_overlay(df=df, *args, **kwargs)


def get_live_deployment_adjustment(feature_df: pd.DataFrame | None = None, *args, **kwargs) -> dict[str, float | str]:
    artifact = _load_model_artifact("deployment_model")
    if artifact is None or feature_df is None or len(feature_df) == 0:
        return {
            "core_multiplier": 1.0,
            "opp_multiplier": 1.0,
            "opportunistic_multiplier": 1.0,
            "monetary_multiplier": 1.0,
            "mon_multiplier": 1.0,
            "core_additive": 0.0,
            "opp_additive": 0.0,
            "opportunistic_additive": 0.0,
            "monetary_additive": 0.0,
            "mon_additive": 0.0,
            "deployment_regime": "neutral",
            "deployment_signal": 0.0,
            "deployment_ml_signal": 0.0,
            "confidence": 0.0,
        }

    try:
        latest = feature_df.tail(1).copy()
        X = _prepare_features_for_model(latest, artifact)
        raw_edge = float(artifact["model"].predict(X)[0])
        clipped_edge = float(np.clip(raw_edge, -0.03, 0.03))
        signal = float(np.clip(0.5 + 10.0 * clipped_edge, 0.0, 1.0))

        core_multiplier = 1.0 + 1.5 * clipped_edge
        opp_multiplier = 1.0 + 3.0 * clipped_edge

        core_multiplier = float(np.clip(core_multiplier, 0.93, 1.10))
        opp_multiplier = float(np.clip(opp_multiplier, 0.88, 1.18))

        if clipped_edge > 0.01:
            regime = "ml_risk_on"
        elif clipped_edge < -0.01:
            regime = "ml_risk_off"
        else:
            regime = "ml_neutral"

        return {
            "core_multiplier": core_multiplier,
            "opp_multiplier": opp_multiplier,
            "opportunistic_multiplier": opp_multiplier,
            "monetary_multiplier": 1.0,
            "mon_multiplier": 1.0,
            "core_additive": 0.0,
            "opp_additive": 0.0,
            "opportunistic_additive": 0.0,
            "monetary_additive": 0.0,
            "mon_additive": 0.0,
            "deployment_regime": regime,
            "deployment_signal": clipped_edge,
            "deployment_ml_signal": signal,
            "confidence": abs(clipped_edge),
        }

    except Exception:
        return {
            "core_multiplier": 1.0,
            "opp_multiplier": 1.0,
            "opportunistic_multiplier": 1.0,
            "monetary_multiplier": 1.0,
            "mon_multiplier": 1.0,
            "core_additive": 0.0,
            "opp_additive": 0.0,
            "opportunistic_additive": 0.0,
            "monetary_additive": 0.0,
            "mon_additive": 0.0,
            "deployment_regime": "neutral",
            "deployment_signal": 0.0,
            "deployment_ml_signal": 0.0,
            "confidence": 0.0,
        }