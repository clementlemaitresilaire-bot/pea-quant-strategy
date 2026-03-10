from __future__ import annotations

from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.ml.config import get_ml_config, is_ml_enabled, model_path


TRADE_MODEL_NAME = "trade_quality_model"
DEPLOYMENT_MODEL_NAME = "deployment_model"
REBALANCE_MODEL_NAME = "rebalance_model"


def _ensure_overlay_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    default_columns: dict[str, Any] = {
        "trade_quality_score": pd.NA,
        "trade_quality_proba": pd.NA,
        "trade_quality_label": pd.NA,
        "trade_quality_overlay_applied": False,
        "rebalance_overlay_applied": False,
        "rebalance_score": pd.NA,
        "rebalance_proba": pd.NA,
        "rebalance_label": pd.NA,
        "ml_trade_quality_edge": pd.NA,
        "ml_rebalance_edge": pd.NA,
    }

    for col, default_value in default_columns.items():
        if col not in out.columns:
            out[col] = default_value

    return out


def _safe_bool_series(series: pd.Series | Any, index: pd.Index) -> pd.Series:
    if isinstance(series, pd.Series):
        return series.fillna(False).astype(bool)
    return pd.Series(False, index=index)


def _safe_numeric_series(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return (
        pd.to_numeric(df[col], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(default)
        .astype(float)
    )


def _safe_text_series(df: pd.DataFrame, col: str, default: str = "") -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=object)
    return df[col].fillna(default).astype(str)


def _load_model_artifact(name: str) -> dict[str, Any] | None:
    path = model_path(name)
    if not path.exists():
        return None

    try:
        artifact = joblib.load(path)
    except Exception:
        return None

    if not isinstance(artifact, dict):
        return None
    if not bool(artifact.get("enabled", False)):
        return None
    if "model" not in artifact:
        return None

    return artifact


def _predict_value(artifact: dict[str, Any], features_df: pd.DataFrame) -> np.ndarray:
    feature_columns = list(artifact.get("feature_columns", []))
    numeric_columns = list(artifact.get("numeric_columns", []))
    categorical_columns = list(artifact.get("categorical_columns", []))

    X = features_df.copy()

    for col in feature_columns:
        if col not in X.columns:
            X[col] = 0.0 if col in numeric_columns else ""

    for col in numeric_columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    for col in categorical_columns:
        X[col] = X[col].fillna("").astype(str)

    X = X[feature_columns].copy()
    pred = artifact["model"].predict(X)
    return np.asarray(pred, dtype=float)


def _build_trade_quality_features(signal_df: pd.DataFrame, bucket: str) -> pd.DataFrame:
    out = pd.DataFrame(index=signal_df.index)

    bucket_lower = str(bucket).lower()

    raw_score = _safe_numeric_series(signal_df, "core_score")
    if raw_score.abs().sum() == 0.0:
        raw_score = _safe_numeric_series(signal_df, "opp_score_net")
    if raw_score.abs().sum() == 0.0:
        raw_score = _safe_numeric_series(signal_df, "score")

    out["bucket"] = bucket_lower
    out["sector"] = _safe_text_series(signal_df, "sector")
    out["raw_score"] = raw_score
    out["mom12_ex1"] = _safe_numeric_series(signal_df, "mom12_ex1")
    out["ret_6m"] = _safe_numeric_series(signal_df, "ret_6m")
    out["ret_3m"] = _safe_numeric_series(signal_df, "ret_3m")
    out["ret_20d"] = _safe_numeric_series(signal_df, "ret_20d")
    out["ret_10d"] = _safe_numeric_series(signal_df, "ret_10d")
    out["ret_5d"] = _safe_numeric_series(signal_df, "ret_5d")
    out["vol60"] = _safe_numeric_series(signal_df, "vol60")
    out["vol20"] = _safe_numeric_series(signal_df, "vol20", default=float(out["vol60"].median()) if len(out) else 0.0)
    out["dd_6m"] = _safe_numeric_series(signal_df, "dd_6m")
    out["dd_12m"] = _safe_numeric_series(signal_df, "dd_12m", default=float(out["dd_6m"].median()) if len(out) else 0.0)
    out["dist_ma200"] = _safe_numeric_series(signal_df, "dist_ma200")
    out["dist_ma20"] = _safe_numeric_series(signal_df, "dist_ma20")
    out["rsi14"] = _safe_numeric_series(signal_df, "rsi14", default=50.0)
    out["ma200_flag"] = _safe_bool_series(signal_df.get("ma200_flag", False), signal_df.index).astype(float)
    out["ma200_slope_flag"] = _safe_bool_series(signal_df.get("ma200_slope_flag", False), signal_df.index).astype(float)
    out["breadth_ma200"] = _safe_numeric_series(signal_df, "breadth_ma200")
    out["breadth_slope"] = _safe_numeric_series(signal_df, "breadth_slope")
    out["benchmark_ret_3m"] = _safe_numeric_series(signal_df, "benchmark_ret_3m")
    out["benchmark_ret_6m"] = _safe_numeric_series(signal_df, "benchmark_ret_6m")
    out["benchmark_mom12_ex1"] = _safe_numeric_series(signal_df, "benchmark_mom12_ex1")

    return out


def apply_trade_quality_overlay(signal_df: pd.DataFrame, bucket: str) -> pd.DataFrame:
    if signal_df is None:
        return signal_df
    if signal_df.empty:
        return _ensure_overlay_columns(signal_df)

    out = _ensure_overlay_columns(signal_df)

    if not is_ml_enabled():
        out["trade_quality_overlay_applied"] = False
        return out

    artifact = _load_model_artifact(TRADE_MODEL_NAME)
    if artifact is None:
        out["trade_quality_overlay_applied"] = False
        return out

    features = _build_trade_quality_features(out, bucket=bucket)
    edge = _predict_value(artifact, features)

    cfg = get_ml_config()
    clipped_edge = np.clip(edge, -0.05, 0.05)
    score_bump = cfg.trade_strength * clipped_edge

    out["trade_quality_proba"] = 0.50 + 10.0 * score_bump
    out["trade_quality_score"] = clipped_edge
    out["ml_trade_quality_edge"] = clipped_edge
    out["trade_quality_label"] = (clipped_edge > 0.0).astype(int)
    out["trade_quality_overlay_applied"] = True

    if "core_score" in out.columns:
        out["core_score"] = _safe_numeric_series(out, "core_score") + score_bump
    if "opp_score_net" in out.columns:
        out["opp_score_net"] = _safe_numeric_series(out, "opp_score_net") + 1.15 * score_bump
    if "opp_score_raw" in out.columns:
        out["opp_score_raw"] = _safe_numeric_series(out, "opp_score_raw") + 0.85 * score_bump
    if "score" in out.columns:
        out["score"] = _safe_numeric_series(out, "score") + score_bump

    bucket_lower = str(bucket).lower()
    edge_series = pd.Series(clipped_edge, index=out.index)

    if bucket_lower == "core" and "selected_core_final" in out.columns:
        mask_selected = out["selected_core_final"].fillna(False).astype(bool)
        weak_mask = mask_selected & (edge_series < -0.004)
        if weak_mask.any() and mask_selected.sum() - weak_mask.sum() >= 4:
            out.loc[weak_mask, "selected_core_final"] = False
            out.loc[weak_mask, "selection_reason"] = "ML_REJECT_NEGATIVE_NET_EDGE"

    if bucket_lower in {"opportunistic", "opp"} and "selected_opp_final" in out.columns:
        mask_selected = out["selected_opp_final"].fillna(False).astype(bool)
        weak_mask = mask_selected & (edge_series < -0.006)
        if weak_mask.any() and mask_selected.sum() - weak_mask.sum() >= 1:
            out.loc[weak_mask, "selected_opp_final"] = False
            out.loc[weak_mask, "selection_reason"] = "ML_REJECT_NEGATIVE_NET_EDGE"

    return out


def maybe_apply_trade_quality_overlay(
    signal_df: pd.DataFrame,
    bucket: str,
    enabled: bool = True,
) -> pd.DataFrame:
    if not enabled or not is_ml_enabled():
        return _ensure_overlay_columns(signal_df)
    return apply_trade_quality_overlay(signal_df=signal_df, bucket=bucket)


def _build_rebalance_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    out["bucket"] = _safe_text_series(df, "bucket")
    out["sector"] = _safe_text_series(df, "sector")
    out["raw_score"] = _safe_numeric_series(df, "score")
    out["weight_gap_proxy"] = _safe_numeric_series(df, "weight_gap").abs()

    effective_order_value = _safe_numeric_series(df, "effective_order_value")
    total_cost_est = _safe_numeric_series(df, "total_cost_est")
    out["cost_ratio_proxy"] = np.where(effective_order_value > 0, total_cost_est / effective_order_value, 0.0)

    action = _safe_text_series(df, "action", default="HOLD").str.upper()
    out["action_side"] = np.where(action == "SELL", "SELL", "BUY")

    out["mom12_ex1"] = _safe_numeric_series(df, "mom12_ex1")
    out["ret_6m"] = _safe_numeric_series(df, "ret_6m")
    out["ret_3m"] = _safe_numeric_series(df, "ret_3m")
    out["ret_20d"] = _safe_numeric_series(df, "ret_20d")
    out["vol60"] = _safe_numeric_series(df, "vol60")
    out["vol20"] = _safe_numeric_series(df, "vol20")
    out["dd_6m"] = _safe_numeric_series(df, "dd_6m")
    out["dist_ma200"] = _safe_numeric_series(df, "dist_ma200")
    out["dist_ma20"] = _safe_numeric_series(df, "dist_ma20")
    out["rsi14"] = _safe_numeric_series(df, "rsi14", default=50.0)
    out["breadth_ma200"] = _safe_numeric_series(df, "breadth_ma200")
    out["breadth_slope"] = _safe_numeric_series(df, "breadth_slope")
    out["benchmark_ret_3m"] = _safe_numeric_series(df, "benchmark_ret_3m")
    out["benchmark_ret_6m"] = _safe_numeric_series(df, "benchmark_ret_6m")

    return out


def apply_rebalance_overlay(df: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
    if df is None:
        return df
    if df.empty:
        return _ensure_overlay_columns(df)

    out = _ensure_overlay_columns(df)

    if not is_ml_enabled():
        out["rebalance_overlay_applied"] = False
        return out

    artifact = _load_model_artifact(REBALANCE_MODEL_NAME)
    if artifact is None:
        out["rebalance_overlay_applied"] = False
        return out

    features = _build_rebalance_features(out)
    edge = _predict_value(artifact, features)

    cfg = get_ml_config()

    out["rebalance_proba"] = 0.50 + 10.0 * np.clip(edge, -0.05, 0.05)
    out["rebalance_score"] = np.clip(edge, -0.05, 0.05)
    out["ml_rebalance_edge"] = edge
    out["rebalance_label"] = (edge > 0.0).astype(int)
    out["rebalance_overlay_applied"] = True

    execute = out["execute"].fillna(False).astype(bool)
    action = _safe_text_series(out, "action", default="HOLD").str.upper()
    alpha_surplus = _safe_numeric_series(out, "alpha_surplus")
    weight_gap_abs = _safe_numeric_series(out, "weight_gap").abs()

    marginal_trade = execute & (action != "HOLD") & (alpha_surplus < 0.03) & (weight_gap_abs < 0.06)
    reject_mask = marginal_trade & (
        (pd.Series(edge, index=out.index) < cfg.rebalance_edge_floor)
        | (alpha_surplus < cfg.rebalance_alpha_surplus_floor)
    )

    if reject_mask.any():
        out.loc[reject_mask, "execute"] = False
        out.loc[reject_mask, "decision_reason"] = "ML_REJECT_NEGATIVE_NET_EDGE"

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


def _build_deployment_feature_row(
    regime_state,
    latest_features_df: pd.DataFrame | None,
    universe_df: pd.DataFrame | None,
) -> pd.DataFrame:
    latest = latest_features_df.copy() if latest_features_df is not None else pd.DataFrame()
    if latest.empty:
        latest = pd.DataFrame(index=[0])

    breadth_ma200 = float(getattr(regime_state, "breadth_ma200", 0.0))
    breadth_slope = float(getattr(regime_state, "breadth_slope", 0.0))
    avg_ret_3m = float(getattr(regime_state, "avg_ret_3m", 0.0))
    avg_ret_6m = float(getattr(regime_state, "avg_ret_6m", 0.0))
    avg_mom12_ex1 = float(getattr(regime_state, "avg_mom12_ex1", 0.0))

    bench_subset = latest.copy()
    if universe_df is not None and not universe_df.empty and "ticker" in latest.columns and "ticker" in universe_df.columns:
        if "benchmark_member" in universe_df.columns and "active" in universe_df.columns:
            members = universe_df.loc[
                (universe_df["benchmark_member"] == True) & (universe_df["active"] == True),
                ["ticker"],
            ].drop_duplicates()
            candidate = latest.merge(members, on="ticker", how="inner")
            if not candidate.empty:
                bench_subset = candidate

    benchmark_ret_3m = float(pd.to_numeric(bench_subset.get("ret_3m", 0.0), errors="coerce").fillna(0.0).mean())
    benchmark_ret_6m = float(pd.to_numeric(bench_subset.get("ret_6m", 0.0), errors="coerce").fillna(0.0).mean())
    benchmark_mom12_ex1 = float(pd.to_numeric(bench_subset.get("mom12_ex1", 0.0), errors="coerce").fillna(0.0).mean())
    median_vol20 = float(pd.to_numeric(latest.get("vol20", 0.0), errors="coerce").fillna(0.0).median())
    cross_sectional_dispersion = float(pd.to_numeric(latest.get("ret_3m", 0.0), errors="coerce").fillna(0.0).std())
    if not np.isfinite(cross_sectional_dispersion):
        cross_sectional_dispersion = 0.0

    return pd.DataFrame(
        [
            {
                "breadth_ma200": breadth_ma200,
                "breadth_slope": breadth_slope,
                "benchmark_ret_3m": benchmark_ret_3m,
                "benchmark_ret_6m": benchmark_ret_6m,
                "benchmark_mom12_ex1": benchmark_mom12_ex1,
                "avg_ret_3m": avg_ret_3m,
                "avg_ret_6m": avg_ret_6m,
                "avg_mom12_ex1": avg_mom12_ex1,
                "median_vol20": median_vol20,
                "cross_sectional_dispersion": cross_sectional_dispersion,
            }
        ]
    )


def get_live_deployment_adjustment(
    regime_state=None,
    latest_features_df: pd.DataFrame | None = None,
    universe_df: pd.DataFrame | None = None,
    *args,
    **kwargs,
) -> dict[str, float | str]:
    neutral = {
        "deployment_ml_proba": 0.50,
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
        "live_score": 0.0,
        "confidence": 0.0,
    }

    if not is_ml_enabled() or regime_state is None:
        return neutral

    artifact = _load_model_artifact(DEPLOYMENT_MODEL_NAME)
    if artifact is None:
        return neutral

    features = _build_deployment_feature_row(
        regime_state=regime_state,
        latest_features_df=latest_features_df,
        universe_df=universe_df,
    )
    edge = float(_predict_value(artifact, features)[0])

    cfg = get_ml_config()

    clipped_edge = float(np.clip(edge, -0.03, 0.03))
    normalized = (clipped_edge + 0.03) / 0.06

    core_multiplier = float(np.interp(normalized, [0.0, 1.0], [cfg.core_deployment_floor, cfg.core_deployment_cap]))
    opp_multiplier = float(np.interp(normalized, [0.0, 1.0], [cfg.opp_deployment_floor, cfg.opp_deployment_cap]))

    if clipped_edge >= 0.006:
        regime = "ml_risk_on"
    elif clipped_edge <= -0.006:
        regime = "ml_risk_off"
    else:
        regime = "ml_neutral"

    confidence = min(1.0, abs(clipped_edge) / 0.03)

    return {
        "deployment_ml_proba": 0.50 + 10.0 * clipped_edge,
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
        "live_score": clipped_edge,
        "confidence": confidence,
    }