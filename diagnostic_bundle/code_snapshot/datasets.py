from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.price_features import compute_price_features
from src.io.loaders import load_universe
from src.io.market_data import load_all_price_data
from src.settings import load_config


COMMON_SIGNAL_FEATURES = [
    "bucket",
    "sector",
    "raw_score",
    "mom12_ex1",
    "ret_6m",
    "ret_3m",
    "ret_20d",
    "ret_10d",
    "ret_5d",
    "vol60",
    "vol20",
    "dd_6m",
    "dd_12m",
    "dist_ma200",
    "dist_ma20",
    "rsi14",
    "ma200_flag",
    "ma200_slope_flag",
    "breadth_ma200",
    "breadth_slope",
    "benchmark_ret_3m",
    "benchmark_ret_6m",
    "benchmark_mom12_ex1",
]

REBALANCE_FEATURES = [
    "bucket",
    "sector",
    "raw_score",
    "weight_gap_proxy",
    "cost_ratio_proxy",
    "action_side",
    "mom12_ex1",
    "ret_6m",
    "ret_3m",
    "ret_20d",
    "vol60",
    "vol20",
    "dd_6m",
    "dist_ma200",
    "dist_ma20",
    "rsi14",
    "breadth_ma200",
    "breadth_slope",
    "benchmark_ret_3m",
    "benchmark_ret_6m",
]

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


def _load_feature_history() -> pd.DataFrame:
    price_df = load_all_price_data().copy()
    price_df["date"] = pd.to_datetime(price_df["date"])
    price_df = price_df.sort_values(["ticker", "date"]).reset_index(drop=True)

    features_df = compute_price_features(price_df)

    if "dd_12m" not in features_df.columns:
        features_df["dd_12m"] = features_df.get("dd_6m", 0.0)

    if "vol20" not in features_df.columns:
        features_df["vol20"] = features_df.get("vol60", 0.0)

    return features_df


def _add_forward_returns(df: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    out = df.sort_values(["ticker", "date"]).copy()
    grouped = out.groupby("ticker", group_keys=False)

    for horizon in horizons:
        out[f"fwd_ret_{horizon}d"] = grouped["adjusted_close"].shift(-horizon) / out["adjusted_close"] - 1.0

    return out


def _month_end_snapshots(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["month"] = out["date"].dt.to_period("M")

    out = (
        out.sort_values(["ticker", "date"])
        .groupby(["ticker", "month"], as_index=False)
        .tail(1)
        .drop(columns="month")
        .reset_index(drop=True)
    )
    return out


def _build_benchmark_feature_row(monthly_df: pd.DataFrame, universe_df: pd.DataFrame) -> pd.DataFrame:
    config = load_config()

    if getattr(config.benchmark, "mode", "") == "single_ticker":
        ticker = getattr(config.benchmark, "ticker", None)
        if ticker is not None:
            bench = monthly_df.loc[monthly_df["ticker"] == ticker].copy()
            if not bench.empty:
                keep_cols = ["date", "ret_3m", "ret_6m", "mom12_ex1", "fwd_ret_21d", "fwd_ret_63d"]
                keep_cols = [c for c in keep_cols if c in bench.columns]
                return bench[keep_cols].rename(
                    columns={
                        "ret_3m": "benchmark_ret_3m",
                        "ret_6m": "benchmark_ret_6m",
                        "mom12_ex1": "benchmark_mom12_ex1",
                        "fwd_ret_21d": "benchmark_fwd_ret_21d",
                        "fwd_ret_63d": "benchmark_fwd_ret_63d",
                    }
                )

    members = universe_df.loc[
        (universe_df["benchmark_member"] == True) & (universe_df["active"] == True),
        ["ticker"],
    ].drop_duplicates()

    bench = monthly_df.merge(members, on="ticker", how="inner")
    if bench.empty:
        bench = monthly_df.copy()

    agg = {"ret_3m": "mean", "ret_6m": "mean", "mom12_ex1": "mean"}
    if "fwd_ret_21d" in bench.columns:
        agg["fwd_ret_21d"] = "mean"
    if "fwd_ret_63d" in bench.columns:
        agg["fwd_ret_63d"] = "mean"

    out = bench.groupby("date", as_index=False).agg(agg).rename(
        columns={
            "ret_3m": "benchmark_ret_3m",
            "ret_6m": "benchmark_ret_6m",
            "mom12_ex1": "benchmark_mom12_ex1",
            "fwd_ret_21d": "benchmark_fwd_ret_21d",
            "fwd_ret_63d": "benchmark_fwd_ret_63d",
        }
    )
    return out


def _build_market_breadth_frame(monthly_df: pd.DataFrame, universe_df: pd.DataFrame) -> pd.DataFrame:
    members = universe_df.loc[
        (universe_df["benchmark_member"] == True) & (universe_df["active"] == True),
        ["ticker"],
    ].drop_duplicates()

    bench = monthly_df.merge(members, on="ticker", how="inner")
    if bench.empty:
        bench = monthly_df.copy()

    out = bench.groupby("date", as_index=False).agg(
        breadth_ma200=("ma200_flag", "mean"),
        breadth_slope=("ma200_slope_flag", "mean"),
        avg_ret_3m=("ret_3m", "mean"),
        avg_ret_6m=("ret_6m", "mean"),
        avg_mom12_ex1=("mom12_ex1", "mean"),
        median_vol20=("vol20", "median"),
        cross_sectional_dispersion=("ret_3m", "std"),
    )
    out["cross_sectional_dispersion"] = out["cross_sectional_dispersion"].fillna(0.0)
    return out


def build_monthly_ml_panel() -> pd.DataFrame:
    universe_df = load_universe().copy()
    features_df = _load_feature_history()

    features_df = _add_forward_returns(features_df, horizons=[21, 63])
    monthly_df = _month_end_snapshots(features_df)

    universe_keep = [
        "ticker",
        "sector",
        "pea_eligible",
        "core_allowed",
        "opp_allowed",
        "benchmark_member",
        "active",
        "liquidity_bucket",
        "spread_proxy",
        "ttf_flag",
    ]
    monthly_df = monthly_df.merge(universe_df[universe_keep], on="ticker", how="left")

    benchmark_df = _build_benchmark_feature_row(monthly_df, universe_df)
    breadth_df = _build_market_breadth_frame(monthly_df, universe_df)

    monthly_df = monthly_df.merge(benchmark_df, on="date", how="left")
    monthly_df = monthly_df.merge(breadth_df, on="date", how="left")

    for col in [
        "benchmark_ret_3m",
        "benchmark_ret_6m",
        "benchmark_mom12_ex1",
        "benchmark_fwd_ret_21d",
        "benchmark_fwd_ret_63d",
        "breadth_ma200",
        "breadth_slope",
        "avg_ret_3m",
        "avg_ret_6m",
        "avg_mom12_ex1",
        "median_vol20",
        "cross_sectional_dispersion",
        "spread_proxy",
    ]:
        if col not in monthly_df.columns:
            monthly_df[col] = 0.0

    return monthly_df


def _safe_numeric(s: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)


def _estimate_cost_ratio(df: pd.DataFrame, bucket: str) -> pd.Series:
    spread_proxy = _safe_numeric(df.get("spread_proxy", 0.0), default=0.0)
    vol20 = _safe_numeric(df.get("vol20", 0.0), default=0.0)
    dist_ma20 = _safe_numeric(df.get("dist_ma20", 0.0), default=0.0).abs()

    if str(bucket).lower() == "core":
        base = 0.0025
        cost = base + 0.35 * spread_proxy + 0.05 * vol20 + 0.01 * dist_ma20
    else:
        base = 0.0040
        cost = base + 0.45 * spread_proxy + 0.07 * vol20 + 0.015 * dist_ma20

    return cost.clip(lower=0.0, upper=0.08)


def _build_raw_score(df: pd.DataFrame, bucket: str) -> pd.Series:
    mom12 = _safe_numeric(df.get("mom12_ex1", 0.0))
    ret6 = _safe_numeric(df.get("ret_6m", 0.0))
    ret3 = _safe_numeric(df.get("ret_3m", 0.0))
    ret20 = _safe_numeric(df.get("ret_20d", 0.0))
    vol60 = _safe_numeric(df.get("vol60", 0.0))
    dd6 = _safe_numeric(df.get("dd_6m", 0.0)).abs()

    if str(bucket).lower() == "core":
        return 0.45 * mom12 + 0.30 * ret6 + 0.15 * ret3 - 0.07 * vol60 - 0.03 * dd6
    return 0.30 * mom12 + 0.25 * ret6 + 0.25 * ret3 + 0.15 * ret20 - 0.05 * vol60


def build_trade_quality_dataset() -> pd.DataFrame:
    panel = build_monthly_ml_panel().copy()
    panel = panel.loc[(panel["pea_eligible"] == True) & (panel["active"] == True)].copy()

    rows: list[pd.DataFrame] = []

    for bucket_name, allowed_col, horizon_col, bench_horizon_col in [
        ("core", "core_allowed", "fwd_ret_63d", "benchmark_fwd_ret_63d"),
        ("opportunistic", "opp_allowed", "fwd_ret_21d", "benchmark_fwd_ret_21d"),
    ]:
        sub = panel.loc[panel[allowed_col] == True].copy()
        if sub.empty:
            continue

        sub["bucket"] = bucket_name
        sub["raw_score"] = _build_raw_score(sub, bucket=bucket_name)
        sub["cost_ratio_proxy"] = _estimate_cost_ratio(sub, bucket=bucket_name)
        sub["future_alpha"] = _safe_numeric(sub[horizon_col], default=np.nan) - _safe_numeric(sub[bench_horizon_col], default=0.0)
        sub["target_value"] = sub["future_alpha"] - sub["cost_ratio_proxy"]

        keep_cols = ["date", "ticker", "target_value"] + COMMON_SIGNAL_FEATURES
        tmp = sub.copy()
        for col in keep_cols:
            if col not in tmp.columns:
                tmp[col] = np.nan
        rows.append(tmp[keep_cols].copy())

    if not rows:
        return pd.DataFrame(columns=["date", "ticker", "target_value"] + COMMON_SIGNAL_FEATURES)

    out = pd.concat(rows, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    out["target_value"] = pd.to_numeric(out["target_value"], errors="coerce")
    out = out.dropna(subset=["date", "target_value"]).copy()

    return out[["date", "ticker", "target_value"] + COMMON_SIGNAL_FEATURES].copy()


def build_deployment_dataset() -> pd.DataFrame:
    panel = build_monthly_ml_panel().copy()

    breadth = (
        panel.groupby("date", as_index=False)
        .agg(
            breadth_ma200=("breadth_ma200", "first"),
            breadth_slope=("breadth_slope", "first"),
            benchmark_ret_3m=("benchmark_ret_3m", "first"),
            benchmark_ret_6m=("benchmark_ret_6m", "first"),
            benchmark_mom12_ex1=("benchmark_mom12_ex1", "first"),
            avg_ret_3m=("avg_ret_3m", "first"),
            avg_ret_6m=("avg_ret_6m", "first"),
            avg_mom12_ex1=("avg_mom12_ex1", "first"),
            median_vol20=("median_vol20", "first"),
            cross_sectional_dispersion=("cross_sectional_dispersion", "first"),
            benchmark_fwd_ret_21d=("benchmark_fwd_ret_21d", "first"),
        )
        .sort_values("date")
        .reset_index(drop=True)
    )

    vol_penalty = 0.20 * _safe_numeric(breadth["median_vol20"], default=0.0)
    disp_penalty = 0.10 * _safe_numeric(breadth["cross_sectional_dispersion"], default=0.0)
    breadth["target_value"] = _safe_numeric(breadth["benchmark_fwd_ret_21d"], default=np.nan) - vol_penalty - disp_penalty
    breadth = breadth.dropna(subset=["target_value"]).copy()

    return breadth[["date", "target_value"] + DEPLOYMENT_FEATURES].copy()


def build_rebalance_dataset() -> pd.DataFrame:
    trade_df = build_trade_quality_dataset().copy()
    if trade_df.empty:
        return pd.DataFrame(columns=["date", "ticker", "target_value"] + REBALANCE_FEATURES)

    trade_df["weight_gap_proxy"] = (
        trade_df.groupby(["date", "bucket"])["raw_score"]
        .transform(lambda s: (s - s.median()).abs())
        .fillna(0.0)
    )

    trade_df["cost_ratio_proxy"] = np.where(
        trade_df["bucket"] == "core",
        0.0025 + 0.30 * trade_df["weight_gap_proxy"].clip(upper=0.10),
        0.0040 + 0.40 * trade_df["weight_gap_proxy"].clip(upper=0.10),
    )

    trade_df["action_side"] = np.where(_safe_numeric(trade_df["raw_score"], default=0.0) >= 0.0, "BUY", "SELL")

    momentum_mix = np.where(
        trade_df["bucket"] == "core",
        0.55 * _safe_numeric(trade_df["ret_20d"], default=0.0) + 0.45 * _safe_numeric(trade_df["ret_3m"], default=0.0),
        0.75 * _safe_numeric(trade_df["ret_20d"], default=0.0) + 0.25 * _safe_numeric(trade_df["ret_10d"], default=0.0),
    )

    trade_df["target_value"] = momentum_mix - _safe_numeric(trade_df["cost_ratio_proxy"], default=0.0)
    trade_df = trade_df.dropna(subset=["target_value"]).copy()

    final_cols = ["date", "ticker", "target_value"] + REBALANCE_FEATURES
    for col in final_cols:
        if col not in trade_df.columns:
            trade_df[col] = np.nan

    return trade_df[final_cols].copy()