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
        out[f"fwd_ret_{horizon}d"] = (
            grouped["adjusted_close"].shift(-horizon) / out["adjusted_close"] - 1.0
        )

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

    agg = {
        "ret_3m": "mean",
        "ret_6m": "mean",
        "mom12_ex1": "mean",
    }
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
    monthly_df = monthly_df.merge(
        universe_df[universe_keep],
        on="ticker",
        how="left",
    )

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
    ]:
        if col not in monthly_df.columns:
            monthly_df[col] = 0.0

    return monthly_df


def build_trade_quality_dataset() -> pd.DataFrame:
    panel = build_monthly_ml_panel().copy()
    panel = panel.loc[(panel["pea_eligible"] == True) & (panel["active"] == True)].copy()

    rows = []

    core_panel = panel.loc[panel["core_allowed"] == True].copy()
    if not core_panel.empty:
        core_panel["future_alpha"] = (
            pd.to_numeric(core_panel["fwd_ret_63d"], errors="coerce").fillna(np.nan)
            - pd.to_numeric(core_panel["benchmark_fwd_ret_63d"], errors="coerce").fillna(0.0)
        )
        core_panel["cost_ratio_proxy"] = 0.0035 + 0.15 * core_panel["vol60"].fillna(0.0).clip(upper=0.30)

        for date, subdf in core_panel.groupby("date"):
            tmp = subdf.copy()

            raw_score = (
                0.45 * pd.to_numeric(tmp["mom12_ex1"], errors="coerce").fillna(0.0)
                + 0.30 * pd.to_numeric(tmp["ret_6m"], errors="coerce").fillna(0.0)
                + 0.15 * pd.to_numeric(tmp["ret_3m"], errors="coerce").fillna(0.0)
                - 0.10 * pd.to_numeric(tmp["vol60"], errors="coerce").fillna(0.0)
            )
            tmp["raw_score"] = raw_score
            tmp["target_value"] = tmp["future_alpha"] - tmp["cost_ratio_proxy"]

            keep_cols = [
                "date", "ticker", "sector", "raw_score", "mom12_ex1", "ret_6m", "ret_3m", "ret_20d",
                "ret_10d", "ret_5d", "vol60", "vol20", "dd_6m", "dd_12m", "dist_ma200", "dist_ma20",
                "rsi14", "ma200_flag", "ma200_slope_flag", "breadth_ma200", "breadth_slope",
                "benchmark_ret_3m", "benchmark_ret_6m", "benchmark_mom12_ex1", "target_value"
            ]
            tmp = tmp[keep_cols].copy()
            tmp["bucket"] = "core"
            rows.append(tmp)

    opp_panel = panel.loc[panel["opp_allowed"] == True].copy()
    if not opp_panel.empty:
        opp_panel["future_alpha"] = (
            pd.to_numeric(opp_panel["fwd_ret_21d"], errors="coerce").fillna(np.nan)
            - pd.to_numeric(opp_panel["benchmark_fwd_ret_21d"], errors="coerce").fillna(0.0)
        )
        opp_panel["cost_ratio_proxy"] = (
            0.006
            + 0.20 * opp_panel["vol20"].fillna(0.0).clip(upper=0.40)
            + 0.002 * opp_panel["ttf_flag"].fillna(False).astype(float)
        )

        for date, subdf in opp_panel.groupby("date"):
            tmp = subdf.copy()

            raw_score = (
                0.35 * pd.to_numeric(tmp["mom12_ex1"], errors="coerce").fillna(0.0)
                + 0.30 * pd.to_numeric(tmp["ret_6m"], errors="coerce").fillna(0.0)
                + 0.20 * pd.to_numeric(tmp["ret_3m"], errors="coerce").fillna(0.0)
                + 0.15 * pd.to_numeric(tmp["ret_20d"], errors="coerce").fillna(0.0)
            )
            tmp["raw_score"] = raw_score
            tmp["target_value"] = tmp["future_alpha"] - tmp["cost_ratio_proxy"]

            keep_cols = [
                "date", "ticker", "sector", "raw_score", "mom12_ex1", "ret_6m", "ret_3m", "ret_20d",
                "ret_10d", "ret_5d", "vol60", "vol20", "dd_6m", "dd_12m", "dist_ma200", "dist_ma20",
                "rsi14", "ma200_flag", "ma200_slope_flag", "breadth_ma200", "breadth_slope",
                "benchmark_ret_3m", "benchmark_ret_6m", "benchmark_mom12_ex1", "target_value"
            ]
            tmp = tmp[keep_cols].copy()
            tmp["bucket"] = "opportunistic"
            rows.append(tmp)

    if not rows:
        return pd.DataFrame(columns=["date", "ticker", "target_value"] + COMMON_SIGNAL_FEATURES)

    out = pd.concat(rows, ignore_index=True)
    out = out.dropna(subset=["date", "target_value"]).copy()

    final_cols = ["date", "ticker", "target_value"] + COMMON_SIGNAL_FEATURES
    for col in final_cols:
        if col not in out.columns:
            out[col] = np.nan

    return out[final_cols].copy()


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
            fwd_bench_21d=("benchmark_fwd_ret_21d", "first"),
        )
        .sort_values("date")
        .reset_index(drop=True)
    )

    future_regime = pd.to_numeric(breadth["fwd_bench_21d"], errors="coerce").fillna(0.0)
    risk_penalty = 0.25 * pd.to_numeric(breadth["median_vol20"], errors="coerce").fillna(0.0)
    dispersion_penalty = 0.15 * pd.to_numeric(breadth["cross_sectional_dispersion"], errors="coerce").fillna(0.0)

    breadth["target_value"] = future_regime - risk_penalty - dispersion_penalty
    return breadth


def build_rebalance_dataset() -> pd.DataFrame:
    panel = build_monthly_ml_panel().copy()
    panel = panel.loc[(panel["pea_eligible"] == True) & (panel["active"] == True)].copy()

    rows = []

    core_panel = panel.loc[panel["core_allowed"] == True].copy()
    if not core_panel.empty:
        core_panel["future_alpha"] = (
            pd.to_numeric(core_panel["fwd_ret_63d"], errors="coerce").fillna(np.nan)
            - pd.to_numeric(core_panel["benchmark_fwd_ret_63d"], errors="coerce").fillna(0.0)
        )

        for date, subdf in core_panel.groupby("date"):
            tmp = subdf.copy()

            tmp["raw_score"] = (
                0.45 * pd.to_numeric(tmp["mom12_ex1"], errors="coerce").fillna(0.0)
                + 0.30 * pd.to_numeric(tmp["ret_6m"], errors="coerce").fillna(0.0)
                + 0.15 * pd.to_numeric(tmp["ret_3m"], errors="coerce").fillna(0.0)
                - 0.10 * pd.to_numeric(tmp["vol60"], errors="coerce").fillna(0.0)
            )

            cross_med = tmp["raw_score"].median()
            tmp["weight_gap_proxy"] = (tmp["raw_score"] - cross_med).abs().clip(upper=0.12)

            tmp["cost_ratio_proxy"] = (
                0.0035
                + 0.10 * pd.to_numeric(tmp["vol60"], errors="coerce").fillna(0.0).clip(upper=0.35)
                + 0.35 * tmp["weight_gap_proxy"]
            )

            tmp["action_side"] = np.where(tmp["raw_score"] >= cross_med, "BUY", "SELL")
            tmp["bucket"] = "core"

            direction = np.where(tmp["action_side"] == "BUY", 1.0, -1.0)
            future_edge = direction * tmp["future_alpha"]
            tmp["target_value"] = future_edge - tmp["cost_ratio_proxy"]

            keep_cols = ["date", "ticker", "target_value"] + REBALANCE_FEATURES
            tmp = tmp[keep_cols].copy()
            rows.append(tmp)

    opp_panel = panel.loc[panel["opp_allowed"] == True].copy()
    if not opp_panel.empty:
        opp_panel["future_alpha"] = (
            pd.to_numeric(opp_panel["fwd_ret_21d"], errors="coerce").fillna(np.nan)
            - pd.to_numeric(opp_panel["benchmark_fwd_ret_21d"], errors="coerce").fillna(0.0)
        )

        for date, subdf in opp_panel.groupby("date"):
            tmp = subdf.copy()

            tmp["raw_score"] = (
                0.35 * pd.to_numeric(tmp["mom12_ex1"], errors="coerce").fillna(0.0)
                + 0.30 * pd.to_numeric(tmp["ret_6m"], errors="coerce").fillna(0.0)
                + 0.20 * pd.to_numeric(tmp["ret_3m"], errors="coerce").fillna(0.0)
                + 0.15 * pd.to_numeric(tmp["ret_20d"], errors="coerce").fillna(0.0)
            )

            cross_med = tmp["raw_score"].median()
            tmp["weight_gap_proxy"] = (tmp["raw_score"] - cross_med).abs().clip(upper=0.15)

            tmp["cost_ratio_proxy"] = (
                0.006
                + 0.12 * pd.to_numeric(tmp["vol20"], errors="coerce").fillna(0.0).clip(upper=0.45)
                + 0.45 * tmp["weight_gap_proxy"]
                + 0.002 * tmp["ttf_flag"].fillna(False).astype(float)
            )

            tmp["action_side"] = np.where(tmp["raw_score"] >= cross_med, "BUY", "SELL")
            tmp["bucket"] = "opportunistic"

            direction = np.where(tmp["action_side"] == "BUY", 1.0, -1.0)
            future_edge = direction * tmp["future_alpha"]
            tmp["target_value"] = future_edge - tmp["cost_ratio_proxy"]

            keep_cols = ["date", "ticker", "target_value"] + REBALANCE_FEATURES
            tmp = tmp[keep_cols].copy()
            rows.append(tmp)

    if not rows:
        return pd.DataFrame(columns=["date", "ticker", "target_value"] + REBALANCE_FEATURES)

    out = pd.concat(rows, ignore_index=True)
    out = out.dropna(subset=["date", "target_value"]).copy()

    final_cols = ["date", "ticker", "target_value"] + REBALANCE_FEATURES
    for col in final_cols:
        if col not in out.columns:
            out[col] = np.nan

    return out[final_cols].copy()


if __name__ == "__main__":
    tq = build_trade_quality_dataset()
    dep = build_deployment_dataset()
    reb = build_rebalance_dataset()

    print("trade_quality:", tq.shape)
    print("deployment:", dep.shape)
    print("rebalance:", reb.shape)