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


def _cross_sectional_binary_target(
    df: pd.DataFrame,
    value_col: str,
    top_q: float,
    bottom_q: float,
) -> pd.Series:
    values = pd.to_numeric(df[value_col], errors="coerce")
    if values.notna().sum() < 8:
        return pd.Series(np.nan, index=df.index)

    q_top = values.quantile(top_q)
    q_bottom = values.quantile(bottom_q)

    target = pd.Series(np.nan, index=df.index, dtype=float)
    target.loc[values >= q_top] = 1.0
    target.loc[values <= q_bottom] = 0.0
    return target


def build_trade_quality_dataset() -> pd.DataFrame:
    panel = build_monthly_ml_panel().copy()
    panel = panel.loc[(panel["pea_eligible"] == True) & (panel["active"] == True)].copy()

    rows = []

    # CORE: predict relative winners vs benchmark on 63d
    core_panel = panel.loc[panel["core_allowed"] == True].copy()
    if not core_panel.empty:
        core_panel["alpha_63d"] = (
            pd.to_numeric(core_panel["fwd_ret_63d"], errors="coerce").fillna(np.nan)
            - pd.to_numeric(core_panel["benchmark_fwd_ret_63d"], errors="coerce").fillna(0.0)
        )

        for date, subdf in core_panel.groupby("date"):
            tmp = subdf.copy()
            tmp["target"] = _cross_sectional_binary_target(tmp, "alpha_63d", top_q=0.70, bottom_q=0.30)

            raw_score = (
                0.45 * pd.to_numeric(tmp["mom12_ex1"], errors="coerce").fillna(0.0)
                + 0.30 * pd.to_numeric(tmp["ret_6m"], errors="coerce").fillna(0.0)
                + 0.15 * pd.to_numeric(tmp["ret_3m"], errors="coerce").fillna(0.0)
                - 0.10 * pd.to_numeric(tmp["vol60"], errors="coerce").fillna(0.0)
            )

            tmp["raw_score"] = raw_score

            keep_cols = [
                "date", "ticker", "sector", "raw_score", "mom12_ex1", "ret_6m", "ret_3m", "ret_20d",
                "ret_10d", "ret_5d", "vol60", "vol20", "dd_6m", "dd_12m", "dist_ma200", "dist_ma20",
                "rsi14", "ma200_flag", "ma200_slope_flag", "breadth_ma200", "breadth_slope",
                "benchmark_ret_3m", "benchmark_ret_6m", "benchmark_mom12_ex1", "target"
            ]
            tmp = tmp[keep_cols].copy()
            tmp["bucket"] = "core"
            rows.append(tmp)

    # OPP: predict strongest short-term relative leaders on 21d
    opp_panel = panel.loc[panel["opp_allowed"] == True].copy()
    if not opp_panel.empty:
        opp_panel["alpha_21d"] = (
            pd.to_numeric(opp_panel["fwd_ret_21d"], errors="coerce").fillna(np.nan)
            - pd.to_numeric(opp_panel["benchmark_fwd_ret_21d"], errors="coerce").fillna(0.0)
        )

        for date, subdf in opp_panel.groupby("date"):
            tmp = subdf.copy()
            tmp["target"] = _cross_sectional_binary_target(tmp, "alpha_21d", top_q=0.75, bottom_q=0.35)

            raw_score = (
                0.35 * pd.to_numeric(tmp["mom12_ex1"], errors="coerce").fillna(0.0)
                + 0.30 * pd.to_numeric(tmp["ret_6m"], errors="coerce").fillna(0.0)
                + 0.20 * pd.to_numeric(tmp["ret_3m"], errors="coerce").fillna(0.0)
                + 0.15 * pd.to_numeric(tmp["ret_20d"], errors="coerce").fillna(0.0)
            )

            tmp["raw_score"] = raw_score

            keep_cols = [
                "date", "ticker", "sector", "raw_score", "mom12_ex1", "ret_6m", "ret_3m", "ret_20d",
                "ret_10d", "ret_5d", "vol60", "vol20", "dd_6m", "dd_12m", "dist_ma200", "dist_ma20",
                "rsi14", "ma200_flag", "ma200_slope_flag", "breadth_ma200", "breadth_slope",
                "benchmark_ret_3m", "benchmark_ret_6m", "benchmark_mom12_ex1", "target"
            ]
            tmp = tmp[keep_cols].copy()
            tmp["bucket"] = "opportunistic"
            rows.append(tmp)

    if not rows:
        return pd.DataFrame(columns=["date", "ticker", "bucket", "sector", "target"] + COMMON_SIGNAL_FEATURES)

    out = pd.concat(rows, ignore_index=True)
    out = out.dropna(subset=["date", "target"]).copy()
    out["target"] = pd.to_numeric(out["target"], errors="coerce")
    out = out.dropna(subset=["target"]).copy()

    final_cols = ["date", "ticker", "target"] + COMMON_SIGNAL_FEATURES
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

    # More selective target: only truly favorable short-term regime gets label 1.
    breadth["target"] = (pd.to_numeric(breadth["fwd_bench_21d"], errors="coerce").fillna(0.0) > 0.015).astype(float)
    return breadth


def build_rebalance_dataset() -> pd.DataFrame:
    trade_df = build_trade_quality_dataset().copy()

    trade_df["weight_gap_proxy"] = (
        trade_df.groupby(["date", "bucket"])["raw_score"]
        .transform(lambda s: (s - s.median()).abs())
        .fillna(0.0)
    )

    trade_df["cost_ratio_proxy"] = np.where(
        trade_df["bucket"] == "core",
        0.004 + 0.45 * trade_df["weight_gap_proxy"].clip(upper=0.10),
        0.006 + 0.55 * trade_df["weight_gap_proxy"].clip(upper=0.10),
    )

    trade_df["action_side"] = np.where(trade_df["raw_score"] >= 0.0, "BUY", "SELL")

    # Only reject trades that look weak after netting a rough cost proxy.
    future_edge = np.where(
        trade_df["bucket"] == "core",
        0.7 * trade_df["ret_20d"].fillna(0.0) + 0.3 * trade_df["ret_3m"].fillna(0.0),
        1.0 * trade_df["ret_20d"].fillna(0.0),
    )
    trade_df["target"] = (future_edge > (trade_df["cost_ratio_proxy"] + 0.004)).astype(float)

    final_cols = ["date", "ticker", "target"] + REBALANCE_FEATURES
    for col in final_cols:
        if col not in trade_df.columns:
            trade_df[col] = np.nan

    return trade_df[final_cols].copy()