from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.features.feature_pipeline import build_full_feature_panel
from src.io.loaders import load_universe
from src.settings import load_config


@dataclass
class RegimeState:
    benchmark_mode: str
    benchmark_label: str
    latest_date: str
    breadth_ma200: float
    breadth_slope: float
    avg_ret_3m: float
    avg_ret_6m: float
    avg_mom12_ex1: float
    regime: str
    target_core_deployment: float


def _get_benchmark_universe(
    features_df: pd.DataFrame,
    universe_df: pd.DataFrame,
) -> pd.DataFrame:
    config = load_config()

    if config.benchmark.mode == "single_ticker":
        benchmark_df = features_df.loc[features_df["ticker"] == config.benchmark.ticker].copy()
        if benchmark_df.empty:
            raise ValueError(f"No data found for benchmark ticker '{config.benchmark.ticker}'")
        return benchmark_df

    members = universe_df.loc[
        (universe_df["benchmark_member"] == True) & (universe_df["active"] == True),
        ["ticker"],
    ].drop_duplicates()

    benchmark_df = features_df.merge(members, on="ticker", how="inner")
    if benchmark_df.empty:
        raise ValueError("Synthetic benchmark universe is empty.")
    return benchmark_df


def determine_market_regime(
    features_df: pd.DataFrame | None = None,
    universe_df: pd.DataFrame | None = None,
) -> RegimeState:
    config = load_config()

    if features_df is None:
        features_df = build_full_feature_panel()

    if universe_df is None:
        universe_df = load_universe()

    benchmark_df = _get_benchmark_universe(features_df, universe_df).copy()
    benchmark_df = benchmark_df.sort_values("date")
    latest_date = pd.to_datetime(benchmark_df["date"]).max()
    latest = benchmark_df.loc[pd.to_datetime(benchmark_df["date"]) == latest_date].copy()

    if latest.empty:
        raise ValueError("No latest benchmark observations available.")

    if config.benchmark.mode == "single_ticker":
        row = latest.iloc[0]

        breadth_ma200 = float(bool(row["ma200_flag"]))
        breadth_slope = float(bool(row["ma200_slope_flag"]))
        avg_ret_3m = float(row["ret_3m"]) if pd.notna(row["ret_3m"]) else 0.0
        avg_ret_6m = float(row["ret_6m"]) if pd.notna(row["ret_6m"]) else 0.0
        avg_mom12_ex1 = float(row["mom12_ex1"]) if pd.notna(row["mom12_ex1"]) else 0.0
        benchmark_label = config.benchmark.ticker

    else:
        breadth_ma200 = float(latest["ma200_flag"].astype(float).mean())
        breadth_slope = float(latest["ma200_slope_flag"].astype(float).mean())
        avg_ret_3m = float(pd.to_numeric(latest["ret_3m"], errors="coerce").fillna(0.0).mean())
        avg_ret_6m = float(pd.to_numeric(latest["ret_6m"], errors="coerce").fillna(0.0).mean())
        avg_mom12_ex1 = float(pd.to_numeric(latest["mom12_ex1"], errors="coerce").fillna(0.0).mean())
        benchmark_label = "synthetic_benchmark_universe"

    if (breadth_ma200 >= 0.60) and (breadth_slope >= 0.55) and (avg_ret_6m > 0):
        regime = "green"
        target_core_deployment = 1.00
    elif (breadth_ma200 <= 0.40) and (breadth_slope <= 0.45) and (avg_ret_3m < 0):
        regime = "red"
        target_core_deployment = 0.85
    else:
        regime = "orange"
        target_core_deployment = 1.00

    return RegimeState(
        benchmark_mode=config.benchmark.mode,
        benchmark_label=benchmark_label,
        latest_date=str(latest_date.date()),
        breadth_ma200=breadth_ma200,
        breadth_slope=breadth_slope,
        avg_ret_3m=avg_ret_3m,
        avg_ret_6m=avg_ret_6m,
        avg_mom12_ex1=avg_mom12_ex1,
        regime=regime,
        target_core_deployment=target_core_deployment,
    )


def print_regime_state(regime_state: RegimeState) -> None:
    print("\n=== Market Regime ===")
    print(f"Benchmark mode          : {regime_state.benchmark_mode}")
    print(f"Benchmark label         : {regime_state.benchmark_label}")
    print(f"Latest date             : {regime_state.latest_date}")
    print(f"Breadth above MA200     : {regime_state.breadth_ma200:.2%}")
    print(f"Breadth positive slope  : {regime_state.breadth_slope:.2%}")
    print(f"Average 3m return       : {regime_state.avg_ret_3m:.2%}")
    print(f"Average 6m return       : {regime_state.avg_ret_6m:.2%}")
    print(f"Average 12m ex 1m mom   : {regime_state.avg_mom12_ex1:.2%}")
    print(f"Regime                  : {regime_state.regime}")
    print(f"Target core deployment  : {regime_state.target_core_deployment:.2%}")


if __name__ == "__main__":
    regime_state = determine_market_regime()
    print_regime_state(regime_state)