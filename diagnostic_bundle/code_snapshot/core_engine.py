from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict

import numpy as np
import pandas as pd

from src.features.feature_pipeline import build_latest_feature_snapshot
from src.io.loaders import load_portfolio_snapshot, load_universe
from src.settings import load_config


@dataclass
class CoreSelectionResult:
    selected_tickers: list[str]
    signal_table: pd.DataFrame


def _safe_zscore(series: pd.Series) -> pd.Series:
    clean = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    mean = clean.mean()
    std = clean.std(ddof=0)

    if pd.isna(std) or std == 0:
        return pd.Series(np.zeros(len(clean)), index=clean.index)

    return (clean - mean) / std


def _prepare_core_universe(
    latest_features_df: pd.DataFrame,
    universe_df: pd.DataFrame,
) -> pd.DataFrame:
    merged = latest_features_df.merge(
        universe_df[
            [
                "ticker",
                "sector",
                "pea_eligible",
                "core_allowed",
                "active",
                "benchmark_member",
                "liquidity_bucket",
                "ttf_flag",
            ]
        ],
        on="ticker",
        how="inner",
    )

    merged = merged.loc[
        (merged["pea_eligible"] == True)
        & (merged["core_allowed"] == True)
        & (merged["active"] == True)
    ].copy()

    if merged.empty:
        raise ValueError("Core universe is empty after filtering.")

    return merged


def _build_benchmark_feature_row(
    latest_features_df: pd.DataFrame,
    universe_df: pd.DataFrame,
) -> pd.Series:
    config = load_config()

    if config.benchmark.mode == "single_ticker":
        bench = latest_features_df.loc[latest_features_df["ticker"] == config.benchmark.ticker].copy()
        if bench.empty:
            raise ValueError(f"Benchmark ticker '{config.benchmark.ticker}' not found in latest features.")
        return bench.iloc[0]

    members = universe_df.loc[
        (universe_df["benchmark_member"] == True) & (universe_df["active"] == True),
        ["ticker"],
    ].drop_duplicates()

    bench = latest_features_df.merge(members, on="ticker", how="inner").copy()
    if bench.empty:
        raise ValueError("Synthetic benchmark universe is empty.")

    numeric_cols = ["mom12_ex1", "ret_6m", "ret_3m", "vol60", "dd_6m"]
    out = {}
    for col in numeric_cols:
        out[col] = float(pd.to_numeric(bench[col], errors="coerce").fillna(0.0).mean())

    return pd.Series(out)


def _compute_relative_strength(df: pd.DataFrame, benchmark_row: pd.Series) -> pd.Series:
    rel_12 = df["mom12_ex1"] - float(benchmark_row["mom12_ex1"])
    rel_6 = df["ret_6m"] - float(benchmark_row["ret_6m"])
    rel_3 = df["ret_3m"] - float(benchmark_row["ret_3m"])
    return 0.50 * rel_12 + 0.30 * rel_6 + 0.20 * rel_3


def _add_current_core_holdings_info(df: pd.DataFrame, portfolio_df: pd.DataFrame) -> pd.DataFrame:
    current_core = portfolio_df.loc[portfolio_df["bucket"] == "core"].copy()

    if current_core.empty:
        df["currently_held_core"] = False
        df["current_core_weight_total"] = 0.0
        return df

    current_core = current_core[["ticker", "current_weight_total"]].drop_duplicates(subset=["ticker"])
    current_core = current_core.rename(columns={"current_weight_total": "current_core_weight_total"})
    current_core["currently_held_core"] = True

    df = df.merge(current_core, on="ticker", how="left")
    df["currently_held_core"] = df["currently_held_core"].fillna(False)
    df["current_core_weight_total"] = df["current_core_weight_total"].fillna(0.0)
    return df


def _add_sector_and_liquidity_penalties(df: pd.DataFrame, portfolio_df: pd.DataFrame) -> pd.DataFrame:
    current_core = portfolio_df.loc[portfolio_df["bucket"] == "core"].copy()

    if current_core.empty:
        df["sector_current_count"] = 0
    else:
        sector_counts = current_core.groupby("sector").size().rename("sector_current_count").reset_index()
        df = df.merge(sector_counts, on="sector", how="left")
        df["sector_current_count"] = df["sector_current_count"].fillna(0)

    df["sector_crowding_penalty"] = 0.04 * df["sector_current_count"]

    liq_map = {"high": 0.00, "standard": 0.03, "lower": 0.07}
    df["liquidity_penalty"] = df["liquidity_bucket"].map(liq_map).fillna(0.03)

    return df


def _stable_core_selection(
    df: pd.DataFrame,
    max_positions: int,
    max_per_sector: int,
    min_replacement_delta: float,
) -> pd.DataFrame:
    working = df.copy()

    working["eligible_for_selection"] = (
        working["entry_candidate"]
        | (working["currently_held_core"] & working["retention_candidate"])
    )

    working["construction_score"] = working["core_score"]
    working.loc[working["currently_held_core"], "construction_score"] += 0.04

    held_pool = working.loc[
        working["currently_held_core"] & working["eligible_for_selection"]
    ].sort_values("construction_score", ascending=False)

    new_pool = working.loc[
        (~working["currently_held_core"]) & working["entry_candidate"]
    ].sort_values("construction_score", ascending=False)

    selected_indices: list[int] = []
    sector_counter: dict[str, int] = defaultdict(int)

    for idx, row in held_pool.iterrows():
        sector = str(row["sector"])
        if sector_counter[sector] >= max_per_sector:
            continue
        selected_indices.append(idx)
        sector_counter[sector] += 1
        if len(selected_indices) >= max_positions:
            break

    for idx, row in new_pool.iterrows():
        if len(selected_indices) < max_positions:
            sector = str(row["sector"])
            if sector_counter[sector] >= max_per_sector:
                continue
            selected_indices.append(idx)
            sector_counter[sector] += 1

    selected_df = working.loc[selected_indices].copy() if selected_indices else working.iloc[0:0].copy()

    remaining_new = new_pool.loc[~new_pool.index.isin(selected_df.index)].copy()

    if not selected_df.empty and not remaining_new.empty:
        for new_idx, new_row in remaining_new.iterrows():
            candidates_to_replace = selected_df.loc[
                (selected_df["currently_held_core"])
                & (~selected_df["entry_candidate"])
            ].sort_values("construction_score", ascending=True)

            if candidates_to_replace.empty:
                continue

            worst_idx = candidates_to_replace.index[0]
            worst_row = candidates_to_replace.iloc[0]

            score_delta = float(new_row["construction_score"] - worst_row["construction_score"])
            new_sector = str(new_row["sector"])
            old_sector = str(worst_row["sector"])

            if new_sector != old_sector and sector_counter[new_sector] >= max_per_sector:
                continue

            if score_delta >= min_replacement_delta:
                selected_df = selected_df.drop(index=worst_idx)
                sector_counter[old_sector] -= 1

                selected_df = pd.concat([selected_df, working.loc[[new_idx]]], axis=0)
                sector_counter[new_sector] += 1

    working["selected_core_final"] = working.index.isin(selected_df.index)

    working["selection_reason"] = "NOT_SELECTED"
    working.loc[
        working["selected_core_final"] & (~working["currently_held_core"]),
        "selection_reason",
    ] = "NEW_ENTRY"
    working.loc[
        working["selected_core_final"] & working["currently_held_core"] & working["entry_candidate"],
        "selection_reason",
    ] = "KEEP_TOP_RANK"
    working.loc[
        working["selected_core_final"] & working["currently_held_core"] & (~working["entry_candidate"]),
        "selection_reason",
    ] = "KEEP_BUFFER_ZONE"
    working.loc[
        (~working["selected_core_final"]) & working["currently_held_core"],
        "selection_reason",
    ] = "DROP_OUT"

    return working


def build_core_signal_table(
    latest_features_df: pd.DataFrame | None = None,
    universe_df: pd.DataFrame | None = None,
    portfolio_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    config = load_config()
    max_positions = min(config.sleeves["core"].max_positions, 9)
    max_per_sector = 3

    if latest_features_df is None:
        latest_features_df = build_latest_feature_snapshot()

    if universe_df is None:
        universe_df = load_universe()

    if portfolio_df is None:
        portfolio_df = load_portfolio_snapshot()

    required_cols = {
        "ticker",
        "mom12_ex1",
        "ret_6m",
        "ret_3m",
        "vol60",
        "dd_6m",
        "ma200_flag",
        "ma200_slope_flag",
    }
    missing = required_cols - set(latest_features_df.columns)
    if missing:
        raise ValueError(f"Missing required columns for core engine: {sorted(missing)}")

    df = _prepare_core_universe(latest_features_df, universe_df)
    df = _add_current_core_holdings_info(df, portfolio_df)
    df = _add_sector_and_liquidity_penalties(df, portfolio_df)

    benchmark_row = _build_benchmark_feature_row(latest_features_df, universe_df)

    numeric_cols = ["mom12_ex1", "ret_6m", "ret_3m", "vol60", "dd_6m"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        med = df[col].median()
        if pd.isna(med):
            med = 0.0
        df[col] = df[col].fillna(med)

    df["rel_strength"] = _compute_relative_strength(df, benchmark_row)

    df["mom12_ex1_z"] = _safe_zscore(df["mom12_ex1"])
    df["mom6_z"] = _safe_zscore(df["ret_6m"])
    df["mom3_z"] = _safe_zscore(df["ret_3m"])
    df["rel_strength_z"] = _safe_zscore(df["rel_strength"])
    df["vol60_z"] = _safe_zscore(df["vol60"])
    df["dd6m_z"] = _safe_zscore(df["dd_6m"])

    df["trend_bonus_value"] = 0.0
    strong_trend_mask = df["ma200_flag"] & df["ma200_slope_flag"]
    weak_trend_mask = (~df["ma200_flag"]) & (~df["ma200_slope_flag"])

    df.loc[strong_trend_mask, "trend_bonus_value"] = 0.20
    df.loc[weak_trend_mask, "trend_bonus_value"] = -0.20

    df["persistence_bonus"] = 0.0
    df.loc[df["currently_held_core"] & df["ma200_flag"], "persistence_bonus"] = 0.04

    df["core_score"] = (
        0.30 * df["mom12_ex1_z"]
        + 0.20 * df["mom6_z"]
        + 0.10 * df["mom3_z"]
        + 0.25 * df["rel_strength_z"]
        - 0.08 * df["vol60_z"]
        - 0.07 * df["dd6m_z"]
        + df["trend_bonus_value"]
        + df["persistence_bonus"]
        - df["sector_crowding_penalty"]
        - df["liquidity_penalty"]
    )

    df = df.sort_values(
        ["core_score", "currently_held_core"],
        ascending=[False, False],
    ).reset_index(drop=True)

    df["rank_core"] = np.arange(1, len(df) + 1)
    df["entry_candidate"] = df["rank_core"] <= min(8, len(df))
    df["retention_candidate"] = df["rank_core"] <= min(14, len(df))

    df = _stable_core_selection(
        df=df,
        max_positions=max_positions,
        max_per_sector=max_per_sector,
        min_replacement_delta=config.alpha_guardrails.core_min_score_delta_for_replacement,
    )

    return df[
        [
            "ticker",
            "sector",
            "currently_held_core",
            "current_core_weight_total",
            "sector_current_count",
            "sector_crowding_penalty",
            "liquidity_penalty",
            "mom12_ex1",
            "ret_6m",
            "ret_3m",
            "rel_strength",
            "vol60",
            "dd_6m",
            "ma200_flag",
            "ma200_slope_flag",
            "trend_bonus_value",
            "persistence_bonus",
            "core_score",
            "rank_core",
            "entry_candidate",
            "retention_candidate",
            "eligible_for_selection",
            "construction_score",
            "selected_core_final",
            "selection_reason",
        ]
    ]


def select_core_names(
    latest_features_df: pd.DataFrame | None = None,
    universe_df: pd.DataFrame | None = None,
    portfolio_df: pd.DataFrame | None = None,
) -> CoreSelectionResult:
    signal_table = build_core_signal_table(
        latest_features_df=latest_features_df,
        universe_df=universe_df,
        portfolio_df=portfolio_df,
    )

    selected_tickers = signal_table.loc[
        signal_table["selected_core_final"], "ticker"
    ].tolist()

    return CoreSelectionResult(
        selected_tickers=selected_tickers,
        signal_table=signal_table,
    )


if __name__ == "__main__":
    result = select_core_names()
    print("\n=== Selected Core Tickers ===")
    print(result.selected_tickers)
    print("\n=== Core Signal Table ===")
    print(result.signal_table)