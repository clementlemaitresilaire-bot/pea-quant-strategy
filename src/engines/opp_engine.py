from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.features.feature_pipeline import build_latest_feature_snapshot
from src.io.loaders import load_portfolio_snapshot, load_universe


@dataclass
class OppSelectionResult:
    selected_tickers: list[str]
    signal_table: pd.DataFrame


def _safe_zscore(series: pd.Series) -> pd.Series:
    clean = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    mean = clean.mean()
    std = clean.std(ddof=0)

    if pd.isna(std) or std == 0:
        return pd.Series(np.zeros(len(clean)), index=clean.index)

    return (clean - mean) / std


def _prepare_opp_etf_universe(
    latest_features_df: pd.DataFrame,
    universe_df: pd.DataFrame,
) -> pd.DataFrame:
    merged = latest_features_df.merge(
        universe_df[
            [
                "ticker",
                "name",
                "sector",
                "industry",
                "pea_eligible",
                "opp_allowed",
                "active",
                "ttf_flag",
                "liquidity_bucket",
                "spread_proxy",
            ]
        ],
        on="ticker",
        how="inner",
    )

    merged = merged.loc[
        (merged["pea_eligible"] == True)
        & (merged["opp_allowed"] == True)
        & (merged["active"] == True)
        & (merged["sector"].astype(str).str.upper() == "ETF")
    ].copy()

    if merged.empty:
        raise ValueError("Opportunistic ETF universe is empty after filtering.")

    return merged


def _add_current_opp_info(df: pd.DataFrame, portfolio_df: pd.DataFrame) -> pd.DataFrame:
    current_opp = portfolio_df.loc[portfolio_df["bucket"] == "opportunistic"].copy()

    if current_opp.empty:
        df["currently_held_opp"] = False
        df["current_opp_weight_total"] = 0.0
        return df

    current_opp_flag = current_opp[["ticker", "current_weight_total"]].drop_duplicates(subset=["ticker"])
    current_opp_flag = current_opp_flag.rename(columns={"current_weight_total": "current_opp_weight_total"})
    current_opp_flag["currently_held_opp"] = True

    df = df.merge(current_opp_flag, on="ticker", how="left")
    df["currently_held_opp"] = df["currently_held_opp"].fillna(False)
    df["current_opp_weight_total"] = df["current_opp_weight_total"].fillna(0.0)

    return df


def _compute_selection_regime(latest_features_df: pd.DataFrame) -> tuple[str, int]:
    breadth_ma200 = float(pd.to_numeric(latest_features_df.get("ma200_flag", 0.0), errors="coerce").fillna(0.0).mean())
    breadth_slope = float(pd.to_numeric(latest_features_df.get("ma200_slope_flag", 0.0), errors="coerce").fillna(0.0).mean())
    avg_ret_3m = float(pd.to_numeric(latest_features_df.get("ret_3m", 0.0), errors="coerce").fillna(0.0).mean())

    if (breadth_ma200 >= 0.60) and (breadth_slope >= 0.55) and (avg_ret_3m > 0.0):
        return "good", 3
    if (breadth_ma200 <= 0.42) and (breadth_slope <= 0.45) and (avg_ret_3m < 0.0):
        return "weak", 2
    return "neutral", 2


def _apply_feature_sanity_filters(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    sanity = pd.Series(True, index=out.index)

    bounds = {
        "ret_5d": (-0.35, 0.35),
        "ret_10d": (-0.45, 0.45),
        "ret_20d": (-0.60, 0.60),
        "ret_3m": (-0.80, 0.90),
        "ret_6m": (-0.90, 1.20),
        "ret_12m": (-0.95, 1.80),
        "mom12_ex1": (-0.95, 1.50),
        "dist_ma20": (-0.30, 0.30),
        "dist_ma200": (-0.45, 0.80),
        "vol20": (0.0, 1.00),
        "vol60": (0.0, 0.80),
        "spread_proxy": (0.0, 0.05),
        "rsi14": (1.0, 99.0),
    }

    for col, (lo, hi) in bounds.items():
        if col not in out.columns:
            continue
        val = pd.to_numeric(out[col], errors="coerce")
        sanity &= val.between(lo, hi, inclusive="both")

    out["feature_sanity_ok"] = sanity
    return out


def build_opp_signal_table(
    latest_features_df: pd.DataFrame | None = None,
    universe_df: pd.DataFrame | None = None,
    portfolio_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Opportunistic sleeve = liquid PEA ETF rotation with strict sanity filters.

    Design goals:
    - keep activity, but never at the cost of absurd data points
    - use 2 names in neutral/weak regimes, 3 names only in strong regimes
    - favor medium-term momentum with explicit risk penalties
    - small persistence bonus to reduce churn, but no unbounded buffer logic
    """
    if latest_features_df is None:
        latest_features_df = build_latest_feature_snapshot()

    if universe_df is None:
        universe_df = load_universe()

    if portfolio_df is None:
        portfolio_df = load_portfolio_snapshot()

    required_cols = {
        "ticker",
        "ret_5d",
        "ret_10d",
        "ret_20d",
        "ret_3m",
        "ret_6m",
        "ret_12m",
        "mom12_ex1",
        "rsi14",
        "dist_ma20",
        "dist_ma200",
        "vol20",
        "vol60",
        "ma200_flag",
        "ma200_slope_flag",
    }
    missing = required_cols - set(latest_features_df.columns)
    if missing:
        raise ValueError(f"Missing required columns for opp ETF engine: {sorted(missing)}")

    df = _prepare_opp_etf_universe(latest_features_df, universe_df)
    df = _add_current_opp_info(df, portfolio_df)

    if "dd_12m" not in df.columns:
        df["dd_12m"] = df["dd_6m"] if "dd_6m" in df.columns else 0.0

    if "vol_ratio_20_60" not in df.columns:
        df["vol_ratio_20_60"] = df["vol20"] / df["vol60"].replace(0, np.nan)

    numeric_cols = [
        "ret_5d",
        "ret_10d",
        "ret_20d",
        "ret_3m",
        "ret_6m",
        "ret_12m",
        "mom12_ex1",
        "rsi14",
        "dist_ma20",
        "dist_ma200",
        "vol20",
        "vol60",
        "vol_ratio_20_60",
        "dd_12m",
        "spread_proxy",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        med = df[col].median()
        if pd.isna(med):
            med = 0.0
        df[col] = df[col].fillna(med)

    df = _apply_feature_sanity_filters(df)

    regime_label, target_count = _compute_selection_regime(latest_features_df)

    df["absolute_momentum"] = (
        0.50 * df["mom12_ex1"]
        + 0.30 * df["ret_6m"]
        + 0.20 * df["ret_3m"]
    )

    df["recent_strength"] = (
        0.55 * df["ret_20d"]
        + 0.30 * df["ret_10d"]
        + 0.15 * df["ret_5d"]
    )

    df["trend_quality"] = (
        0.50 * df["dist_ma200"]
        + 0.20 * df["dist_ma20"]
        + 0.15 * np.where(df["ma200_flag"], 1.0, -1.0)
        + 0.15 * np.where(df["ma200_slope_flag"], 1.0, -1.0)
    )

    df["risk_quality"] = (
        -0.45 * df["vol20"]
        -0.20 * df["vol60"]
        -0.20 * (df["vol_ratio_20_60"] - 1.0).clip(lower=0.0)
        -0.10 * abs(df["dd_12m"])
        -0.05 * df["spread_proxy"]
    )

    df["abs_mom_z"] = _safe_zscore(df["absolute_momentum"])
    df["recent_strength_z"] = _safe_zscore(df["recent_strength"])
    df["trend_z"] = _safe_zscore(df["trend_quality"])
    df["risk_z"] = _safe_zscore(df["risk_quality"])

    df["opp_score_raw"] = (
        0.46 * df["abs_mom_z"]
        + 0.14 * df["recent_strength_z"]
        + 0.25 * df["trend_z"]
        + 0.15 * df["risk_z"]
    )

    df["absolute_momentum_ok"] = (df["mom12_ex1"] > -0.03) & (df["ret_6m"] > -0.03)
    df["trend_ok"] = (df["dist_ma200"] > -0.04) & (df["ma200_flag"] | df["ma200_slope_flag"])
    df["risk_ok"] = (
        (df["vol20"] < 0.45)
        & (df["vol60"] < 0.38)
        & (df["vol_ratio_20_60"] < 1.40)
        & (df["dd_12m"] > -0.30)
    )
    df["not_overextended"] = (df["rsi14"] < 78.0) & (df["dist_ma20"] < 0.14)

    df["persistence_bonus"] = 0.0
    df.loc[df["currently_held_opp"] & (df["opp_score_raw"] > -0.10), "persistence_bonus"] = 0.03

    df["opp_score_net"] = df["opp_score_raw"] + df["persistence_bonus"]

    entry_threshold = 0.05 if regime_label == "weak" else 0.00
    keep_threshold = -0.12 if regime_label == "weak" else -0.08

    df["eligible_opp_candidate"] = (
        df["feature_sanity_ok"]
        & (
            (
                (df["opp_score_net"] > entry_threshold)
                & df["absolute_momentum_ok"]
                & df["trend_ok"]
                & df["risk_ok"]
                & df["not_overextended"]
            )
            | (
                df["currently_held_opp"]
                & (df["opp_score_net"] > keep_threshold)
                & (df["dist_ma200"] > -0.06)
                & (df["vol20"] < 0.50)
            )
        )
    )

    # Hard reject names with implausible short-term moves even if other metrics look attractive.
    df["anomaly_flag"] = (
        (~df["feature_sanity_ok"])
        | (df["ret_5d"].abs() > 0.25)
        | (df["ret_10d"].abs() > 0.35)
        | (df["ret_20d"].abs() > 0.45)
    )
    df.loc[df["anomaly_flag"], "eligible_opp_candidate"] = False

    df = df.sort_values(
        ["eligible_opp_candidate", "opp_score_net", "currently_held_opp"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    df["rank_opp"] = np.arange(1, len(df) + 1)

    selected = df.loc[df["eligible_opp_candidate"]].head(target_count).copy()

    # Fallback only on sane names; never use anomalous ETFs as fallback.
    if selected.empty:
        fallback = df.loc[
            df["feature_sanity_ok"]
            & (df["dist_ma200"] > -0.03)
            & (df["vol20"] < 0.42)
        ].copy()
        selected = fallback.sort_values(
            ["currently_held_opp", "opp_score_net"],
            ascending=[False, False],
        ).head(max(1, min(target_count, 2))).copy()

    selected_tickers = set(selected["ticker"].tolist())
    df["selected_opp_final"] = df["ticker"].isin(selected_tickers)

    df["selection_reason"] = "NOT_SELECTED"
    df.loc[df["selected_opp_final"] & (~df["currently_held_opp"]), "selection_reason"] = "NEW_OPP_ENTRY"
    df.loc[df["selected_opp_final"] & df["currently_held_opp"], "selection_reason"] = "KEEP_OPP"
    df.loc[(~df["selected_opp_final"]) & df["currently_held_opp"], "selection_reason"] = "EXIT_OPP"
    df.loc[df["anomaly_flag"], "selection_reason"] = "REJECT_ANOMALY"

    return df[
        [
            "ticker",
            "name",
            "sector",
            "industry",
            "currently_held_opp",
            "current_opp_weight_total",
            "ret_5d",
            "ret_10d",
            "ret_20d",
            "ret_3m",
            "ret_6m",
            "ret_12m",
            "mom12_ex1",
            "rsi14",
            "dist_ma20",
            "dist_ma200",
            "dd_12m",
            "vol20",
            "vol60",
            "vol_ratio_20_60",
            "spread_proxy",
            "ttf_flag",
            "liquidity_bucket",
            "absolute_momentum",
            "recent_strength",
            "trend_quality",
            "risk_quality",
            "absolute_momentum_ok",
            "trend_ok",
            "risk_ok",
            "not_overextended",
            "feature_sanity_ok",
            "anomaly_flag",
            "opp_score_raw",
            "persistence_bonus",
            "opp_score_net",
            "rank_opp",
            "eligible_opp_candidate",
            "selected_opp_final",
            "selection_reason",
        ]
    ]


def select_opp_names(
    latest_features_df: pd.DataFrame | None = None,
    universe_df: pd.DataFrame | None = None,
    portfolio_df: pd.DataFrame | None = None,
) -> OppSelectionResult:
    signal_table = build_opp_signal_table(
        latest_features_df=latest_features_df,
        universe_df=universe_df,
        portfolio_df=portfolio_df,
    )

    selected_tickers = signal_table.loc[
        signal_table["selected_opp_final"], "ticker"
    ].tolist()

    return OppSelectionResult(
        selected_tickers=selected_tickers,
        signal_table=signal_table,
    )


if __name__ == "__main__":
    result = select_opp_names()
    print("\n=== Selected Opportunistic ETF Tickers ===")
    print(result.selected_tickers)
    print("\n=== Opportunistic ETF Signal Table ===")
    print(result.signal_table)