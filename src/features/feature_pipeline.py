from __future__ import annotations

import pandas as pd

from src.features.price_features import compute_price_features
from src.io.loaders import load_universe
from src.io.market_data import load_all_price_data
from src.settings import load_config


def get_required_feature_tickers() -> list[str]:
    """
    Build the canonical ticker list required by the live strategy pipeline.

    Includes:
    - active universe names
    - ETF sleeve names
    - benchmark ticker when benchmark mode is single_ticker
    """
    config = load_config()
    universe_df = load_universe()

    tickers = set(
        universe_df.loc[universe_df["active"] == True, "ticker"]
        .dropna()
        .astype(str)
        .str.strip()
        .tolist()
    )

    for etf in config.etf_sleeve.etfs:
        tickers.add(str(etf.ticker).strip())

    if config.benchmark.mode == "single_ticker" and config.benchmark.ticker:
        tickers.add(str(config.benchmark.ticker).strip())

    return sorted(t for t in tickers if t)


def build_full_feature_panel(
    price_df: pd.DataFrame | None = None,
    *,
    tickers: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    auto_refresh: bool = True,
    restrict_to_required_tickers: bool = True,
) -> pd.DataFrame:
    """
    Load raw market data and compute the full historical feature panel.

    By default, this uses the canonical live ticker universe instead of blindly
    loading every cached CSV on disk.
    """
    if price_df is None:
        effective_tickers = tickers
        if effective_tickers is None and restrict_to_required_tickers:
            effective_tickers = get_required_feature_tickers()

        price_df = load_all_price_data(
            tickers=effective_tickers,
            start_date=start_date,
            end_date=end_date,
            auto_refresh=auto_refresh,
        )

    if price_df is None or price_df.empty:
        return pd.DataFrame()

    features_df = compute_price_features(price_df)
    if features_df.empty:
        return features_df

    features_df = (
        features_df.sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )
    return features_df


def get_latest_features(
    features_df: pd.DataFrame,
    as_of_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Keep only the latest available feature row per ticker.

    If as_of_date is provided, only observations up to that date are considered.
    """
    if features_df is None or features_df.empty:
        raise ValueError("features_df is empty")

    working = features_df.copy()
    working["date"] = pd.to_datetime(working["date"], errors="coerce")
    working = working.dropna(subset=["date"]).copy()

    if as_of_date is not None:
        cutoff = pd.Timestamp(as_of_date)
        working = working.loc[working["date"] <= cutoff].copy()

    if working.empty:
        raise ValueError("No feature rows available for the requested cutoff date.")

    latest = (
        working.sort_values(["ticker", "date"])
        .groupby("ticker", as_index=False)
        .tail(1)
        .sort_values("ticker")
        .reset_index(drop=True)
    )
    return latest


def build_latest_feature_snapshot(
    price_df: pd.DataFrame | None = None,
    *,
    tickers: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    as_of_date: str | pd.Timestamp | None = None,
    auto_refresh: bool = True,
    restrict_to_required_tickers: bool = True,
) -> pd.DataFrame:
    """
    End-to-end live feature pipeline:
    - load the canonical price data
    - compute all features
    - keep only the latest row per ticker
    """
    features_df = build_full_feature_panel(
        price_df=price_df,
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        auto_refresh=auto_refresh,
        restrict_to_required_tickers=restrict_to_required_tickers,
    )

    if features_df.empty:
        return features_df

    latest_df = get_latest_features(features_df, as_of_date=as_of_date)
    return latest_df


if __name__ == "__main__":
    latest_snapshot = build_latest_feature_snapshot()
    print(latest_snapshot.tail())