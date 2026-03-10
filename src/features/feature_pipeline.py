from __future__ import annotations

import pandas as pd

from src.features.price_features import compute_price_features
from src.io.market_data import load_all_price_data


def build_full_feature_panel() -> pd.DataFrame:
    """
    Load raw market data and compute the full historical feature panel.
    """
    price_df = load_all_price_data()
    features_df = compute_price_features(price_df)
    return features_df


def get_latest_features(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only the latest available feature row per ticker.
    """
    if features_df.empty:
        raise ValueError("features_df is empty")

    latest = (
        features_df.sort_values(["ticker", "date"])
        .groupby("ticker", as_index=False)
        .tail(1)
        .sort_values("ticker")
        .reset_index(drop=True)
    )
    return latest


def build_latest_feature_snapshot() -> pd.DataFrame:
    """
    End-to-end pipeline:
    - load prices
    - compute all features
    - keep only latest row per ticker
    """
    features_df = build_full_feature_panel()
    latest_df = get_latest_features(features_df)
    return latest_df


if __name__ == "__main__":
    latest_snapshot = build_latest_feature_snapshot()
    print(latest_snapshot.T)