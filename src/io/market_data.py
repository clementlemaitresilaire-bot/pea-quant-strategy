from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data_providers.factory import get_market_data_provider
from src.settings import RAW_DATA_DIR


REQUIRED_PRICE_COLUMNS = {
    "date",
    "ticker",
    "open",
    "high",
    "low",
    "close",
    "adjusted_close",
    "volume",
}


def load_price_csv(csv_path: Path) -> pd.DataFrame:
    """
    Load a single price CSV and validate required columns.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Price file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    missing = REQUIRED_PRICE_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns in {csv_path.name}: {sorted(missing)}"
        )

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    return df


def load_all_price_data(prices_dir: Path | None = None) -> pd.DataFrame:
    """
    Load all price data from the configured provider.

    For CSV provider:
    - reads from data/raw/prices/

    For Yahoo provider:
    - reads tickers from the raw/prices directory only if files already exist
    - so the normal workflow is to call update_prices first
    """
    provider = get_market_data_provider()

    # If provider is CSV, keep previous behavior
    provider_name = provider.__class__.__name__.lower()
    if "csv" in provider_name:
        directory = prices_dir or (RAW_DATA_DIR / "prices")

        if not directory.exists():
            raise FileNotFoundError(f"Prices directory not found: {directory}")

        csv_files = sorted(directory.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV price files found in: {directory}")

        frames: list[pd.DataFrame] = []
        for csv_file in csv_files:
            frames.append(load_price_csv(csv_file))

        combined = pd.concat(frames, ignore_index=True)
        combined = combined.sort_values(["ticker", "date"]).reset_index(drop=True)
        return combined

    # For non-CSV providers, we still rely on raw/prices cache for now
    # so that the rest of the engine remains stable and deterministic.
    directory = prices_dir or (RAW_DATA_DIR / "prices")

    if not directory.exists():
        raise FileNotFoundError(f"Prices directory not found: {directory}")

    csv_files = sorted(directory.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No cached CSV price files found in: {directory}. "
            f"Run `python -m src.io.update_prices` first."
        )

    frames: list[pd.DataFrame] = []
    for csv_file in csv_files:
        frames.append(load_price_csv(csv_file))

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["ticker", "date"]).reset_index(drop=True)

    return combined


def get_latest_price_snapshot(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return the latest available row per ticker.
    """
    if price_df.empty:
        raise ValueError("price_df is empty")

    latest = (
        price_df.sort_values(["ticker", "date"])
        .groupby("ticker", as_index=False)
        .tail(1)
        .sort_values("ticker")
        .reset_index(drop=True)
    )
    return latest


if __name__ == "__main__":
    price_data = load_all_price_data()
    print(price_data.head())
    print("\n--- Latest snapshot ---")
    print(get_latest_price_snapshot(price_data))