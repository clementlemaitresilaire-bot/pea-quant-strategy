from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from src.settings import RAW_DATA_DIR


PRICE_COLUMNS = [
    "date",
    "ticker",
    "open",
    "high",
    "low",
    "close",
    "adjusted_close",
    "volume",
]

REQUIRED_PRICE_COLUMNS = set(PRICE_COLUMNS)
NUMERIC_PRICE_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "adjusted_close",
    "volume",
]


def _empty_price_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=PRICE_COLUMNS)


def _coerce_datetime(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, errors="coerce")
    if getattr(dt.dt, "tz", None) is not None:
        dt = dt.dt.tz_convert(None)
    return dt


def normalize_price_dataframe(
    df: pd.DataFrame,
    *,
    source_name: str = "<memory>",
    expected_ticker: str | None = None,
) -> pd.DataFrame:
    """
    Normalize and validate a price dataframe to the internal canonical schema.
    """
    if df is None or df.empty:
        return _empty_price_frame()

    out = df.copy()

    missing = REQUIRED_PRICE_COLUMNS - set(out.columns)
    if missing:
        raise ValueError(
            f"Missing required columns in {source_name}: {sorted(missing)}"
        )

    out = out[PRICE_COLUMNS].copy()

    out["date"] = _coerce_datetime(out["date"])
    out["ticker"] = out["ticker"].astype("string").str.strip()

    for col in NUMERIC_PRICE_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    # Fallback if adjusted_close exists but has holes while close is present
    mask_missing_adj = out["adjusted_close"].isna() & out["close"].notna()
    if mask_missing_adj.any():
        out.loc[mask_missing_adj, "adjusted_close"] = out.loc[mask_missing_adj, "close"]

    out = out.dropna(subset=["date", "ticker", "adjusted_close"]).copy()

    if expected_ticker is not None:
        bad_tickers = (
            out.loc[out["ticker"] != expected_ticker, "ticker"]
            .dropna()
            .unique()
            .tolist()
        )
        if bad_tickers:
            raise ValueError(
                f"{source_name} contains unexpected ticker values: {bad_tickers}. "
                f"Expected only '{expected_ticker}'."
            )

    out = (
        out.drop_duplicates(subset=["ticker", "date"], keep="last")
        .sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )

    return out


def load_price_csv(csv_path: Path) -> pd.DataFrame:
    """
    Load and normalize a single cached price CSV.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Price file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    expected_ticker = csv_path.stem
    return normalize_price_dataframe(
        df,
        source_name=csv_path.name,
        expected_ticker=expected_ticker,
    )


def list_price_cache_files(prices_dir: Path | None = None) -> list[Path]:
    directory = prices_dir or (RAW_DATA_DIR / "prices")

    if not directory.exists():
        raise FileNotFoundError(f"Prices directory not found: {directory}")

    csv_files = sorted(directory.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV price files found in: {directory}")

    return csv_files


def list_cached_tickers(prices_dir: Path | None = None) -> list[str]:
    return [path.stem for path in list_price_cache_files(prices_dir)]


def load_cached_price_history(
    ticker: str,
    prices_dir: Path | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    directory = prices_dir or (RAW_DATA_DIR / "prices")
    csv_path = directory / f"{ticker}.csv"
    df = load_price_csv(csv_path)

    if start_date is not None:
        df = df.loc[df["date"] >= pd.to_datetime(start_date)]
    if end_date is not None:
        df = df.loc[df["date"] <= pd.to_datetime(end_date)]

    return df.reset_index(drop=True)


def load_all_price_data(
    prices_dir: Path | None = None,
    tickers: Iterable[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """
    Load cached local price data only.

    This function is intentionally deterministic and does not call external APIs.
    All Yahoo downloading/updating must happen in src.io.update_prices.
    """
    directory = prices_dir or (RAW_DATA_DIR / "prices")

    if tickers is None:
        csv_files = list_price_cache_files(directory)
        frames = [load_price_csv(path) for path in csv_files]
    else:
        ticker_list = sorted({str(t).strip() for t in tickers if str(t).strip()})
        if not ticker_list:
            return _empty_price_frame()
        frames = [
            load_cached_price_history(
                ticker=ticker,
                prices_dir=directory,
                start_date=start_date,
                end_date=end_date,
            )
            for ticker in ticker_list
            if (directory / f"{ticker}.csv").exists()
        ]

    if not frames:
        return _empty_price_frame()

    combined = pd.concat(frames, ignore_index=True)
    combined = normalize_price_dataframe(combined, source_name="combined_cache")

    if start_date is not None:
        combined = combined.loc[combined["date"] >= pd.to_datetime(start_date)]
    if end_date is not None:
        combined = combined.loc[combined["date"] <= pd.to_datetime(end_date)]

    return combined.reset_index(drop=True)


def get_latest_cached_date(
    ticker: str,
    prices_dir: Path | None = None,
) -> pd.Timestamp | None:
    directory = prices_dir or (RAW_DATA_DIR / "prices")
    csv_path = directory / f"{ticker}.csv"

    if not csv_path.exists():
        return None

    df = load_price_csv(csv_path)
    if df.empty:
        return None

    return pd.Timestamp(df["date"].max())


def upsert_price_history_cache(
    df: pd.DataFrame,
    prices_dir: Path | None = None,
) -> None:
    """
    Merge new price data into the local CSV cache without losing history.
    """
    directory = prices_dir or (RAW_DATA_DIR / "prices")
    directory.mkdir(parents=True, exist_ok=True)

    normalized = normalize_price_dataframe(df, source_name="upsert_payload")
    if normalized.empty:
        return

    for ticker, new_df in normalized.groupby("ticker", sort=True):
        csv_path = directory / f"{ticker}.csv"

        if csv_path.exists():
            old_df = load_price_csv(csv_path)
            merged = pd.concat([old_df, new_df], ignore_index=True)
        else:
            merged = new_df.copy()

        merged = normalize_price_dataframe(
            merged,
            source_name=f"merged_cache:{ticker}",
            expected_ticker=ticker,
        )
        merged.to_csv(csv_path, index=False)


def get_latest_price_snapshot(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return the latest available row per ticker.
    """
    if price_df is None or price_df.empty:
        raise ValueError("price_df is empty")

    normalized = normalize_price_dataframe(price_df, source_name="price_snapshot_input")

    latest = (
        normalized.sort_values(["ticker", "date"])
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