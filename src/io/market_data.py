from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from src.settings import RAW_DATA_DIR, load_config


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


def _get_global_cache_latest_date(prices_dir: Path | None = None) -> pd.Timestamp | None:
    directory = prices_dir or (RAW_DATA_DIR / "prices")

    if not directory.exists():
        return None

    dates: list[pd.Timestamp] = []
    for csv_path in directory.glob("*.csv"):
        try:
            df = load_price_csv(csv_path)
            if not df.empty:
                dates.append(pd.Timestamp(df["date"].max()))
        except Exception:
            continue

    if not dates:
        return None

    return max(dates)


def _cache_is_stale(prices_dir: Path | None = None) -> bool:
    cfg = load_config().data_provider

    latest = _get_global_cache_latest_date(prices_dir)
    if latest is None:
        return True

    today = pd.Timestamp.today().normalize()
    age_days = (today - latest.normalize()).days
    return age_days > cfg.max_cache_staleness_days


def _maybe_refresh_cache(
    prices_dir: Path | None = None,
    tickers: Iterable[str] | None = None,
) -> None:
    cfg = load_config().data_provider

    if cfg.provider != "yahoo":
        return

    if not cfg.auto_update_on_load:
        return

    if not _cache_is_stale(prices_dir):
        return

    from src.io.update_prices import update_prices

    ticker_list = None
    if tickers is not None:
        ticker_list = sorted({str(t).strip() for t in tickers if str(t).strip()}) or None

    update_prices(
        tickers=ticker_list,
        start_date=cfg.default_download_start_date,
        provider_name=cfg.provider,
        incremental=True,
        overlap_days=cfg.overlap_days,
    )


def load_all_price_data(
    prices_dir: Path | None = None,
    tickers: Iterable[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    auto_refresh: bool = True,
) -> pd.DataFrame:
    """
    Load canonical local price data.

    If configured, this function may refresh the local cache from Yahoo before
    reading the CSV files.
    """
    directory = prices_dir or (RAW_DATA_DIR / "prices")

    ticker_list = None
    if tickers is not None:
        ticker_list = sorted({str(t).strip() for t in tickers if str(t).strip()})
        if not ticker_list:
            return _empty_price_frame()

    if auto_refresh:
        _maybe_refresh_cache(directory, ticker_list)

    if ticker_list is None:
        csv_files = list_price_cache_files(directory)
        frames = [load_price_csv(path) for path in csv_files]
    else:
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


def upsert_price_history_cache(
    df: pd.DataFrame,
    prices_dir: Path | None = None,
) -> None:
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