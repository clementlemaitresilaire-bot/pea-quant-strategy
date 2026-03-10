from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data_providers.base import MarketDataProvider
from src.io.market_data import (
    PRICE_COLUMNS,
    load_all_price_data,
    load_cached_price_history,
    normalize_price_dataframe,
    upsert_price_history_cache,
)
from src.settings import RAW_DATA_DIR


class CsvMarketDataProvider(MarketDataProvider):
    """
    Local CSV cache provider.

    This provider is deterministic and never calls external APIs.
    It reads and writes the canonical local cache used by the rest of the project.
    """

    def __init__(self, prices_dir: Path | None = None) -> None:
        self.prices_dir = prices_dir or (RAW_DATA_DIR / "prices")

    @staticmethod
    def _empty_price_frame() -> pd.DataFrame:
        return pd.DataFrame(columns=PRICE_COLUMNS)

    def fetch_price_history(
        self,
        ticker: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        return load_cached_price_history(
            ticker=ticker,
            prices_dir=self.prices_dir,
            start_date=start_date,
            end_date=end_date,
        )

    def fetch_many_price_histories(
        self,
        tickers: list[str],
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        if not tickers:
            return self._empty_price_frame()

        clean_tickers = sorted({str(t).strip() for t in tickers if str(t).strip()})
        if not clean_tickers:
            return self._empty_price_frame()

        return load_all_price_data(
            prices_dir=self.prices_dir,
            tickers=clean_tickers,
            start_date=start_date,
            end_date=end_date,
        )

    def save_price_history(
        self,
        df: pd.DataFrame,
        output_dir: Path,
    ) -> None:
        normalized = normalize_price_dataframe(df, source_name="csv_provider_save")
        if normalized.empty:
            return

        upsert_price_history_cache(normalized, prices_dir=output_dir)


if __name__ == "__main__":
    provider = CsvMarketDataProvider()
    print(type(provider).__name__)