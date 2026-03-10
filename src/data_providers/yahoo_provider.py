from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from src.data_providers.base import MarketDataProvider
from src.io.market_data import (
    PRICE_COLUMNS,
    normalize_price_dataframe,
    upsert_price_history_cache,
)


class YahooMarketDataProvider(MarketDataProvider):
    """
    Yahoo Finance provider using yfinance as the primary upstream source.

    Design:
    - Yahoo is the main external source of truth for market data acquisition.
    - Downloaded data is normalized to the project's canonical schema.
    - Persistence is handled as an upsert into the local CSV cache.
    """

    def __init__(
        self,
        *,
        max_retries: int = 3,
        retry_sleep_seconds: float = 1.5,
        timeout_seconds: int = 30,
    ) -> None:
        try:
            import yfinance as yf
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "YahooMarketDataProvider requires the 'yfinance' package."
            ) from exc

        self.yf = yf
        self.max_retries = max_retries
        self.retry_sleep_seconds = retry_sleep_seconds
        self.timeout_seconds = timeout_seconds

    @staticmethod
    def _empty_price_frame() -> pd.DataFrame:
        return pd.DataFrame(columns=PRICE_COLUMNS)

    @staticmethod
    def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Flatten yfinance MultiIndex columns if present.

        Typical case:
        [('Open', 'MC.PA'), ('High', 'MC.PA'), ...] -> ['Open', 'High', ...]
        """
        if not isinstance(df.columns, pd.MultiIndex):
            return df

        flattened: list[str] = []
        for col in df.columns:
            parts = [str(x) for x in col if x not in ("", None)]
            if not parts:
                flattened.append("unknown")
                continue

            # Prefer the first level if it matches expected OHLCV naming
            first = parts[0]
            flattened.append(first)

        df = df.copy()
        df.columns = flattened
        return df

    def _normalize_history(self, raw_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        if raw_df is None or raw_df.empty:
            return self._empty_price_frame()

        df = self._flatten_columns(raw_df).reset_index().copy()

        rename_map = {
            "Date": "date",
            "Datetime": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adjusted_close",
            "Volume": "volume",
        }
        df = df.rename(columns=rename_map)

        if "adjusted_close" not in df.columns and "close" in df.columns:
            df["adjusted_close"] = df["close"]

        df["ticker"] = ticker

        for col in PRICE_COLUMNS:
            if col not in df.columns:
                df[col] = pd.NA

        df = df[PRICE_COLUMNS].copy()

        return normalize_price_dataframe(
            df,
            source_name=f"yahoo:{ticker}",
            expected_ticker=ticker,
        )

    def _download_once(
        self,
        ticker: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        raw_df = self.yf.download(
            tickers=ticker,
            start=start_date,
            end=end_date,
            interval="1d",
            auto_adjust=False,
            actions=False,
            progress=False,
            threads=False,
            group_by="column",
            timeout=self.timeout_seconds,
        )

        return self._normalize_history(raw_df, ticker)

    def fetch_price_history(
        self,
        ticker: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """
        Fetch daily price history for one ticker from Yahoo with retries.
        """
        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                df = self._download_once(
                    ticker=ticker,
                    start_date=start_date,
                    end_date=end_date,
                )
                return df
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(self.retry_sleep_seconds * attempt)

        raise RuntimeError(
            f"Yahoo download failed for ticker '{ticker}' after "
            f"{self.max_retries} attempts."
        ) from last_error

    def fetch_many_price_histories(
        self,
        tickers: list[str],
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """
        Fetch many tickers sequentially.

        Sequential download is chosen intentionally for robustness:
        a single bad ticker should not corrupt the whole batch.
        """
        frames: list[pd.DataFrame] = []

        clean_tickers = sorted({str(t).strip() for t in tickers if str(t).strip()})
        for ticker in clean_tickers:
            df = self.fetch_price_history(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
            )
            if not df.empty:
                frames.append(df)

        if not frames:
            return self._empty_price_frame()

        combined = pd.concat(frames, ignore_index=True)
        return normalize_price_dataframe(combined, source_name="yahoo:batch")

    def save_price_history(
        self,
        df: pd.DataFrame,
        output_dir: Path,
    ) -> None:
        """
        Persist downloaded Yahoo data into the local CSV cache via upsert.

        This preserves old history and only merges new rows.
        """
        upsert_price_history_cache(df, prices_dir=output_dir)


if __name__ == "__main__":
    provider = YahooMarketDataProvider()
    df = provider.fetch_price_history("MC.PA", start_date="2025-01-01")
    print(df.tail())