from __future__ import annotations

from pathlib import Path
import pandas as pd
import yfinance as yf

from src.data_providers.base import MarketDataProvider


class YahooMarketDataProvider(MarketDataProvider):
    """
    Yahoo Finance provider using yfinance.
    """

    def _flatten_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Flatten yfinance MultiIndex columns if present.
        """
        if isinstance(df.columns, pd.MultiIndex):
            flattened = []
            for col in df.columns:
                # keep first non-empty meaningful level
                parts = [str(x) for x in col if x not in ("", None)]
                flattened.append(parts[0] if parts else "unknown")
            df.columns = flattened
        return df

    def _normalize_history(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(
                columns=["date", "ticker", "open", "high", "low", "close", "adjusted_close", "volume"]
            )

        df = self._flatten_columns(df)
        df = df.reset_index().copy()

        rename_map = {
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adjusted_close",
            "Volume": "volume",
        }
        df = df.rename(columns=rename_map)

        # if adjusted close is absent, fallback to close
        if "adjusted_close" not in df.columns and "close" in df.columns:
            df["adjusted_close"] = df["close"]

        df["ticker"] = ticker

        required_cols = ["date", "ticker", "open", "high", "low", "close", "adjusted_close", "volume"]
        for col in required_cols:
            if col not in df.columns:
                df[col] = pd.NA

        df = df[required_cols].copy()
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None, ambiguous="NaT", nonexistent="NaT")

        # force numeric types
        numeric_cols = ["open", "high", "low", "close", "adjusted_close", "volume"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["date", "adjusted_close"]).sort_values("date").reset_index(drop=True)

        return df

    def fetch_price_history(
        self,
        ticker: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        history = yf.download(
            tickers=ticker,
            start=start_date,
            end=end_date,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
            group_by="column",
        )

        return self._normalize_history(history, ticker)

    def fetch_many_price_histories(
        self,
        tickers: list[str],
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        frames = []
        for ticker in tickers:
            try:
                df = self.fetch_price_history(ticker, start_date, end_date)
                if not df.empty:
                    frames.append(df)
            except Exception:
                continue

        if not frames:
            return pd.DataFrame()

        return pd.concat(frames, ignore_index=True).sort_values(["ticker", "date"]).reset_index(drop=True)

    def save_price_history(
        self,
        df: pd.DataFrame,
        output_dir: Path,
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        for ticker, subdf in df.groupby("ticker"):
            subdf.to_csv(output_dir / f"{ticker}.csv", index=False)


if __name__ == "__main__":
    provider = YahooMarketDataProvider()
    df = provider.fetch_many_price_histories(["AIR.PA", "MC.PA"], start_date="2024-01-01")
    print(df.dtypes)
    print(df.head())