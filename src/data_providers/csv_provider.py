from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.data_providers.base import MarketDataProvider
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


class CsvMarketDataProvider(MarketDataProvider):
    def __init__(self, prices_dir: Path | None = None) -> None:
        self.prices_dir = prices_dir or (RAW_DATA_DIR / "prices")

    def _load_price_csv(self, csv_path: Path) -> pd.DataFrame:
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

    def fetch_price_history(
        self,
        ticker: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        path = self.prices_dir / f"{ticker}.csv"
        df = self._load_price_csv(path)

        if start_date is not None:
            df = df.loc[pd.to_datetime(df["date"]) >= pd.to_datetime(start_date)]
        if end_date is not None:
            df = df.loc[pd.to_datetime(df["date"]) <= pd.to_datetime(end_date)]

        return df.reset_index(drop=True)

    def fetch_many_price_histories(
        self,
        tickers: list[str],
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        if not tickers:
            return pd.DataFrame()

        frames = []
        for ticker in tickers:
            try:
                frames.append(self.fetch_price_history(ticker, start_date, end_date))
            except FileNotFoundError:
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
    provider = CsvMarketDataProvider()
    print("csv provider ok")