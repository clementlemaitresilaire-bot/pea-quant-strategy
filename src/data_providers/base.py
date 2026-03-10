from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
import pandas as pd


class MarketDataProvider(ABC):
    @abstractmethod
    def fetch_price_history(
        self,
        ticker: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        pass

    @abstractmethod
    def fetch_many_price_histories(
        self,
        tickers: list[str],
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        pass

    @abstractmethod
    def save_price_history(
        self,
        df: pd.DataFrame,
        output_dir: Path,
    ) -> None:
        pass