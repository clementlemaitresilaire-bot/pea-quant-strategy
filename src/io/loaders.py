from __future__ import annotations

from pathlib import Path
from typing import Type

import pandas as pd
from pydantic import BaseModel, ValidationError

from src.schemas import (
    CashStateRow,
    PortfolioSnapshotRow,
    TradeHistoryRow,
    UniverseRow,
)
from src.settings import DATA_DIR


def _validate_dataframe_rows(
    df: pd.DataFrame,
    schema: Type[BaseModel],
    dataset_name: str,
) -> pd.DataFrame:
    """
    Validate each row of a DataFrame against a Pydantic schema.

    Returns a normalized DataFrame built from validated rows.
    Raises ValueError if at least one row is invalid.
    """
    validated_rows: list[dict] = []
    errors: list[str] = []

    for idx, row in df.iterrows():
        row_dict = row.to_dict()
        try:
            validated = schema.model_validate(row_dict)
            validated_rows.append(validated.model_dump())
        except ValidationError as exc:
            errors.append(f"{dataset_name} - row {idx}: {exc}")

    if errors:
        error_message = "\n\n".join(errors[:10])
        raise ValueError(
            f"Validation failed for dataset '{dataset_name}'.\n"
            f"Showing up to 10 errors:\n{error_message}"
        )

    return pd.DataFrame(validated_rows)


def _load_csv(csv_path: Path, dataset_name: str) -> pd.DataFrame:
    """
    Load a CSV file and return a raw DataFrame.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"{dataset_name} file not found: {csv_path}")

    return pd.read_csv(csv_path)


def load_universe(csv_path: Path | None = None) -> pd.DataFrame:
    """
    Load and validate the investment universe.
    """
    path = csv_path or (DATA_DIR / "state" / "universe.csv")
    df = _load_csv(path, "universe")
    return _validate_dataframe_rows(df, UniverseRow, "universe")


def load_portfolio_snapshot(csv_path: Path | None = None) -> pd.DataFrame:
    """
    Load and validate the current portfolio snapshot.
    """
    path = csv_path or (DATA_DIR / "state" / "portfolio_snapshot.csv")
    df = _load_csv(path, "portfolio_snapshot")
    return _validate_dataframe_rows(df, PortfolioSnapshotRow, "portfolio_snapshot")


def load_cash_state(csv_path: Path | None = None) -> pd.DataFrame:
    """
    Load and validate the current cash state.
    """
    path = csv_path or (DATA_DIR / "state" / "cash_state.csv")
    df = _load_csv(path, "cash_state")
    return _validate_dataframe_rows(df, CashStateRow, "cash_state")


def load_trades_history(csv_path: Path | None = None) -> pd.DataFrame:
    """
    Load and validate the trade history.
    """
    path = csv_path or (DATA_DIR / "state" / "trades_history.csv")
    df = _load_csv(path, "trades_history")
    return _validate_dataframe_rows(df, TradeHistoryRow, "trades_history")


if __name__ == "__main__":
    print("Loader module imported successfully.")