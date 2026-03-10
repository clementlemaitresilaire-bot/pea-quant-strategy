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
from src.settings import STATE_DATA_DIR


def _load_csv(csv_path: Path, dataset_name: str) -> pd.DataFrame:
    """
    Load a CSV file as a raw DataFrame.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"{dataset_name} file not found: {csv_path}")

    return pd.read_csv(csv_path)


def _validate_dataframe_rows(
    df: pd.DataFrame,
    schema: Type[BaseModel],
    dataset_name: str,
) -> pd.DataFrame:
    """
    Validate each row against a Pydantic schema and rebuild a normalized DataFrame.
    """
    if df is None or df.empty:
        return pd.DataFrame()

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


def _convert_datetime_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="raise")
    return out


def _assert_unique_keys(
    df: pd.DataFrame,
    dataset_name: str,
    key_columns: list[str],
) -> None:
    if df.empty:
        return

    dup_mask = df.duplicated(subset=key_columns, keep=False)
    if dup_mask.any():
        dup_preview = df.loc[dup_mask, key_columns].head(10).to_dict(orient="records")
        raise ValueError(
            f"Duplicate key rows found in dataset '{dataset_name}' for keys {key_columns}. "
            f"Examples: {dup_preview}"
        )


def load_universe(csv_path: Path | None = None) -> pd.DataFrame:
    """
    Load and validate the investment universe.
    Global integrity rules:
    - one row per ticker
    """
    path = csv_path or (STATE_DATA_DIR / "universe.csv")
    df = _load_csv(path, "universe")
    df = _validate_dataframe_rows(df, UniverseRow, "universe")

    if df.empty:
        return df

    _assert_unique_keys(df, "universe", ["ticker"])

    df = (
        df.sort_values(["ticker"])
        .reset_index(drop=True)
    )
    return df


def load_portfolio_snapshot(csv_path: Path | None = None) -> pd.DataFrame:
    """
    Load and validate the current portfolio snapshot.
    Global integrity rules:
    - one row per (date, ticker)
    """
    path = csv_path or (STATE_DATA_DIR / "portfolio_snapshot.csv")
    df = _load_csv(path, "portfolio_snapshot")
    df = _validate_dataframe_rows(df, PortfolioSnapshotRow, "portfolio_snapshot")

    if df.empty:
        return df

    df = _convert_datetime_columns(df, ["date", "entry_date"])
    _assert_unique_keys(df, "portfolio_snapshot", ["date", "ticker"])

    df = (
        df.sort_values(["date", "bucket", "ticker"])
        .reset_index(drop=True)
    )
    return df


def load_cash_state(csv_path: Path | None = None) -> pd.DataFrame:
    """
    Load and validate the cash state history.
    Global integrity rules:
    - one row per date
    """
    path = csv_path or (STATE_DATA_DIR / "cash_state.csv")
    df = _load_csv(path, "cash_state")
    df = _validate_dataframe_rows(df, CashStateRow, "cash_state")

    if df.empty:
        return df

    df = _convert_datetime_columns(df, ["date"])
    _assert_unique_keys(df, "cash_state", ["date"])

    df = (
        df.sort_values(["date"])
        .reset_index(drop=True)
    )
    return df


def load_trades_history(csv_path: Path | None = None) -> pd.DataFrame:
    """
    Load and validate the trade history.
    Global integrity rules:
    - no exact duplicate trade rows on the same key fields
    """
    path = csv_path or (STATE_DATA_DIR / "trades_history.csv")
    df = _load_csv(path, "trades_history")
    df = _validate_dataframe_rows(df, TradeHistoryRow, "trades_history")

    if df.empty:
        return df

    df = _convert_datetime_columns(df, ["trade_date"])

    _assert_unique_keys(
        df,
        "trades_history",
        ["trade_date", "ticker", "side", "quantity", "price", "reason_code"],
    )

    df = (
        df.sort_values(["trade_date", "ticker", "side"])
        .reset_index(drop=True)
    )
    return df


if __name__ == "__main__":
    print("Loader module imported successfully.")