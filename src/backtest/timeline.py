from __future__ import annotations

import pandas as pd


def build_trading_calendar(price_df: pd.DataFrame) -> list[pd.Timestamp]:
    """
    Build the full trading calendar from available price dates.
    """
    if "date" not in price_df.columns:
        raise ValueError("price_df must contain a 'date' column")

    dates = pd.to_datetime(price_df["date"]).drop_duplicates().sort_values()
    return list(dates)


def build_rebalance_calendar(
    price_df: pd.DataFrame,
    frequency: str = "monthly",
) -> list[pd.Timestamp]:
    """
    Build rebalance dates from available trading dates.

    Supported frequencies:
    - daily
    - weekly
    - monthly
    """
    trading_dates = pd.Series(build_trading_calendar(price_df), name="date")
    calendar_df = pd.DataFrame({"date": trading_dates})
    calendar_df["year"] = calendar_df["date"].dt.year
    calendar_df["month"] = calendar_df["date"].dt.month
    calendar_df["week"] = calendar_df["date"].dt.isocalendar().week.astype(int)

    if frequency == "daily":
        rebalance_dates = calendar_df["date"]

    elif frequency == "weekly":
        rebalance_dates = calendar_df.groupby(["year", "week"])["date"].max().reset_index(drop=True)

    elif frequency == "monthly":
        rebalance_dates = calendar_df.groupby(["year", "month"])["date"].max().reset_index(drop=True)

    else:
        raise ValueError("frequency must be one of: daily, weekly, monthly")

    return list(rebalance_dates)


if __name__ == "__main__":
    sample = pd.DataFrame(
        {"date": pd.date_range("2025-01-01", periods=40, freq="B")}
    )
    print(build_trading_calendar(sample)[:5])
    print(build_rebalance_calendar(sample, frequency="monthly"))