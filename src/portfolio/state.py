from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.io.loaders import load_cash_state, load_portfolio_snapshot


@dataclass
class PortfolioSummary:
    total_positions_value: float
    total_cash: float
    total_portfolio_value: float
    core_positions_value: float
    opportunistic_positions_value: float
    core_cash: float
    opportunistic_cash: float
    core_total_value: float
    opportunistic_total_value: float
    core_weight_total: float
    opportunistic_weight_total: float
    cash_weight_total: float


def compute_portfolio_state(
    portfolio_df: pd.DataFrame | None = None,
    cash_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, PortfolioSummary]:
    """
    Compute an updated portfolio state from positions + cash.

    Returns:
    - updated positions DataFrame with recalculated values/weights
    - PortfolioSummary object with portfolio-level metrics
    """
    if portfolio_df is None:
        portfolio_df = load_portfolio_snapshot()

    if cash_df is None:
        cash_df = load_cash_state()

    positions = portfolio_df.copy()
    cash = cash_df.copy()

    if cash.empty:
        raise ValueError("cash_state is empty")

    latest_cash = cash.iloc[-1]

    positions["recalc_market_value"] = positions["quantity"] * positions["last_price"]

    total_positions_value = float(positions["recalc_market_value"].sum())
    total_cash = float(latest_cash["cash_total"])
    total_portfolio_value = total_positions_value + total_cash

    if total_portfolio_value <= 0:
        raise ValueError("Total portfolio value must be strictly positive")

    positions["recalc_weight_total"] = positions["recalc_market_value"] / total_portfolio_value

    core_mask = positions["bucket"] == "core"
    opp_mask = positions["bucket"] == "opportunistic"

    core_positions_value = float(positions.loc[core_mask, "recalc_market_value"].sum())
    opportunistic_positions_value = float(positions.loc[opp_mask, "recalc_market_value"].sum())

    core_cash = float(latest_cash["cash_core"])
    opportunistic_cash = float(latest_cash["cash_opp"])

    core_total_value = core_positions_value + core_cash
    opportunistic_total_value = opportunistic_positions_value + opportunistic_cash

    positions["recalc_weight_bucket"] = 0.0

    if core_total_value > 0:
        positions.loc[core_mask, "recalc_weight_bucket"] = (
            positions.loc[core_mask, "recalc_market_value"] / core_total_value
        )

    if opportunistic_total_value > 0:
        positions.loc[opp_mask, "recalc_weight_bucket"] = (
            positions.loc[opp_mask, "recalc_market_value"] / opportunistic_total_value
        )

    core_weight_total = core_total_value / total_portfolio_value
    opportunistic_weight_total = opportunistic_total_value / total_portfolio_value
    cash_weight_total = total_cash / total_portfolio_value

    summary = PortfolioSummary(
        total_positions_value=total_positions_value,
        total_cash=total_cash,
        total_portfolio_value=total_portfolio_value,
        core_positions_value=core_positions_value,
        opportunistic_positions_value=opportunistic_positions_value,
        core_cash=core_cash,
        opportunistic_cash=opportunistic_cash,
        core_total_value=core_total_value,
        opportunistic_total_value=opportunistic_total_value,
        core_weight_total=core_weight_total,
        opportunistic_weight_total=opportunistic_weight_total,
        cash_weight_total=cash_weight_total,
    )

    return positions, summary


def print_portfolio_summary(summary: PortfolioSummary) -> None:
    """
    Print a simple human-readable portfolio summary.
    """
    print("\n=== Portfolio Summary ===")
    print(f"Total positions value     : {summary.total_positions_value:,.2f} EUR")
    print(f"Total cash               : {summary.total_cash:,.2f} EUR")
    print(f"Total portfolio value    : {summary.total_portfolio_value:,.2f} EUR")
    print(f"Core total value         : {summary.core_total_value:,.2f} EUR")
    print(f"Opportunistic total value: {summary.opportunistic_total_value:,.2f} EUR")
    print(f"Core weight total        : {summary.core_weight_total:.2%}")
    print(f"Opp weight total         : {summary.opportunistic_weight_total:.2%}")
    print(f"Cash weight total        : {summary.cash_weight_total:.2%}")


if __name__ == "__main__":
    positions_df, summary = compute_portfolio_state()
    print(positions_df)
    print_portfolio_summary(summary)