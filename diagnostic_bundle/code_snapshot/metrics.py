from __future__ import annotations

import numpy as np
import pandas as pd


def _clean_numeric_series(series: pd.Series) -> pd.Series:
    """
    Convert a pandas Series to numeric and remove missing / infinite values.
    """
    cleaned = pd.to_numeric(series, errors="coerce")
    cleaned = cleaned.replace([np.inf, -np.inf], np.nan)
    return cleaned.dropna()


def _annualized_return_from_series(series: pd.Series) -> float:
    """
    Compute annualized return from a value series.
    Missing / invalid values are removed first.
    """
    clean = _clean_numeric_series(series)

    if len(clean) < 2:
        return 0.0

    start = float(clean.iloc[0])
    end = float(clean.iloc[-1])

    if start <= 0 or end <= 0:
        return 0.0

    years = len(clean) / 252
    if years <= 0:
        return 0.0

    return float((end / start) ** (1 / years) - 1)


def _max_drawdown(series: pd.Series) -> float:
    """
    Compute max drawdown from a value series.
    Missing / invalid values are removed first.
    """
    clean = _clean_numeric_series(series)

    if len(clean) == 0:
        return 0.0

    running_max = clean.cummax()
    drawdown = (clean / running_max) - 1.0
    return float(drawdown.min())


def compute_backtest_metrics(
    equity_curve: pd.DataFrame,
    orders_history: pd.DataFrame | None = None,
) -> dict[str, float]:
    """
    Compute serious backtest metrics.

    Expected columns in equity_curve:
    - date
    - portfolio_value
    - benchmark_value (optional)
    - daily_turnover (optional)
    - costs_paid_this_day (optional)
    """
    if equity_curve.empty:
        raise ValueError("equity_curve is empty")

    df = equity_curve.copy().sort_values("date").reset_index(drop=True)

    if "portfolio_value" not in df.columns:
        raise ValueError("equity_curve must contain 'portfolio_value'")

    df["portfolio_value"] = pd.to_numeric(df["portfolio_value"], errors="coerce")
    df["portfolio_value"] = df["portfolio_value"].replace([np.inf, -np.inf], np.nan)

    portfolio_series = df["portfolio_value"].dropna()

    if len(portfolio_series) < 2:
        raise ValueError("portfolio_value must contain at least two valid observations")

    df["portfolio_return"] = df["portfolio_value"].pct_change().fillna(0.0)

    total_return = float(portfolio_series.iloc[-1] / portfolio_series.iloc[0] - 1.0)
    cagr = float(_annualized_return_from_series(df["portfolio_value"]))
    annualized_volatility = float(df["portfolio_return"].std(ddof=0) * np.sqrt(252))

    sharpe = 0.0
    if annualized_volatility > 0:
        sharpe = float((df["portfolio_return"].mean() * 252) / annualized_volatility)

    max_drawdown = _max_drawdown(df["portfolio_value"])

    metrics = {
        "total_return": total_return,
        "cagr": cagr,
        "annualized_volatility": annualized_volatility,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
    }

    if "benchmark_value" in df.columns:
        df["benchmark_value"] = pd.to_numeric(df["benchmark_value"], errors="coerce")
        df["benchmark_value"] = df["benchmark_value"].replace([np.inf, -np.inf], np.nan)

        benchmark_series = df["benchmark_value"].dropna()

        if len(benchmark_series) >= 2:
            df["benchmark_return"] = df["benchmark_value"].pct_change().fillna(0.0)

            benchmark_total_return = float(benchmark_series.iloc[-1] / benchmark_series.iloc[0] - 1.0)
            benchmark_cagr = float(_annualized_return_from_series(df["benchmark_value"]))
            active_return = float(total_return - benchmark_total_return)

            active_daily_df = df[["portfolio_return", "benchmark_return"]].copy()
            active_daily_df = active_daily_df.replace([np.inf, -np.inf], np.nan).dropna()

            if len(active_daily_df) > 0:
                active_daily = active_daily_df["portfolio_return"] - active_daily_df["benchmark_return"]
                tracking_error = float(active_daily.std(ddof=0) * np.sqrt(252))
            else:
                active_daily = pd.Series(dtype=float)
                tracking_error = 0.0

            information_ratio = 0.0
            if tracking_error > 0 and len(active_daily) > 0:
                information_ratio = float((active_daily.mean() * 252) / tracking_error)

            metrics.update(
                {
                    "benchmark_total_return": benchmark_total_return,
                    "benchmark_cagr": benchmark_cagr,
                    "active_return": active_return,
                    "tracking_error": tracking_error,
                    "information_ratio": information_ratio,
                }
            )
        else:
            metrics.update(
                {
                    "benchmark_total_return": 0.0,
                    "benchmark_cagr": 0.0,
                    "active_return": 0.0,
                    "tracking_error": 0.0,
                    "information_ratio": 0.0,
                }
            )

    if "daily_turnover" in df.columns:
        df["daily_turnover"] = pd.to_numeric(df["daily_turnover"], errors="coerce").fillna(0.0)
        rebalance_days = df.loc[df["daily_turnover"] > 0, "daily_turnover"]
        metrics["average_rebalance_turnover"] = float(rebalance_days.mean()) if not rebalance_days.empty else 0.0
        metrics["total_turnover"] = float(df["daily_turnover"].sum())

    if "costs_paid_this_day" in df.columns:
        df["costs_paid_this_day"] = pd.to_numeric(df["costs_paid_this_day"], errors="coerce").fillna(0.0)
        total_costs_paid = float(df["costs_paid_this_day"].sum())
        metrics["total_costs_paid"] = total_costs_paid
        initial_value = float(portfolio_series.iloc[0])
        metrics["total_costs_as_pct_initial"] = total_costs_paid / initial_value if initial_value > 0 else 0.0

    if orders_history is not None and not orders_history.empty:
        metrics["num_orders"] = float(len(orders_history))
        metrics["num_executed_orders"] = (
            float((orders_history["execute"] == True).sum())
            if "execute" in orders_history.columns
            else float(len(orders_history))
        )

    return metrics


if __name__ == "__main__":
    sample = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=6, freq="B"),
            "portfolio_value": [100, 101, 99, 103, 104, 106],
            "benchmark_value": [100, 100.5, 100.2, 101, 101.3, 102],
            "daily_turnover": [0, 0.10, 0, 0.08, 0, 0],
            "costs_paid_this_day": [0, 1, 0, 1.2, 0, 0],
        }
    )
    print(compute_backtest_metrics(sample))