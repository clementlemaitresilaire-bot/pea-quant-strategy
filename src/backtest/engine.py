from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.backtest.metrics import compute_backtest_metrics
from src.backtest.simulator import (
    build_order_proposals_at_date,
    execute_orders,
    get_latest_prices_until_date,
    initialize_portfolio_state,
    initialize_portfolio_state_from_target_weights_at_date,
    mark_to_market_portfolio,
)
from src.engines.allocation_engine import build_full_target_allocation
from src.features.feature_pipeline import get_latest_features
from src.features.price_features import compute_price_features
from src.io.loaders import load_universe
from src.io.market_data import load_all_price_data
from src.settings import load_config


@dataclass
class BacktestResult:
    equity_curve: pd.DataFrame
    orders_history: pd.DataFrame
    target_history: pd.DataFrame
    metrics: dict


def _get_all_trading_dates(price_df: pd.DataFrame) -> list[pd.Timestamp]:
    dates = pd.to_datetime(price_df["date"], errors="coerce").dropna().sort_values().unique()
    return list(pd.to_datetime(dates))


def _get_rebalance_dates(trading_dates: list[pd.Timestamp], rebalance_frequency: str) -> set[pd.Timestamp]:
    dates = pd.Series(pd.to_datetime(trading_dates)).sort_values().reset_index(drop=True)

    if rebalance_frequency == "daily":
        return set(dates.tolist())

    if rebalance_frequency == "weekly":
        week_period = dates.dt.to_period("W")
        rebalance = dates.groupby(week_period).max()
        return set(pd.to_datetime(rebalance).tolist())

    if rebalance_frequency == "monthly":
        month_period = dates.dt.to_period("M")
        rebalance = dates.groupby(month_period).max()
        return set(pd.to_datetime(rebalance).tolist())

    raise ValueError(f"Unsupported rebalance_frequency: {rebalance_frequency}")


def _get_next_trading_date(
    trading_dates: list[pd.Timestamp],
    current_date: pd.Timestamp,
) -> pd.Timestamp | None:
    current_date = pd.Timestamp(current_date)
    for d in trading_dates:
        if pd.Timestamp(d) > current_date:
            return pd.Timestamp(d)
    return None


def _build_signal_execution_schedule(
    trading_dates: list[pd.Timestamp],
    rebalance_dates: set[pd.Timestamp],
    signal_start_date: pd.Timestamp,
) -> dict[pd.Timestamp, pd.Timestamp]:
    """
    Build a mapping:
        execution_date -> signal_date

    Signal is computed on rebalance date t, and executed on next trading day t+1.
    """
    trading_dates = sorted(pd.to_datetime(trading_dates))
    signal_start_date = pd.Timestamp(signal_start_date)

    schedule: dict[pd.Timestamp, pd.Timestamp] = {}

    for signal_date in sorted(pd.to_datetime(list(rebalance_dates))):
        if signal_date < signal_start_date:
            continue

        execution_date = _get_next_trading_date(trading_dates, signal_date)
        if execution_date is None:
            continue

        schedule[pd.Timestamp(execution_date)] = pd.Timestamp(signal_date)

    return schedule


def _find_first_eligible_rebalance_date(
    full_features_df: pd.DataFrame,
    rebalance_dates: set[pd.Timestamp],
) -> pd.Timestamp:
    sorted_rebalances = sorted(pd.to_datetime(list(rebalance_dates)))
    if not sorted_rebalances:
        raise ValueError("No rebalance dates available.")

    required_feature_cols = [
        "mom12_ex1",
        "ret_6m",
        "ret_3m",
        "ret_20d",
        "vol60",
        "vol20",
        "dd_6m",
        "dist_ma200",
        "dist_ma20",
        "rsi14",
    ]

    for current_date in sorted_rebalances:
        sliced_features = full_features_df.loc[
            pd.to_datetime(full_features_df["date"], errors="coerce") <= current_date
        ].copy()

        if sliced_features.empty:
            continue

        latest_features_df = get_latest_features(sliced_features)

        available_cols = [c for c in required_feature_cols if c in latest_features_df.columns]
        if available_cols:
            valid_count = int(latest_features_df[available_cols].notna().all(axis=1).sum())
            if valid_count == 0:
                continue

        return pd.Timestamp(current_date)

    raise ValueError("Could not find any eligible rebalance date after warm-up.")


def _select_official_signal_start_date(
    eligible_start_date: pd.Timestamp,
    last_trading_date: pd.Timestamp,
    user_start_date: str | None = None,
    target_years: int = 10,
) -> pd.Timestamp:
    if user_start_date is not None:
        return max(pd.Timestamp(user_start_date), eligible_start_date)

    target_start = pd.Timestamp(last_trading_date) - pd.DateOffset(years=target_years)
    return max(target_start, eligible_start_date)


def _normalize_target_weights_to_full_investment(target_df: pd.DataFrame) -> pd.DataFrame:
    out = target_df.copy()

    if "target_weight_final" not in out.columns:
        return out

    out["target_weight_final"] = pd.to_numeric(out["target_weight_final"], errors="coerce").fillna(0.0)

    positive_mask = out["target_weight_final"] > 0
    total_weight = float(out.loc[positive_mask, "target_weight_final"].sum())

    if total_weight > 0:
        out.loc[positive_mask, "target_weight_final"] = (
            out.loc[positive_mask, "target_weight_final"] / total_weight
        )

    if "target_weight_raw" in out.columns:
        out["target_weight_raw"] = pd.to_numeric(out["target_weight_raw"], errors="coerce").fillna(0.0)
        raw_positive_mask = out["target_weight_raw"] > 0
        raw_total = float(out.loc[raw_positive_mask, "target_weight_raw"].sum())

        if raw_total > 0:
            out.loc[raw_positive_mask, "target_weight_raw"] = (
                out.loc[raw_positive_mask, "target_weight_raw"] / raw_total
            )

    return out


def _build_single_ticker_benchmark_curve(
    price_df: pd.DataFrame,
    trading_dates: list[pd.Timestamp],
    initial_capital: float,
    ticker: str,
) -> pd.DataFrame:
    bench_df = price_df.loc[price_df["ticker"].astype(str) == str(ticker)].copy()
    bench_df["date"] = pd.to_datetime(bench_df["date"], errors="coerce")
    bench_df["adjusted_close"] = pd.to_numeric(bench_df["adjusted_close"], errors="coerce")
    bench_df = bench_df.dropna(subset=["date", "adjusted_close"]).copy()

    if bench_df.empty:
        raise ValueError(
            f"Benchmark ticker '{ticker}' has no price history in the loaded price data. "
            "Either add this ticker to your data source or switch benchmark.mode."
        )

    bench_df = (
        bench_df[["date", "adjusted_close"]]
        .rename(columns={"adjusted_close": "benchmark_level"})
        .sort_values("date")
        .reset_index(drop=True)
    )

    out = pd.DataFrame({"date": trading_dates})
    out = out.merge(bench_df, on="date", how="left")
    out["benchmark_level"] = out["benchmark_level"].ffill().bfill()

    first_level = float(out["benchmark_level"].iloc[0]) if not out.empty else np.nan
    if pd.isna(first_level) or first_level <= 0:
        out["benchmark_value"] = initial_capital
    else:
        out["benchmark_value"] = initial_capital * out["benchmark_level"] / first_level

    return out[["date", "benchmark_value"]].copy()


def _build_synthetic_benchmark_curve(
    price_df: pd.DataFrame,
    trading_dates: list[pd.Timestamp],
    initial_capital: float,
) -> pd.DataFrame:
    universe_df = load_universe()
    benchmark_members = universe_df.loc[
        (universe_df["benchmark_member"] == True) & (universe_df["active"] == True),
        "ticker",
    ].drop_duplicates()

    bench_df = price_df.loc[price_df["ticker"].isin(benchmark_members)].copy()
    bench_df["date"] = pd.to_datetime(bench_df["date"], errors="coerce")
    bench_df["adjusted_close"] = pd.to_numeric(bench_df["adjusted_close"], errors="coerce")
    bench_df = bench_df.dropna(subset=["date", "adjusted_close"]).copy()

    if bench_df.empty:
        return pd.DataFrame({"date": trading_dates, "benchmark_value": [initial_capital] * len(trading_dates)})

    daily_mean = (
        bench_df.groupby("date", as_index=False)["adjusted_close"]
        .mean()
        .rename(columns={"adjusted_close": "benchmark_level"})
        .sort_values("date")
        .reset_index(drop=True)
    )

    out = pd.DataFrame({"date": trading_dates})
    out = out.merge(daily_mean[["date", "benchmark_level"]], on="date", how="left")
    out["benchmark_level"] = out["benchmark_level"].ffill().bfill()

    first_level = float(out["benchmark_level"].iloc[0]) if not out.empty else np.nan
    if pd.isna(first_level) or first_level <= 0:
        out["benchmark_value"] = initial_capital
    else:
        out["benchmark_value"] = initial_capital * out["benchmark_level"] / first_level

    return out[["date", "benchmark_value"]].copy()


def _build_benchmark_curve(
    price_df: pd.DataFrame,
    trading_dates: list[pd.Timestamp],
    initial_capital: float,
) -> pd.DataFrame:
    cfg = load_config()

    if cfg.benchmark.mode == "single_ticker":
        return _build_single_ticker_benchmark_curve(
            price_df=price_df,
            trading_dates=trading_dates,
            initial_capital=initial_capital,
            ticker=cfg.benchmark.ticker,
        )

    if cfg.benchmark.mode == "synthetic_universe":
        return _build_synthetic_benchmark_curve(
            price_df=price_df,
            trading_dates=trading_dates,
            initial_capital=initial_capital,
        )

    raise ValueError(f"Unsupported benchmark mode: {cfg.benchmark.mode}")


def run_backtest(
    initial_capital: float = 50_000.0,
    rebalance_frequency: str = "monthly",
    start_date: str | None = None,
    end_date: str | None = None,
    use_seed_state: bool = False,
    target_backtest_years: int = 10,
) -> BacktestResult:
    universe_df = load_universe().copy()

    price_df = load_all_price_data().copy()
    price_df["date"] = pd.to_datetime(price_df["date"], errors="coerce")
    price_df["adjusted_close"] = pd.to_numeric(price_df["adjusted_close"], errors="coerce")
    price_df = price_df.dropna(subset=["date", "adjusted_close"]).copy()
    price_df = price_df.sort_values(["date", "ticker"]).reset_index(drop=True)

    all_trading_dates_full = _get_all_trading_dates(price_df)
    if not all_trading_dates_full:
        raise ValueError("No trading dates found in price data.")

    if end_date is not None:
        last_trading_date = max(d for d in all_trading_dates_full if d <= pd.Timestamp(end_date))
    else:
        last_trading_date = all_trading_dates_full[-1]

    usable_price_df = price_df.loc[price_df["date"] <= last_trading_date].copy()
    all_trading_dates = _get_all_trading_dates(usable_price_df)
    rebalance_dates_all = _get_rebalance_dates(all_trading_dates, rebalance_frequency)

    full_features_df = compute_price_features(usable_price_df.copy())
    full_features_df["date"] = pd.to_datetime(full_features_df["date"], errors="coerce")
    full_features_df = full_features_df.dropna(subset=["date"]).copy()

    eligible_signal_start_date = _find_first_eligible_rebalance_date(
        full_features_df=full_features_df,
        rebalance_dates=rebalance_dates_all,
    )

    official_signal_start_date = _select_official_signal_start_date(
        eligible_start_date=eligible_signal_start_date,
        last_trading_date=last_trading_date,
        user_start_date=start_date,
        target_years=target_backtest_years,
    )

    execution_schedule_all = _build_signal_execution_schedule(
        trading_dates=all_trading_dates,
        rebalance_dates=rebalance_dates_all,
        signal_start_date=official_signal_start_date,
    )

    trading_dates = [
        d for d in all_trading_dates
        if d > official_signal_start_date and (end_date is None or d <= pd.Timestamp(end_date))
    ]
    trading_dates = [pd.Timestamp(d) for d in trading_dates if pd.Timestamp(d) in execution_schedule_all or pd.Timestamp(d) > official_signal_start_date]

    if not trading_dates:
        raise ValueError("No trading dates found for the measured backtest window after signal/execution shift.")

    first_execution_date = trading_dates[0]

    execution_schedule = {
        pd.Timestamp(exec_date): pd.Timestamp(signal_date)
        for exec_date, signal_date in execution_schedule_all.items()
        if pd.Timestamp(exec_date) in set(trading_dates)
    }

    if use_seed_state:
        state = initialize_portfolio_state(initial_capital=initial_capital)
        initial_target_df = pd.DataFrame()
    else:
        sliced_features = full_features_df.loc[
            pd.to_datetime(full_features_df["date"], errors="coerce") <= official_signal_start_date
        ].copy()
        latest_features_df = get_latest_features(sliced_features)

        initial_allocation_result = build_full_target_allocation(
            latest_features_df=latest_features_df,
            universe_df=universe_df,
        )
        initial_target_df = _normalize_target_weights_to_full_investment(
            initial_allocation_result.combined_target_weights_table.copy()
        )

        state = initialize_portfolio_state_from_target_weights_at_date(
            target_df=initial_target_df,
            initial_capital=initial_capital,
            start_date=first_execution_date,
            price_df=usable_price_df,
        )

    equity_rows: list[dict] = []
    orders_rows: list[pd.DataFrame] = []
    target_rows: list[pd.DataFrame] = []

    if not use_seed_state and not initial_target_df.empty:
        initial_target_snapshot = initial_target_df.copy()
        initial_target_snapshot["date"] = first_execution_date
        initial_target_snapshot["signal_date"] = official_signal_start_date
        initial_target_snapshot["execution_date"] = first_execution_date
        target_rows.append(initial_target_snapshot)

    first_prices = get_latest_prices_until_date(usable_price_df, first_execution_date)
    starting_portfolio_value = mark_to_market_portfolio(state, first_prices)["portfolio_value"]
    if starting_portfolio_value <= 0:
        starting_portfolio_value = initial_capital

    benchmark_curve = _build_benchmark_curve(
        price_df=usable_price_df,
        trading_dates=trading_dates,
        initial_capital=starting_portfolio_value,
    )
    benchmark_map = dict(zip(pd.to_datetime(benchmark_curve["date"]), benchmark_curve["benchmark_value"]))

    for current_date in trading_dates:
        latest_prices = get_latest_prices_until_date(usable_price_df, current_date)

        if current_date in execution_schedule:
            signal_date = execution_schedule[current_date]

            sliced_features = full_features_df.loc[
                pd.to_datetime(full_features_df["date"], errors="coerce") <= signal_date
            ].copy()
            latest_features_df = get_latest_features(sliced_features)

            allocation_result = build_full_target_allocation(
                latest_features_df=latest_features_df,
                universe_df=universe_df,
            )
            target_df = _normalize_target_weights_to_full_investment(
                allocation_result.combined_target_weights_table.copy()
            )
            target_df["date"] = current_date
            target_df["signal_date"] = signal_date
            target_df["execution_date"] = current_date
            target_rows.append(target_df)

            orders_df = build_order_proposals_at_date(
                target_df=target_df,
                state=state,
                latest_prices=latest_prices,
                universe_df=universe_df,
                latest_features_df=latest_features_df,
            )
            orders_df["date"] = current_date
            orders_df["signal_date"] = signal_date
            orders_rows.append(orders_df)

            state, _, costs_paid, gross_turnover = execute_orders(
                state=state,
                orders_df=orders_df,
                latest_prices=latest_prices,
            )
        else:
            costs_paid = 0.0
            gross_turnover = 0.0

        state.date = current_date

        mtm = mark_to_market_portfolio(state, latest_prices)
        portfolio_value = float(mtm["portfolio_value"])

        equity_rows.append(
            {
                "date": current_date,
                "benchmark_value": benchmark_map.get(pd.Timestamp(current_date), np.nan),
                "general_cash": mtm.get("general_cash", np.nan),
                "core_cash": mtm["core_cash"],
                "monetary_cash": mtm["monetary_cash"],
                "opp_cash": mtm["opp_cash"],
                "core_positions_value": mtm["core_positions_value"],
                "monetary_positions_value": mtm["monetary_positions_value"],
                "opp_positions_value": mtm["opp_positions_value"],
                "core_total_value": mtm["core_total_value"],
                "monetary_total_value": mtm["monetary_total_value"],
                "opp_total_value": mtm["opp_total_value"],
                "portfolio_value": portfolio_value,
                "daily_turnover": gross_turnover / portfolio_value if portfolio_value > 0 else 0.0,
                "costs_paid_this_day": costs_paid,
                "cumulative_costs_paid": state.cumulative_costs_paid,
                "num_core_positions": len(state.core.holdings),
                "num_monetary_positions": len(state.monetary.holdings),
                "num_opp_positions": len(state.opportunistic.holdings),
            }
        )

    equity_curve = pd.DataFrame(equity_rows).sort_values("date").reset_index(drop=True)

    orders_history = (
        pd.concat(orders_rows, ignore_index=True)
        if len(orders_rows) > 0
        else pd.DataFrame()
    )
    target_history = (
        pd.concat(target_rows, ignore_index=True)
        if len(target_rows) > 0
        else pd.DataFrame()
    )

    metrics = compute_backtest_metrics(
        equity_curve=equity_curve,
        orders_history=orders_history,
    )

    metrics["official_start_date"] = str(pd.Timestamp(first_execution_date).date())
    metrics["official_end_date"] = str(pd.Timestamp(trading_dates[-1]).date())
    metrics["warmup_start_date"] = str(pd.Timestamp(all_trading_dates[0]).date())
    metrics["eligible_start_date"] = str(pd.Timestamp(eligible_signal_start_date).date())
    metrics["signal_start_date"] = str(pd.Timestamp(official_signal_start_date).date())
    metrics["target_backtest_years"] = float(target_backtest_years)

    return BacktestResult(
        equity_curve=equity_curve,
        orders_history=orders_history,
        target_history=target_history,
        metrics=metrics,
    )


if __name__ == "__main__":
    result = run_backtest()

    print("\n=== BACKTEST METRICS ===")
    for k, v in result.metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.6f}")
        else:
            print(f"{k}: {v}")