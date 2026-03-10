from __future__ import annotations

import argparse
import heapq
import math
from functools import reduce
from math import gcd

import pandas as pd

from src.engines.allocation_engine import build_full_target_allocation
from src.features.feature_pipeline import build_latest_feature_snapshot
from src.io.loaders import load_universe
from src.io.market_data import get_latest_price_snapshot, load_all_price_data
from src.settings import STATE_DATA_DIR, ensure_project_directories


DEFAULT_INITIAL_CAPITAL = 50_000.0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate portfolio_snapshot.csv and cash_state.csv from current engines."
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=DEFAULT_INITIAL_CAPITAL,
        help="Initial capital to allocate (default: 50000).",
    )
    parser.add_argument(
        "--min-order-eur",
        type=float,
        default=50.0,
        help="Minimum target line size in EUR before normalization (default: 50).",
    )
    return parser.parse_args()


def _sanitize_existing_state_files() -> None:
    """
    Core engine may read the existing state files while the seed is being built.
    So we sanitize any pre-existing snapshot/cash files *before* calling
    build_full_target_allocation().
    """
    portfolio_path = STATE_DATA_DIR / "portfolio_snapshot.csv"
    cash_path = STATE_DATA_DIR / "cash_state.csv"

    if portfolio_path.exists():
        df = pd.read_csv(portfolio_path)

        required_defaults = {
            "date": pd.Timestamp.today().date().isoformat(),
            "ticker": "",
            "bucket": "unknown",
            "quantity": 0,
            "avg_price": 0.0,
            "last_price": 0.0,
            "market_value": 0.0,
            "current_weight_total": 0.0,
            "current_weight_bucket": 0.0,
            "unrealized_pnl_pct": 0.0,
            "days_held": 0,
            "entry_date": pd.Timestamp.today().date().isoformat(),
            "entry_signal_type": "",
            "sector": "Unknown",
            "ttf_flag": False,
        }

        for col, default in required_defaults.items():
            if col not in df.columns:
                df[col] = default

        df["ticker"] = df["ticker"].fillna("").astype(str)
        df["bucket"] = df["bucket"].fillna("unknown").astype(str)
        df["sector"] = df["sector"].fillna("Unknown").astype(str)
        df.loc[df["sector"].str.strip() == "", "sector"] = "Unknown"

        numeric_cols = [
            "quantity",
            "avg_price",
            "last_price",
            "market_value",
            "current_weight_total",
            "current_weight_bucket",
            "unrealized_pnl_pct",
            "days_held",
        ]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        df["ttf_flag"] = df["ttf_flag"].fillna(False).astype(bool)

        ordered_cols = [
            "date",
            "ticker",
            "bucket",
            "quantity",
            "avg_price",
            "last_price",
            "market_value",
            "current_weight_total",
            "current_weight_bucket",
            "unrealized_pnl_pct",
            "days_held",
            "entry_date",
            "entry_signal_type",
            "sector",
            "ttf_flag",
        ]
        df = df[ordered_cols].copy()
        df.to_csv(portfolio_path, index=False)

    if cash_path.exists():
        df = pd.read_csv(cash_path)

        required_defaults = {
            "date": pd.Timestamp.today().date().isoformat(),
            "cash_total": 0.0,
            "cash_general": 0.0,
            "cash_core": 0.0,
            "cash_opp": 0.0,
            "cash_monetary": 0.0,
        }

        for col, default in required_defaults.items():
            if col not in df.columns:
                df[col] = default

        for col in ["cash_total", "cash_general", "cash_core", "cash_opp", "cash_monetary"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        df.to_csv(cash_path, index=False)


def _build_latest_price_map() -> dict[str, float]:
    all_prices = load_all_price_data()
    latest_snapshot = get_latest_price_snapshot(all_prices)
    latest_snapshot["adjusted_close"] = pd.to_numeric(latest_snapshot["adjusted_close"], errors="coerce")
    latest_snapshot = latest_snapshot.dropna(subset=["ticker", "adjusted_close"]).copy()
    latest_snapshot = latest_snapshot.loc[latest_snapshot["adjusted_close"] > 0].copy()
    latest_snapshot["ticker"] = latest_snapshot["ticker"].astype(str)
    return dict(zip(latest_snapshot["ticker"], latest_snapshot["adjusted_close"]))


def _eur_to_cents(x: float) -> int:
    return int(round(float(x) * 100))


def _solve_exact_best_fill_unbounded(prices_cents: list[int], budget_cents: int) -> tuple[list[int], int]:
    n = len(prices_cents)
    if n == 0 or budget_cents <= 0:
        return [0] * n, 0

    affordable_idx = [i for i, p in enumerate(prices_cents) if p <= budget_cents]
    if not affordable_idx:
        return [0] * n, 0

    aff_prices = [prices_cents[i] for i in affordable_idx]
    common_gcd = reduce(gcd, aff_prices)
    scaled_prices = [p // common_gcd for p in aff_prices]
    scaled_budget = budget_cents // common_gcd

    min_scaled_price = min(scaled_prices)
    cheapest_local_idx = scaled_prices.index(min_scaled_price)

    inf = 10**30
    dist = [inf] * min_scaled_price
    prev_residue = [-1] * min_scaled_price
    prev_coin_local = [-1] * min_scaled_price

    dist[0] = 0
    heap: list[tuple[int, int]] = [(0, 0)]

    while heap:
        current_dist, residue = heapq.heappop(heap)
        if current_dist != dist[residue]:
            continue

        for local_idx, coin in enumerate(scaled_prices):
            new_dist = current_dist + coin
            if new_dist > scaled_budget:
                continue

            new_residue = new_dist % min_scaled_price
            if new_dist < dist[new_residue]:
                dist[new_residue] = new_dist
                prev_residue[new_residue] = residue
                prev_coin_local[new_residue] = local_idx
                heapq.heappush(heap, (new_dist, new_residue))

    best_scaled_spend = 0
    best_residue = 0

    for residue in range(min_scaled_price):
        d = dist[residue]
        if d == inf or d > scaled_budget:
            continue

        k = (scaled_budget - d) // min_scaled_price
        candidate = d + k * min_scaled_price

        if candidate > best_scaled_spend:
            best_scaled_spend = candidate
            best_residue = residue

    local_counts = [0] * len(affordable_idx)
    residue = best_residue

    while residue != 0:
        coin_local = prev_coin_local[residue]
        if coin_local == -1:
            break
        local_counts[coin_local] += 1
        residue = prev_residue[residue]

    used_scaled = sum(c * p for c, p in zip(local_counts, scaled_prices))
    extra_cheapest = (best_scaled_spend - used_scaled) // min_scaled_price
    local_counts[cheapest_local_idx] += extra_cheapest

    counts = [0] * n
    for local_idx, global_idx in enumerate(affordable_idx):
        counts[global_idx] = local_counts[local_idx]

    best_spent_cents = best_scaled_spend * common_gcd
    return counts, best_spent_cents


def _allocate_seed_positions(
    target_df: pd.DataFrame,
    latest_price_map: dict[str, float],
    initial_capital: float,
    min_order_eur: float,
) -> pd.DataFrame:
    if initial_capital <= 0:
        raise ValueError("Initial capital must be strictly positive.")
    if min_order_eur < 0:
        raise ValueError("min_order_eur must be non-negative.")

    work = target_df.copy()

    work["ticker"] = work["ticker"].astype(str)
    work["target_weight_final"] = pd.to_numeric(work["target_weight_final"], errors="coerce").fillna(0.0)
    work = work.loc[work["target_weight_final"] > 0].copy()

    work["last_price"] = work["ticker"].map(latest_price_map)
    work["last_price"] = pd.to_numeric(work["last_price"], errors="coerce")
    work = work.dropna(subset=["last_price"]).copy()
    work = work.loc[work["last_price"] > 0].copy()

    if work.empty:
        raise ValueError("No valid target rows with prices available.")

    original_weight_sum = float(work["target_weight_final"].sum())

    work["target_eur_pre"] = initial_capital * work["target_weight_final"]
    work = work.loc[work["target_eur_pre"] >= min_order_eur].copy()

    if work.empty:
        raise ValueError("All target rows filtered out by min_order_eur.")

    retained_sum = float(work["target_weight_final"].sum())
    if retained_sum <= 0:
        raise ValueError("Retained target weights sum to zero.")

    work["seed_weight"] = work["target_weight_final"] / retained_sum
    work["target_eur"] = initial_capital * work["seed_weight"]

    work["quantity"] = (work["target_eur"] / work["last_price"]).apply(math.floor).astype(int)

    work = work.loc[work["quantity"] > 0].copy()
    if work.empty:
        raise ValueError("All retained rows floor to zero shares.")

    retained_sum_2 = float(work["seed_weight"].sum())
    if retained_sum_2 <= 0:
        raise ValueError("Investable seed weights sum to zero after zero-share filtering.")

    work["seed_weight"] = work["seed_weight"] / retained_sum_2
    work["target_eur"] = initial_capital * work["seed_weight"]
    work["quantity"] = (work["target_eur"] / work["last_price"]).apply(math.floor).astype(int)
    work["quantity"] = work["quantity"].clip(lower=1)

    work["market_value"] = work["quantity"] * work["last_price"]
    work["remaining_gap"] = work["target_eur"] - work["market_value"]

    residual_cash = float(initial_capital - work["market_value"].sum())
    min_price = float(work["last_price"].min())

    max_iterations = 500_000
    iterations = 0

    while residual_cash + 1e-12 >= min_price and iterations < max_iterations:
        affordable = work.loc[work["last_price"] <= residual_cash + 1e-12].copy()
        if affordable.empty:
            break

        affordable = affordable.sort_values(
            ["remaining_gap", "seed_weight", "last_price"],
            ascending=[False, False, True],
        )

        idx = affordable.index[0]
        px = float(work.loc[idx, "last_price"])

        if px > residual_cash + 1e-12:
            break

        work.loc[idx, "quantity"] += 1
        work.loc[idx, "market_value"] = float(work.loc[idx, "quantity"]) * px
        work.loc[idx, "remaining_gap"] = float(work.loc[idx, "target_eur"]) - float(work.loc[idx, "market_value"])

        residual_cash -= px
        iterations += 1

    residual_cash = max(0.0, residual_cash)
    residual_cash_cents = _eur_to_cents(residual_cash)

    prices_cents = [_eur_to_cents(px) for px in work["last_price"].tolist()]
    extra_counts, exact_spent_cents = _solve_exact_best_fill_unbounded(
        prices_cents=prices_cents,
        budget_cents=residual_cash_cents,
    )

    if exact_spent_cents > 0:
        for row_pos, extra_qty in enumerate(extra_counts):
            if extra_qty > 0:
                idx = work.index[row_pos]
                work.loc[idx, "quantity"] += int(extra_qty)

    work["market_value"] = work["quantity"] * work["last_price"]
    work["remaining_gap"] = work["target_eur"] - work["market_value"]
    work["original_weight_sum"] = original_weight_sum

    final_invested_value = float(work["market_value"].sum())
    final_residual_cash = float(initial_capital - final_invested_value)
    final_min_price = float(work["last_price"].min())

    if final_residual_cash + 1e-8 >= final_min_price:
        raise RuntimeError(
            "Residual cash is still greater than or equal to the cheapest retained price. "
            "This means the seed allocation failed to deploy all investable cash."
        )

    return work


def main() -> None:
    args = _parse_args()
    ensure_project_directories()

    _sanitize_existing_state_files()

    initial_capital = float(args.capital)
    min_order_eur = float(args.min_order_eur)

    universe_df = load_universe().copy()
    universe_map = {}
    if not universe_df.empty and "ticker" in universe_df.columns:
        universe_df = universe_df.copy()
        universe_df["ticker"] = universe_df["ticker"].astype(str)
        universe_map = universe_df.set_index("ticker").to_dict(orient="index")

    latest_features_df = build_latest_feature_snapshot()
    allocation_result = build_full_target_allocation(
        latest_features_df=latest_features_df,
        universe_df=universe_df,
    )

    target_df = allocation_result.combined_target_weights_table.copy()
    if target_df.empty:
        raise ValueError("Combined target allocation is empty.")

    latest_price_map = _build_latest_price_map()
    latest_date = pd.to_datetime(latest_features_df["date"]).max().date()

    allocated_df = _allocate_seed_positions(
        target_df=target_df,
        latest_price_map=latest_price_map,
        initial_capital=initial_capital,
        min_order_eur=min_order_eur,
    )

    allocated_invested_value = float(allocated_df["market_value"].sum())
    allocated_cash_left = float(initial_capital - allocated_invested_value)

    rows = []
    missing_universe_tickers: list[str] = []

    for _, item in allocated_df.iterrows():
        ticker = str(item["ticker"])
        bucket = str(item["bucket"])

        info = universe_map.get(ticker, {})
        if ticker not in universe_map:
            missing_universe_tickers.append(ticker)

        sector_value = info.get("sector", "Unknown")
        if pd.isna(sector_value) or sector_value is None or str(sector_value).strip() == "":
            sector_value = "Unknown"

        rows.append(
            {
                "date": latest_date,
                "ticker": ticker,
                "bucket": bucket,
                "quantity": int(item["quantity"]),
                "avg_price": float(item["last_price"]),
                "last_price": float(item["last_price"]),
                "market_value": float(item["market_value"]),
                "entry_signal_type": "seed_from_current_strategy",
                "sector": str(sector_value),
                "ttf_flag": bool(info.get("ttf_flag", False)),
            }
        )

    positions_df = pd.DataFrame(rows)
    if positions_df.empty:
        raise ValueError("No seeded positions were created.")

    positions_df["sector"] = positions_df["sector"].fillna("Unknown").astype(str)
    positions_df["bucket"] = positions_df["bucket"].fillna("unknown").astype(str)
    positions_df["ticker"] = positions_df["ticker"].astype(str)

    invested_value = float(positions_df["market_value"].sum())
    cash_general = round(max(0.0, initial_capital - invested_value), 2)
    total_portfolio_value = invested_value + cash_general

    positions_df["current_weight_total"] = positions_df["market_value"] / total_portfolio_value
    positions_df["current_weight_bucket"] = 0.0

    for bucket_name in positions_df["bucket"].dropna().unique():
        bucket_mask = positions_df["bucket"] == bucket_name
        bucket_invested = float(positions_df.loc[bucket_mask, "market_value"].sum())
        if bucket_invested > 0:
            positions_df.loc[bucket_mask, "current_weight_bucket"] = (
                positions_df.loc[bucket_mask, "market_value"] / bucket_invested
            )

    positions_df["unrealized_pnl_pct"] = 0.0
    positions_df["days_held"] = 0
    positions_df["entry_date"] = latest_date

    portfolio_snapshot_df = positions_df[
        [
            "date",
            "ticker",
            "bucket",
            "quantity",
            "avg_price",
            "last_price",
            "market_value",
            "current_weight_total",
            "current_weight_bucket",
            "unrealized_pnl_pct",
            "days_held",
            "entry_date",
            "entry_signal_type",
            "sector",
            "ttf_flag",
        ]
    ].copy()

    portfolio_snapshot_df["sector"] = portfolio_snapshot_df["sector"].fillna("Unknown").astype(str)
    portfolio_snapshot_df["bucket"] = portfolio_snapshot_df["bucket"].fillna("unknown").astype(str)
    portfolio_snapshot_df["ticker"] = portfolio_snapshot_df["ticker"].astype(str)

    cash_state_df = pd.DataFrame(
        [
            {
                "date": latest_date,
                "cash_total": cash_general,
                "cash_general": cash_general,
                "cash_core": 0.0,
                "cash_opp": 0.0,
                "cash_monetary": 0.0,
            }
        ]
    )

    portfolio_path = STATE_DATA_DIR / "portfolio_snapshot.csv"
    cash_path = STATE_DATA_DIR / "cash_state.csv"

    portfolio_snapshot_df.to_csv(portfolio_path, index=False)
    cash_state_df.to_csv(cash_path, index=False)

    summary_df = (
        portfolio_snapshot_df.groupby("bucket", as_index=False)
        .agg(
            num_lines=("ticker", "size"),
            invested_value=("market_value", "sum"),
        )
        .sort_values("bucket")
        .reset_index(drop=True)
    )

    print(f"Saved: {portfolio_path}")
    print(f"Saved: {cash_path}")
    print(f"Portfolio date: {latest_date}")
    print(f"Initial capital: {initial_capital:.2f} EUR")
    print(f"Original target weight sum: {allocated_df['original_weight_sum'].iloc[0]:.6f}")
    print(f"Normalized retained weight sum: {allocated_df['seed_weight'].sum():.6f}")
    print(f"Min retained price: {allocated_df['last_price'].min():.2f} EUR")
    print(f"Allocated invested value: {allocated_invested_value:.2f} EUR")
    print(f"Allocated residual cash: {allocated_cash_left:.2f} EUR")
    print(f"Snapshot invested value: {invested_value:.2f} EUR")
    print(f"Cash general: {cash_general:.2f} EUR")
    print(f"Total portfolio value: {total_portfolio_value:.2f} EUR")

    if missing_universe_tickers:
        unique_missing = sorted(set(missing_universe_tickers))
        print("\nTickers missing from universe but kept in seeded portfolio:")
        print(", ".join(unique_missing))

    print("\nSeeded portfolio by bucket:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()