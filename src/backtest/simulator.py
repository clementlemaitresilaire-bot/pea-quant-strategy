from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from functools import reduce
from math import gcd
from pathlib import Path

import pandas as pd

from src.features.cost_features import estimate_transaction_cost
from src.ml.overlays import apply_rebalance_overlay
from src.settings import STATE_DATA_DIR, load_config


@dataclass
class SleeveState:
    cash: float
    holdings: dict[str, float] = field(default_factory=dict)


@dataclass
class PortfolioState:
    date: pd.Timestamp | None
    core: SleeveState
    monetary: SleeveState
    opportunistic: SleeveState
    general_cash: float = 0.0
    cumulative_costs_paid: float = 0.0


def initialize_portfolio_state(initial_capital: float) -> PortfolioState:
    return PortfolioState(
        date=None,
        core=SleeveState(cash=0.0, holdings={}),
        monetary=SleeveState(cash=0.0, holdings={}),
        opportunistic=SleeveState(cash=0.0, holdings={}),
        general_cash=float(initial_capital),
        cumulative_costs_paid=0.0,
    )


def _eur_to_cents(x: float) -> int:
    return int(round(float(x) * 100))


def _load_seed_snapshot_raw() -> pd.DataFrame:
    path = Path(STATE_DATA_DIR) / "portfolio_snapshot.csv"
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)

    required_defaults = {
        "ticker": "",
        "bucket": "unknown",
        "quantity": 0.0,
        "last_price": 0.0,
        "market_value": 0.0,
        "current_weight_total": 0.0,
        "sector": "Unknown",
    }

    for col, default in required_defaults.items():
        if col not in df.columns:
            df[col] = default

    df["ticker"] = df["ticker"].fillna("").astype(str)
    df["bucket"] = df["bucket"].fillna("unknown").astype(str)
    df["sector"] = df["sector"].fillna("Unknown").astype(str)
    df.loc[df["sector"].str.strip() == "", "sector"] = "Unknown"

    for col in ["quantity", "last_price", "market_value", "current_weight_total"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df = df.loc[df["ticker"].str.strip() != ""].copy()
    return df


def get_latest_prices_until_date(price_df: pd.DataFrame, current_date: pd.Timestamp) -> dict[str, float]:
    df = price_df.loc[pd.to_datetime(price_df["date"], errors="coerce") <= current_date].copy()
    if df.empty:
        return {}

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["adjusted_close"] = pd.to_numeric(df["adjusted_close"], errors="coerce")
    df = df.dropna(subset=["date", "adjusted_close"]).copy()

    latest = (
        df.sort_values(["ticker", "date"])
        .groupby("ticker", as_index=False)
        .tail(1)
    )

    latest["ticker"] = latest["ticker"].astype(str)
    return dict(zip(latest["ticker"], latest["adjusted_close"]))


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


def _allocate_integer_positions_from_target_weights(
    target_df: pd.DataFrame,
    latest_prices: dict[str, float],
    initial_capital: float,
) -> tuple[pd.DataFrame, float]:
    work = target_df.copy()

    work["ticker"] = work["ticker"].astype(str)
    work["target_weight_final"] = pd.to_numeric(work["target_weight_final"], errors="coerce").fillna(0.0)
    work = work.loc[work["target_weight_final"] > 0].copy()

    work["hist_price"] = work["ticker"].map(latest_prices)
    work["hist_price"] = pd.to_numeric(work["hist_price"], errors="coerce")
    work = work.dropna(subset=["hist_price"]).copy()
    work = work.loc[work["hist_price"] > 0].copy()

    if work.empty:
        return pd.DataFrame(), float(initial_capital)

    weight_sum = float(work["target_weight_final"].sum())
    if weight_sum <= 0:
        return pd.DataFrame(), float(initial_capital)

    work["seed_weight"] = work["target_weight_final"] / weight_sum
    work["target_eur"] = initial_capital * work["seed_weight"]
    work["quantity"] = (work["target_eur"] / work["hist_price"]).apply(math.floor).astype(int)

    work = work.loc[work["quantity"] > 0].copy()
    if work.empty:
        return pd.DataFrame(), float(initial_capital)

    weight_sum_2 = float(work["seed_weight"].sum())
    work["seed_weight"] = work["seed_weight"] / weight_sum_2
    work["target_eur"] = initial_capital * work["seed_weight"]
    work["quantity"] = (work["target_eur"] / work["hist_price"]).apply(math.floor).astype(int)
    work["quantity"] = work["quantity"].clip(lower=1)

    work["market_value_hist"] = work["quantity"] * work["hist_price"]
    work["remaining_gap"] = work["target_eur"] - work["market_value_hist"]

    residual_cash = float(initial_capital - work["market_value_hist"].sum())
    min_price = float(work["hist_price"].min())

    max_iterations = 500_000
    iterations = 0

    while residual_cash + 1e-12 >= min_price and iterations < max_iterations:
        affordable = work.loc[work["hist_price"] <= residual_cash + 1e-12].copy()
        if affordable.empty:
            break

        affordable = affordable.sort_values(
            ["remaining_gap", "seed_weight", "hist_price"],
            ascending=[False, False, True],
        )

        idx = affordable.index[0]
        px = float(work.loc[idx, "hist_price"])

        if px > residual_cash + 1e-12:
            break

        work.loc[idx, "quantity"] += 1
        work.loc[idx, "market_value_hist"] = float(work.loc[idx, "quantity"]) * px
        work.loc[idx, "remaining_gap"] = float(work.loc[idx, "target_eur"]) - float(work.loc[idx, "market_value_hist"])

        residual_cash -= px
        iterations += 1

    residual_cash = max(0.0, residual_cash)
    residual_cash_cents = _eur_to_cents(residual_cash)
    prices_cents = [_eur_to_cents(px) for px in work["hist_price"].tolist()]

    extra_counts, _ = _solve_exact_best_fill_unbounded(
        prices_cents=prices_cents,
        budget_cents=residual_cash_cents,
    )

    for row_pos, extra_qty in enumerate(extra_counts):
        if extra_qty > 0:
            idx = work.index[row_pos]
            work.loc[idx, "quantity"] += int(extra_qty)

    work["market_value_hist"] = work["quantity"] * work["hist_price"]
    final_residual_cash = float(initial_capital - work["market_value_hist"].sum())

    return work, max(0.0, final_residual_cash)


_INITIAL_BACKTEST_BUCKET_WEIGHTS = {
    "core": 0.50,
    "monetary": 0.30,
    "opportunistic": 0.20,
}


def _allocate_integer_positions_from_target_weights_by_bucket(
    target_df: pd.DataFrame,
    latest_prices: dict[str, float],
    initial_capital: float,
    bucket_budget_weights: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, float]:
    work = target_df.copy()
    if work.empty:
        return pd.DataFrame(), float(initial_capital)

    work["ticker"] = work["ticker"].astype(str)
    work["bucket"] = work.get("bucket", "").fillna("").astype(str)
    work["target_weight_final"] = pd.to_numeric(work.get("target_weight_final", 0.0), errors="coerce").fillna(0.0)
    work = work.loc[work["target_weight_final"] > 0].copy()

    if work.empty:
        return pd.DataFrame(), float(initial_capital)

    bucket_budget_weights = dict(bucket_budget_weights or _INITIAL_BACKTEST_BUCKET_WEIGHTS)

    allocated_frames: list[pd.DataFrame] = []
    total_residual_cash = 0.0

    for bucket, bucket_weight in bucket_budget_weights.items():
        bucket_budget = float(initial_capital) * float(bucket_weight)
        if bucket_budget <= 0:
            continue

        bucket_target = work.loc[work["bucket"] == bucket].copy()
        if bucket_target.empty:
            total_residual_cash += bucket_budget
            continue

        bucket_weight_sum = float(bucket_target["target_weight_final"].sum())
        if bucket_weight_sum <= 0:
            total_residual_cash += bucket_budget
            continue

        bucket_target["target_weight_final"] = bucket_target["target_weight_final"] / bucket_weight_sum

        bucket_allocated, bucket_residual_cash = _allocate_integer_positions_from_target_weights(
            target_df=bucket_target,
            latest_prices=latest_prices,
            initial_capital=bucket_budget,
        )

        if not bucket_allocated.empty:
            bucket_allocated["bucket"] = bucket
            allocated_frames.append(bucket_allocated)

        total_residual_cash += float(bucket_residual_cash)

    if not allocated_frames:
        return pd.DataFrame(), float(initial_capital)

    allocated_df = pd.concat(allocated_frames, ignore_index=True)
    return allocated_df, max(0.0, float(total_residual_cash))


def initialize_portfolio_state_from_target_weights_at_date(
    target_df: pd.DataFrame,
    initial_capital: float,
    start_date: pd.Timestamp,
    price_df: pd.DataFrame,
) -> PortfolioState:
    start_prices = get_latest_prices_until_date(price_df, pd.Timestamp(start_date))
    if not start_prices:
        return initialize_portfolio_state(initial_capital=initial_capital)

    work, final_residual_cash = _allocate_integer_positions_from_target_weights_by_bucket(
        target_df=target_df,
        latest_prices=start_prices,
        initial_capital=initial_capital,
        bucket_budget_weights=_INITIAL_BACKTEST_BUCKET_WEIGHTS,
    )

    if work.empty:
        return initialize_portfolio_state(initial_capital=initial_capital)

    core_holdings: dict[str, float] = {}
    monetary_holdings: dict[str, float] = {}
    opp_holdings: dict[str, float] = {}

    for _, row in work.iterrows():
        ticker = str(row["ticker"])
        bucket = str(row["bucket"])
        qty = float(row["quantity"])

        if qty <= 0:
            continue

        if bucket == "core":
            core_holdings[ticker] = qty
        elif bucket == "monetary":
            monetary_holdings[ticker] = qty
        elif bucket == "opportunistic":
            opp_holdings[ticker] = qty

    return PortfolioState(
        date=None,
        core=SleeveState(cash=0.0, holdings=core_holdings),
        monetary=SleeveState(cash=0.0, holdings=monetary_holdings),
        opportunistic=SleeveState(cash=0.0, holdings=opp_holdings),
        general_cash=final_residual_cash,
        cumulative_costs_paid=0.0,
    )


def initialize_portfolio_state_from_seed_at_date(
    initial_capital: float,
    start_date: pd.Timestamp,
    price_df: pd.DataFrame,
) -> PortfolioState:
    seed_df = _load_seed_snapshot_raw()
    if seed_df.empty:
        return initialize_portfolio_state(initial_capital=initial_capital)

    start_prices = get_latest_prices_until_date(price_df, pd.Timestamp(start_date))
    if not start_prices:
        return initialize_portfolio_state(initial_capital=initial_capital)

    work = seed_df.copy()
    work["hist_price"] = work["ticker"].map(start_prices)
    work["hist_price"] = pd.to_numeric(work["hist_price"], errors="coerce")
    work = work.dropna(subset=["hist_price"]).copy()
    work = work.loc[work["hist_price"] > 0].copy()

    if work.empty:
        return initialize_portfolio_state(initial_capital=initial_capital)

    if "current_weight_total" in work.columns and float(work["current_weight_total"].sum()) > 0:
        work["seed_weight"] = pd.to_numeric(work["current_weight_total"], errors="coerce").fillna(0.0)
    else:
        work["seed_weight"] = pd.to_numeric(work["market_value"], errors="coerce").fillna(0.0)

    work = work.loc[work["seed_weight"] > 0].copy()
    if work.empty:
        return initialize_portfolio_state(initial_capital=initial_capital)

    work["target_weight_final"] = work["seed_weight"]
    return initialize_portfolio_state_from_target_weights_at_date(
        target_df=work[["ticker", "bucket", "target_weight_final"]].copy(),
        initial_capital=initial_capital,
        start_date=start_date,
        price_df=price_df,
    )


def initialize_portfolio_state_from_seed() -> PortfolioState:
    seed_df = _load_seed_snapshot_raw()
    if seed_df.empty:
        return initialize_portfolio_state(initial_capital=0.0)

    core_holdings: dict[str, float] = {}
    monetary_holdings: dict[str, float] = {}
    opp_holdings: dict[str, float] = {}

    for _, row in seed_df.iterrows():
        ticker = str(row["ticker"])
        bucket = str(row["bucket"])
        qty = float(row["quantity"])

        if qty <= 0:
            continue

        if bucket == "core":
            core_holdings[ticker] = qty
        elif bucket == "monetary":
            monetary_holdings[ticker] = qty
        elif bucket == "opportunistic":
            opp_holdings[ticker] = qty

    return PortfolioState(
        date=None,
        core=SleeveState(cash=0.0, holdings=core_holdings),
        monetary=SleeveState(cash=0.0, holdings=monetary_holdings),
        opportunistic=SleeveState(cash=0.0, holdings=opp_holdings),
        general_cash=0.0,
        cumulative_costs_paid=0.0,
    )


def _get_sleeve_ref(state: PortfolioState, bucket: str) -> SleeveState:
    if bucket == "core":
        return state.core
    if bucket == "monetary":
        return state.monetary
    if bucket == "opportunistic":
        return state.opportunistic
    raise ValueError(f"Unknown bucket: {bucket}")


def mark_to_market_holdings(holdings: dict[str, float], latest_prices: dict[str, float]) -> float:
    return float(sum(float(qty) * float(latest_prices.get(ticker, 0.0)) for ticker, qty in holdings.items()))


def build_synthetic_portfolio_snapshot(
    state: PortfolioState,
    latest_prices: dict[str, float],
    current_date: pd.Timestamp,
    universe_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    total_value = mark_to_market_portfolio(state, latest_prices)["portfolio_value"]
    if total_value <= 0:
        total_value = 1.0

    bucket_map = {
        "core": state.core,
        "monetary": state.monetary,
        "opportunistic": state.opportunistic,
    }

    for bucket, sleeve in bucket_map.items():
        bucket_value = sum(float(qty) * float(latest_prices.get(ticker, 0.0)) for ticker, qty in sleeve.holdings.items())
        if bucket_value <= 0:
            bucket_value = 1.0

        for ticker, qty in sleeve.holdings.items():
            price = float(latest_prices.get(ticker, 0.0))
            mv = float(qty) * price

            uni = universe_df.loc[universe_df["ticker"] == ticker]
            if not uni.empty and "sector" in uni.columns and pd.notna(uni["sector"].iloc[0]):
                sector = str(uni["sector"].iloc[0])
            else:
                sector = "Unknown"
            if sector.strip() == "":
                sector = "Unknown"

            ttf_flag = bool(uni["ttf_flag"].iloc[0]) if not uni.empty and "ttf_flag" in uni.columns else False

            rows.append(
                {
                    "date": current_date,
                    "ticker": str(ticker),
                    "bucket": str(bucket),
                    "quantity": qty,
                    "avg_price": price,
                    "last_price": price,
                    "market_value": mv,
                    "current_weight_total": mv / total_value,
                    "current_weight_bucket": mv / bucket_value,
                    "unrealized_pnl_pct": 0.0,
                    "days_held": 0,
                    "entry_date": current_date,
                    "entry_signal_type": "",
                    "sector": sector,
                    "ttf_flag": ttf_flag,
                }
            )

    out = pd.DataFrame(rows)
    if not out.empty:
        out["ticker"] = out["ticker"].astype(str)
        out["bucket"] = out["bucket"].fillna("unknown").astype(str)
        out["sector"] = out["sector"].fillna("Unknown").astype(str)
        out.loc[out["sector"].str.strip() == "", "sector"] = "Unknown"
    return out


def mark_to_market_portfolio(state: PortfolioState, latest_prices: dict[str, float]) -> dict[str, float]:
    core_positions_value = mark_to_market_holdings(state.core.holdings, latest_prices)
    monetary_positions_value = mark_to_market_holdings(state.monetary.holdings, latest_prices)
    opp_positions_value = mark_to_market_holdings(state.opportunistic.holdings, latest_prices)

    core_total_value = core_positions_value
    monetary_total_value = monetary_positions_value
    opp_total_value = opp_positions_value

    portfolio_value = core_total_value + monetary_total_value + opp_total_value + state.general_cash

    return {
        "general_cash": state.general_cash,
        "core_cash": 0.0,
        "monetary_cash": 0.0,
        "opp_cash": 0.0,
        "core_positions_value": core_positions_value,
        "monetary_positions_value": monetary_positions_value,
        "opp_positions_value": opp_positions_value,
        "core_total_value": core_total_value,
        "monetary_total_value": monetary_total_value,
        "opp_total_value": opp_total_value,
        "portfolio_value": portfolio_value,
    }


def estimate_alpha_metrics(
    bucket: str,
    score: float,
    effective_order_value: float,
    total_cost_est: float,
    annual_fee_rate: float = 0.0,
    sector: str = "",
) -> dict[str, float]:
    config = load_config()
    guard = config.alpha_guardrails

    if effective_order_value <= 0:
        return {
            "expected_holding_months": 0.0,
            "expected_gross_alpha_horizon": 0.0,
            "required_net_alpha_horizon": 0.0,
            "roundtrip_cost_ratio": 0.0,
            "expected_net_alpha_horizon": 0.0,
            "alpha_surplus": 0.0,
        }

    is_opp_etf_rotation = (bucket == "opportunistic") and (str(sector).upper() == "ETF")

    if bucket == "core":
        holding_months = guard.core_expected_holding_months
        required_net_alpha_horizon = guard.core_required_net_alpha_annual * (holding_months / 12.0)
        annual_gross_alpha = min(max(0.06, 0.075 + 0.035 * float(score)), 0.22)
        roundtrip_cost_ratio = (2.0 + guard.core_roundtrip_cost_buffer) * total_cost_est / effective_order_value

    elif is_opp_etf_rotation:
        holding_months = max(4.0, guard.opp_expected_holding_months)
        required_net_alpha_horizon = 0.02 * (holding_months / 12.0)
        annual_gross_alpha = min(max(0.06, 0.08 + 0.06 * float(score)), 0.20)
        annual_gross_alpha -= float(annual_fee_rate)
        roundtrip_cost_ratio = 2.0 * total_cost_est / effective_order_value

    elif bucket == "opportunistic":
        holding_months = guard.opp_expected_holding_months
        required_net_alpha_horizon = guard.opp_required_net_alpha_annual * (holding_months / 12.0)
        annual_gross_alpha = min(max(0.10, 0.14 + 0.09 * float(score)), 0.45)
        annual_gross_alpha -= float(annual_fee_rate)
        roundtrip_cost_ratio = (2.0 + guard.opp_roundtrip_cost_buffer) * total_cost_est / effective_order_value

    else:
        holding_months = 18.0
        required_net_alpha_horizon = 0.04 * (holding_months / 12.0)
        annual_gross_alpha = min(max(0.05, 0.06 + 0.02 * float(score)), 0.15)
        annual_gross_alpha -= float(annual_fee_rate)
        roundtrip_cost_ratio = 2.0 * total_cost_est / effective_order_value

    expected_gross_alpha_horizon = annual_gross_alpha * (holding_months / 12.0)
    expected_net_alpha_horizon = expected_gross_alpha_horizon - roundtrip_cost_ratio
    alpha_surplus = expected_net_alpha_horizon - required_net_alpha_horizon

    return {
        "expected_holding_months": holding_months,
        "expected_gross_alpha_horizon": expected_gross_alpha_horizon,
        "required_net_alpha_horizon": required_net_alpha_horizon,
        "roundtrip_cost_ratio": roundtrip_cost_ratio,
        "expected_net_alpha_horizon": expected_net_alpha_horizon,
        "alpha_surplus": alpha_surplus,
    }


def build_order_proposals_at_date(
    target_df: pd.DataFrame,
    state: PortfolioState,
    latest_prices: dict[str, float],
    universe_df: pd.DataFrame,
    latest_features_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    config = load_config()

    current_rows = []
    mtm = mark_to_market_portfolio(state, latest_prices)
    total_portfolio_value = float(mtm["portfolio_value"])
    if total_portfolio_value <= 0:
        total_portfolio_value = 1.0

    for bucket_name, sleeve in [
        ("core", state.core),
        ("monetary", state.monetary),
        ("opportunistic", state.opportunistic),
    ]:
        for ticker, qty in sleeve.holdings.items():
            price = float(latest_prices.get(ticker, 0.0))
            mv = qty * price
            current_rows.append(
                {
                    "ticker": ticker,
                    "bucket": bucket_name,
                    "quantity": qty,
                    "last_price": price,
                    "current_market_value": mv,
                    "current_weight_total": mv / total_portfolio_value,
                }
            )

    current_positions = pd.DataFrame(current_rows)
    if current_positions.empty:
        current_positions = pd.DataFrame(
            columns=["ticker", "bucket", "quantity", "last_price", "current_market_value", "current_weight_total"]
        )

    universe_info = universe_df[["ticker", "ttf_flag", "liquidity_bucket", "sector"]].drop_duplicates().copy()

    merged = target_df.merge(
        current_positions,
        on=["ticker", "bucket"],
        how="outer",
    ).merge(
        universe_info,
        on="ticker",
        how="left",
    )

    merged["last_price"] = pd.to_numeric(merged["last_price"], errors="coerce")
    merged["last_price"] = merged["last_price"].fillna(merged["ticker"].map(latest_prices)).fillna(0.0)

    merged["target_weight_final"] = pd.to_numeric(merged.get("target_weight_final", 0.0), errors="coerce").fillna(0.0)
    merged["target_weight_raw"] = pd.to_numeric(merged.get("target_weight_raw", 0.0), errors="coerce").fillna(0.0)
    merged["current_weight_total"] = pd.to_numeric(merged.get("current_weight_total", 0.0), errors="coerce").fillna(0.0)
    merged["current_market_value"] = pd.to_numeric(merged.get("current_market_value", 0.0), errors="coerce").fillna(0.0)
    merged["quantity"] = pd.to_numeric(merged.get("quantity", 0.0), errors="coerce").fillna(0.0)
    merged["ttf_flag"] = merged["ttf_flag"].fillna(False)
    merged["liquidity_bucket"] = merged["liquidity_bucket"].fillna("standard")
    merged["sector"] = merged["sector"].fillna("Unknown").astype(str)
    merged.loc[merged["sector"].str.strip() == "", "sector"] = "Unknown"
    merged["reason"] = merged.get("reason", "EXIT_NON_TARGET")
    merged["reason"] = merged["reason"].fillna("EXIT_NON_TARGET")
    merged["bucket"] = merged["bucket"].fillna("core")
    merged["score"] = pd.to_numeric(merged.get("score", 0.0), errors="coerce").fillna(0.0)
    merged["annual_fee_rate"] = pd.to_numeric(merged.get("annual_fee_rate", 0.0), errors="coerce").fillna(0.0)

    defaults = {
        "mom12_ex1": 0.0,
        "ret_6m": 0.0,
        "ret_3m": 0.0,
        "ret_20d": 0.0,
        "vol60": 0.0,
        "vol20": 0.0,
        "dd_6m": 0.0,
        "dist_ma200": 0.0,
        "dist_ma20": 0.0,
        "rsi14": 50.0,
    }
    if latest_features_df is not None and not latest_features_df.empty:
        feat_cols = ["ticker"] + list(defaults.keys())
        avail = [c for c in feat_cols if c in latest_features_df.columns]
        merged = merged.merge(
            latest_features_df[avail].drop_duplicates(subset=["ticker"]),
            on="ticker",
            how="left",
        )

    for col, default in defaults.items():
        if col not in merged.columns:
            merged[col] = default
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(default)

    if latest_features_df is not None and not latest_features_df.empty:
        merged["breadth_ma200"] = float(pd.to_numeric(latest_features_df.get("ma200_flag", 0.0), errors="coerce").fillna(0.0).mean())
        merged["breadth_slope"] = float(pd.to_numeric(latest_features_df.get("ma200_slope_flag", 0.0), errors="coerce").fillna(0.0).mean())
        merged["benchmark_ret_3m"] = float(pd.to_numeric(latest_features_df.get("ret_3m", 0.0), errors="coerce").fillna(0.0).mean())
        merged["benchmark_ret_6m"] = float(pd.to_numeric(latest_features_df.get("ret_6m", 0.0), errors="coerce").fillna(0.0).mean())
    else:
        merged["breadth_ma200"] = 0.0
        merged["breadth_slope"] = 0.0
        merged["benchmark_ret_3m"] = 0.0
        merged["benchmark_ret_6m"] = 0.0

    merged["target_market_value"] = merged["target_weight_final"] * total_portfolio_value
    merged["weight_gap"] = merged["target_weight_final"] - merged["current_weight_total"]
    merged["order_value"] = merged["target_market_value"] - merged["current_market_value"]

    rows = []

    for _, row in merged.iterrows():
        bucket = str(row["bucket"])
        min_order = config.sleeves[bucket].min_order_eur
        deadband = 0.03 if bucket == "monetary" else (config.allocation.core_rebalance_deadband if bucket == "core" else 0.0)

        current_weight = float(row["current_weight_total"])
        target_weight = float(row["target_weight_final"])
        weight_gap = target_weight - current_weight
        weight_gap_abs = abs(weight_gap)
        current_mv = float(row["current_market_value"])
        target_mv = float(row["target_market_value"])
        order_value = target_mv - current_mv
        order_value_abs = abs(order_value)
        last_price = float(row["last_price"])
        ttf_flag = bool(row["ttf_flag"])
        score = float(row["score"])
        annual_fee_rate = float(row["annual_fee_rate"])
        sector = str(row["sector"])

        if target_weight > current_weight:
            action = "BUY"
        elif target_weight < current_weight:
            action = "SELL"
        else:
            action = "HOLD"

        shares_est = math.floor(order_value_abs / last_price) if action != "HOLD" and last_price > 0 else 0
        effective_order_value = shares_est * last_price

        cost_est = estimate_transaction_cost(
            order_value=effective_order_value,
            side="BUY" if action == "BUY" else "SELL",
            ttf_flag=ttf_flag,
            liquidity_bucket=str(row["liquidity_bucket"]),
        )

        alpha = estimate_alpha_metrics(
            bucket=bucket,
            score=score,
            effective_order_value=effective_order_value,
            total_cost_est=cost_est.total_cost_est,
            annual_fee_rate=annual_fee_rate,
            sector=sector,
        )

        execute = True
        decision_reason = str(row["reason"])

        if action == "HOLD":
            execute = False
            decision_reason = "NO_ACTION"
        elif shares_est <= 0:
            execute = False
            decision_reason = "REJECT_ZERO_SHARES"
        elif effective_order_value < min_order:
            execute = False
            decision_reason = "REJECT_SMALL_ORDER"
        elif weight_gap_abs < deadband:
            execute = False
            decision_reason = "REJECT_DEADBAND"
        else:
            if bucket == "monetary":
                if current_weight == 0.0 and target_weight > 0.0:
                    execute = True
                    decision_reason = "INIT_PERMANENT_ETF"
                elif alpha["alpha_surplus"] <= 0:
                    execute = False
                    decision_reason = "REJECT_ALPHA_HURDLE"
            elif action == "BUY":
                if alpha["alpha_surplus"] <= 0:
                    execute = False
                    decision_reason = "REJECT_ALPHA_HURDLE"
            elif action == "SELL":
                if decision_reason in {"DROP_OUT", "EXIT_OPP", "EXIT_NON_TARGET"}:
                    execute = True
                elif alpha["alpha_surplus"] <= 0:
                    execute = False
                    decision_reason = "REJECT_ALPHA_HURDLE"

        rows.append(
            {
                "ticker": row["ticker"],
                "bucket": bucket,
                "score": score,
                "annual_fee_rate": annual_fee_rate,
                "sector": sector,
                "current_weight_total": current_weight,
                "target_weight_final": target_weight,
                "weight_gap": weight_gap,
                "current_market_value": current_mv,
                "target_market_value": target_mv,
                "order_value": order_value,
                "effective_order_value": effective_order_value,
                "action": action,
                "shares_est": shares_est,
                "brokerage_est": cost_est.brokerage_est,
                "ttf_est": cost_est.ttf_est,
                "spread_est": cost_est.spread_est,
                "total_cost_est": cost_est.total_cost_est,
                "expected_holding_months": alpha["expected_holding_months"],
                "expected_gross_alpha_horizon": alpha["expected_gross_alpha_horizon"],
                "required_net_alpha_horizon": alpha["required_net_alpha_horizon"],
                "roundtrip_cost_ratio": alpha["roundtrip_cost_ratio"],
                "expected_net_alpha_horizon": alpha["expected_net_alpha_horizon"],
                "alpha_surplus": alpha["alpha_surplus"],
                "mom12_ex1": float(row["mom12_ex1"]),
                "ret_6m": float(row["ret_6m"]),
                "ret_3m": float(row["ret_3m"]),
                "ret_20d": float(row["ret_20d"]),
                "vol60": float(row["vol60"]),
                "vol20": float(row["vol20"]),
                "dd_6m": float(row["dd_6m"]),
                "dist_ma200": float(row["dist_ma200"]),
                "dist_ma20": float(row["dist_ma20"]),
                "rsi14": float(row["rsi14"]),
                "breadth_ma200": float(row["breadth_ma200"]),
                "breadth_slope": float(row["breadth_slope"]),
                "benchmark_ret_3m": float(row["benchmark_ret_3m"]),
                "benchmark_ret_6m": float(row["benchmark_ret_6m"]),
                "execute": execute,
                "decision_reason": decision_reason,
            }
        )

    orders_df = pd.DataFrame(rows)

    if not orders_df.empty:
        orders_df = apply_rebalance_overlay(orders_df)

    return orders_df


def execute_orders(
    state: PortfolioState,
    orders_df: pd.DataFrame,
    latest_prices: dict[str, float],
) -> tuple[PortfolioState, pd.DataFrame, float, float]:
    state = PortfolioState(
        date=state.date,
        core=SleeveState(cash=0.0, holdings=dict(state.core.holdings)),
        monetary=SleeveState(cash=0.0, holdings=dict(state.monetary.holdings)),
        opportunistic=SleeveState(cash=0.0, holdings=dict(state.opportunistic.holdings)),
        general_cash=state.general_cash,
        cumulative_costs_paid=state.cumulative_costs_paid,
    )

    executable = orders_df.loc[orders_df["execute"] == True].copy()
    if executable.empty:
        return state, pd.DataFrame(), 0.0, 0.0

    fills = []
    costs_paid = 0.0
    gross_turnover = 0.0

    executable["action_order"] = executable["action"].map({"SELL": 0, "BUY": 1, "HOLD": 2})
    executable = executable.sort_values(["action_order", "bucket", "ticker"]).reset_index(drop=True)

    for _, row in executable.iterrows():
        ticker = str(row["ticker"])
        bucket = str(row["bucket"])
        action = str(row["action"])
        shares = float(row["shares_est"])
        price = float(latest_prices.get(ticker, 0.0))
        total_cost = float(row["total_cost_est"])

        if price <= 0 or shares <= 0:
            continue

        sleeve = _get_sleeve_ref(state, bucket)
        current_qty = sleeve.holdings.get(ticker, 0.0)

        if action == "SELL":
            shares_to_sell = min(current_qty, shares)
            if shares_to_sell <= 0:
                continue

            gross = shares_to_sell * price
            state.general_cash += gross - total_cost
            new_qty = current_qty - shares_to_sell

            if new_qty > 0:
                sleeve.holdings[ticker] = new_qty
            else:
                sleeve.holdings.pop(ticker, None)

            costs_paid += total_cost
            gross_turnover += gross

            fills.append(
                {
                    "ticker": ticker,
                    "bucket": bucket,
                    "action": "SELL",
                    "shares_filled": shares_to_sell,
                    "fill_price": price,
                    "gross_amount": gross,
                    "total_cost": total_cost,
                    "net_cash_impact": gross - total_cost,
                }
            )

    for _, row in executable.iterrows():
        ticker = str(row["ticker"])
        bucket = str(row["bucket"])
        action = str(row["action"])
        shares = float(row["shares_est"])
        price = float(latest_prices.get(ticker, 0.0))
        total_cost = float(row["total_cost_est"])

        if action != "BUY":
            continue
        if price <= 0 or shares <= 0:
            continue

        sleeve = _get_sleeve_ref(state, bucket)
        gross = shares * price
        cash_needed = gross + total_cost

        if state.general_cash < cash_needed:
            max_affordable_shares = math.floor(max((state.general_cash - total_cost), 0.0) / price)
            if max_affordable_shares <= 0:
                continue

            shares = float(max_affordable_shares)
            gross = shares * price
            if float(row["shares_est"]) > 0:
                total_cost = total_cost * (shares / float(row["shares_est"]))
            cash_needed = gross + total_cost

            if state.general_cash < cash_needed:
                continue

        state.general_cash -= cash_needed
        sleeve.holdings[ticker] = sleeve.holdings.get(ticker, 0.0) + shares

        costs_paid += total_cost
        gross_turnover += gross

        fills.append(
            {
                "ticker": ticker,
                "bucket": bucket,
                "action": "BUY",
                "shares_filled": shares,
                "fill_price": price,
                "gross_amount": gross,
                "total_cost": total_cost,
                "net_cash_impact": -(gross + total_cost),
            }
        )

    state.cumulative_costs_paid += costs_paid
    fills_df = pd.DataFrame(fills)

    return state, fills_df, costs_paid, gross_turnover