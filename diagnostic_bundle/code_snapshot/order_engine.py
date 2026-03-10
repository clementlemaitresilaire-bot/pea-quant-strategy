from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.engines.allocation_engine import build_full_target_allocation
from src.features.cost_features import estimate_transaction_cost
from src.features.feature_pipeline import build_latest_feature_snapshot
from src.io.loaders import load_cash_state, load_portfolio_snapshot, load_universe
from src.io.market_data import get_latest_price_snapshot, load_all_price_data
from src.ml.overlays import apply_rebalance_overlay
from src.settings import load_config


def _load_cash_total() -> float:
    try:
        cash_df = load_cash_state()
        if cash_df.empty:
            return 0.0

        row = cash_df.iloc[-1]

        if "cash_general" in row.index and pd.notna(row["cash_general"]):
            return float(row["cash_general"])

        if "cash_total" in row.index and pd.notna(row["cash_total"]):
            return float(row["cash_total"])

        return 0.0
    except Exception:
        return 0.0


def _get_min_order_by_bucket(bucket: str) -> float:
    config = load_config()
    return config.sleeves[bucket].min_order_eur


def _get_deadband_by_bucket(bucket: str) -> float:
    config = load_config()
    if bucket == "core":
        return config.allocation.core_rebalance_deadband
    if bucket == "monetary":
        return 0.03
    if bucket == "opportunistic":
        return 0.01
    raise ValueError(f"Unknown bucket: {bucket}")


def _build_live_market_context(
    latest_features_df: pd.DataFrame | None = None,
) -> dict[str, float]:
    if latest_features_df is None or latest_features_df.empty:
        try:
            latest_features_df = build_latest_feature_snapshot()
        except Exception:
            return {
                "breadth_ma200": 0.0,
                "breadth_slope": 0.0,
                "benchmark_ret_3m": 0.0,
                "benchmark_ret_6m": 0.0,
            }

    df = latest_features_df.copy()

    breadth_ma200 = float(pd.to_numeric(df.get("ma200_flag", 0.0), errors="coerce").fillna(0.0).mean())
    breadth_slope = float(pd.to_numeric(df.get("ma200_slope_flag", 0.0), errors="coerce").fillna(0.0).mean())
    benchmark_ret_3m = float(pd.to_numeric(df.get("ret_3m", 0.0), errors="coerce").fillna(0.0).mean())
    benchmark_ret_6m = float(pd.to_numeric(df.get("ret_6m", 0.0), errors="coerce").fillna(0.0).mean())

    return {
        "breadth_ma200": breadth_ma200,
        "breadth_slope": breadth_slope,
        "benchmark_ret_3m": benchmark_ret_3m,
        "benchmark_ret_6m": benchmark_ret_6m,
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
        # ETF rotation should not be blocked by a hurdle calibrated for noisy stock-picking
        holding_months = max(4.0, guard.opp_expected_holding_months)
        required_net_alpha_horizon = 0.02 * (holding_months / 12.0)
        annual_gross_alpha = min(max(0.06, 0.08 + 0.06 * float(score)), 0.20)
        annual_gross_alpha -= float(annual_fee_rate)
        roundtrip_cost_ratio = 2.0 * total_cost_est / effective_order_value

    elif bucket == "opportunistic":
        holding_months = guard.opp_expected_holding_months
        required_net_alpha_horizon = guard.opp_required_net_alpha_annual * (holding_months / 12.0)
        annual_gross_alpha = min(max(0.10, 0.14 + 0.09 * float(score)), 0.45)
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


def build_order_proposals(
    portfolio_df: pd.DataFrame | None = None,
    universe_df: pd.DataFrame | None = None,
    combined_target_df: pd.DataFrame | None = None,
    price_snapshot_df: pd.DataFrame | None = None,
    latest_features_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if portfolio_df is None:
        portfolio_df = load_portfolio_snapshot()

    if universe_df is None:
        universe_df = load_universe()

    if combined_target_df is None:
        combined_target_df = build_full_target_allocation(
            universe_df=universe_df,
            latest_features_df=latest_features_df,
        ).combined_target_weights_table

    if price_snapshot_df is None:
        price_snapshot_df = get_latest_price_snapshot(load_all_price_data())

    if latest_features_df is None:
        try:
            latest_features_df = build_latest_feature_snapshot()
        except Exception:
            latest_features_df = pd.DataFrame()

    latest_price_map = dict(zip(price_snapshot_df["ticker"], price_snapshot_df["adjusted_close"]))
    cash_total = _load_cash_total()

    current_positions = portfolio_df.copy()
    if current_positions.empty:
        current_positions = pd.DataFrame(columns=["ticker", "bucket", "quantity", "last_price", "market_value"])

    if "quantity" not in current_positions.columns:
        current_positions["quantity"] = 0.0

    current_positions["quantity"] = pd.to_numeric(current_positions["quantity"], errors="coerce").fillna(0.0)
    current_positions["last_price"] = current_positions["ticker"].map(latest_price_map)
    current_positions["last_price"] = pd.to_numeric(current_positions["last_price"], errors="coerce").fillna(0.0)
    current_positions["current_market_value"] = current_positions["quantity"] * current_positions["last_price"]

    total_positions_value = float(current_positions["current_market_value"].sum())
    total_portfolio_value = total_positions_value + cash_total
    if total_portfolio_value <= 0:
        total_portfolio_value = 1.0

    current_positions["current_weight_total"] = current_positions["current_market_value"] / total_portfolio_value

    universe_info = universe_df[
        ["ticker", "ttf_flag", "liquidity_bucket", "sector"]
    ].drop_duplicates().copy()

    merged = combined_target_df.merge(
        current_positions[["ticker", "bucket", "quantity", "last_price", "current_market_value", "current_weight_total"]],
        on=["ticker", "bucket"],
        how="outer",
    ).merge(
        universe_info,
        on="ticker",
        how="left",
    )

    # enrich with latest features so the ML rebalance filter sees real market state
    if latest_features_df is not None and not latest_features_df.empty:
        feature_cols = [
            "ticker",
            "mom12_ex1",
            "ret_6m",
            "ret_3m",
            "ret_20d",
            "ret_10d",
            "ret_5d",
            "vol60",
            "vol20",
            "dd_6m",
            "dist_ma200",
            "dist_ma20",
            "rsi14",
        ]
        available_cols = [c for c in feature_cols if c in latest_features_df.columns]
        if "ticker" in available_cols:
            merged = merged.merge(
                latest_features_df[available_cols].drop_duplicates(subset=["ticker"]),
                on="ticker",
                how="left",
            )

    market_context = _build_live_market_context(latest_features_df)
    merged["breadth_ma200"] = market_context["breadth_ma200"]
    merged["breadth_slope"] = market_context["breadth_slope"]
    merged["benchmark_ret_3m"] = market_context["benchmark_ret_3m"]
    merged["benchmark_ret_6m"] = market_context["benchmark_ret_6m"]

    merged["market_last_price"] = merged["ticker"].map(latest_price_map)
    merged["last_price"] = pd.to_numeric(merged["last_price"], errors="coerce").fillna(merged["market_last_price"])
    merged["last_price"] = pd.to_numeric(merged["last_price"], errors="coerce").fillna(0.0)

    merged["current_weight_total"] = merged["current_weight_total"].fillna(0.0)
    merged["current_market_value"] = merged["current_market_value"].fillna(0.0)
    merged["quantity"] = merged["quantity"].fillna(0.0)
    merged["target_weight_final"] = pd.to_numeric(merged["target_weight_final"], errors="coerce").fillna(0.0)
    merged["target_weight_raw"] = pd.to_numeric(merged["target_weight_raw"], errors="coerce").fillna(0.0)
    merged["ttf_flag"] = merged["ttf_flag"].fillna(False)
    merged["liquidity_bucket"] = merged["liquidity_bucket"].fillna("standard")
    merged["sector"] = merged["sector"].fillna("")
    merged["reason"] = merged["reason"].fillna("EXIT_NON_TARGET")
    merged["bucket"] = merged["bucket"].fillna("core")
    merged["score"] = pd.to_numeric(merged["score"], errors="coerce").fillna(0.0)
    merged["annual_fee_rate"] = pd.to_numeric(merged.get("annual_fee_rate", 0.0), errors="coerce").fillna(0.0)

    feature_fill_defaults = {
        "mom12_ex1": 0.0,
        "ret_6m": 0.0,
        "ret_3m": 0.0,
        "ret_20d": 0.0,
        "ret_10d": 0.0,
        "ret_5d": 0.0,
        "vol60": 0.0,
        "vol20": 0.0,
        "dd_6m": 0.0,
        "dist_ma200": 0.0,
        "dist_ma20": 0.0,
        "rsi14": 50.0,
    }
    for col, default in feature_fill_defaults.items():
        if col not in merged.columns:
            merged[col] = default
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(default)

    merged["target_market_value"] = merged["target_weight_final"] * total_portfolio_value
    merged["weight_gap"] = merged["target_weight_final"] - merged["current_weight_total"]
    merged["order_value"] = merged["target_market_value"] - merged["current_market_value"]

    rows = []

    for _, row in merged.iterrows():
        bucket = str(row["bucket"])
        min_order = _get_min_order_by_bucket(bucket)
        deadband = _get_deadband_by_bucket(bucket)

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

        shares_est = math.floor(order_value_abs / last_price) if last_price > 0 and action != "HOLD" else 0
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

    output = pd.DataFrame(rows)
    output = output.sort_values(
        ["execute", "bucket", "action", "score", "ticker"],
        ascending=[False, True, True, False, True],
    ).reset_index(drop=True)

    # ML overlay acts only as a final filter on marginal rebalances
    output = apply_rebalance_overlay(output)

    return output


if __name__ == "__main__":
    proposals = build_order_proposals()
    print("\n=== Consolidated Order Proposals ===")
    print(proposals)