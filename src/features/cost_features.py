from __future__ import annotations

from dataclasses import dataclass

from src.settings import load_config


@dataclass
class TransactionCostEstimate:
    brokerage_est: float
    ttf_est: float
    spread_est: float
    total_cost_est: float


def _bps_to_rate(bps: float) -> float:
    return float(bps) / 10_000.0


def estimate_transaction_cost(
    order_value: float,
    side: str,
    ttf_flag: bool,
    liquidity_bucket: str,
) -> TransactionCostEstimate:
    """
    Estimate all-in transaction costs.

    Inputs:
    - order_value: absolute gross traded value in EUR
    - side: BUY or SELL
    - ttf_flag: whether French FTT applies
    - liquidity_bucket: high / standard / lower
    """
    config = load_config()
    costs = config.costs

    order_value = abs(float(order_value))
    side = str(side).upper()
    liquidity_bucket = str(liquidity_bucket).lower()

    brokerage_rate = _bps_to_rate(costs.brokerage_bps)

    spread_rate_map = {
        "high": _bps_to_rate(costs.spread_bps_high),
        "standard": _bps_to_rate(costs.spread_bps_standard),
        "lower": _bps_to_rate(costs.spread_bps_lower),
    }
    spread_rate = spread_rate_map.get(liquidity_bucket, _bps_to_rate(costs.spread_bps_standard))

    brokerage_est = order_value * brokerage_rate

    # Apply French TTF only on BUY orders for flagged names
    if side == "BUY" and bool(ttf_flag):
        ttf_est = order_value * float(costs.ttf_france_rate)
    else:
        ttf_est = 0.0

    spread_est = order_value * spread_rate
    total_cost_est = brokerage_est + ttf_est + spread_est

    return TransactionCostEstimate(
        brokerage_est=brokerage_est,
        ttf_est=ttf_est,
        spread_est=spread_est,
        total_cost_est=total_cost_est,
    )


if __name__ == "__main__":
    example = estimate_transaction_cost(
        order_value=10_000,
        side="BUY",
        ttf_flag=True,
        liquidity_bucket="high",
    )
    print(example)