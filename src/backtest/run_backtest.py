from __future__ import annotations

from src.backtest.engine import run_backtest


def main() -> None:
    result = run_backtest(
        initial_capital=50_000.0,
        rebalance_frequency="monthly",
    )

    print("\n=== BACKTEST METRICS ===")
    for key, value in result.metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")

    print("\n=== EQUITY CURVE (last rows) ===")
    print(result.equity_curve.tail())

    print("\n=== ORDERS HISTORY (last rows) ===")
    print(result.orders_history.tail())

    print("\n=== TARGET HISTORY (last rows) ===")
    print(result.target_history.tail())


if __name__ == "__main__":
    main()