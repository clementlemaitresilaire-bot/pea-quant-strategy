from __future__ import annotations

import argparse
import subprocess

import pandas as pd

from src.backtest.engine import run_backtest
from src.backtest.export_backtest import export_backtest_outputs
from src.backtest.reports import build_backtest_text_summary
from src.ml.config import set_ml_enabled
from src.settings import EXPORT_DATA_DIR, ensure_project_directories, load_config


def _run_single_backtest(
    *,
    use_ml: bool,
    initial_capital: float,
    rebalance_frequency: str,
):
    set_ml_enabled(use_ml)
    return run_backtest(
        initial_capital=initial_capital,
        rebalance_frequency=rebalance_frequency,
    )


def _export_ml_comparison_csv(result_no_ml, result_ml) -> None:
    metrics_no_ml = result_no_ml.metrics.copy()
    metrics_no_ml["label"] = "without_ml"

    metrics_ml = result_ml.metrics.copy()
    metrics_ml["label"] = "with_ml"

    df = pd.DataFrame([metrics_no_ml, metrics_ml])

    if {"label", "num_executed_orders"}.issubset(df.columns):
        no_ml_orders = float(
            df.loc[df["label"] == "without_ml", "num_executed_orders"].iloc[0]
        )
        df["executed_orders_delta_vs_without_ml"] = (
            df["num_executed_orders"] - no_ml_orders
        )

    out_dir = EXPORT_DATA_DIR / "backtest"
    out_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_dir / "ml_comparison_metrics.csv", index=False)

    debug_df = pd.DataFrame(
        [
            {
                "label": "without_ml",
                "num_orders_history_rows": len(result_no_ml.orders_history),
                "num_target_history_rows": len(result_no_ml.target_history),
            },
            {
                "label": "with_ml",
                "num_orders_history_rows": len(result_ml.orders_history),
                "num_target_history_rows": len(result_ml.target_history),
            },
        ]
    )
    debug_df.to_csv(out_dir / "ml_comparison_debug.csv", index=False)


def _print_ml_comparison_summary() -> None:
    print("\n=== ML COMPARISON SUMMARY ===")
    try:
        ml_path = EXPORT_DATA_DIR / "backtest" / "ml_comparison_metrics.csv"
        df = pd.read_csv(ml_path)

        print(df.to_string(index=False))

        if {"label", "total_return"}.issubset(df.columns):
            no_ml_ret = float(df.loc[df["label"] == "without_ml", "total_return"].iloc[0])
            ml_ret = float(df.loc[df["label"] == "with_ml", "total_return"].iloc[0])
            print(f"\nDelta total return : {(ml_ret - no_ml_ret):.4%}")

        if {"label", "num_executed_orders"}.issubset(df.columns):
            no_ml_orders = float(
                df.loc[df["label"] == "without_ml", "num_executed_orders"].iloc[0]
            )
            ml_orders = float(
                df.loc[df["label"] == "with_ml", "num_executed_orders"].iloc[0]
            )
            print(f"Delta executed orders : {ml_orders - no_ml_orders:.0f}")

    except Exception as exc:
        print(f"Could not display ML comparison summary: {exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run backtests and export reports from the local cache only."
    )
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=50_000.0,
        help="Initial capital used in the backtest.",
    )
    parser.add_argument(
        "--rebalance-frequency",
        type=str,
        default="monthly",
        choices=["weekly", "monthly", "quarterly"],
        help="Rebalancing frequency for the backtest.",
    )
    parser.add_argument(
        "--compare-ml",
        action="store_true",
        help="Run both without ML and with ML, then export a comparison.",
    )
    parser.add_argument(
        "--open-report",
        action="store_true",
        help="Open the generated HTML report automatically when available.",
    )
    return parser.parse_args()


def main() -> None:
    ensure_project_directories()
    _ = load_config()  # validate config at startup

    args = parse_args()

    print("\n=== BACKTEST ENTRY POINT ===")
    print("Mode                : cache-only")
    print("Remote refresh      : disabled")
    print("Data source         : local CSV cache")
    print(f"Initial capital     : {args.initial_capital:,.2f}")
    print(f"Rebalance frequency : {args.rebalance_frequency}")

    if args.compare_ml:
        print("\n=== RUNNING BACKTEST WITHOUT ML ===")
        result_no_ml = _run_single_backtest(
            use_ml=False,
            initial_capital=args.initial_capital,
            rebalance_frequency=args.rebalance_frequency,
        )

        print("\n=== RUNNING BACKTEST WITH ML ===")
        result_ml = _run_single_backtest(
            use_ml=True,
            initial_capital=args.initial_capital,
            rebalance_frequency=args.rebalance_frequency,
        )

        _export_ml_comparison_csv(result_no_ml, result_ml)

        final_result = result_ml
        set_ml_enabled(True)
    else:
        print("\n=== RUNNING BACKTEST WITH CURRENT ML SETTING ===")
        final_result = _run_single_backtest(
            use_ml=True,
            initial_capital=args.initial_capital,
            rebalance_frequency=args.rebalance_frequency,
        )
        set_ml_enabled(True)

    print("\n=== BACKTEST SUMMARY ===\n")
    print(build_backtest_text_summary(final_result))

    exported = export_backtest_outputs(
        initial_capital=args.initial_capital,
        rebalance_frequency=args.rebalance_frequency,
        result=final_result,
    )

    print("\n=== FILES GENERATED ===")
    for key, path in exported.items():
        print(f"{key}: {path}")

    if args.compare_ml:
        _print_ml_comparison_summary()

    html_report = exported.get("visual_report_html")
    if args.open_report and html_report is not None:
        try:
            subprocess.run(["open", str(html_report)], check=False)
            print(f"\nOpened HTML report: {html_report}")
        except Exception as exc:
            print(f"\nCould not open HTML report automatically: {exc}")


if __name__ == "__main__":
    main()