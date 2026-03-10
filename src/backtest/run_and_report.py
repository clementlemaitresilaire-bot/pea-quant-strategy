from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd

from src.backtest.engine import run_backtest
from src.backtest.export_backtest import export_backtest_outputs
from src.backtest.reports import build_backtest_text_summary
from src.ml.config import set_ml_enabled
from src.settings import EXPORT_DATA_DIR


def _build_row(result, label: str) -> dict:
    row = dict(result.metrics)

    row["label"] = label
    row["num_orders_history_rows"] = len(result.orders_history)
    row["num_target_history_rows"] = len(result.target_history)

    return row


def _run_single_backtest(use_ml: bool):
    set_ml_enabled(use_ml)
    return run_backtest(
        initial_capital=50_000.0,
        rebalance_frequency="monthly",
        target_backtest_years=10,
    )


def _export_ml_comparison_csv(result_no_ml, result_ml) -> Path:
    rows = [
        _build_row(result_no_ml, "without_ml"),
        _build_row(result_ml, "with_ml"),
    ]

    df = pd.DataFrame(rows)

    if {"label", "num_executed_orders"}.issubset(df.columns):
        no_ml_orders = float(df.loc[df["label"] == "without_ml", "num_executed_orders"].iloc[0])
        df["executed_orders_delta_vs_without_ml"] = df["num_executed_orders"] - no_ml_orders
    else:
        df["executed_orders_delta_vs_without_ml"] = 0.0

    out_dir = EXPORT_DATA_DIR / "backtest"
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = out_dir / "ml_comparison_metrics.csv"
    debug_path = out_dir / "ml_comparison_debug.csv"

    df.to_csv(metrics_path, index=False)

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
    debug_df.to_csv(debug_path, index=False)

    return metrics_path


def main() -> None:
    print("\n=== RUNNING BACKTEST WITHOUT ML ===")
    result_no_ml = _run_single_backtest(use_ml=False)

    print("\n=== SUMMARY WITHOUT ML ===\n")
    print(build_backtest_text_summary(result_no_ml))

    print("\n=== RUNNING BACKTEST WITH ML ===")
    result_ml = _run_single_backtest(use_ml=True)

    print("\n=== SUMMARY WITH ML ===\n")
    print(build_backtest_text_summary(result_ml))

    comparison_path = _export_ml_comparison_csv(result_no_ml, result_ml)

    # On remet explicitement ML=True pour que l'export final soit cohérent
    # avec le run "with_ml" et avec la logique du dashboard final.
    set_ml_enabled(True)

    exported = export_backtest_outputs(
        initial_capital=50_000.0,
        rebalance_frequency="monthly",
        result=result_ml,
    )

    print("\n=== ML COMPARISON METRICS ===")
    try:
        df = pd.read_csv(comparison_path)
        print(df.to_string(index=False))

        if {"label", "total_return"}.issubset(df.columns):
            no_ml_ret = float(df.loc[df["label"] == "without_ml", "total_return"].iloc[0])
            ml_ret = float(df.loc[df["label"] == "with_ml", "total_return"].iloc[0])
            print(f"\nDelta total return : {(ml_ret - no_ml_ret):.4%}")

        if {"label", "sharpe"}.issubset(df.columns):
            no_ml_sharpe = float(df.loc[df["label"] == "without_ml", "sharpe"].iloc[0])
            ml_sharpe = float(df.loc[df["label"] == "with_ml", "sharpe"].iloc[0])
            print(f"Delta sharpe       : {ml_sharpe - no_ml_sharpe:.4f}")

        if {"label", "num_executed_orders"}.issubset(df.columns):
            no_ml_orders = float(df.loc[df["label"] == "without_ml", "num_executed_orders"].iloc[0])
            ml_orders = float(df.loc[df["label"] == "with_ml", "num_executed_orders"].iloc[0])
            print(f"Delta executed orders : {ml_orders - no_ml_orders:.0f}")

    except Exception as e:
        print(f"Could not display ML comparison summary: {e}")

    print("\n=== FILES GENERATED ===")
    for key, path in exported.items():
        print(f"{key}: {path}")

    html_report = exported.get("visual_report_html")
    if html_report is not None:
        try:
            subprocess.run(["open", str(html_report)], check=False)
            print(f"\nOpened HTML report: {html_report}")
        except Exception as e:
            print(f"\nCould not open HTML report automatically: {e}")


if __name__ == "__main__":
    main()