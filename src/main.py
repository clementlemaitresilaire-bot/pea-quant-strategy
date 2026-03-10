from __future__ import annotations

import argparse

from src.io.run_data_pipeline import run_data_pipeline
from src.reports.exports import export_all_strategy_outputs
from src.reports.summaries import build_run_summary
from src.settings import ensure_project_directories, load_config


def _run_live_cycle(mode_label: str) -> None:
    """
    Unified live execution path:
    1. refresh / validate market data
    2. export current strategy outputs
    3. print a human-readable run summary
    """
    data_summary = run_data_pipeline(
        run_update=True,
        run_diagnostics=True,
        build_features=True,
        export_outputs=True,
    )

    exported = export_all_strategy_outputs()
    summary = build_run_summary()

    print(f"\n=== {mode_label.upper()} RUN COMPLETED ===")

    print("\n--- Data pipeline summary ---")
    print(f"Tickers requested    : {data_summary.get('tickers_requested', 0)}")
    print(f"Price rows loaded    : {data_summary.get('prices', {}).get('rows', 0)}")
    print(f"Price tickers loaded : {data_summary.get('prices', {}).get('tickers_loaded', 0)}")
    print(f"Price max date       : {data_summary.get('prices', {}).get('max_date', '')}")

    features = data_summary.get("features", {})
    if features:
        print(f"Feature full rows    : {features.get('full_rows', 0)}")
        print(f"Feature latest rows  : {features.get('latest_rows', 0)}")
        print(f"Feature max date     : {features.get('latest_max_date', '')}")

    diagnostics = data_summary.get("diagnostics", {})
    if diagnostics:
        print(f"Cache tickers        : {diagnostics.get('tickers_in_cache', 0)}")
        print(f"Missing required     : {diagnostics.get('required_tickers_missing', 0)}")
        print(f"Stale tickers        : {diagnostics.get('stale_count', 0)}")

    print("\n--- Exported strategy files ---")
    for name, path in exported.items():
        print(f"{name}: {path}")

    print()
    print(summary)


def run_daily() -> None:
    _run_live_cycle("daily")


def run_weekly() -> None:
    _run_live_cycle("weekly")


def run_monthly() -> None:
    _run_live_cycle("monthly")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PEA Strategy Platform main entry point."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="monthly",
        choices=["daily", "weekly", "monthly"],
        help="Run mode for the strategy platform.",
    )
    return parser.parse_args()


def main() -> None:
    ensure_project_directories()
    _ = load_config()  # validate config at startup

    args = parse_args()

    if args.mode == "daily":
        run_daily()
    elif args.mode == "weekly":
        run_weekly()
    elif args.mode == "monthly":
        run_monthly()
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()