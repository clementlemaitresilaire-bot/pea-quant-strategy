from __future__ import annotations

import argparse

from src.reports.exports import export_all_strategy_outputs
from src.reports.summaries import build_run_summary
from src.settings import ensure_project_directories, load_config


def run_daily() -> None:
    """
    Daily mode:
    refresh data-dependent outputs and export current strategy state.
    """
    exported = export_all_strategy_outputs()
    summary = build_run_summary()

    print("\n=== DAILY RUN COMPLETED ===")
    for name, path in exported.items():
        print(f"{name}: {path}")

    print()
    print(summary)


def run_weekly() -> None:
    """
    Weekly mode:
    currently same execution path as daily, but intended for
    opportunistic review extensions later.
    """
    exported = export_all_strategy_outputs()
    summary = build_run_summary()

    print("\n=== WEEKLY RUN COMPLETED ===")
    for name, path in exported.items():
        print(f"{name}: {path}")

    print()
    print(summary)


def run_monthly() -> None:
    """
    Monthly mode:
    full strategic review including core sleeve review.
    """
    exported = export_all_strategy_outputs()
    summary = build_run_summary()

    print("\n=== MONTHLY RUN COMPLETED ===")
    for name, path in exported.items():
        print(f"{name}: {path}")

    print()
    print(summary)


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