from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.backtest.engine import BacktestResult, run_backtest
from src.backtest.plots import generate_all_backtest_plots
from src.backtest.reports import (
    build_backtest_text_summary,
    build_drawdown_table,
    build_monthly_returns_table,
    build_orders_summary_table,
    build_performance_summary_table,
    build_plot_commentaries,
    build_plot_report_html,
    build_portfolio_overview_table,
    build_positions_diagnostics_table,
    build_return_diagnostics_table,
    build_sleeve_performance_table,
    build_turnover_cost_table,
    build_yearly_returns_table,
)
from src.settings import EXPORT_DATA_DIR, ensure_project_directories


def _write_text(text: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _export_df(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def export_backtest_outputs(
    initial_capital: float = 50_000.0,
    rebalance_frequency: str = "monthly",
    start_date: str | None = None,
    end_date: str | None = None,
    result: BacktestResult | None = None,
) -> dict[str, Path]:
    ensure_project_directories()

    if result is None:
        result = run_backtest(
            initial_capital=initial_capital,
            rebalance_frequency=rebalance_frequency,
            start_date=start_date,
            end_date=end_date,
        )

    output_dir = EXPORT_DATA_DIR / "backtest"
    plots_dir = output_dir / "plots"
    tables_dir = output_dir / "tables"
    comments_dir = output_dir / "plot_comments"

    summary_text = build_backtest_text_summary(result)

    performance_summary = build_performance_summary_table(result)
    portfolio_overview = build_portfolio_overview_table(result)
    yearly_returns = build_yearly_returns_table(result)
    monthly_returns = build_monthly_returns_table(result)
    drawdown_table = build_drawdown_table(result)
    turnover_cost_table = build_turnover_cost_table(result)
    orders_summary = build_orders_summary_table(result)
    positions_diagnostics = build_positions_diagnostics_table(result)
    sleeve_performance = build_sleeve_performance_table(result)
    return_diagnostics = build_return_diagnostics_table(result)

    exported = {
        "summary_txt": _write_text(summary_text, output_dir / "backtest_summary.txt"),
        "equity_curve_csv": _export_df(result.equity_curve, tables_dir / "equity_curve.csv"),
        "orders_history_csv": _export_df(result.orders_history, tables_dir / "orders_history.csv"),
        "target_history_csv": _export_df(result.target_history, tables_dir / "target_history.csv"),
        "performance_summary_csv": _export_df(performance_summary, tables_dir / "performance_summary.csv"),
        "portfolio_overview_csv": _export_df(portfolio_overview, tables_dir / "portfolio_overview.csv"),
        "yearly_returns_csv": _export_df(yearly_returns, tables_dir / "yearly_returns.csv"),
        "monthly_returns_csv": _export_df(monthly_returns, tables_dir / "monthly_returns.csv"),
        "drawdown_table_csv": _export_df(drawdown_table, tables_dir / "drawdown_table.csv"),
        "turnover_cost_table_csv": _export_df(turnover_cost_table, tables_dir / "turnover_cost_table.csv"),
        "orders_summary_csv": _export_df(orders_summary, tables_dir / "orders_summary.csv"),
        "positions_diagnostics_csv": _export_df(positions_diagnostics, tables_dir / "positions_diagnostics.csv"),
        "sleeve_performance_csv": _export_df(sleeve_performance, tables_dir / "sleeve_performance.csv"),
        "return_diagnostics_csv": _export_df(return_diagnostics, tables_dir / "return_diagnostics.csv"),
    }

    plot_paths = generate_all_backtest_plots(result, plots_dir)
    for key, path in plot_paths.items():
        exported[f"{key}_plot"] = path

    plot_commentaries = build_plot_commentaries(result)
    for key, commentary in plot_commentaries.items():
        exported[f"{key}_commentary_txt"] = _write_text(commentary, comments_dir / f"{key}.txt")

    relative_plot_paths = {key: Path("plots") / path.name for key, path in plot_paths.items()}
    visual_report_html = build_plot_report_html(result, relative_plot_paths)
    exported["visual_report_html"] = _write_text(visual_report_html, output_dir / "backtest_visual_report.html")

    return exported


def export_backtest_results(
    initial_capital: float = 50_000.0,
    rebalance_frequency: str = "monthly",
    start_date: str | None = None,
    end_date: str | None = None,
    result: BacktestResult | None = None,
) -> dict[str, Path]:
    return export_backtest_outputs(
        initial_capital=initial_capital,
        rebalance_frequency=rebalance_frequency,
        start_date=start_date,
        end_date=end_date,
        result=result,
    )


if __name__ == "__main__":
    exported = export_backtest_outputs()

    print("\n=== EXPORTED BACKTEST OUTPUTS ===")
    for key, path in exported.items():
        print(f"{key}: {path}")