from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.backtest.engine import BacktestResult


def _prepare_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


def _prepare_equity(result: BacktestResult) -> pd.DataFrame:
    equity = result.equity_curve.copy().sort_values("date").reset_index(drop=True)
    equity["date"] = pd.to_datetime(equity["date"], errors="coerce")
    equity = equity.dropna(subset=["date"]).copy()

    numeric_cols = [
        "portfolio_value",
        "benchmark_value",
        "daily_turnover",
        "costs_paid_this_day",
        "cumulative_costs_paid",
        "core_total_value",
        "monetary_total_value",
        "opp_total_value",
        "general_cash",
        "num_core_positions",
        "num_monetary_positions",
        "num_opp_positions",
    ]
    for col in numeric_cols:
        if col in equity.columns:
            equity[col] = pd.to_numeric(equity[col], errors="coerce")

    if "portfolio_value" in equity.columns:
        equity["portfolio_return"] = equity["portfolio_value"].pct_change()
    if "benchmark_value" in equity.columns:
        equity["benchmark_return"] = equity["benchmark_value"].pct_change()
    if "portfolio_return" in equity.columns and "benchmark_return" in equity.columns:
        equity["active_return"] = equity["portfolio_return"] - equity["benchmark_return"]

    if {"core_total_value", "monetary_total_value", "opp_total_value", "portfolio_value"}.issubset(equity.columns):
        total = equity["portfolio_value"].replace(0.0, np.nan)
        equity["core_weight"] = equity["core_total_value"] / total
        equity["monetary_weight"] = equity["monetary_total_value"] / total
        equity["opp_weight"] = equity["opp_total_value"] / total
        if "general_cash" in equity.columns:
            equity["cash_weight"] = equity["general_cash"] / total

    equity["month"] = equity["date"].dt.to_period("M").astype(str)
    return equity


def _save_fig(fig: plt.Figure, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_equity_vs_benchmark(result: BacktestResult, output_dir: Path) -> Path:
    equity = _prepare_equity(result)
    _prepare_output_dir(output_dir)

    fig, ax = plt.subplots(figsize=(12.5, 6.2))
    ax.plot(equity["date"], equity["portfolio_value"], linewidth=2.4, label="Portefeuille")
    if "benchmark_value" in equity.columns and equity["benchmark_value"].notna().any():
        ax.plot(equity["date"], equity["benchmark_value"], linewidth=1.8, label="Benchmark")

    ax.set_title("Valeur du portefeuille vs benchmark")
    ax.set_xlabel("Date")
    ax.set_ylabel("Valeur (EUR)")
    ax.grid(True, alpha=0.25)
    ax.legend()

    return _save_fig(fig, output_dir / "equity_vs_benchmark.png")


def plot_indexed_equity_vs_benchmark(result: BacktestResult, output_dir: Path) -> Path:
    equity = _prepare_equity(result)
    _prepare_output_dir(output_dir)

    fig, ax = plt.subplots(figsize=(12.5, 6.2))

    port = equity["portfolio_value"].dropna()
    if len(port) > 0 and float(port.iloc[0]) > 0:
        ax.plot(
            equity["date"],
            equity["portfolio_value"] / float(port.iloc[0]) * 100.0,
            linewidth=2.4,
            label="Portefeuille (Base 100)",
        )

    if "benchmark_value" in equity.columns and equity["benchmark_value"].notna().any():
        bench = equity["benchmark_value"].dropna()
        if len(bench) > 0 and float(bench.iloc[0]) > 0:
            ax.plot(
                equity["date"],
                equity["benchmark_value"] / float(bench.iloc[0]) * 100.0,
                linewidth=1.8,
                label="Benchmark (Base 100)",
            )

    ax.set_title("Performance cumulée rebasée")
    ax.set_xlabel("Date")
    ax.set_ylabel("Base 100")
    ax.grid(True, alpha=0.25)
    ax.legend()

    return _save_fig(fig, output_dir / "indexed_equity_vs_benchmark.png")


def plot_cumulative_active_return(result: BacktestResult, output_dir: Path) -> Path:
    equity = _prepare_equity(result)
    _prepare_output_dir(output_dir)

    fig, ax = plt.subplots(figsize=(12.5, 6.2))
    if "active_return" in equity.columns:
        cum_active = (1.0 + equity["active_return"].fillna(0.0)).cumprod() - 1.0
        ax.plot(equity["date"], cum_active, linewidth=2.2)
        ax.axhline(0.0, linewidth=1.0, alpha=0.6)

    ax.set_title("Performance active cumulée")
    ax.set_xlabel("Date")
    ax.set_ylabel("Performance active")
    ax.grid(True, alpha=0.25)

    return _save_fig(fig, output_dir / "cumulative_active_return.png")


def plot_drawdown(result: BacktestResult, output_dir: Path) -> Path:
    equity = _prepare_equity(result)
    _prepare_output_dir(output_dir)

    fig, ax = plt.subplots(figsize=(12.5, 6.2))
    if "portfolio_value" in equity.columns:
        running_max = equity["portfolio_value"].cummax()
        drawdown = equity["portfolio_value"] / running_max - 1.0
        ax.fill_between(equity["date"], drawdown, 0.0, alpha=0.75)
        ax.axhline(0.0, linewidth=1.0, alpha=0.6)

    ax.set_title("Drawdown du portefeuille")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.grid(True, alpha=0.25)

    return _save_fig(fig, output_dir / "drawdown.png")


def plot_rolling_return(result: BacktestResult, output_dir: Path, window: int = 252) -> Path:
    equity = _prepare_equity(result)
    _prepare_output_dir(output_dir)

    fig, ax = plt.subplots(figsize=(12.5, 6.2))

    if "portfolio_value" in equity.columns:
        rolling_port = equity["portfolio_value"] / equity["portfolio_value"].shift(window) - 1.0
        ax.plot(equity["date"], rolling_port, linewidth=2.2, label="Portefeuille")

    if "benchmark_value" in equity.columns and equity["benchmark_value"].notna().any():
        rolling_bench = equity["benchmark_value"] / equity["benchmark_value"].shift(window) - 1.0
        ax.plot(equity["date"], rolling_bench, linewidth=1.6, label="Benchmark")

    ax.axhline(0.0, linewidth=1.0, alpha=0.6)
    ax.set_title(f"Performance glissante sur {window} jours")
    ax.set_xlabel("Date")
    ax.set_ylabel("Performance")
    ax.grid(True, alpha=0.25)
    ax.legend()

    return _save_fig(fig, output_dir / "rolling_252d_return.png")


def plot_rolling_volatility(result: BacktestResult, output_dir: Path, window: int = 252) -> Path:
    equity = _prepare_equity(result)
    _prepare_output_dir(output_dir)

    fig, ax = plt.subplots(figsize=(12.5, 6.2))

    if "portfolio_return" in equity.columns:
        rolling_port_vol = equity["portfolio_return"].rolling(window).std(ddof=0) * np.sqrt(252)
        ax.plot(equity["date"], rolling_port_vol, linewidth=2.2, label="Portefeuille")

    if "benchmark_return" in equity.columns and equity["benchmark_return"].notna().any():
        rolling_bench_vol = equity["benchmark_return"].rolling(window).std(ddof=0) * np.sqrt(252)
        ax.plot(equity["date"], rolling_bench_vol, linewidth=1.6, label="Benchmark")

    ax.set_title(f"Volatilité annualisée glissante sur {window} jours")
    ax.set_xlabel("Date")
    ax.set_ylabel("Volatilité")
    ax.grid(True, alpha=0.25)
    ax.legend()

    return _save_fig(fig, output_dir / "rolling_252d_volatility.png")


def plot_rolling_sharpe(result: BacktestResult, output_dir: Path, window: int = 252) -> Path:
    equity = _prepare_equity(result)
    _prepare_output_dir(output_dir)

    fig, ax = plt.subplots(figsize=(12.5, 6.2))

    if "portfolio_return" in equity.columns:
        rolling_mean = equity["portfolio_return"].rolling(window).mean() * 252
        rolling_vol = equity["portfolio_return"].rolling(window).std(ddof=0) * np.sqrt(252)
        rolling_sharpe = rolling_mean / rolling_vol.replace(0.0, np.nan)
        ax.plot(equity["date"], rolling_sharpe, linewidth=2.2)

    ax.axhline(0.0, linewidth=1.0, alpha=0.6)
    ax.set_title(f"Sharpe glissant sur {window} jours")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sharpe")
    ax.grid(True, alpha=0.25)

    return _save_fig(fig, output_dir / "rolling_252d_sharpe.png")


def plot_turnover_and_costs(result: BacktestResult, output_dir: Path) -> Path:
    equity = _prepare_equity(result)
    _prepare_output_dir(output_dir)

    monthly = (
        equity.groupby("month", as_index=False)
        .agg(
            monthly_turnover=("daily_turnover", "sum"),
            monthly_costs=("costs_paid_this_day", "sum"),
        )
    )
    monthly["month_dt"] = pd.to_datetime(monthly["month"] + "-01")

    fig, axes = plt.subplots(2, 1, figsize=(13, 8.6), sharex=True)

    axes[0].bar(monthly["month_dt"], monthly["monthly_turnover"], width=20)
    axes[0].set_title("Turnover mensuel")
    axes[0].set_ylabel("Turnover")
    axes[0].grid(True, axis="y", alpha=0.25)

    axes[1].bar(monthly["month_dt"], monthly["monthly_costs"], width=20)
    axes[1].set_title("Coûts de transaction mensuels")
    axes[1].set_ylabel("Coûts (EUR)")
    axes[1].set_xlabel("Date")
    axes[1].grid(True, axis="y", alpha=0.25)

    return _save_fig(fig, output_dir / "turnover_and_costs.png")


def plot_cumulative_costs(result: BacktestResult, output_dir: Path) -> Path:
    equity = _prepare_equity(result)
    _prepare_output_dir(output_dir)

    fig, ax = plt.subplots(figsize=(12.5, 6.2))
    if "costs_paid_this_day" in equity.columns:
        cum_costs = equity["costs_paid_this_day"].fillna(0.0).cumsum()
        ax.plot(equity["date"], cum_costs, linewidth=2.2)

    ax.set_title("Coûts de transaction cumulés")
    ax.set_xlabel("Date")
    ax.set_ylabel("EUR")
    ax.grid(True, alpha=0.25)

    return _save_fig(fig, output_dir / "cumulative_costs.png")


def plot_sleeve_values(result: BacktestResult, output_dir: Path) -> Path:
    equity = _prepare_equity(result)
    _prepare_output_dir(output_dir)

    fig, ax = plt.subplots(figsize=(12.5, 6.2))
    for col, label in [
        ("core_total_value", "Core"),
        ("monetary_total_value", "Monétaire"),
        ("opp_total_value", "Opportuniste"),
        ("general_cash", "Cash"),
    ]:
        if col in equity.columns and equity[col].notna().any():
            ax.plot(equity["date"], equity[col], linewidth=2.0, label=label)

    ax.set_title("Valeur des poches au cours du temps")
    ax.set_xlabel("Date")
    ax.set_ylabel("Valeur (EUR)")
    ax.grid(True, alpha=0.25)
    ax.legend()

    return _save_fig(fig, output_dir / "sleeve_values.png")


def plot_sleeve_weights(result: BacktestResult, output_dir: Path) -> Path:
    equity = _prepare_equity(result)
    _prepare_output_dir(output_dir)

    fig, ax = plt.subplots(figsize=(12.5, 6.2))

    cols = []
    labels = []
    for col, label in [
        ("core_weight", "Core"),
        ("monetary_weight", "Monétaire"),
        ("opp_weight", "Opportuniste"),
        ("cash_weight", "Cash"),
    ]:
        if col in equity.columns and equity[col].notna().any():
            cols.append(equity[col].fillna(0.0).to_numpy())
            labels.append(label)

    if cols:
        ax.stackplot(equity["date"], *cols, labels=labels, alpha=0.88)
        ax.set_ylim(0, 1)

    ax.set_title("Poids des poches dans le portefeuille")
    ax.set_xlabel("Date")
    ax.set_ylabel("Poids")
    ax.grid(True, alpha=0.20)
    ax.legend(loc="upper right")

    return _save_fig(fig, output_dir / "sleeve_weights.png")


def plot_monthly_returns_heatmap(result: BacktestResult, output_dir: Path) -> Path:
    equity = _prepare_equity(result)
    _prepare_output_dir(output_dir)

    if "portfolio_value" not in equity.columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.set_title("Heatmap des rendements mensuels")
        return _save_fig(fig, output_dir / "monthly_returns_heatmap.png")

    work = equity[["date", "portfolio_value"]].copy()
    work["year"] = work["date"].dt.year
    work["month_num"] = work["date"].dt.month
    work["month_label"] = work["date"].dt.strftime("%b")

    rows = []
    for (year, month_num, month_label), subdf in work.groupby(["year", "month_num", "month_label"]):
        start = float(subdf["portfolio_value"].iloc[0])
        end = float(subdf["portfolio_value"].iloc[-1])
        ret = np.nan if start <= 0 else end / start - 1.0
        rows.append({"year": year, "month_num": month_num, "month_label": month_label, "ret": ret})

    monthly = pd.DataFrame(rows)
    if monthly.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.set_title("Heatmap des rendements mensuels")
        return _save_fig(fig, output_dir / "monthly_returns_heatmap.png")

    pivot = monthly.pivot(index="year", columns="month_label", values="ret")
    ordered_months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    pivot = pivot.reindex(columns=ordered_months)

    fig, ax = plt.subplots(figsize=(14, max(4.5, 0.55 * len(pivot.index))))
    im = ax.imshow(pivot.fillna(0.0).to_numpy(), aspect="auto")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index.astype(str))
    ax.set_title("Heatmap des rendements mensuels")

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j]
            text = "" if pd.isna(val) else f"{100 * val:.1f}%"
            ax.text(j, i, text, ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

    return _save_fig(fig, output_dir / "monthly_returns_heatmap.png")


def plot_positions_counts(result: BacktestResult, output_dir: Path) -> Path:
    equity = _prepare_equity(result)
    _prepare_output_dir(output_dir)

    fig, ax = plt.subplots(figsize=(12.5, 6.2))
    for col, label in [
        ("num_core_positions", "Positions Core"),
        ("num_monetary_positions", "Positions Monétaires"),
        ("num_opp_positions", "Positions Opportunistes"),
    ]:
        if col in equity.columns and equity[col].notna().any():
            ax.plot(equity["date"], equity[col], linewidth=2.0, label=label)

    ax.set_title("Nombre de positions par poche")
    ax.set_xlabel("Date")
    ax.set_ylabel("Nombre de positions")
    ax.grid(True, alpha=0.25)
    ax.legend()

    return _save_fig(fig, output_dir / "positions_counts.png")


def plot_orders_by_reason(result: BacktestResult, output_dir: Path) -> Path:
    _prepare_output_dir(output_dir)
    orders = result.orders_history.copy()

    fig, ax = plt.subplots(figsize=(12.5, 7.2))
    if not orders.empty and "decision_reason" in orders.columns:
        counts = (
            orders["decision_reason"]
            .fillna("UNKNOWN")
            .value_counts()
            .head(15)
            .sort_values(ascending=True)
        )
        ax.barh(counts.index.astype(str), counts.values)

    ax.set_title("Principales raisons de décision")
    ax.set_xlabel("Nombre d'occurrences")
    ax.grid(True, axis="x", alpha=0.25)

    return _save_fig(fig, output_dir / "orders_by_reason.png")


def plot_monthly_active_return(result: BacktestResult, output_dir: Path) -> Path:
    equity = _prepare_equity(result)
    _prepare_output_dir(output_dir)

    monthly = (
        equity.groupby("month", as_index=False)
        .agg(
            port_start=("portfolio_value", "first"),
            port_end=("portfolio_value", "last"),
            bench_start=("benchmark_value", "first"),
            bench_end=("benchmark_value", "last"),
        )
    )
    monthly["month_dt"] = pd.to_datetime(monthly["month"] + "-01")
    monthly["portfolio_return"] = monthly["port_end"] / monthly["port_start"] - 1.0
    monthly["benchmark_return"] = monthly["bench_end"] / monthly["bench_start"] - 1.0
    monthly["active_return"] = monthly["portfolio_return"] - monthly["benchmark_return"]

    fig, ax = plt.subplots(figsize=(13, 6.2))
    ax.bar(monthly["month_dt"], monthly["active_return"], width=20)
    ax.axhline(0.0, linewidth=1.0, alpha=0.6)
    ax.set_title("Performance active mensuelle")
    ax.set_xlabel("Date")
    ax.set_ylabel("Performance active")
    ax.grid(True, axis="y", alpha=0.25)

    return _save_fig(fig, output_dir / "monthly_active_return.png")


def generate_all_backtest_plots(result: BacktestResult, output_dir: Path) -> dict[str, Path]:
    _prepare_output_dir(output_dir)

    return {
        "equity_vs_benchmark": plot_equity_vs_benchmark(result, output_dir),
        "indexed_equity_vs_benchmark": plot_indexed_equity_vs_benchmark(result, output_dir),
        "cumulative_active_return": plot_cumulative_active_return(result, output_dir),
        "drawdown": plot_drawdown(result, output_dir),
        "rolling_252d_return": plot_rolling_return(result, output_dir, window=252),
        "rolling_252d_volatility": plot_rolling_volatility(result, output_dir, window=252),
        "rolling_252d_sharpe": plot_rolling_sharpe(result, output_dir, window=252),
        "turnover_and_costs": plot_turnover_and_costs(result, output_dir),
        "cumulative_costs": plot_cumulative_costs(result, output_dir),
        "sleeve_values": plot_sleeve_values(result, output_dir),
        "sleeve_weights": plot_sleeve_weights(result, output_dir),
        "positions_counts": plot_positions_counts(result, output_dir),
        "orders_by_reason": plot_orders_by_reason(result, output_dir),
        "monthly_active_return": plot_monthly_active_return(result, output_dir),
        "monthly_returns_heatmap": plot_monthly_returns_heatmap(result, output_dir),
    }