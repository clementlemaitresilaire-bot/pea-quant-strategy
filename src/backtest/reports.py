from __future__ import annotations

import html
from pathlib import Path

import numpy as np
import pandas as pd

from src.settings import EXPORT_DATA_DIR


def _get_result_field(result, field_name: str):
    if hasattr(result, field_name):
        return getattr(result, field_name)
    if isinstance(result, dict):
        return result.get(field_name)
    return None


def _clean_numeric_series(series: pd.Series) -> pd.Series:
    cleaned = pd.to_numeric(series, errors="coerce")
    cleaned = cleaned.replace([np.inf, -np.inf], np.nan)
    return cleaned.dropna()


def _safe_last(series: pd.Series) -> float | None:
    clean = _clean_numeric_series(series)
    if len(clean) == 0:
        return None
    return float(clean.iloc[-1])


def _first_last_return(series: pd.Series) -> float:
    clean = _clean_numeric_series(series)
    if len(clean) < 2:
        return np.nan
    start = float(clean.iloc[0])
    end = float(clean.iloc[-1])
    if start <= 0:
        return np.nan
    return float(end / start - 1.0)


def _annualized_return_from_series(series: pd.Series) -> float:
    clean = _clean_numeric_series(series)
    if len(clean) < 2:
        return np.nan
    start = float(clean.iloc[0])
    end = float(clean.iloc[-1])
    if start <= 0 or end <= 0:
        return np.nan
    years = len(clean) / 252
    if years <= 0:
        return np.nan
    return float((end / start) ** (1 / years) - 1.0)


def _annualized_volatility_from_series(series: pd.Series) -> float:
    clean = _clean_numeric_series(series)
    if len(clean) < 2:
        return np.nan
    returns = clean.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if len(returns) == 0:
        return np.nan
    return float(returns.std(ddof=0) * np.sqrt(252))


def _max_drawdown_from_series(series: pd.Series) -> float:
    clean = _clean_numeric_series(series)
    if len(clean) == 0:
        return np.nan
    running_max = clean.cummax()
    drawdown = clean / running_max - 1.0
    return float(drawdown.min())


def _fmt_pct(x: float | int | None) -> str:
    if x is None or pd.isna(x):
        return "N/A"
    return f"{100 * float(x):.2f}%"


def _fmt_num(x: float | int | None, decimals: int = 2) -> str:
    if x is None or pd.isna(x):
        return "N/A"
    return f"{float(x):,.{decimals}f}".replace(",", " ").replace(".", ",")


def _fmt_int(x: float | int | None) -> str:
    if x is None or pd.isna(x):
        return "N/A"
    return f"{int(round(float(x))):,}".replace(",", " ")


def _fmt_date(x) -> str:
    if x is None or pd.isna(x):
        return "N/A"
    return pd.to_datetime(x).strftime("%Y-%m-%d")


def _prepare_equity_curve(result) -> pd.DataFrame:
    equity_curve = _get_result_field(result, "equity_curve")
    if equity_curve is None or equity_curve.empty:
        return pd.DataFrame()

    df = equity_curve.copy().sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()

    numeric_candidates = [
        "portfolio_value",
        "benchmark_value",
        "daily_turnover",
        "costs_paid_this_day",
        "cumulative_costs_paid",
        "core_total_value",
        "monetary_total_value",
        "opp_total_value",
        "core_cash",
        "monetary_cash",
        "opp_cash",
        "num_core_positions",
        "num_monetary_positions",
        "num_opp_positions",
    ]
    for col in numeric_candidates:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "portfolio_value" in df.columns:
        df["portfolio_return"] = df["portfolio_value"].pct_change()
    if "benchmark_value" in df.columns:
        df["benchmark_return"] = df["benchmark_value"].pct_change()
    if "portfolio_return" in df.columns and "benchmark_return" in df.columns:
        df["active_return"] = df["portfolio_return"] - df["benchmark_return"]

    if {"core_total_value", "monetary_total_value", "opp_total_value", "portfolio_value"}.issubset(df.columns):
        total = df["portfolio_value"].replace(0.0, np.nan)
        df["core_weight"] = df["core_total_value"] / total
        df["monetary_weight"] = df["monetary_total_value"] / total
        df["opp_weight"] = df["opp_total_value"] / total

    return df


def _build_sleeve_performance_rows(df: pd.DataFrame) -> list[dict]:
    if df.empty:
        return []

    sleeve_map = [
        ("Portefeuille", "portfolio_value"),
        ("Core", "core_total_value"),
        ("Monétaire", "monetary_total_value"),
        ("Opportuniste", "opp_total_value"),
    ]

    rows: list[dict] = []
    portfolio_end = _safe_last(df["portfolio_value"]) if "portfolio_value" in df.columns else None

    for sleeve_name, col in sleeve_map:
        if col not in df.columns:
            continue

        clean = _clean_numeric_series(df[col])
        if len(clean) == 0:
            continue

        start_value = float(clean.iloc[0])
        end_value = float(clean.iloc[-1])

        row = {
            "poche": sleeve_name,
            "valeur_initiale": start_value,
            "valeur_finale": end_value,
            "performance_totale": _first_last_return(df[col]),
            "cagr": _annualized_return_from_series(df[col]),
            "volatilite_annualisee": _annualized_volatility_from_series(df[col]),
            "max_drawdown": _max_drawdown_from_series(df[col]),
        }

        if portfolio_end is not None and portfolio_end > 0:
            row["poids_final_dans_portefeuille"] = end_value / portfolio_end
        else:
            row["poids_final_dans_portefeuille"] = np.nan

        rows.append(row)

    return rows


def build_yearly_returns_table(result) -> pd.DataFrame:
    df = _prepare_equity_curve(result)
    columns = [
        "annee",
        "performance_portefeuille",
        "performance_benchmark",
        "performance_active",
        "performance_core",
        "performance_monetaire",
        "performance_opportuniste",
    ]
    if df.empty or "portfolio_value" not in df.columns:
        return pd.DataFrame(columns=columns)

    df["annee"] = df["date"].dt.year
    rows: list[dict] = []

    for year, subdf in df.groupby("annee"):
        portfolio_return = _first_last_return(subdf["portfolio_value"])
        benchmark_return = _first_last_return(subdf["benchmark_value"]) if "benchmark_value" in subdf.columns else np.nan
        rows.append(
            {
                "annee": int(year),
                "performance_portefeuille": portfolio_return,
                "performance_benchmark": benchmark_return,
                "performance_active": portfolio_return - benchmark_return if pd.notna(portfolio_return) and pd.notna(benchmark_return) else np.nan,
                "performance_core": _first_last_return(subdf["core_total_value"]) if "core_total_value" in subdf.columns else np.nan,
                "performance_monetaire": _first_last_return(subdf["monetary_total_value"]) if "monetary_total_value" in subdf.columns else np.nan,
                "performance_opportuniste": _first_last_return(subdf["opp_total_value"]) if "opp_total_value" in subdf.columns else np.nan,
            }
        )

    return pd.DataFrame(rows).sort_values("annee").reset_index(drop=True)


def build_monthly_returns_table(result) -> pd.DataFrame:
    df = _prepare_equity_curve(result)
    columns = [
        "mois",
        "performance_portefeuille",
        "performance_benchmark",
        "performance_active",
        "performance_core",
        "performance_monetaire",
        "performance_opportuniste",
    ]
    if df.empty or "portfolio_value" not in df.columns:
        return pd.DataFrame(columns=columns)

    df["mois"] = df["date"].dt.to_period("M").astype(str)
    rows: list[dict] = []

    for month, subdf in df.groupby("mois"):
        portfolio_return = _first_last_return(subdf["portfolio_value"])
        benchmark_return = _first_last_return(subdf["benchmark_value"]) if "benchmark_value" in subdf.columns else np.nan
        rows.append(
            {
                "mois": month,
                "performance_portefeuille": portfolio_return,
                "performance_benchmark": benchmark_return,
                "performance_active": portfolio_return - benchmark_return if pd.notna(portfolio_return) and pd.notna(benchmark_return) else np.nan,
                "performance_core": _first_last_return(subdf["core_total_value"]) if "core_total_value" in subdf.columns else np.nan,
                "performance_monetaire": _first_last_return(subdf["monetary_total_value"]) if "monetary_total_value" in subdf.columns else np.nan,
                "performance_opportuniste": _first_last_return(subdf["opp_total_value"]) if "opp_total_value" in subdf.columns else np.nan,
            }
        )

    return pd.DataFrame(rows).sort_values("mois").reset_index(drop=True)


def build_drawdown_table(result, top_n: int = 15) -> pd.DataFrame:
    df = _prepare_equity_curve(result)
    if df.empty or "portfolio_value" not in df.columns:
        return pd.DataFrame(columns=["date", "valeur_portefeuille", "plus_haut_historique", "drawdown"])

    work = df[["date", "portfolio_value"]].dropna().copy()
    work["plus_haut_historique"] = work["portfolio_value"].cummax()
    work["drawdown"] = work["portfolio_value"] / work["plus_haut_historique"] - 1.0
    work = work.rename(columns={"portfolio_value": "valeur_portefeuille"})
    return work.sort_values("drawdown").head(top_n).reset_index(drop=True)


def build_performance_summary_table(result) -> pd.DataFrame:
    metrics = _get_result_field(result, "metrics") or {}
    rows = []

    mapping = {
        "total_return": "Performance totale",
        "cagr": "CAGR",
        "annualized_volatility": "Volatilité annualisée",
        "sharpe": "Sharpe",
        "max_drawdown": "Max drawdown",
        "benchmark_total_return": "Performance benchmark",
        "benchmark_cagr": "CAGR benchmark",
        "active_return": "Performance active",
        "tracking_error": "Tracking error",
        "information_ratio": "Information ratio",
        "average_rebalance_turnover": "Turnover moyen de rebalance",
        "total_turnover": "Turnover total",
        "total_costs_paid": "Coûts totaux payés",
        "total_costs_as_pct_initial": "Coûts / capital initial",
        "num_orders": "Nombre d'ordres",
        "num_executed_orders": "Nombre d'ordres exécutés",
    }

    for key, label in mapping.items():
        if key in metrics:
            rows.append({"categorie": "Metrics globales", "indicateur": label, "valeur": metrics.get(key)})

    return pd.DataFrame(rows)


def build_portfolio_overview_table(result) -> pd.DataFrame:
    df = _prepare_equity_curve(result)
    if df.empty:
        return pd.DataFrame(columns=["element", "valeur"])

    rows: list[dict] = []
    rows.append({"element": "Date de début", "valeur": str(df["date"].iloc[0].date())})
    rows.append({"element": "Date de fin", "valeur": str(df["date"].iloc[-1].date())})

    mappings = [
        ("Valeur finale portefeuille", "portfolio_value"),
        ("Valeur finale benchmark", "benchmark_value"),
        ("Valeur finale poche core", "core_total_value"),
        ("Valeur finale poche monétaire", "monetary_total_value"),
        ("Valeur finale poche opportuniste", "opp_total_value"),
        ("Cash final core", "core_cash"),
        ("Cash final monétaire", "monetary_cash"),
        ("Cash final opportuniste", "opp_cash"),
        ("Nombre final positions core", "num_core_positions"),
        ("Nombre final positions monétaires", "num_monetary_positions"),
        ("Nombre final positions opportunistes", "num_opp_positions"),
        ("Coûts cumulés payés", "cumulative_costs_paid"),
    ]
    for label, col in mappings:
        if col in df.columns:
            rows.append({"element": label, "valeur": _safe_last(df[col])})

    portfolio_last = _safe_last(df["portfolio_value"]) if "portfolio_value" in df.columns else None
    if portfolio_last is not None and portfolio_last > 0:
        if "core_total_value" in df.columns:
            rows.append({"element": "Poids final core", "valeur": (_safe_last(df["core_total_value"]) or 0.0) / portfolio_last})
        if "monetary_total_value" in df.columns:
            rows.append({"element": "Poids final monétaire", "valeur": (_safe_last(df["monetary_total_value"]) or 0.0) / portfolio_last})
        if "opp_total_value" in df.columns:
            rows.append({"element": "Poids final opportuniste", "valeur": (_safe_last(df["opp_total_value"]) or 0.0) / portfolio_last})

    return pd.DataFrame(rows)


def build_turnover_cost_table(result) -> pd.DataFrame:
    df = _prepare_equity_curve(result)
    if df.empty:
        return pd.DataFrame(columns=["indicateur", "valeur"])

    total_turnover = float(df["daily_turnover"].fillna(0.0).sum()) if "daily_turnover" in df.columns else 0.0
    average_daily_turnover = float(df["daily_turnover"].fillna(0.0).mean()) if "daily_turnover" in df.columns else 0.0
    rebalance_days = df.loc[df.get("daily_turnover", 0.0).fillna(0.0) > 0, "daily_turnover"] if "daily_turnover" in df.columns else pd.Series(dtype=float)
    average_rebalance_turnover = float(rebalance_days.mean()) if not rebalance_days.empty else 0.0
    max_daily_turnover = float(df["daily_turnover"].fillna(0.0).max()) if "daily_turnover" in df.columns else 0.0
    total_costs_paid = float(df["costs_paid_this_day"].fillna(0.0).sum()) if "costs_paid_this_day" in df.columns else 0.0
    average_cost_per_day = float(df["costs_paid_this_day"].fillna(0.0).mean()) if "costs_paid_this_day" in df.columns else 0.0
    max_daily_cost = float(df["costs_paid_this_day"].fillna(0.0).max()) if "costs_paid_this_day" in df.columns else 0.0

    return pd.DataFrame(
        [
            {"indicateur": "Turnover total", "valeur": total_turnover},
            {"indicateur": "Turnover moyen quotidien", "valeur": average_daily_turnover},
            {"indicateur": "Turnover moyen sur jours de rebalance", "valeur": average_rebalance_turnover},
            {"indicateur": "Turnover journalier maximum", "valeur": max_daily_turnover},
            {"indicateur": "Coûts totaux payés", "valeur": total_costs_paid},
            {"indicateur": "Coût moyen par jour", "valeur": average_cost_per_day},
            {"indicateur": "Coût journalier maximum", "valeur": max_daily_cost},
        ]
    )


def build_orders_summary_table(result) -> pd.DataFrame:
    orders_history = _get_result_field(result, "orders_history")
    if orders_history is None or orders_history.empty:
        return pd.DataFrame(columns=["execute", "raison_decision", "nombre", "volume_total", "volume_moyen"])

    df = orders_history.copy()
    df["effective_order_value"] = pd.to_numeric(df.get("effective_order_value", 0.0), errors="coerce").fillna(0.0)
    df["execute"] = df.get("execute", True)
    df["decision_reason"] = df.get("decision_reason", "UNKNOWN").fillna("UNKNOWN")

    out = (
        df.groupby(["execute", "decision_reason"], dropna=False)
        .agg(
            nombre=("decision_reason", "size"),
            volume_total=("effective_order_value", "sum"),
            volume_moyen=("effective_order_value", "mean"),
        )
        .reset_index()
        .rename(columns={"decision_reason": "raison_decision"})
        .sort_values(["execute", "nombre"], ascending=[False, False])
        .reset_index(drop=True)
    )
    return out


def build_positions_diagnostics_table(result) -> pd.DataFrame:
    df = _prepare_equity_curve(result)
    if df.empty:
        return pd.DataFrame(columns=["indicateur", "valeur"])

    def _last_or_zero(col: str) -> float:
        if col not in df.columns:
            return 0.0
        clean = _clean_numeric_series(df[col])
        return float(clean.iloc[-1]) if len(clean) > 0 else 0.0

    def _mean_or_zero(col: str) -> float:
        if col not in df.columns:
            return 0.0
        return float(pd.to_numeric(df[col], errors="coerce").fillna(0.0).mean())

    return pd.DataFrame(
        [
            {"indicateur": "Positions finales core", "valeur": _last_or_zero("num_core_positions")},
            {"indicateur": "Positions finales monétaires", "valeur": _last_or_zero("num_monetary_positions")},
            {"indicateur": "Positions finales opportunistes", "valeur": _last_or_zero("num_opp_positions")},
            {"indicateur": "Moyenne positions core", "valeur": _mean_or_zero("num_core_positions")},
            {"indicateur": "Moyenne positions monétaires", "valeur": _mean_or_zero("num_monetary_positions")},
            {"indicateur": "Moyenne positions opportunistes", "valeur": _mean_or_zero("num_opp_positions")},
        ]
    )


def build_sleeve_performance_table(result) -> pd.DataFrame:
    df = _prepare_equity_curve(result)
    columns = [
        "poche",
        "valeur_initiale",
        "valeur_finale",
        "performance_totale",
        "cagr",
        "volatilite_annualisee",
        "max_drawdown",
        "poids_final_dans_portefeuille",
    ]
    if df.empty:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(_build_sleeve_performance_rows(df), columns=columns)


def build_return_diagnostics_table(result) -> pd.DataFrame:
    df = _prepare_equity_curve(result)
    columns = ["indicateur", "valeur"]
    if df.empty or "portfolio_value" not in df.columns:
        return pd.DataFrame(columns=columns)

    monthly = build_monthly_returns_table(result)
    yearly = build_yearly_returns_table(result)

    portfolio_returns = pd.to_numeric(df["portfolio_return"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    active_returns = pd.to_numeric(df.get("active_return", pd.Series(dtype=float)), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()

    rows: list[dict] = []

    if not monthly.empty:
        port_month = pd.to_numeric(monthly["performance_portefeuille"], errors="coerce").dropna()
        rows.extend([
            {"indicateur": "Nombre de mois", "valeur": float(len(port_month))},
            {"indicateur": "Proportion de mois positifs", "valeur": float((port_month > 0).mean()) if len(port_month) > 0 else np.nan},
            {"indicateur": "Meilleur mois", "valeur": float(port_month.max()) if len(port_month) > 0 else np.nan},
            {"indicateur": "Pire mois", "valeur": float(port_month.min()) if len(port_month) > 0 else np.nan},
        ])

    if not yearly.empty:
        port_year = pd.to_numeric(yearly["performance_portefeuille"], errors="coerce").dropna()
        rows.extend([
            {"indicateur": "Nombre d'années", "valeur": float(len(port_year))},
            {"indicateur": "Proportion d'années positives", "valeur": float((port_year > 0).mean()) if len(port_year) > 0 else np.nan},
            {"indicateur": "Meilleure année", "valeur": float(port_year.max()) if len(port_year) > 0 else np.nan},
            {"indicateur": "Pire année", "valeur": float(port_year.min()) if len(port_year) > 0 else np.nan},
        ])

    if len(portfolio_returns) > 0:
        rows.extend([
            {"indicateur": "Meilleur jour", "valeur": float(portfolio_returns.max())},
            {"indicateur": "Pire jour", "valeur": float(portfolio_returns.min())},
        ])

    if len(active_returns) > 0:
        rows.extend([
            {"indicateur": "Proportion de jours à alpha positif", "valeur": float((active_returns > 0).mean())},
            {"indicateur": "Meilleur jour actif", "valeur": float(active_returns.max())},
            {"indicateur": "Pire jour actif", "valeur": float(active_returns.min())},
        ])

    return pd.DataFrame(rows, columns=columns)


def _load_ml_comparison_table() -> pd.DataFrame:
    path = EXPORT_DATA_DIR / "backtest" / "ml_comparison_metrics.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def build_plot_commentaries(result) -> dict[str, str]:
    df = _prepare_equity_curve(result)
    metrics = _get_result_field(result, "metrics") or {}
    out: dict[str, str] = {}

    if df.empty:
        return out

    start_date = _fmt_date(df["date"].iloc[0])
    end_date = _fmt_date(df["date"].iloc[-1])

    portfolio_total_return = metrics.get("total_return", _first_last_return(df["portfolio_value"])) if "portfolio_value" in df.columns else np.nan
    benchmark_total_return = metrics.get("benchmark_total_return", _first_last_return(df["benchmark_value"])) if "benchmark_value" in df.columns else np.nan
    active_total_return = metrics.get("active_return", portfolio_total_return - benchmark_total_return)

    out["equity_vs_benchmark"] = (
        f"Ce graphique montre la trajectoire de la valeur absolue du portefeuille par rapport au benchmark entre "
        f"{start_date} et {end_date}. Le portefeuille termine sur une performance totale de {_fmt_pct(portfolio_total_return)} "
        f"contre {_fmt_pct(benchmark_total_return)} pour le benchmark, soit une performance active finale de {_fmt_pct(active_total_return)}."
    )

    out["indexed_equity_vs_benchmark"] = (
        "La vue rebasée en base 100 permet de comparer la dynamique de performance indépendamment du capital de départ. "
        "C’est la meilleure lecture pour voir si la surperformance est régulière ou concentrée sur quelques périodes."
    )

    if "active_return" in df.columns:
        cum_active = (1.0 + df["active_return"].fillna(0.0)).cumprod() - 1.0
        out["cumulative_active_return"] = (
            f"La performance active cumulée se termine à {_fmt_pct(cum_active.iloc[-1])}. "
            "Cette figure permet de juger si l’alpha est structurel ou simplement ponctuel."
        )

    if "portfolio_value" in df.columns:
        running_max = df["portfolio_value"].cummax()
        drawdown = df["portfolio_value"] / running_max - 1.0
        out["drawdown"] = (
            f"Le drawdown maximum atteint {_fmt_pct(drawdown.min())}. "
            "Cette figure est essentielle pour mesurer le coût psychologique et financier de la stratégie en phase défavorable."
        )

    out["rolling_252d_return"] = (
        "Les performances glissantes sur un an permettent de repérer les périodes de forte création de valeur "
        "et celles où la stratégie perd temporairement son avantage."
    )
    out["rolling_252d_volatility"] = (
        "La volatilité glissante annualisée montre si le profil de risque est stable ou s’il varie fortement selon les régimes de marché."
    )
    out["rolling_252d_sharpe"] = (
        "Le Sharpe glissant permet de savoir si la performance a été obtenue efficacement par unité de risque prise."
    )
    out["turnover_and_costs"] = (
        "Le turnover mensuel et les coûts mensuels sont séparés pour rester lisibles. "
        "Cette figure aide à identifier les épisodes de sur-rotation et les périodes où l’exécution détruit du rendement."
    )
    out["cumulative_costs"] = (
        "Les coûts cumulés mesurent la part de performance brute qui a été abandonnée en frais d’exécution au fil du temps."
    )
    out["sleeve_values"] = (
        "Cette figure décompose la contribution des différentes poches à la croissance de la valeur globale du portefeuille."
    )
    out["sleeve_weights"] = (
        "La vue empilée des poids permet de comprendre comment l’allocation entre les poches évolue au cours du backtest."
    )
    out["positions_counts"] = (
        "Le nombre de positions par poche permet d’analyser la concentration, la diversification effective et l’activité réelle de chaque sleeve."
    )
    out["orders_by_reason"] = (
        "Le détail des raisons de décision est particulièrement utile pour diagnostiquer les freins du moteur : deadband, coûts, alpha hurdle ou sorties forcées."
    )
    out["monthly_active_return"] = (
        "La performance active mensuelle donne une vue plus exploitable que le bruit quotidien pour juger de la régularité de l’alpha."
    )
    out["monthly_returns_heatmap"] = (
        "La heatmap mensuelle synthétise très rapidement la régularité de la performance, la fréquence des mois négatifs et la concentration des gains."
    )

    return out


def build_french_global_conclusion(result) -> str:
    metrics = _get_result_field(result, "metrics") or {}
    df = _prepare_equity_curve(result)
    ml_df = _load_ml_comparison_table()

    total_return = metrics.get("total_return", np.nan)
    cagr = metrics.get("cagr", np.nan)
    vol = metrics.get("annualized_volatility", np.nan)
    sharpe = metrics.get("sharpe", np.nan)
    mdd = metrics.get("max_drawdown", np.nan)
    active = metrics.get("active_return", np.nan)
    info_ratio = metrics.get("information_ratio", np.nan)
    costs = metrics.get("total_costs_paid", np.nan)
    turnover = metrics.get("total_turnover", np.nan)

    sleeve_table = build_sleeve_performance_table(result)
    best_sleeve_sentence = ""
    if not sleeve_table.empty:
        sleeve_work = sleeve_table.loc[sleeve_table["poche"] != "Portefeuille"].copy()
        if not sleeve_work.empty:
            best_row = sleeve_work.sort_values("performance_totale", ascending=False).iloc[0]
            worst_row = sleeve_work.sort_values("performance_totale", ascending=True).iloc[0]
            best_sleeve_sentence = (
                f"La meilleure contribution par poche provient de {best_row['poche']} avec une performance totale de "
                f"{_fmt_pct(best_row['performance_totale'])}, tandis que la poche la plus faible est {worst_row['poche']} "
                f"avec {_fmt_pct(worst_row['performance_totale'])}. "
            )

    ml_sentence = ""
    if not ml_df.empty and {"label", "total_return"}.issubset(ml_df.columns):
        try:
            no_ml = ml_df.loc[ml_df["label"] == "without_ml"].iloc[0]
            yes_ml = ml_df.loc[ml_df["label"] == "with_ml"].iloc[0]
            delta_ret = float(yes_ml["total_return"] - no_ml["total_return"])
            delta_sharpe = float(yes_ml["sharpe"] - no_ml["sharpe"]) if "sharpe" in ml_df.columns else np.nan
            ml_sentence = (
                f"Le comparatif ML vs non-ML montre une amélioration de {_fmt_pct(delta_ret)} de performance totale "
                f"et une variation de Sharpe de {_fmt_num(delta_sharpe)}. "
            )
        except Exception:
            ml_sentence = ""

    if df.empty:
        return "Aucune conclusion disponible faute de données de backtest."

    return (
        f"Au global, la stratégie délivre une performance totale de {_fmt_pct(total_return)} sur la période étudiée, "
        f"soit un CAGR de {_fmt_pct(cagr)}, avec une volatilité annualisée de {_fmt_pct(vol)} et un Sharpe de {_fmt_num(sharpe)}. "
        f"La surperformance par rapport au benchmark atteint {_fmt_pct(active)}, pour un information ratio de {_fmt_num(info_ratio)}. "
        f"Le principal coût de cette performance reste un drawdown maximum de {_fmt_pct(mdd)} ainsi qu’un turnover total de {_fmt_num(turnover)} "
        f"et des coûts d’exécution cumulés de {_fmt_num(costs)} EUR. "
        f"{best_sleeve_sentence}"
        f"{ml_sentence}"
        "En synthèse, la stratégie apparaît cohérente : elle crée de l’alpha sans dégrader excessivement le profil de risque, "
        "mais son efficacité dépend encore fortement de la discipline d’exécution, de la maîtrise des coûts et de la qualité du moteur de sélection."
    )


def _html_table(
    df: pd.DataFrame,
    pct_cols: set[str] | None = None,
    int_cols: set[str] | None = None,
    max_rows: int | None = 20,
) -> str:
    if df is None or df.empty:
        return '<div class="empty">Aucune donnée disponible.</div>'

    work = df.copy()
    if max_rows is not None:
        work = work.head(max_rows).copy()

    pct_cols = pct_cols or set()
    int_cols = int_cols or set()

    def _fmt(val, col):
        if pd.isna(val):
            return ""
        if col in pct_cols:
            return _fmt_pct(val)
        if col in int_cols:
            return _fmt_int(val)
        if isinstance(val, (int, float, np.integer, np.floating)):
            return _fmt_num(val)
        return html.escape(str(val))

    headers = "".join(f"<th>{html.escape(str(c))}</th>" for c in work.columns)
    rows = []
    for _, row in work.iterrows():
        tds = "".join(f"<td>{_fmt(row[c], c)}</td>" for c in work.columns)
        rows.append(f"<tr>{tds}</tr>")

    return f"""
    <div class="table-wrap">
      <table class="data-table">
        <thead><tr>{headers}</tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
    </div>
    """


def build_plot_report_html(result, plot_paths: dict[str, str] | dict) -> str:
    metrics = _get_result_field(result, "metrics") or {}
    equity = _prepare_equity_curve(result)
    plot_commentaries = build_plot_commentaries(result)

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
    ml_comparison = _load_ml_comparison_table()
    conclusion_text = build_french_global_conclusion(result)

    start_date = _fmt_date(equity["date"].iloc[0]) if not equity.empty else "N/A"
    end_date = _fmt_date(equity["date"].iloc[-1]) if not equity.empty else "N/A"

    kpis = [
        ("Performance totale", _fmt_pct(metrics.get("total_return"))),
        ("CAGR", _fmt_pct(metrics.get("cagr"))),
        ("Volatilité", _fmt_pct(metrics.get("annualized_volatility"))),
        ("Sharpe", _fmt_num(metrics.get("sharpe"))),
        ("Max drawdown", _fmt_pct(metrics.get("max_drawdown"))),
        ("Performance active", _fmt_pct(metrics.get("active_return"))),
        ("Tracking error", _fmt_pct(metrics.get("tracking_error"))),
        ("Information ratio", _fmt_num(metrics.get("information_ratio"))),
        ("Coûts payés", f"{_fmt_num(metrics.get('total_costs_paid'))} EUR"),
        ("Ordres exécutés", _fmt_int(metrics.get("num_executed_orders"))),
    ]

    section_order = [
        "equity_vs_benchmark",
        "indexed_equity_vs_benchmark",
        "cumulative_active_return",
        "drawdown",
        "rolling_252d_return",
        "rolling_252d_volatility",
        "rolling_252d_sharpe",
        "monthly_active_return",
        "monthly_returns_heatmap",
        "sleeve_values",
        "sleeve_weights",
        "positions_counts",
        "turnover_and_costs",
        "cumulative_costs",
        "orders_by_reason",
    ]

    title_map = {
        "equity_vs_benchmark": "Valeur du portefeuille vs benchmark",
        "indexed_equity_vs_benchmark": "Performance cumulée rebasée",
        "cumulative_active_return": "Performance active cumulée",
        "drawdown": "Drawdown",
        "rolling_252d_return": "Performance glissante 252 jours",
        "rolling_252d_volatility": "Volatilité glissante 252 jours",
        "rolling_252d_sharpe": "Sharpe glissant 252 jours",
        "monthly_active_return": "Performance active mensuelle",
        "monthly_returns_heatmap": "Heatmap des rendements mensuels",
        "sleeve_values": "Valeur des poches",
        "sleeve_weights": "Poids des poches",
        "positions_counts": "Nombre de positions",
        "turnover_and_costs": "Turnover et coûts",
        "cumulative_costs": "Coûts cumulés",
        "orders_by_reason": "Raisons des décisions",
    }

    plot_cards = []
    for key in section_order:
        if key not in plot_paths:
            continue
        rel_path = str(plot_paths[key]).replace("\\", "/")
        plot_cards.append(
            f"""
            <section class="plot-card">
              <h3>{html.escape(title_map.get(key, key))}</h3>
              <img src="{html.escape(rel_path)}" alt="{html.escape(title_map.get(key, key))}">
              <p class="commentary">{html.escape(plot_commentaries.get(key, ""))}</p>
            </section>
            """
        )

    ml_block = ""
    if not ml_comparison.empty:
        ml_block = f"""
        <section class="panel">
          <div class="section-head">
            <h2>Comparatif ML vs non-ML</h2>
            <p>Cette section permet de vérifier si la couche ML modifie réellement la stratégie en performance, en risque et en exécution.</p>
          </div>
          {_html_table(
              ml_comparison,
              pct_cols={"total_return","cagr","annualized_volatility","max_drawdown","benchmark_total_return","benchmark_cagr","active_return","tracking_error"},
              int_cols={"num_orders","num_executed_orders","executed_orders_delta_vs_without_ml"},
              max_rows=None,
          )}
        </section>
        """

    html_out = f"""
<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Rapport visuel de backtest</title>
<style>
:root {{
  --bg: #f6f8fb;
  --panel: #ffffff;
  --text: #0f172a;
  --muted: #64748b;
  --border: #e2e8f0;
  --accent: #1d4ed8;
  --shadow: 0 10px 24px rgba(15, 23, 42, 0.05);
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
  background: var(--bg);
  color: var(--text);
}}
header {{
  background: linear-gradient(135deg, #0f172a, #1e3a8a);
  color: white;
  padding: 34px 42px;
}}
header h1 {{
  margin: 0 0 10px 0;
  font-size: 34px;
}}
header p {{
  margin: 6px 0;
  color: #dbeafe;
}}
main {{
  max-width: 1540px;
  margin: 0 auto;
  padding: 28px 24px 64px;
}}
.section-head {{
  margin-bottom: 14px;
}}
.section-head h2 {{
  margin: 0 0 8px 0;
}}
.section-head p {{
  margin: 0;
  color: var(--muted);
  line-height: 1.55;
}}
.kpi-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(185px, 1fr));
  gap: 14px;
  margin-bottom: 24px;
}}
.kpi {{
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 16px 18px;
  box-shadow: var(--shadow);
}}
.kpi .label {{
  color: var(--muted);
  font-size: 13px;
  margin-bottom: 8px;
}}
.kpi .value {{
  font-size: 24px;
  font-weight: 700;
}}
.panel {{
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 22px 22px 18px;
  margin-bottom: 22px;
  box-shadow: var(--shadow);
}}
.two-col {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 22px;
}}
.three-col {{
  display: grid;
  grid-template-columns: 1fr 1fr 1fr;
  gap: 22px;
}}
@media (max-width: 1150px) {{
  .two-col, .three-col {{ grid-template-columns: 1fr; }}
}}
.table-wrap {{
  overflow-x: auto;
}}
.data-table {{
  width: 100%;
  border-collapse: collapse;
  font-size: 14px;
  min-width: 640px;
}}
.data-table th, .data-table td {{
  border-bottom: 1px solid var(--border);
  padding: 10px 10px;
  text-align: left;
  vertical-align: top;
  white-space: nowrap;
}}
.data-table th {{
  background: #f8fafc;
  font-weight: 700;
  position: sticky;
  top: 0;
}}
.data-table tr:hover td {{
  background: #f8fbff;
}}
.plot-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(520px, 1fr));
  gap: 22px;
}}
@media (max-width: 700px) {{
  .plot-grid {{ grid-template-columns: 1fr; }}
}}
.plot-card {{
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 16px;
  box-shadow: var(--shadow);
}}
.plot-card h3 {{
  margin-top: 0;
  margin-bottom: 12px;
}}
.plot-card img {{
  width: 100%;
  height: auto;
  border-radius: 10px;
  border: 1px solid var(--border);
  background: white;
}}
.commentary {{
  margin: 12px 0 4px;
  color: #334155;
  line-height: 1.6;
}}
.summary-box {{
  background: #eff6ff;
  border: 1px solid #bfdbfe;
  border-radius: 16px;
  padding: 18px 20px;
  line-height: 1.7;
  color: #1e3a8a;
}}
.empty {{
  color: var(--muted);
  font-style: italic;
}}
footer {{
  color: var(--muted);
  margin-top: 26px;
  font-size: 13px;
}}
</style>
</head>
<body>
<header>
  <h1>Rapport visuel de backtest</h1>
  <p><strong>Période :</strong> {html.escape(start_date)} → {html.escape(end_date)}</p>
  <p><strong>Objet :</strong> performance, risque, allocation par poches, exécution, coûts et diagnostics ML.</p>
</header>

<main>

  <section class="panel">
    <div class="section-head">
      <h2>Vue d'ensemble</h2>
      <p>Cette première section donne les indicateurs clés du backtest et permet de juger immédiatement la qualité globale de la stratégie.</p>
    </div>
    <section class="kpi-grid">
      {"".join(f'<div class="kpi"><div class="label">{html.escape(label)}</div><div class="value">{html.escape(value)}</div></div>' for label, value in kpis)}
    </section>
  </section>

  <section class="two-col">
    <section class="panel">
      <div class="section-head">
        <h2>Résumé du portefeuille</h2>
        <p>Valeurs finales, poids de fin de période, cash et structure globale du portefeuille.</p>
      </div>
      {_html_table(portfolio_overview, max_rows=None)}
    </section>

    <section class="panel">
      <div class="section-head">
        <h2>Résumé des performances</h2>
        <p>Tableau synthétique des indicateurs de performance et de risque du backtest complet.</p>
      </div>
      {_html_table(performance_summary, pct_cols={"valeur"}, int_cols=set(), max_rows=None)}
    </section>
  </section>

  <section class="three-col">
    <section class="panel">
      <div class="section-head">
        <h2>Performance par poche</h2>
        <p>Contribution relative des différentes sleeves en performance, risque et poids final.</p>
      </div>
      {_html_table(
          sleeve_performance,
          pct_cols={"performance_totale","cagr","volatilite_annualisee","max_drawdown","poids_final_dans_portefeuille"},
          max_rows=None,
      )}
    </section>

    <section class="panel">
      <div class="section-head">
        <h2>Diagnostics rendement / risque</h2>
        <p>Régularité de la stratégie à travers les mois, années, journées et performance active.</p>
      </div>
      {_html_table(return_diagnostics, pct_cols={"valeur"}, max_rows=None)}
    </section>

    <section class="panel">
      <div class="section-head">
        <h2>Diagnostics de positions</h2>
        <p>Analyse de la concentration et du niveau moyen d’investissement par poche.</p>
      </div>
      {_html_table(positions_diagnostics, max_rows=None)}
    </section>
  </section>

  <section class="two-col">
    <section class="panel">
      <div class="section-head">
        <h2>Turnover et coûts</h2>
        <p>Mesure de l’intensité de rotation de la stratégie et de l’impact des coûts de transaction.</p>
      </div>
      {_html_table(turnover_cost_table, max_rows=None)}
    </section>

    <section class="panel">
      <div class="section-head">
        <h2>Résumé des décisions d’ordres</h2>
        <p>Vue consolidée des raisons d’exécution ou de rejet des ordres proposés par le moteur.</p>
      </div>
      {_html_table(orders_summary, int_cols={"nombre"}, max_rows=20)}
    </section>
  </section>

  {ml_block}

  <section class="two-col">
    <section class="panel">
      <div class="section-head">
        <h2>Performances annuelles</h2>
        <p>Vue annuelle du portefeuille, du benchmark et des poches pour identifier les régimes les plus favorables.</p>
      </div>
      {_html_table(
          yearly_returns,
          pct_cols={"performance_portefeuille","performance_benchmark","performance_active","performance_core","performance_monetaire","performance_opportuniste"},
          max_rows=None,
      )}
    </section>

    <section class="panel">
      <div class="section-head">
        <h2>Pires drawdowns</h2>
        <p>Les épisodes de baisse les plus marqués du portefeuille, classés du plus sévère au moins sévère.</p>
      </div>
      {_html_table(drawdown_table, pct_cols={"drawdown"}, max_rows=15)}
    </section>
  </section>

  <section class="panel">
    <div class="section-head">
      <h2>Performances mensuelles</h2>
      <p>Tableau détaillé des performances mensuelles du portefeuille, du benchmark et de chaque poche.</p>
    </div>
    {_html_table(
        monthly_returns,
        pct_cols={"performance_portefeuille","performance_benchmark","performance_active","performance_core","performance_monetaire","performance_opportuniste"},
        max_rows=36,
    )}
  </section>

  <section class="panel">
    <div class="section-head">
      <h2>Figures et commentaires</h2>
      <p>Chaque figure ci-dessous est accompagnée d’un commentaire d’interprétation pour rendre le rapport directement exploitable.</p>
    </div>
    <div class="plot-grid">
      {''.join(plot_cards)}
    </div>
  </section>

  <section class="panel">
    <div class="section-head">
      <h2>Résumé global en français</h2>
      <p>Conclusion rédigée automatiquement à partir des résultats du backtest.</p>
    </div>
    <div class="summary-box">
      {html.escape(conclusion_text)}
    </div>
  </section>

  <footer>
    Rapport généré automatiquement depuis la chaîne d’exports de backtest.
  </footer>
</main>
</body>
</html>
"""
    return html_out


def build_backtest_text_summary(result) -> str:
    metrics = _get_result_field(result, "metrics") or {}
    equity_curve = _prepare_equity_curve(result)
    lines: list[str] = []
    lines.append("=== BACKTEST SUMMARY ===")
    lines.append("")

    if not equity_curve.empty and "date" in equity_curve.columns:
        lines.append(f"Période                    : {equity_curve['date'].iloc[0].date()} -> {equity_curve['date'].iloc[-1].date()}")
        if "portfolio_value" in equity_curve.columns:
            lines.append(f"Valeur finale portefeuille : {_fmt_num(_safe_last(equity_curve['portfolio_value']))} EUR")
        if "benchmark_value" in equity_curve.columns:
            lines.append(f"Valeur finale benchmark    : {_fmt_num(_safe_last(equity_curve['benchmark_value']))} EUR")
        lines.append("")

    for key, label in [
        ("total_return", "Performance totale"),
        ("cagr", "CAGR"),
        ("annualized_volatility", "Volatilité annualisée"),
        ("sharpe", "Sharpe"),
        ("max_drawdown", "Max drawdown"),
        ("benchmark_total_return", "Performance benchmark"),
        ("benchmark_cagr", "CAGR benchmark"),
        ("active_return", "Performance active"),
        ("tracking_error", "Tracking error"),
        ("information_ratio", "Information ratio"),
    ]:
        if key in metrics:
            value = metrics.get(key)
            if key in {"sharpe", "information_ratio"}:
                lines.append(f"{label:<27}: {_fmt_num(value)}")
            else:
                lines.append(f"{label:<27}: {_fmt_pct(value)}")

    lines.append("")
    for key, label in [
        ("total_costs_paid", "Coûts totaux payés"),
        ("total_turnover", "Turnover total"),
        ("average_rebalance_turnover", "Turnover moyen rebalance"),
        ("num_orders", "Nombre d'ordres"),
        ("num_executed_orders", "Nombre d'ordres exécutés"),
    ]:
        if key in metrics:
            value = metrics.get(key)
            if key == "total_costs_paid":
                lines.append(f"{label:<27}: {_fmt_num(value)} EUR")
            elif "num_" in key:
                lines.append(f"{label:<27}: {_fmt_int(value)}")
            else:
                lines.append(f"{label:<27}: {_fmt_num(value)}")

    lines.append("")
    lines.append("Conclusion :")
    lines.append(build_french_global_conclusion(result))

    return "\n".join(lines)