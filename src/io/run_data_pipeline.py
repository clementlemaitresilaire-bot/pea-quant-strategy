from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.features.feature_pipeline import (
    build_full_feature_panel,
    build_latest_feature_snapshot,
    get_required_feature_tickers,
)
from src.io.data_quality import export_price_cache_diagnostics
from src.io.market_data import load_all_price_data
from src.io.update_prices import update_prices
from src.settings import EXPORT_DATA_DIR, load_config


def _normalize_ticker_list(tickers: list[str] | None) -> list[str]:
    if tickers is None:
        return get_required_feature_tickers()

    normalized = sorted({str(t).strip() for t in tickers if str(t).strip()})
    return normalized


def _export_dataframe(df: pd.DataFrame, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return str(path)


def run_data_pipeline(
    tickers: list[str] | None = None,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    run_update: bool = True,
    run_diagnostics: bool = True,
    build_features: bool = True,
    export_outputs: bool = True,
) -> dict[str, object]:
    """
    Unified entry point for the project data pipeline.

    Pipeline:
    1. optionally refresh prices from Yahoo into local CSV cache
    2. optionally run cache diagnostics
    3. load canonical local price data
    4. optionally compute full feature panel + latest feature snapshot
    5. optionally export outputs

    Important:
    - If run_update=True, later loading is forced with auto_refresh=False to avoid
      an immediate second refresh.
    """
    cfg = load_config()
    pipeline_tickers = _normalize_ticker_list(tickers)

    if not pipeline_tickers:
        raise ValueError("No tickers resolved for the data pipeline.")

    exports_dir = EXPORT_DATA_DIR / "data_pipeline"
    exports_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, object] = {
        "tickers_requested": len(pipeline_tickers),
        "tickers": pipeline_tickers,
        "update": {},
        "diagnostics": {},
        "prices": {},
        "features": {},
        "exports": {},
    }

    if run_update:
        update_result = update_prices(
            tickers=pipeline_tickers,
            start_date=start_date or cfg.data_provider.default_download_start_date,
            end_date=end_date,
            provider_name=cfg.data_provider.provider,
            incremental=True,
            overlap_days=cfg.data_provider.overlap_days,
            run_diagnostics_after_update=run_diagnostics,
        )
        summary["update"] = update_result

    if run_diagnostics and not run_update:
        diagnostics_result = export_price_cache_diagnostics()
        summary["diagnostics"] = diagnostics_result
    elif run_update:
        summary["diagnostics"] = summary["update"].get("diagnostics", {})

    price_df = load_all_price_data(
        tickers=pipeline_tickers,
        start_date=start_date,
        end_date=end_date,
        auto_refresh=False if run_update else True,
    )

    summary["prices"] = {
        "rows": int(len(price_df)),
        "tickers_loaded": int(price_df["ticker"].nunique()) if not price_df.empty else 0,
        "min_date": (
            pd.Timestamp(price_df["date"].min()).strftime("%Y-%m-%d")
            if not price_df.empty
            else ""
        ),
        "max_date": (
            pd.Timestamp(price_df["date"].max()).strftime("%Y-%m-%d")
            if not price_df.empty
            else ""
        ),
    }

    if export_outputs:
        prices_path = exports_dir / "loaded_prices_snapshot.csv"
        summary["exports"]["prices_path"] = _export_dataframe(price_df, prices_path)

    if build_features:
        full_features_df = build_full_feature_panel(
            price_df=price_df,
            tickers=pipeline_tickers,
            start_date=start_date,
            end_date=end_date,
            auto_refresh=False,
            restrict_to_required_tickers=False,
        )

        latest_features_df = (
            build_latest_feature_snapshot(
                price_df=price_df,
                tickers=pipeline_tickers,
                start_date=start_date,
                end_date=end_date,
                auto_refresh=False,
                restrict_to_required_tickers=False,
            )
            if not full_features_df.empty
            else pd.DataFrame()
        )

        summary["features"] = {
            "full_rows": int(len(full_features_df)),
            "latest_rows": int(len(latest_features_df)),
            "feature_tickers": int(latest_features_df["ticker"].nunique())
            if not latest_features_df.empty
            else 0,
            "latest_max_date": (
                pd.Timestamp(latest_features_df["date"].max()).strftime("%Y-%m-%d")
                if not latest_features_df.empty
                else ""
            ),
        }

        if export_outputs:
            full_features_path = exports_dir / "full_feature_panel.csv"
            latest_features_path = exports_dir / "latest_feature_snapshot.csv"

            summary["exports"]["full_features_path"] = _export_dataframe(
                full_features_df, full_features_path
            )
            summary["exports"]["latest_features_path"] = _export_dataframe(
                latest_features_df, latest_features_path
            )

    run_summary_path = exports_dir / "data_pipeline_run_summary.csv"
    summary_rows = []

    for section_name in ["update", "diagnostics", "prices", "features"]:
        section = summary.get(section_name, {})
        if isinstance(section, dict):
            for key, value in section.items():
                if isinstance(value, (list, dict)):
                    continue
                summary_rows.append(
                    {
                        "section": section_name,
                        "metric": key,
                        "value": value,
                    }
                )

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary["exports"]["run_summary_path"] = _export_dataframe(
            summary_df, run_summary_path
        )

    print(f"Tickers requested    : {summary['tickers_requested']}")
    print(f"Price rows loaded    : {summary['prices'].get('rows', 0)}")
    print(f"Price tickers loaded : {summary['prices'].get('tickers_loaded', 0)}")
    print(f"Price max date       : {summary['prices'].get('max_date', '')}")

    if summary["features"]:
        print(f"Feature full rows    : {summary['features'].get('full_rows', 0)}")
        print(f"Feature latest rows  : {summary['features'].get('latest_rows', 0)}")
        print(f"Feature max date     : {summary['features'].get('latest_max_date', '')}")

    if summary["exports"]:
        print(f"Exports dir          : {exports_dir}")

    return summary


if __name__ == "__main__":
    run_data_pipeline()