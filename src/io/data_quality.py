from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.io.loaders import load_universe
from src.io.market_data import PRICE_COLUMNS, load_price_csv
from src.settings import EXPORT_DATA_DIR, RAW_DATA_DIR, load_config


def _required_pipeline_tickers() -> list[str]:
    cfg = load_config()
    universe_df = load_universe()

    tickers = set(
        universe_df.loc[universe_df["active"] == True, "ticker"]
        .dropna()
        .astype(str)
        .str.strip()
        .tolist()
    )

    for etf in cfg.etf_sleeve.etfs:
        tickers.add(str(etf.ticker).strip())

    if cfg.benchmark.mode == "single_ticker" and cfg.benchmark.ticker:
        tickers.add(str(cfg.benchmark.ticker).strip())

    return sorted(t for t in tickers if t)


def _safe_to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def _analyze_single_price_file(csv_path: Path) -> dict[str, object]:
    ticker = csv_path.stem

    raw_rows = 0
    raw_duplicate_dates = 0
    raw_missing_required_columns = ""
    normalized_rows = 0
    min_date = None
    max_date = None
    max_gap_days = None
    stale_days = None
    status = "ok"
    load_error = ""

    try:
        raw_df = pd.read_csv(csv_path)
        raw_rows = int(len(raw_df))

        missing_cols = [col for col in PRICE_COLUMNS if col not in raw_df.columns]
        raw_missing_required_columns = ",".join(missing_cols)

        if "date" in raw_df.columns:
            raw_dates = _safe_to_datetime(raw_df["date"])
            raw_duplicate_dates = int(
                pd.DataFrame({"date": raw_dates})
                .dropna()
                .duplicated(subset=["date"], keep=False)
                .sum()
            )

        normalized_df = load_price_csv(csv_path)
        normalized_rows = int(len(normalized_df))

        if not normalized_df.empty:
            dates = pd.to_datetime(normalized_df["date"], errors="coerce").dropna().sort_values()
            if not dates.empty:
                min_date = pd.Timestamp(dates.iloc[0]).normalize()
                max_date = pd.Timestamp(dates.iloc[-1]).normalize()

                if len(dates) >= 2:
                    gaps = dates.diff().dropna().dt.days
                    max_gap_days = int(gaps.max()) if not gaps.empty else 0
                else:
                    max_gap_days = 0

                today = pd.Timestamp.today().normalize()
                stale_days = int((today - max_date).days)

        if raw_missing_required_columns:
            status = "schema_issue"
        elif normalized_rows == 0:
            status = "empty_after_normalization"
        elif stale_days is not None and stale_days > load_config().data_provider.max_cache_staleness_days:
            status = "stale"

    except Exception as exc:
        status = "load_failed"
        load_error = f"{type(exc).__name__}: {exc}"

    return {
        "ticker": ticker,
        "status": status,
        "load_error": load_error,
        "raw_rows": raw_rows,
        "normalized_rows": normalized_rows,
        "raw_duplicate_dates": raw_duplicate_dates,
        "raw_missing_required_columns": raw_missing_required_columns,
        "min_date": min_date.strftime("%Y-%m-%d") if min_date is not None else "",
        "max_date": max_date.strftime("%Y-%m-%d") if max_date is not None else "",
        "max_gap_days": max_gap_days if max_gap_days is not None else "",
        "stale_days": stale_days if stale_days is not None else "",
        "file_path": str(csv_path),
    }


def build_price_cache_diagnostics(
    prices_dir: Path | None = None,
) -> pd.DataFrame:
    directory = prices_dir or (RAW_DATA_DIR / "prices")
    directory.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for csv_path in sorted(directory.glob("*.csv")):
        rows.append(_analyze_single_price_file(csv_path))

    diagnostics_df = pd.DataFrame(rows)

    if diagnostics_df.empty:
        return diagnostics_df

    required = set(_required_pipeline_tickers())
    diagnostics_df["required_by_pipeline"] = diagnostics_df["ticker"].isin(required)

    diagnostics_df = diagnostics_df.sort_values(
        ["required_by_pipeline", "status", "ticker"],
        ascending=[False, True, True],
    ).reset_index(drop=True)

    return diagnostics_df


def build_missing_required_tickers_report(
    diagnostics_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if diagnostics_df is None:
        diagnostics_df = build_price_cache_diagnostics()

    required = set(_required_pipeline_tickers())
    present = set(diagnostics_df["ticker"].tolist()) if not diagnostics_df.empty else set()

    missing = sorted(required - present)
    return pd.DataFrame({"ticker": missing})


def export_price_cache_diagnostics(
    prices_dir: Path | None = None,
) -> dict[str, object]:
    diagnostics_df = build_price_cache_diagnostics(prices_dir=prices_dir)
    missing_df = build_missing_required_tickers_report(diagnostics_df)

    export_dir = EXPORT_DATA_DIR / "data_quality"
    export_dir.mkdir(parents=True, exist_ok=True)

    diagnostics_path = export_dir / "price_cache_diagnostics.csv"
    missing_path = export_dir / "price_cache_missing_required.csv"

    diagnostics_df.to_csv(diagnostics_path, index=False)
    missing_df.to_csv(missing_path, index=False)

    summary = {
        "tickers_in_cache": int(len(diagnostics_df)),
        "required_tickers_missing": int(len(missing_df)),
        "load_failed_count": int((diagnostics_df["status"] == "load_failed").sum()) if not diagnostics_df.empty else 0,
        "schema_issue_count": int((diagnostics_df["status"] == "schema_issue").sum()) if not diagnostics_df.empty else 0,
        "stale_count": int((diagnostics_df["status"] == "stale").sum()) if not diagnostics_df.empty else 0,
        "diagnostics_path": str(diagnostics_path),
        "missing_required_path": str(missing_path),
    }

    return summary


if __name__ == "__main__":
    print(export_price_cache_diagnostics())