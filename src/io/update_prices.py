from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data_providers.factory import (
    get_market_data_provider,
    resolve_market_data_provider_name,
)
from src.io.data_quality import export_price_cache_diagnostics
from src.io.loaders import load_universe
from src.io.market_data import get_latest_cached_date
from src.settings import (
    EXPORT_DATA_DIR,
    RAW_DATA_DIR,
    ensure_project_directories,
    load_config,
)


DEFAULT_HISTORY_START_DATE = "2018-01-01"
DEFAULT_OVERLAP_DAYS = 7


def _get_required_tickers() -> list[str]:
    config = load_config()
    universe_df = load_universe()

    tickers = set(
        universe_df.loc[universe_df["active"] == True, "ticker"]
        .dropna()
        .astype(str)
        .str.strip()
        .tolist()
    )

    for etf in config.etf_sleeve.etfs:
        tickers.add(etf.ticker)

    if config.benchmark.ticker:
        tickers.add(config.benchmark.ticker)

    return sorted(t for t in tickers if t)


def _normalize_ticker_list(tickers: list[str] | None) -> list[str]:
    if tickers is None:
        return _get_required_tickers()

    normalized = sorted({str(t).strip() for t in tickers if str(t).strip()})
    return normalized


def _to_timestamp(value: str | None) -> pd.Timestamp | None:
    if value is None:
        return None
    return pd.Timestamp(value).normalize()


def _to_datestr(value: pd.Timestamp | None) -> str | None:
    if value is None:
        return None
    return value.strftime("%Y-%m-%d")


def _resolve_download_window(
    ticker: str,
    prices_dir: Path,
    start_date: str | None,
    end_date: str | None,
    incremental: bool,
    overlap_days: int,
) -> dict[str, pd.Timestamp | None]:
    requested_start_ts = _to_timestamp(start_date) or pd.Timestamp(DEFAULT_HISTORY_START_DATE)
    requested_end_ts = _to_timestamp(end_date)

    if requested_end_ts is not None and requested_start_ts > requested_end_ts:
        raise ValueError(
            f"Invalid date window for {ticker}: start_date={start_date} > end_date={end_date}"
        )

    latest_cached_ts = get_latest_cached_date(ticker=ticker, prices_dir=prices_dir)
    latest_cached_ts = latest_cached_ts.normalize() if latest_cached_ts is not None else None

    if incremental and latest_cached_ts is not None:
        overlap_start_ts = latest_cached_ts - pd.Timedelta(days=overlap_days)
        download_start_ts = max(requested_start_ts, overlap_start_ts)
    else:
        download_start_ts = requested_start_ts

    yahoo_end_exclusive_ts = (
        requested_end_ts + pd.Timedelta(days=1)
        if requested_end_ts is not None
        else None
    )

    return {
        "requested_start_ts": requested_start_ts,
        "requested_end_ts": requested_end_ts,
        "cache_latest_before_ts": latest_cached_ts,
        "download_start_ts": download_start_ts,
        "yahoo_end_exclusive_ts": yahoo_end_exclusive_ts,
    }


def update_prices(
    tickers: list[str] | None = None,
    start_date: str | None = DEFAULT_HISTORY_START_DATE,
    end_date: str | None = None,
    *,
    provider_name: str | None = None,
    incremental: bool = True,
    overlap_days: int = DEFAULT_OVERLAP_DAYS,
    run_diagnostics_after_update: bool = True,
) -> dict[str, object]:
    """
    Download/update market data from the configured remote provider (Yahoo by default)
    and persist it into the local CSV cache.

    Recommended daily usage:
        update_prices()

    Recommended targeted test:
        update_prices(tickers=["MC.PA"], start_date="2025-01-01")
    """
    ensure_project_directories()

    resolved_provider = resolve_market_data_provider_name(provider_name)
    if resolved_provider == "csv":
        raise ValueError(
            "update_prices requires a remote provider. "
            "provider='csv' can read local cache but cannot download fresh data."
        )

    if overlap_days < 0:
        raise ValueError(f"overlap_days must be >= 0, got {overlap_days}")

    prices_dir = RAW_DATA_DIR / "prices"
    tickers_to_update = _normalize_ticker_list(tickers)

    if not tickers_to_update:
        raise ValueError("No tickers provided or resolved from the active universe.")

    provider = get_market_data_provider(resolved_provider)

    report_rows: list[dict[str, object]] = []

    for ticker in tickers_to_update:
        window = _resolve_download_window(
            ticker=ticker,
            prices_dir=prices_dir,
            start_date=start_date,
            end_date=end_date,
            incremental=incremental,
            overlap_days=overlap_days,
        )

        requested_start_ts = window["requested_start_ts"]
        requested_end_ts = window["requested_end_ts"]
        cache_latest_before_ts = window["cache_latest_before_ts"]
        download_start_ts = window["download_start_ts"]
        yahoo_end_exclusive_ts = window["yahoo_end_exclusive_ts"]

        status = "failed"
        error_message = ""
        rows_downloaded = 0
        downloaded_min_date = None
        downloaded_max_date = None
        cache_latest_after_ts = cache_latest_before_ts

        try:
            df = provider.fetch_price_history(
                ticker=ticker,
                start_date=_to_datestr(download_start_ts),
                end_date=_to_datestr(yahoo_end_exclusive_ts),
            )

            if df.empty:
                status = "empty"
            else:
                provider.save_price_history(df, prices_dir)
                cache_latest_after_ts = get_latest_cached_date(
                    ticker=ticker,
                    prices_dir=prices_dir,
                )
                cache_latest_after_ts = (
                    cache_latest_after_ts.normalize()
                    if cache_latest_after_ts is not None
                    else None
                )

                status = "success"
                rows_downloaded = int(len(df))
                downloaded_min_date = pd.Timestamp(df["date"].min()).normalize()
                downloaded_max_date = pd.Timestamp(df["date"].max()).normalize()

        except Exception as exc:
            error_message = f"{type(exc).__name__}: {exc}"

        report_rows.append(
            {
                "ticker": ticker,
                "status": status,
                "error_message": error_message,
                "rows_downloaded": rows_downloaded,
                "requested_start_date": _to_datestr(requested_start_ts),
                "requested_end_date": _to_datestr(requested_end_ts),
                "download_start_date": _to_datestr(download_start_ts),
                "yahoo_end_date_exclusive": _to_datestr(yahoo_end_exclusive_ts),
                "cache_latest_before": _to_datestr(cache_latest_before_ts),
                "downloaded_min_date": _to_datestr(downloaded_min_date),
                "downloaded_max_date": _to_datestr(downloaded_max_date),
                "cache_latest_after": _to_datestr(cache_latest_after_ts),
                "downloaded_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            }
        )

    report_df = pd.DataFrame(report_rows).sort_values(["status", "ticker"]).reset_index(drop=True)

    export_dir = EXPORT_DATA_DIR / "data_quality"
    export_dir.mkdir(parents=True, exist_ok=True)

    report_path = export_dir / "price_download_report.csv"
    report_df.to_csv(report_path, index=False)

    success_count = int((report_df["status"] == "success").sum())
    empty_count = int((report_df["status"] == "empty").sum())
    failed_count = int((report_df["status"] == "failed").sum())

    diagnostics_summary: dict[str, object] = {}
    if run_diagnostics_after_update:
        diagnostics_summary = export_price_cache_diagnostics(prices_dir=prices_dir)

    print(f"Provider used      : {resolved_provider}")
    print(f"Tickers requested  : {len(tickers_to_update)}")
    print(f"Tickers success    : {success_count}")
    print(f"Tickers empty      : {empty_count}")
    print(f"Tickers failed     : {failed_count}")
    print(f"Report saved to    : {report_path}")

    if diagnostics_summary:
        print(f"Cache tickers      : {diagnostics_summary['tickers_in_cache']}")
        print(f"Missing required   : {diagnostics_summary['required_tickers_missing']}")
        print(f"Stale tickers      : {diagnostics_summary['stale_count']}")
        print(f"Diag saved to      : {diagnostics_summary['diagnostics_path']}")

    return {
        "provider": resolved_provider,
        "requested_count": len(tickers_to_update),
        "success_count": success_count,
        "empty_count": empty_count,
        "failed_count": failed_count,
        "success_tickers": report_df.loc[report_df["status"] == "success", "ticker"].tolist(),
        "empty_tickers": report_df.loc[report_df["status"] == "empty", "ticker"].tolist(),
        "failed_tickers": report_df.loc[report_df["status"] == "failed", "ticker"].tolist(),
        "report_path": str(report_path),
        "diagnostics": diagnostics_summary,
    }


if __name__ == "__main__":
    update_prices()