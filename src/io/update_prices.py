from __future__ import annotations

import pandas as pd

from src.data_providers.factory import get_market_data_provider
from src.io.loaders import load_universe
from src.settings import EXPORT_DATA_DIR, RAW_DATA_DIR, ensure_project_directories, load_config


def _get_required_tickers() -> list[str]:
    config = load_config()
    universe_df = load_universe()

    tickers = set(
        universe_df.loc[universe_df["active"] == True, "ticker"]
        .drop_duplicates()
        .tolist()
    )

    for etf in config.etf_sleeve.etfs:
        tickers.add(etf.ticker)

    if config.benchmark.ticker:
        tickers.add(config.benchmark.ticker)

    return sorted(tickers)


def update_prices(
    tickers: list[str] | None = None,
    start_date: str | None = "2018-01-01",
    end_date: str | None = None,
) -> dict[str, list[str] | int]:
    ensure_project_directories()

    provider = get_market_data_provider()

    if tickers is None:
        tickers = _get_required_tickers()

    success_tickers: list[str] = []
    failed_tickers: list[str] = []
    empty_tickers: list[str] = []
    all_frames: list[pd.DataFrame] = []

    for ticker in tickers:
        try:
            df = provider.fetch_price_history(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
            )

            if df.empty:
                empty_tickers.append(ticker)
                continue

            all_frames.append(df)
            success_tickers.append(ticker)

        except Exception:
            failed_tickers.append(ticker)

    if all_frames:
        combined_df = pd.concat(all_frames, ignore_index=True).sort_values(["ticker", "date"]).reset_index(drop=True)
        provider.save_price_history(combined_df, RAW_DATA_DIR / "prices")
    else:
        combined_df = pd.DataFrame()

    report_df = pd.DataFrame(
        {
            "ticker": tickers,
            "status": [
                "success" if t in success_tickers
                else "empty" if t in empty_tickers
                else "failed"
                for t in tickers
            ],
        }
    )

    export_dir = EXPORT_DATA_DIR / "data_quality"
    export_dir.mkdir(parents=True, exist_ok=True)
    report_path = export_dir / "price_download_report.csv"
    report_df.to_csv(report_path, index=False)

    print(f"Tickers requested : {len(tickers)}")
    print(f"Tickers success   : {len(success_tickers)}")
    print(f"Tickers empty     : {len(empty_tickers)}")
    print(f"Tickers failed    : {len(failed_tickers)}")
    print(f"Report saved to   : {report_path}")

    return {
        "requested_count": len(tickers),
        "success_count": len(success_tickers),
        "empty_count": len(empty_tickers),
        "failed_count": len(failed_tickers),
        "success_tickers": success_tickers,
        "empty_tickers": empty_tickers,
        "failed_tickers": failed_tickers,
    }


if __name__ == "__main__":
    update_prices()