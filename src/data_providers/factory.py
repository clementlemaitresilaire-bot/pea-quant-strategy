from __future__ import annotations

from typing import Literal

from src.data_providers.base import MarketDataProvider
from src.settings import load_config


ProviderName = Literal["csv", "yahoo", "eodhd", "euronext", "alphavantage"]

SUPPORTED_PROVIDERS: tuple[str, ...] = (
    "csv",
    "yahoo",
    "eodhd",
    "euronext",
    "alphavantage",
)


def resolve_market_data_provider_name(provider_name: str | None = None) -> str:
    """
    Resolve provider name from explicit argument or project config.
    """
    if provider_name is None:
        provider_name = load_config().data_provider.provider

    normalized = str(provider_name).strip().lower()
    if normalized not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"Unknown provider: {provider_name!r}. "
            f"Supported providers: {list(SUPPORTED_PROVIDERS)}"
        )

    return normalized


def get_market_data_provider(
    provider_name: str | None = None,
) -> MarketDataProvider:
    """
    Return a market data provider instance.

    Important:
    - Yahoo provider is imported lazily, so the project can still run in CSV mode
      even if yfinance is not installed.
    """
    resolved = resolve_market_data_provider_name(provider_name)

    if resolved == "csv":
        from src.data_providers.csv_provider import CsvMarketDataProvider

        return CsvMarketDataProvider()

    if resolved == "yahoo":
        try:
            from src.data_providers.yahoo_provider import YahooMarketDataProvider
        except ModuleNotFoundError as exc:
            if exc.name == "yfinance":
                raise ModuleNotFoundError(
                    "The Yahoo provider requires the 'yfinance' package. "
                    "Install it to download/update prices, or use provider='csv' "
                    "to work only from the local cache."
                ) from exc
            raise

        return YahooMarketDataProvider()

    if resolved == "eodhd":
        raise NotImplementedError("EODHD provider not implemented in this branch yet.")

    if resolved == "euronext":
        raise NotImplementedError("Euronext provider not implemented in this branch yet.")

    if resolved == "alphavantage":
        raise NotImplementedError("Alpha Vantage provider not implemented yet.")

    raise RuntimeError(f"Unhandled provider branch: {resolved}")


if __name__ == "__main__":
    provider = get_market_data_provider("csv")
    print(type(provider).__name__)