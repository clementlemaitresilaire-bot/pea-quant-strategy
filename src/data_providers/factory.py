from __future__ import annotations

from src.data_providers.base import MarketDataProvider
from src.data_providers.csv_provider import CsvMarketDataProvider
from src.data_providers.yahoo_provider import YahooMarketDataProvider
from src.settings import load_config


def get_market_data_provider() -> MarketDataProvider:
    config = load_config()
    provider_name = config.data_provider.provider

    if provider_name == "csv":
        return CsvMarketDataProvider()

    if provider_name == "yahoo":
        return YahooMarketDataProvider()

    if provider_name == "eodhd":
        raise NotImplementedError("EODHD provider not implemented in this branch yet.")

    if provider_name == "euronext":
        raise NotImplementedError("Euronext provider not implemented in this branch yet.")

    if provider_name == "alphavantage":
        raise NotImplementedError("Alpha Vantage provider not implemented yet.")

    raise ValueError(f"Unknown provider: {provider_name}")


if __name__ == "__main__":
    provider = get_market_data_provider()
    print(type(provider).__name__)