from __future__ import annotations

import numpy as np
import pandas as pd


TRADING_DAYS_1M = 21
TRADING_DAYS_3M = 63
TRADING_DAYS_6M = 126
TRADING_DAYS_12M = 252


def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """
    Compute RSI on a price series.
    """
    delta = series.diff()

    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    avg_gain = gains.rolling(window=window, min_periods=window).mean()
    avg_loss = losses.rolling(window=window, min_periods=window).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    return rsi.fillna(50.0)


def compute_price_features(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily price-based features from a normalized price DataFrame.

    Required columns:
    - date
    - ticker
    - adjusted_close
    - volume
    """
    required_cols = {"date", "ticker", "adjusted_close", "volume"}
    missing = required_cols - set(price_df.columns)
    if missing:
        raise ValueError(f"Missing required columns in price_df: {sorted(missing)}")

    df = price_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    numeric_cols = ["open", "high", "low", "close", "adjusted_close", "volume"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["adjusted_close"]).copy()

    grouped = df.groupby("ticker", group_keys=False)

    # Returns
    df["ret_1d"] = grouped["adjusted_close"].pct_change(1)
    df["ret_5d"] = grouped["adjusted_close"].pct_change(5)
    df["ret_10d"] = grouped["adjusted_close"].pct_change(10)
    df["ret_20d"] = grouped["adjusted_close"].pct_change(20)
    df["ret_3m"] = grouped["adjusted_close"].pct_change(TRADING_DAYS_3M)
    df["ret_6m"] = grouped["adjusted_close"].pct_change(TRADING_DAYS_6M)
    df["ret_12m"] = grouped["adjusted_close"].pct_change(TRADING_DAYS_12M)

    # 12m excluding the most recent month: P(t-21) / P(t-252) - 1
    df["price_lag_21"] = grouped["adjusted_close"].shift(TRADING_DAYS_1M)
    df["price_lag_252"] = grouped["adjusted_close"].shift(TRADING_DAYS_12M)
    df["mom12_ex1"] = (df["price_lag_21"] / df["price_lag_252"]) - 1.0

    # Moving averages
    df["ma20"] = grouped["adjusted_close"].transform(
        lambda s: s.rolling(window=20, min_periods=20).mean()
    )
    df["ma200"] = grouped["adjusted_close"].transform(
        lambda s: s.rolling(window=200, min_periods=200).mean()
    )

    # MA200 slope proxy
    df["ma200_lag20"] = grouped["ma200"].shift(20)
    df["ma200_slope"] = df["ma200"] - df["ma200_lag20"]

    # Daily returns and realized volatility
    df["daily_return"] = grouped["adjusted_close"].pct_change()

    df["vol20"] = grouped["daily_return"].transform(
        lambda s: s.rolling(window=20, min_periods=20).std()
    ) * np.sqrt(252)

    df["vol60"] = grouped["daily_return"].transform(
        lambda s: s.rolling(window=60, min_periods=60).std()
    ) * np.sqrt(252)

    df["vol_ratio_20_60"] = df["vol20"] / df["vol60"].replace(0, np.nan)

    # Drawdowns
    df["rolling_max_6m"] = grouped["adjusted_close"].transform(
        lambda s: s.rolling(window=TRADING_DAYS_6M, min_periods=20).max()
    )
    df["dd_6m"] = (df["adjusted_close"] / df["rolling_max_6m"]) - 1.0

    df["rolling_max_12m"] = grouped["adjusted_close"].transform(
        lambda s: s.rolling(window=TRADING_DAYS_12M, min_periods=60).max()
    )
    df["dd_12m"] = (df["adjusted_close"] / df["rolling_max_12m"]) - 1.0

    # Distance to moving averages
    df["dist_ma20"] = (df["adjusted_close"] / df["ma20"]) - 1.0
    df["dist_ma200"] = (df["adjusted_close"] / df["ma200"]) - 1.0

    # Flags
    df["ma200_flag"] = df["adjusted_close"] > df["ma200"]
    df["ma200_slope_flag"] = df["ma200_slope"] > 0

    # RSI
    df["rsi14"] = grouped["adjusted_close"].transform(lambda s: compute_rsi(s, window=14))

    # Volume spike
    df["avg_volume_20"] = grouped["volume"].transform(
        lambda s: s.rolling(window=20, min_periods=20).mean()
    )
    df["volume_spike"] = df["volume"] / df["avg_volume_20"]

    return df


if __name__ == "__main__":
    dates = pd.date_range("2025-01-01", periods=300, freq="B")

    sample = pd.DataFrame(
        {
            "date": list(dates) * 2,
            "ticker": ["AIR"] * len(dates) + ["MC"] * len(dates),
            "adjusted_close": np.concatenate(
                [
                    np.linspace(100, 170, len(dates)),
                    np.linspace(200, 260, len(dates)),
                ]
            ),
            "volume": np.concatenate(
                [
                    np.full(len(dates), 1_000_000),
                    np.full(len(dates), 800_000),
                ]
            ),
        }
    )

    features = compute_price_features(sample)
    print(features.tail())