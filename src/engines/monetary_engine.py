from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from src.features.feature_pipeline import build_latest_feature_snapshot, get_latest_features
from src.features.price_features import compute_price_features
from src.io.market_data import load_all_price_data
from src.settings import load_config


@dataclass
class MonetarySelectionResult:
    selected_tickers: list[str]
    signal_table: pd.DataFrame


def _softmax_weights(values: pd.Series, temperature: float = 1.0) -> pd.Series:
    x = pd.to_numeric(values, errors="coerce").fillna(0.0).astype(float)
    x = x - x.max()
    exp_x = np.exp(x / max(temperature, 1e-6))
    total = exp_x.sum()
    if total <= 0:
        return pd.Series(np.ones(len(x)) / len(x), index=x.index)
    return exp_x / total


def _cap_and_renorm_internal(weights: pd.Series, min_w: float, max_w: float) -> pd.Series:
    w = weights.copy().astype(float)
    w = w.clip(lower=min_w, upper=max_w)

    if w.sum() <= 0:
        return w

    w = w / w.sum()

    for _ in range(10):
        changed = False

        too_high = w > max_w
        if too_high.any():
            excess = float((w.loc[too_high] - max_w).sum())
            w.loc[too_high] = max_w
            under = w < max_w
            if under.any() and excess > 0:
                room = max_w - w.loc[under]
                room_sum = room.sum()
                if room_sum > 0:
                    w.loc[under] += excess * (room / room_sum)
            changed = True

        too_low = w < min_w
        if too_low.any():
            deficit = float((min_w - w.loc[too_low]).sum())
            w.loc[too_low] = min_w
            over = w > min_w
            if over.any() and deficit > 0:
                room = w.loc[over] - min_w
                room_sum = room.sum()
                if room_sum > 0:
                    w.loc[over] -= deficit * (room / room_sum)
            changed = True

        if not changed:
            break

    return w / w.sum()


def _get_full_latest_features() -> pd.DataFrame:
    price_df = load_all_price_data()
    price_df["date"] = pd.to_datetime(price_df["date"])
    price_df = price_df.sort_values(["ticker", "date"]).reset_index(drop=True)
    full_features = compute_price_features(price_df)
    return get_latest_features(full_features)


def _ensure_etf_features(latest_features_df: pd.DataFrame | None) -> pd.DataFrame:
    config = load_config()
    etf_tickers = [etf.ticker for etf in config.etf_sleeve.etfs]

    if latest_features_df is None:
        latest_features_df = _get_full_latest_features()

    latest_features_df = latest_features_df.copy()
    current_tickers = set(latest_features_df["ticker"].astype(str).tolist())

    missing = [ticker for ticker in etf_tickers if ticker not in current_tickers]
    if not missing:
        return latest_features_df

    fallback_latest = _get_full_latest_features()
    supplement = fallback_latest.loc[fallback_latest["ticker"].isin(missing)].copy()

    if supplement.empty:
        return latest_features_df

    combined = pd.concat([latest_features_df, supplement], ignore_index=True)
    combined = combined.drop_duplicates(subset=["ticker"], keep="last").reset_index(drop=True)
    return combined


def build_monetary_signal_table(
    latest_features_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    config = load_config()
    latest_features_df = _ensure_etf_features(latest_features_df)

    required_cols = {
        "ticker",
        "ret_3m",
        "ret_6m",
        "mom12_ex1",
        "ma200_flag",
        "ma200_slope_flag",
    }
    missing = required_cols - set(latest_features_df.columns)
    if missing:
        raise ValueError(f"Missing required columns for monetary engine: {sorted(missing)}")

    etf_meta = pd.DataFrame(
        [
            {
                "ticker": etf.ticker,
                "etf_name": etf.name,
                "annual_fee_rate": etf.annual_fee_rate,
                "base_internal_weight": etf.base_weight,
            }
            for etf in config.etf_sleeve.etfs
        ]
    )

    df = latest_features_df.merge(etf_meta, on="ticker", how="inner").copy()

    expected_tickers = set(etf_meta["ticker"].tolist())
    found_tickers = set(df["ticker"].tolist())
    missing_tickers = sorted(expected_tickers - found_tickers)
    if missing_tickers:
        raise ValueError(
            f"Could not build monetary sleeve: missing ETF feature rows for {missing_tickers}"
        )

    for col in ["ret_3m", "ret_6m", "mom12_ex1"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df["trend_bonus"] = 0.0
    df.loc[df["ma200_flag"] & df["ma200_slope_flag"], "trend_bonus"] = 0.03
    df.loc[df["ma200_flag"] & (~df["ma200_slope_flag"]), "trend_bonus"] = 0.01
    df.loc[(~df["ma200_flag"]) & (~df["ma200_slope_flag"]), "trend_bonus"] = -0.03

    df["etf_score"] = (
        0.45 * df["ret_6m"]
        + 0.35 * df["mom12_ex1"]
        + 0.20 * df["ret_3m"]
        + df["trend_bonus"]
        - df["annual_fee_rate"]
    )

    score_weights = _softmax_weights(df["etf_score"], temperature=0.08)
    base_weights = df["base_internal_weight"] / df["base_internal_weight"].sum()

    df["target_internal_weight_raw"] = 0.80 * base_weights + 0.20 * score_weights
    df["target_internal_weight_final"] = _cap_and_renorm_internal(
        df["target_internal_weight_raw"],
        min_w=config.etf_sleeve.min_internal_weight,
        max_w=config.etf_sleeve.max_internal_weight,
    )

    df = df.sort_values("etf_score", ascending=False).reset_index(drop=True)
    df["rank_monetary"] = np.arange(1, len(df) + 1)
    df["selected_monetary_final"] = True
    df["selection_reason"] = "PERMANENT_ETF"

    return df[
        [
            "ticker",
            "etf_name",
            "annual_fee_rate",
            "base_internal_weight",
            "ret_3m",
            "ret_6m",
            "mom12_ex1",
            "ma200_flag",
            "ma200_slope_flag",
            "trend_bonus",
            "etf_score",
            "rank_monetary",
            "target_internal_weight_raw",
            "target_internal_weight_final",
            "selected_monetary_final",
            "selection_reason",
        ]
    ]


def select_monetary_etfs(
    latest_features_df: pd.DataFrame | None = None,
) -> MonetarySelectionResult:
    signal_table = build_monetary_signal_table(latest_features_df=latest_features_df)
    selected_tickers = signal_table["ticker"].tolist()

    return MonetarySelectionResult(
        selected_tickers=selected_tickers,
        signal_table=signal_table,
    )


if __name__ == "__main__":
    result = select_monetary_etfs()
    print("\n=== Selected Monetary ETFs ===")
    print(result.selected_tickers)
    print("\n=== Monetary Signal Table ===")
    print(result.signal_table)