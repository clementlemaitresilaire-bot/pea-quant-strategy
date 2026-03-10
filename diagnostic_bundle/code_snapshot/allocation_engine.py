from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.engines.core_engine import build_core_signal_table
from src.engines.monetary_engine import build_monetary_signal_table
from src.engines.opp_engine import build_opp_signal_table
from src.engines.regime_engine import determine_market_regime
from src.ml.overlays import apply_trade_quality_overlay, get_live_deployment_adjustment


@dataclass
class AllocationResult:
    core_target_weights_table: pd.DataFrame
    monetary_target_weights_table: pd.DataFrame
    opp_target_weights_table: pd.DataFrame
    combined_target_weights_table: pd.DataFrame
    regime: str
    target_core_deployment: float
    target_core_weight_total: float
    target_monetary_weight_total: float
    target_opp_weight_total: float
    target_total_invested_weight: float
    implied_cash_weight: float


def _softmax_weights(scores: pd.Series, temperature: float = 1.0) -> pd.Series:
    x = pd.to_numeric(scores, errors="coerce").fillna(0.0).astype(float)
    x = x - x.max()
    exp_x = np.exp(x / max(float(temperature), 1e-6))
    total = float(exp_x.sum())
    if total <= 0:
        return pd.Series(np.ones(len(x)) / len(x), index=x.index)
    return exp_x / total


def _iterative_cap_and_renorm(
    weights: pd.Series,
    cap: float,
    target_total: float,
    max_iter: int = 20,
) -> pd.Series:
    w = weights.copy().astype(float)
    if w.sum() <= 0:
        return w

    w = w / w.sum() * float(target_total)

    for _ in range(max_iter):
        over = w > float(cap)
        if not over.any():
            break

        w.loc[over] = float(cap)
        remaining_target = float(target_total) - float(w.loc[over].sum())
        under = ~over

        if under.any() and remaining_target > 0:
            base = w.loc[under]
            if base.sum() > 0:
                w.loc[under] = base / base.sum() * remaining_target

    return w


def _build_core_target_weights(
    core_signal_df: pd.DataFrame,
    regime_state,
    deployment_multiplier: float = 1.0,
) -> tuple[pd.DataFrame, float]:
    from src.settings import load_config

    config = load_config()
    alloc_cfg = config.allocation
    sleeve_target = config.sleeves["core"].target_weight

    selected = core_signal_df.loc[core_signal_df["selected_core_final"] == True].copy()
    if selected.empty:
        raise ValueError("No selected core names available for allocation.")

    selected = selected.sort_values("rank_core").reset_index(drop=True)
    selected["rank_floor"] = selected["rank_core"].map(
        lambda r: 1.0 if r <= 2 else (0.8 if r <= 4 else (0.6 if r <= 6 else 0.45))
    )

    score_weights = _softmax_weights(selected["core_score"], temperature=0.70)
    rank_weights = selected["rank_floor"] / selected["rank_floor"].sum()
    selected["target_weight_raw"] = 0.65 * score_weights + 0.35 * rank_weights

    target_core_weight_total = sleeve_target * float(regime_state.target_core_deployment) * float(deployment_multiplier)
    target_core_weight_total = min(max(target_core_weight_total, 0.0), sleeve_target)

    selected["target_weight_final"] = _iterative_cap_and_renorm(
        selected["target_weight_raw"],
        cap=alloc_cfg.max_core_position_weight_total,
        target_total=target_core_weight_total,
    )

    selected["bucket"] = "core"
    selected["reason"] = selected["selection_reason"]

    out = selected[
        [
            "ticker",
            "bucket",
            "rank_core",
            "core_score",
            "target_weight_raw",
            "target_weight_final",
            "reason",
        ]
    ].rename(columns={"rank_core": "rank", "core_score": "score"}).sort_values("rank").reset_index(drop=True)

    if "ml_trade_quality_proba" in selected.columns:
        out = out.merge(
            selected[["ticker", "ml_trade_quality_proba"]],
            on="ticker",
            how="left",
        )

    return out, float(out["target_weight_final"].sum())


def _build_monetary_target_weights(monetary_signal_df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    from src.settings import load_config

    config = load_config()
    sleeve_target = config.sleeves["monetary"].target_weight

    df = monetary_signal_df.copy().sort_values("rank_monetary").reset_index(drop=True)
    if df.empty:
        raise ValueError("Monetary sleeve signal table is empty.")

    df["target_weight_raw"] = df["target_internal_weight_raw"] * sleeve_target
    df["target_weight_final"] = df["target_internal_weight_final"] * sleeve_target

    df["bucket"] = "monetary"
    df["reason"] = df["selection_reason"]
    df["score"] = df["etf_score"]
    df["rank"] = df["rank_monetary"]

    out = df[
        [
            "ticker",
            "bucket",
            "rank",
            "score",
            "annual_fee_rate",
            "target_weight_raw",
            "target_weight_final",
            "reason",
        ]
    ].copy()

    return out, float(out["target_weight_final"].sum())


def _build_opp_target_weights(
    opp_signal_df: pd.DataFrame,
    regime_state,
    deployment_multiplier: float = 1.0,
) -> tuple[pd.DataFrame, float]:
    from src.settings import load_config

    config = load_config()
    alloc_cfg = config.allocation
    sleeve_target = config.sleeves["opportunistic"].target_weight

    selected = opp_signal_df.loc[opp_signal_df["selected_opp_final"] == True].copy()
    if selected.empty:
        empty = pd.DataFrame(
            columns=[
                "ticker",
                "bucket",
                "rank",
                "score",
                "annual_fee_rate",
                "target_weight_raw",
                "target_weight_final",
                "reason",
            ]
        )
        return empty, 0.0

    selected = selected.sort_values("rank_opp").reset_index(drop=True)

    inv_vol = 1.0 / selected["vol20"].replace(0.0, np.nan)
    inv_vol = inv_vol.replace([np.inf, -np.inf], np.nan).fillna(inv_vol.median())
    if inv_vol.isna().all():
        inv_vol = pd.Series(np.ones(len(selected)), index=selected.index)

    score_weights = _softmax_weights(selected["opp_score_net"], temperature=0.65)
    inv_vol_weights = inv_vol / inv_vol.sum()
    equal_weights = pd.Series(np.ones(len(selected)) / len(selected), index=selected.index)

    selected["target_weight_raw"] = (
        0.50 * score_weights
        + 0.30 * inv_vol_weights
        + 0.20 * equal_weights
    )

    # Règle demandée :
    # - sleeve opp déployée à 100% en green/orange
    # - réduite seulement en red
    if regime_state.regime == "red":
        opp_deployment = 0.35
        hard_cap = min(float(alloc_cfg.max_opp_position_weight_total), 0.08)
    else:
        opp_deployment = 1.00
        hard_cap = min(float(alloc_cfg.max_opp_position_weight_total), 0.10)

    target_opp_weight_total = sleeve_target * opp_deployment * float(deployment_multiplier)
    target_opp_weight_total = min(max(target_opp_weight_total, 0.0), sleeve_target)

    if target_opp_weight_total <= 0:
        empty = pd.DataFrame(
            columns=[
                "ticker",
                "bucket",
                "rank",
                "score",
                "annual_fee_rate",
                "target_weight_raw",
                "target_weight_final",
                "reason",
            ]
        )
        return empty, 0.0

    selected["target_weight_final"] = _iterative_cap_and_renorm(
        selected["target_weight_raw"],
        cap=hard_cap,
        target_total=target_opp_weight_total,
    )

    if "annual_fee_rate" not in selected.columns:
        selected["annual_fee_rate"] = 0.0030

    selected["bucket"] = "opportunistic"
    selected["reason"] = selected["selection_reason"]

    out = selected[
        [
            "ticker",
            "bucket",
            "rank_opp",
            "opp_score_net",
            "annual_fee_rate",
            "target_weight_raw",
            "target_weight_final",
            "reason",
        ]
    ].rename(columns={"rank_opp": "rank", "opp_score_net": "score"}).sort_values("rank").reset_index(drop=True)

    if "ml_trade_quality_proba" in selected.columns:
        out = out.merge(
            selected[["ticker", "ml_trade_quality_proba"]],
            on="ticker",
            how="left",
        )

    return out, float(out["target_weight_final"].sum())


def build_full_target_allocation(
    core_signal_df: pd.DataFrame | None = None,
    monetary_signal_df: pd.DataFrame | None = None,
    opp_signal_df: pd.DataFrame | None = None,
    latest_features_df: pd.DataFrame | None = None,
    universe_df: pd.DataFrame | None = None,
) -> AllocationResult:
    if core_signal_df is None:
        core_signal_df = build_core_signal_table(
            latest_features_df=latest_features_df,
            universe_df=universe_df,
        )

    if monetary_signal_df is None:
        monetary_signal_df = build_monetary_signal_table(
            latest_features_df=latest_features_df,
        )

    if opp_signal_df is None:
        opp_signal_df = build_opp_signal_table(
            latest_features_df=latest_features_df,
            universe_df=universe_df,
        )

    core_signal_df = apply_trade_quality_overlay(core_signal_df, bucket="core")
    opp_signal_df = apply_trade_quality_overlay(opp_signal_df, bucket="opportunistic")

    regime_state = determine_market_regime(
        features_df=latest_features_df,
        universe_df=universe_df,
    )

    try:
        deploy_adj = get_live_deployment_adjustment(
            regime_state=regime_state,
            latest_features_df=latest_features_df if latest_features_df is not None else pd.DataFrame(),
            universe_df=universe_df if universe_df is not None else pd.DataFrame(),
        )
    except Exception:
        deploy_adj = {
            "deployment_ml_proba": 0.50,
            "core_multiplier": 1.00,
            "opp_multiplier": 1.00,
        }

    core_table, target_core_weight_total = _build_core_target_weights(
        core_signal_df,
        regime_state,
        deployment_multiplier=float(deploy_adj["core_multiplier"]),
    )
    monetary_table, target_monetary_weight_total = _build_monetary_target_weights(monetary_signal_df)
    opp_table, target_opp_weight_total = _build_opp_target_weights(
        opp_signal_df,
        regime_state,
        deployment_multiplier=float(deploy_adj["opp_multiplier"]),
    )

    frames = [core_table, monetary_table]
    if not opp_table.empty:
        frames.append(opp_table)

    combined = pd.concat(frames, ignore_index=True).copy()
    combined = combined.sort_values(["bucket", "rank", "ticker"]).reset_index(drop=True)

    combined["deployment_ml_proba"] = float(deploy_adj.get("deployment_ml_proba", 0.50))
    combined["core_deployment_multiplier_ml"] = float(deploy_adj.get("core_multiplier", 1.00))
    combined["opp_deployment_multiplier_ml"] = float(deploy_adj.get("opp_multiplier", 1.00))

    target_total_invested_weight = (
        target_core_weight_total
        + target_monetary_weight_total
        + target_opp_weight_total
    )
    implied_cash_weight = 1.0 - target_total_invested_weight

    return AllocationResult(
        core_target_weights_table=core_table,
        monetary_target_weights_table=monetary_table,
        opp_target_weights_table=opp_table,
        combined_target_weights_table=combined,
        regime=regime_state.regime,
        target_core_deployment=regime_state.target_core_deployment,
        target_core_weight_total=target_core_weight_total,
        target_monetary_weight_total=target_monetary_weight_total,
        target_opp_weight_total=target_opp_weight_total,
        target_total_invested_weight=target_total_invested_weight,
        implied_cash_weight=implied_cash_weight,
    )


if __name__ == "__main__":
    result = build_full_target_allocation()

    print("\n=== Allocation Regime ===")
    print(f"Regime                     : {result.regime}")
    print(f"Target core deployment     : {result.target_core_deployment:.2%}")
    print(f"Target core weight total   : {result.target_core_weight_total:.2%}")
    print(f"Target monetary weight     : {result.target_monetary_weight_total:.2%}")
    print(f"Target opp weight total    : {result.target_opp_weight_total:.2%}")
    print(f"Target invested total      : {result.target_total_invested_weight:.2%}")
    print(f"Implied cash weight        : {result.implied_cash_weight:.2%}")

    print("\n=== Core Target Weights ===")
    print(result.core_target_weights_table)

    print("\n=== Monetary Target Weights ===")
    print(result.monetary_target_weights_table)

    print("\n=== Opp Target Weights ===")
    print(result.opp_target_weights_table)