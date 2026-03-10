from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.settings import DATA_DIR


@dataclass(frozen=True)
class MLConfig:
    monthly_only: bool = True
    min_train_months: int = 36
    validation_block_months: int = 6

    trade_horizon_core_days: int = 63
    trade_horizon_opp_days: int = 21
    deployment_horizon_days: int = 21
    rebalance_horizon_days: int = 21

    trade_auc_floor: float = 0.52
    deployment_auc_floor: float = 0.51
    rebalance_auc_floor: float = 0.52

    # Stronger than before, but still bounded inside overlays.
    trade_strength: float = 0.26

    # Deployment ML now matters a bit more, but still cannot dominate.
    core_deployment_floor: float = 0.93
    core_deployment_cap: float = 1.10
    opp_deployment_floor: float = 0.88
    opp_deployment_cap: float = 1.18

    # Rebalance ML remains conservative: only reject weak marginal trades.
    rebalance_probability_floor: float = 0.47

    artifacts_dir: Path = DATA_DIR / "artifacts" / "ml"


_CFG = MLConfig()
_ML_ENABLED = True


def get_ml_config() -> MLConfig:
    _CFG.artifacts_dir.mkdir(parents=True, exist_ok=True)
    return _CFG


def is_ml_enabled() -> bool:
    return _ML_ENABLED


def set_ml_enabled(flag: bool) -> None:
    global _ML_ENABLED
    _ML_ENABLED = bool(flag)


def model_path(name: str) -> Path:
    cfg = get_ml_config()
    return cfg.artifacts_dir / f"{name}.joblib"


def metrics_path(name: str) -> Path:
    cfg = get_ml_config()
    return cfg.artifacts_dir / f"{name}_metrics.csv"