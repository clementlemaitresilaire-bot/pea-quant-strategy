from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.settings import DATA_DIR


@dataclass(frozen=True)
class MLConfig:
    monthly_only: bool = True
    min_train_months: int = 48
    validation_block_months: int = 6

    trade_horizon_core_days: int = 63
    trade_horizon_opp_days: int = 21
    deployment_horizon_days: int = 21
    rebalance_horizon_days: int = 21

    trade_information_coeff_floor: float = 0.01
    deployment_information_coeff_floor: float = 0.005
    rebalance_information_coeff_floor: float = 0.005

    trade_strength: float = 0.12

    core_deployment_floor: float = 0.97
    core_deployment_cap: float = 1.05
    opp_deployment_floor: float = 0.90
    opp_deployment_cap: float = 1.10

    rebalance_edge_floor: float = 0.0025
    rebalance_alpha_surplus_floor: float = 0.0030

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