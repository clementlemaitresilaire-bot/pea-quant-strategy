from __future__ import annotations

from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
STATE_DATA_DIR = DATA_DIR / "state"
EXPORT_DATA_DIR = DATA_DIR / "exports"


class ProjectConfig(BaseModel):
    name: str
    base_currency: str = "EUR"


class SleeveConfig(BaseModel):
    target_weight: float = Field(..., ge=0.0, le=1.0)
    max_positions: int = Field(..., ge=1)
    min_order_eur: float = Field(..., ge=0.0)


class UniverseConfig(BaseModel):
    max_names_core: int = 8
    max_names_opportunistic: int = 2


class BenchmarkConfig(BaseModel):
    mode: Literal["single_ticker", "synthetic_universe"] = "single_ticker"
    ticker: str = "^STOXX"
    synthetic_members_only: bool = False


class ETFItemConfig(BaseModel):
    ticker: str
    name: str
    annual_fee_rate: float = Field(..., ge=0.0)
    base_weight: float = Field(..., ge=0.0, le=1.0)


class ETFSleeveConfig(BaseModel):
    min_internal_weight: float = Field(0.15, ge=0.0, le=1.0)
    max_internal_weight: float = Field(0.50, ge=0.0, le=1.0)
    etfs: list[ETFItemConfig]

    @field_validator("etfs")
    @classmethod
    def validate_etfs(cls, etfs: list[ETFItemConfig]) -> list[ETFItemConfig]:
        if len(etfs) != 3:
            raise ValueError("etf_sleeve must contain exactly 3 ETFs")
        total_base = sum(etf.base_weight for etf in etfs)
        if abs(total_base - 1.0) > 1e-9:
            raise ValueError("ETF base weights must sum to 1.0")
        return etfs


class AllocationConfig(BaseModel):
    max_core_position_weight_total: float = Field(..., ge=0.0, le=1.0)
    max_opp_position_weight_total: float = Field(..., ge=0.0, le=1.0)
    core_rebalance_deadband: float = Field(..., ge=0.0, le=1.0)


class CoreSignalConfig(BaseModel):
    mom12_ex1_weight: float
    mom6_weight: float
    mom3_weight: float
    rel_strength_weight: float
    vol60_weight: float
    dd6m_weight: float
    trend_bonus: float
    entry_rank_cutoff: int
    retention_rank_cutoff: int


class OppSignalConfig(BaseModel):
    ret5d_trigger: float
    ret10d_trigger: float
    ret20d_trigger: float


class OrderRulesConfig(BaseModel):
    core_cost_hurdle_multiple: float = Field(..., ge=0.0)
    opp_cost_hurdle_multiple: float = Field(..., ge=0.0)
    opp_ttf_cost_hurdle_multiple: float = Field(..., ge=0.0)


class AlphaGuardrailsConfig(BaseModel):
    core_expected_holding_months: float = 10.0
    opp_expected_holding_months: float = 2.5

    core_required_net_alpha_annual: float = 0.055
    opp_required_net_alpha_annual: float = 0.09

    core_min_score_delta_for_replacement: float = 0.45
    opp_entry_score_threshold: float = 0.45
    opp_keep_score_threshold: float = 0.08

    core_roundtrip_cost_buffer: float = 1.00
    opp_roundtrip_cost_buffer: float = 1.35

    core_weight_inertia: float = 0.25
    opp_weight_inertia: float = 0.15

    core_min_weight_change: float = 0.020
    opp_min_weight_change: float = 0.015


class CostsConfig(BaseModel):
    brokerage_bps: float = 10.0
    spread_bps_high: float = 5.0
    spread_bps_standard: float = 10.0
    spread_bps_lower: float = 20.0
    ttf_france_rate: float = 0.003


class DataProviderConfig(BaseModel):
    provider: Literal["csv", "yahoo"] = "yahoo"

    # If True, the app may refresh the local cache automatically when loading prices.
    auto_update_on_load: bool = True

    # Number of calendar days after which the cache is considered stale.
    # 0 means "always refresh before use".
    max_cache_staleness_days: int = Field(1, ge=0)

    # Historical bootstrap date for first full download.
    default_download_start_date: str = "2018-01-01"

    # When refreshing incrementally, reload a few extra trailing days to capture:
    # - late Yahoo adjustments
    # - splits/dividends reflected in adjusted prices
    # - missing days / partial corrections
    overlap_days: int = Field(7, ge=0)

    @field_validator("default_download_start_date")
    @classmethod
    def validate_default_download_start_date(cls, value: str) -> str:
        try:
            date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(
                "default_download_start_date must be in YYYY-MM-DD format"
            ) from exc
        return value


class ReportingConfig(BaseModel):
    export_signals: bool = True
    export_orders: bool = True
    export_backtests: bool = True


class AppConfig(BaseModel):
    project: ProjectConfig
    universe: UniverseConfig
    sleeves: dict[str, SleeveConfig]
    benchmark: BenchmarkConfig
    etf_sleeve: ETFSleeveConfig
    allocation: AllocationConfig
    core_signal: CoreSignalConfig
    opp_signal: OppSignalConfig
    order_rules: OrderRulesConfig
    alpha_guardrails: AlphaGuardrailsConfig = AlphaGuardrailsConfig()
    costs: CostsConfig = CostsConfig()
    data_provider: DataProviderConfig = DataProviderConfig()
    reporting: ReportingConfig = ReportingConfig()

    @field_validator("sleeves")
    @classmethod
    def validate_sleeves(cls, sleeves: dict[str, SleeveConfig]) -> dict[str, SleeveConfig]:
        required = {"core", "monetary", "opportunistic"}
        missing = required - set(sleeves)
        if missing:
            raise ValueError(f"Missing sleeve config(s): {sorted(missing)}")

        total_weight = sum(s.target_weight for s in sleeves.values())
        if total_weight > 1.0 + 1e-9:
            raise ValueError(
                f"Total sleeve target weight cannot exceed 1.0, got {total_weight:.4f}"
            )

        return sleeves


def ensure_project_directories() -> None:
    for directory in [
        CONFIG_DIR,
        DATA_DIR,
        RAW_DATA_DIR,
        RAW_DATA_DIR / "prices",
        STATE_DATA_DIR,
        EXPORT_DATA_DIR,
        EXPORT_DATA_DIR / "data_quality",
    ]:
        directory.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def load_config() -> AppConfig:
    config_path = CONFIG_DIR / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    return AppConfig(**raw)


if __name__ == "__main__":
    ensure_project_directories()
    cfg = load_config()
    print(cfg.model_dump())