from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field


BucketType = Literal["core", "monetary", "opportunistic"]


class UniverseRow(BaseModel):
    ticker: str
    name: str
    isin: str
    exchange: str
    sector: str
    industry: str
    country: str
    pea_eligible: bool
    ttf_flag: bool
    liquidity_bucket: Literal["high", "standard", "lower"]
    spread_proxy: float = Field(ge=0.0)
    core_allowed: bool
    opp_allowed: bool
    benchmark_member: bool
    active: bool


class PortfolioSnapshotRow(BaseModel):
    date: date
    ticker: str
    bucket: BucketType
    quantity: float = Field(ge=0.0)
    avg_price: float = Field(ge=0.0)
    last_price: float = Field(ge=0.0)
    market_value: float = Field(ge=0.0)
    current_weight_total: float = Field(ge=0.0, le=1.0)
    current_weight_bucket: float = Field(ge=0.0, le=1.0)
    unrealized_pnl_pct: float
    days_held: int = Field(ge=0)
    entry_date: date
    entry_signal_type: str
    sector: str
    ttf_flag: bool


class CashStateRow(BaseModel):
    date: date
    cash_total: float = Field(ge=0.0)
    cash_general: float = Field(ge=0.0, default=0.0)
    cash_core: float = Field(ge=0.0, default=0.0)
    cash_opp: float = Field(ge=0.0, default=0.0)
    cash_monetary: float = Field(ge=0.0, default=0.0)


class TradeHistoryRow(BaseModel):
    trade_date: date
    ticker: str
    bucket: BucketType
    side: Literal["BUY", "SELL"]
    quantity: float = Field(gt=0.0)
    price: float = Field(gt=0.0)
    gross_amount: float = Field(gt=0.0)
    brokerage: float = Field(ge=0.0)
    ttf: float = Field(ge=0.0)
    spread_proxy: float = Field(ge=0.0)
    total_cost: float = Field(ge=0.0)
    net_amount: float = Field(gt=0.0)
    reason_code: str


class CoreSignalRow(BaseModel):
    date: date
    ticker: str
    mom12_ex1: float
    mom6: float
    mom3: float
    rel_strength: float
    vol60: float = Field(ge=0.0)
    dd6m: float
    ma200_flag: bool
    ma200_slope_flag: bool
    core_score: float
    rank_core: int = Field(ge=1)
    selected_core: bool


class OppSignalRow(BaseModel):
    date: date
    ticker: str
    ret5d: float
    ret10d: float
    ret20d: float
    rsi14: float = Field(ge=0.0, le=100.0)
    dist_ma20: float
    dist_ma200: float
    vol20: float = Field(ge=0.0)
    volume_spike: float = Field(ge=0.0)
    long_trend_flag: bool
    opp_score_raw: float
    cost_rate_est: float = Field(ge=0.0)
    risk_penalty: float = Field(ge=0.0)
    opp_score_net: float
    selected_opp: bool


class TargetWeightRow(BaseModel):
    date: date
    ticker: str
    bucket: BucketType
    current_weight: float = Field(ge=0.0, le=1.0)
    target_weight_raw: float = Field(ge=0.0, le=1.0)
    target_weight_final: float = Field(ge=0.0, le=1.0)
    rebalance_needed: bool
    reason: str


class ProposedOrderRow(BaseModel):
    date: date
    ticker: str
    bucket: BucketType
    action: Literal["BUY", "SELL", "HOLD"]
    shares_est: float = Field(ge=0.0)
    order_value: float = Field(ge=0.0)
    brokerage_est: float = Field(ge=0.0)
    ttf_est: float = Field(ge=0.0)
    spread_est: float = Field(ge=0.0)
    total_cost_est: float = Field(ge=0.0)
    expected_holding_period: str
    required_return_min: float
    execute: bool
    reason: str


class DecisionReportRow(BaseModel):
    date: date
    bucket: BucketType
    ticker: str
    decision: str
    explanation: str