from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class FeeSchedule:
    rate: float
    exponent: float
    taker_only: bool = True
    rebate_rate: float | None = None


@dataclass(frozen=True)
class OutcomeMarket:
    outcome: str
    token_id: str
    last_traded_price: float | None
    settlement_price: float | None = None


@dataclass(frozen=True)
class Market:
    condition_id: str
    question: str
    slug: str
    end_time: datetime
    liquidity: float
    volume: float
    enable_order_book: bool
    outcomes: tuple[OutcomeMarket, ...]
    closed: bool = False
    fees_enabled: bool | None = None
    fee_schedule: FeeSchedule | None = None
    fee_type: str | None = None


@dataclass(frozen=True)
class TokenMetrics:
    midpoint: float | None
    spread: float | None


@dataclass(frozen=True)
class Observation:
    recorded_at: datetime
    price: float
    spread: float | None


@dataclass(frozen=True)
class TradeSignal:
    strategy_name: str
    condition_id: str
    token_id: str
    question: str
    outcome: str
    price: float
    spread: float
    minutes_to_expiry: float
    momentum_delta: float
    liquidity: float
    score: float
    reason: str
