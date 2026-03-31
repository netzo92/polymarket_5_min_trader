from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


def _as_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class BotConfig:
    host: str
    gamma_url: str
    chain_id: int
    market_limit: int
    scan_interval_seconds: int
    min_minutes_to_expiry: int
    max_minutes_to_expiry: int
    target_minutes_to_expiry: int
    min_liquidity: float
    max_spread: float
    min_midpoint: float
    max_midpoint: float
    momentum_window_minutes: int
    min_observations: int
    min_edge: float
    strategy_name: str
    late_leader_horizon_seconds: int
    late_leader_window_seconds: int
    order_amount: float
    recent_trade_cooldown_minutes: int
    state_path: Path
    lock_path: Path
    dry_run: bool
    live_enabled: bool
    private_key: str | None
    funder_address: str | None
    signature_type: int
    api_key: str | None
    api_secret: str | None
    api_passphrase: str | None
    builder_api_key: str | None
    builder_secret: str | None
    builder_passphrase: str | None
    relayer_url: str
    auto_claim_winners: bool = False
    order_size_pct: float = 0.0
    max_consecutive_losses: int = 0
    log_path: Path = Path("data/bot.log")
    log_backup_count: int = 5

    @classmethod
    def from_env(cls) -> "BotConfig":
        load_dotenv(override=True)
        return cls(
            host=os.getenv("POLYMARKET_HOST", "https://clob.polymarket.com"),
            gamma_url=os.getenv("POLYMARKET_GAMMA_URL", "https://gamma-api.polymarket.com"),
            chain_id=int(os.getenv("POLYMARKET_CHAIN_ID", "137")),
            market_limit=int(os.getenv("POLYMARKET_MARKET_LIMIT", "500")),
            scan_interval_seconds=int(os.getenv("POLYMARKET_SCAN_INTERVAL_SECONDS", "60")),
            min_minutes_to_expiry=int(os.getenv("POLYMARKET_MIN_MINUTES_TO_EXPIRY", "2")),
            max_minutes_to_expiry=int(os.getenv("POLYMARKET_MAX_MINUTES_TO_EXPIRY", "8")),
            target_minutes_to_expiry=int(os.getenv("POLYMARKET_TARGET_MINUTES_TO_EXPIRY", "5")),
            min_liquidity=float(os.getenv("POLYMARKET_MIN_LIQUIDITY", "5000")),
            max_spread=float(os.getenv("POLYMARKET_MAX_SPREAD", "0.08")),
            min_midpoint=float(os.getenv("POLYMARKET_MIN_MIDPOINT", "0.10")),
            max_midpoint=float(os.getenv("POLYMARKET_MAX_MIDPOINT", "0.90")),
            momentum_window_minutes=int(os.getenv("POLYMARKET_MOMENTUM_WINDOW_MINUTES", "3")),
            min_observations=int(os.getenv("POLYMARKET_MIN_OBSERVATIONS", "3")),
            min_edge=float(os.getenv("POLYMARKET_MIN_EDGE", "0.015")),
            strategy_name=os.getenv("POLYMARKET_STRATEGY", "momentum_follow").strip(),
            late_leader_horizon_seconds=int(
                os.getenv("POLYMARKET_LATE_LEADER_HORIZON_SECONDS", "39")
            ),
            late_leader_window_seconds=int(
                os.getenv("POLYMARKET_LATE_LEADER_WINDOW_SECONDS", "5")
            ),
            order_amount=float(os.getenv("POLYMARKET_ORDER_AMOUNT", "25")),
            order_size_pct=float(os.getenv("POLYMARKET_ORDER_SIZE_PCT", "0")),
            recent_trade_cooldown_minutes=int(
                os.getenv("POLYMARKET_RECENT_TRADE_COOLDOWN_MINUTES", "20")
            ),
            state_path=Path(os.getenv("POLYMARKET_STATE_PATH", "data/state.json")),
            lock_path=Path(os.getenv("POLYMARKET_LOCK_PATH", "data/live.lock")),
            log_path=Path(os.getenv("POLYMARKET_LOG_PATH", "data/bot.log")),
            dry_run=_as_bool(os.getenv("POLYMARKET_DRY_RUN"), True),
            live_enabled=_as_bool(os.getenv("POLYMARKET_LIVE_ENABLED"), False),
            private_key=os.getenv("POLYMARKET_PRIVATE_KEY"),
            funder_address=os.getenv("POLYMARKET_FUNDER_ADDRESS"),
            signature_type=int(os.getenv("POLYMARKET_SIGNATURE_TYPE", "0")),
            api_key=os.getenv("POLYMARKET_API_KEY"),
            api_secret=os.getenv("POLYMARKET_API_SECRET"),
            api_passphrase=os.getenv("POLYMARKET_API_PASSPHRASE"),
            builder_api_key=os.getenv("POLYMARKET_BUILDER_API_KEY")
            or os.getenv("POLYMARKET_API_KEY"),
            builder_secret=os.getenv("POLYMARKET_BUILDER_SECRET")
            or os.getenv("POLYMARKET_API_SECRET"),
            builder_passphrase=os.getenv("POLYMARKET_BUILDER_PASSPHRASE")
            or os.getenv("POLYMARKET_API_PASSPHRASE"),
            relayer_url=os.getenv(
                "POLYMARKET_RELAYER_URL", "https://relayer-v2.polymarket.com"
            ),
            auto_claim_winners=_as_bool(
                os.getenv("POLYMARKET_AUTO_CLAIM_WINNERS"), False
            ),
            max_consecutive_losses=int(
                os.getenv("POLYMARKET_MAX_CONSECUTIVE_LOSSES", "0")
            ),
            log_backup_count=int(os.getenv("POLYMARKET_LOG_BACKUP_COUNT", "5")),
        )
