from datetime import datetime, timedelta, timezone
from pathlib import Path

from polymarket_5_min_trader.config import BotConfig
from polymarket_5_min_trader.models import Market, Observation, OutcomeMarket, TokenMetrics
from polymarket_5_min_trader.state import BotState
from polymarket_5_min_trader.strategy import (
    build_trade_signals,
    effective_late_leader_window_seconds,
    filter_tradeable_markets,
    late_leader_expiry_bounds_seconds,
    list_strategy_names,
    market_depth,
)


def make_config() -> BotConfig:
    return BotConfig(
        host="https://clob.polymarket.com",
        gamma_url="https://gamma-api.polymarket.com",
        chain_id=137,
        market_limit=500,
        scan_interval_seconds=60,
        min_minutes_to_expiry=2,
        max_minutes_to_expiry=8,
        target_minutes_to_expiry=5,
        min_liquidity=5000,
        max_spread=0.08,
        min_midpoint=0.10,
        max_midpoint=0.90,
        momentum_window_minutes=3,
        min_observations=3,
        min_edge=0.015,
        strategy_name="momentum_follow",
        late_leader_horizon_seconds=39,
        late_leader_window_seconds=5,
        order_amount=25.0,
        recent_trade_cooldown_minutes=20,
        state_path=Path("data/state.json"),
        lock_path=Path("data/live.lock"),
        dry_run=True,
        live_enabled=False,
        private_key=None,
        funder_address=None,
        signature_type=0,
        api_key=None,
        api_secret=None,
        api_passphrase=None,
        builder_api_key=None,
        builder_secret=None,
        builder_passphrase=None,
        relayer_url="https://relayer-v2.polymarket.com",
        auto_claim_winners=False,
    )


def test_filter_tradeable_markets_keeps_short_expiry_liquid_markets() -> None:
    now = datetime(2026, 3, 30, 20, 0, tzinfo=timezone.utc)
    config = make_config()
    market = Market(
        condition_id="0xmarket",
        question="Will BTC be above 85k at 2:05 PM?",
        slug="btc-above-85k",
        end_time=now + timedelta(minutes=5),
        liquidity=6000,
        volume=10000,
        enable_order_book=True,
        outcomes=(
            OutcomeMarket("Yes", "yes-token", 0.49),
            OutcomeMarket("No", "no-token", 0.51),
        ),
    )
    assert filter_tradeable_markets([market], config=config, now=now) == [market]


def test_build_trade_signals_prefers_positive_momentum() -> None:
    now = datetime(2026, 3, 30, 20, 0, tzinfo=timezone.utc)
    config = make_config()
    market = Market(
        condition_id="0xmarket",
        question="Will BTC be above 85k at 2:05 PM?",
        slug="btc-above-85k",
        end_time=now + timedelta(minutes=5),
        liquidity=7000,
        volume=10000,
        enable_order_book=True,
        outcomes=(
            OutcomeMarket("Yes", "yes-token", 0.49),
            OutcomeMarket("No", "no-token", 0.51),
        ),
    )
    state = BotState(
        observations={
            "yes-token": [
                Observation(now - timedelta(minutes=3), 0.42, 0.02),
                Observation(now - timedelta(minutes=2), 0.44, 0.02),
                Observation(now - timedelta(minutes=1), 0.46, 0.02),
            ],
            "no-token": [
                Observation(now - timedelta(minutes=3), 0.58, 0.03),
                Observation(now - timedelta(minutes=2), 0.56, 0.03),
                Observation(now - timedelta(minutes=1), 0.54, 0.03),
            ],
        }
    )
    token_metrics = {
        "yes-token": TokenMetrics(midpoint=0.48, spread=0.02),
        "no-token": TokenMetrics(midpoint=0.52, spread=0.02),
    }

    signals = build_trade_signals([market], token_metrics, state, config=config, now=now)

    assert len(signals) == 1
    assert signals[0].token_id == "yes-token"
    assert signals[0].momentum_delta > 0
    assert signals[0].strategy_name == "momentum_follow"


def test_mean_reversion_strategy_buys_bouncing_loser() -> None:
    now = datetime(2026, 3, 30, 20, 0, tzinfo=timezone.utc)
    config = make_config()
    market = Market(
        condition_id="0xmarket",
        question="Bitcoin Up or Down - March 30, 4:00PM-4:05PM ET",
        slug="btc-updown-5m-1774900800",
        end_time=now + timedelta(minutes=5),
        liquidity=7000,
        volume=10000,
        enable_order_book=True,
        outcomes=(
            OutcomeMarket("Up", "up-token", 0.49),
            OutcomeMarket("Down", "down-token", 0.51),
        ),
    )
    state = BotState(
        observations={
            "up-token": [
                Observation(now - timedelta(minutes=3), 0.60, 0.02),
                Observation(now - timedelta(minutes=2), 0.45, 0.02),
                Observation(now - timedelta(minutes=1), 0.42, 0.02),
            ],
            "down-token": [
                Observation(now - timedelta(minutes=3), 0.40, 0.02),
                Observation(now - timedelta(minutes=2), 0.55, 0.02),
                Observation(now - timedelta(minutes=1), 0.58, 0.02),
            ],
        }
    )
    token_metrics = {
        "up-token": TokenMetrics(midpoint=0.44, spread=0.02),
        "down-token": TokenMetrics(midpoint=0.56, spread=0.02),
    }

    signals = build_trade_signals(
        [market],
        token_metrics,
        state,
        config=config,
        now=now,
        strategy_name="mean_reversion",
    )

    assert len(signals) == 1
    assert signals[0].token_id == "up-token"
    assert signals[0].strategy_name == "mean_reversion"


def test_market_depth_falls_back_to_volume_when_liquidity_missing() -> None:
    now = datetime(2026, 3, 30, 20, 0, tzinfo=timezone.utc)
    market = Market(
        condition_id="0xmarket",
        question="Bitcoin Up or Down - March 30, 4:00PM-4:05PM ET",
        slug="btc-updown-5m-1774900800",
        end_time=now + timedelta(minutes=5),
        liquidity=0,
        volume=125000,
        enable_order_book=True,
        outcomes=(
            OutcomeMarket("Up", "up-token", 0.49),
            OutcomeMarket("Down", "down-token", 0.51),
        ),
    )
    assert market_depth(market) == 125000


def test_strategy_registry_lists_multiple_named_strategies() -> None:
    assert list_strategy_names() == [
        "late_leader",
        "momentum_follow",
        "relative_strength",
        "cheap_momentum",
        "mean_reversion",
        "short_term_flip",
        "steady_trend",
        "breakout_continuation",
        "pullback_reclaim",
        "underdog_reclaim",
    ]


def test_filter_tradeable_markets_supports_late_leader_seconds_window() -> None:
    now = datetime(2026, 3, 30, 20, 0, tzinfo=timezone.utc)
    config = BotConfig(
        **{
            **make_config().__dict__,
            "scan_interval_seconds": 1,
            "late_leader_window_seconds": 5,
        }
    )
    lower, upper = late_leader_expiry_bounds_seconds(config)
    market = Market(
        condition_id="0xlate",
        question="Bitcoin Up or Down - March 30, 4:00PM-4:05PM ET",
        slug="btc-updown-5m-1774900800",
        end_time=now + timedelta(seconds=(lower + upper) / 2),
        liquidity=7000,
        volume=10000,
        enable_order_book=True,
        outcomes=(
            OutcomeMarket("Up", "up-token", 0.75),
            OutcomeMarket("Down", "down-token", 0.25),
        ),
    )

    assert filter_tradeable_markets(
        [market],
        config=config,
        now=now,
        strategy_name="late_leader",
    ) == [market]


def test_late_leader_window_tracks_scan_interval() -> None:
    config = BotConfig(
        **{
            **make_config().__dict__,
            "scan_interval_seconds": 1,
            "late_leader_window_seconds": 5,
        }
    )

    assert effective_late_leader_window_seconds(config) == 1
    assert late_leader_expiry_bounds_seconds(config) == (38, 40)


def test_build_trade_signals_supports_live_late_leader_strategy() -> None:
    now = datetime(2026, 3, 30, 20, 0, tzinfo=timezone.utc)
    config = make_config()
    market = Market(
        condition_id="0xlate",
        question="Bitcoin Up or Down - March 30, 4:00PM-4:05PM ET",
        slug="btc-updown-5m-1774900800",
        end_time=now + timedelta(seconds=config.late_leader_horizon_seconds),
        liquidity=7000,
        volume=10000,
        enable_order_book=True,
        outcomes=(
            OutcomeMarket("Up", "up-token", 0.75),
            OutcomeMarket("Down", "down-token", 0.25),
        ),
    )

    signals = build_trade_signals(
        [market],
        {
            "up-token": TokenMetrics(midpoint=0.75, spread=0.01),
            "down-token": TokenMetrics(midpoint=0.25, spread=0.01),
        },
        BotState(),
        config=config,
        now=now,
        strategy_name="late_leader",
    )

    assert len(signals) == 1
    assert signals[0].strategy_name == "late_leader"
    assert signals[0].token_id == "up-token"
