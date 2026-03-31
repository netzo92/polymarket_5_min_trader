from datetime import datetime, timedelta, timezone
from pathlib import Path

from polymarket_5_min_trader.backtest import (
    calculate_buy_fee_usdc,
    list_backtest_strategy_names,
    run_execution_scenarios,
    run_strategy_backtests,
    simulate_market,
)
from polymarket_5_min_trader.config import BotConfig
from polymarket_5_min_trader.history import HistoricalDatasetStore, HistoricalMarketBundle
from polymarket_5_min_trader.models import FeeSchedule, Market, OutcomeMarket


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


def test_simulate_market_returns_winning_trade() -> None:
    end_time = datetime(2026, 3, 30, 20, 5, tzinfo=timezone.utc)
    market = Market(
        condition_id="0xmarket",
        question="Will BTC be above 85k at 2:05 PM?",
        slug="btc-above-85k",
        end_time=end_time,
        liquidity=7000,
        volume=10000,
        enable_order_book=True,
        closed=True,
        outcomes=(
            OutcomeMarket("Yes", "yes-token", 0.49, settlement_price=1.0),
            OutcomeMarket("No", "no-token", 0.51, settlement_price=0.0),
        ),
    )
    start = int((end_time - timedelta(minutes=6)).timestamp())
    bundle = HistoricalMarketBundle(
        market=market,
        price_history={
            "yes-token": [
                {"t": start, "p": 0.42},
                {"t": start + 60, "p": 0.44},
                {"t": start + 120, "p": 0.46},
                {"t": start + 180, "p": 0.48},
            ],
            "no-token": [
                {"t": start, "p": 0.58},
                {"t": start + 60, "p": 0.56},
                {"t": start + 120, "p": 0.54},
                {"t": start + 180, "p": 0.52},
            ],
        },
        trades=[],
    )

    trade = simulate_market(bundle, config=make_config())

    assert trade is not None
    assert trade.strategy_name == "momentum_follow"
    assert trade.outcome == "Yes"
    assert trade.pnl > 0


def test_run_strategy_backtests_compares_multiple_strategies(tmp_path: Path) -> None:
    end_time = datetime(2026, 3, 30, 20, 5, tzinfo=timezone.utc)
    market = Market(
        condition_id="0xmarket",
        question="Bitcoin Up or Down - March 30, 4:00PM-4:05PM ET",
        slug="btc-updown-5m-1774900800",
        end_time=end_time,
        liquidity=7000,
        volume=10000,
        enable_order_book=True,
        closed=True,
        outcomes=(
            OutcomeMarket("Up", "up-token", 0.49, settlement_price=1.0),
            OutcomeMarket("Down", "down-token", 0.51, settlement_price=0.0),
        ),
    )
    start = int((end_time - timedelta(minutes=6)).timestamp())
    bundle = HistoricalMarketBundle(
        market=market,
        price_history={
            "up-token": [
                {"t": start, "p": 0.42},
                {"t": start + 60, "p": 0.44},
                {"t": start + 120, "p": 0.46},
                {"t": start + 180, "p": 0.48},
            ],
            "down-token": [
                {"t": start, "p": 0.58},
                {"t": start + 60, "p": 0.56},
                {"t": start + 120, "p": 0.54},
                {"t": start + 180, "p": 0.52},
            ],
        },
        trades=[],
    )
    HistoricalDatasetStore(tmp_path).save_bundle(bundle)

    summaries = run_strategy_backtests(
        tmp_path,
        config=make_config(),
        strategy_names=["momentum_follow", "cheap_momentum"],
    )

    assert [summary.strategy_name for summary in summaries] == [
        "momentum_follow",
        "cheap_momentum",
    ]
    assert all(summary.markets_seen == 1 for summary in summaries)


def test_simulate_market_supports_late_leader_strategy() -> None:
    end_time = datetime(2026, 3, 30, 20, 5, tzinfo=timezone.utc)
    t0 = int((end_time - timedelta(minutes=2)).timestamp())
    market = Market(
        condition_id="0xlate",
        question="Bitcoin Up or Down - March 30, 4:00PM-4:05PM ET",
        slug="btc-updown-5m-1774900800",
        end_time=end_time,
        liquidity=7000,
        volume=10000,
        enable_order_book=True,
        closed=True,
        outcomes=(
            OutcomeMarket("Up", "up-token", 0.7, settlement_price=1.0),
            OutcomeMarket("Down", "down-token", 0.3, settlement_price=0.0),
        ),
    )
    bundle = HistoricalMarketBundle(
        market=market,
        price_history={
            "up-token": [
                {"t": t0, "p": 0.52},
                {"t": t0 + 60, "p": 0.70},
                {"t": t0 + 90, "p": 0.80},
            ],
            "down-token": [
                {"t": t0, "p": 0.48},
                {"t": t0 + 60, "p": 0.30},
                {"t": t0 + 90, "p": 0.20},
            ],
        },
        trades=[],
    )

    trade = simulate_market(bundle, config=make_config(), strategy_name="late_leader_60s")

    assert trade is not None
    assert trade.strategy_name == "late_leader_60s"
    assert trade.outcome == "Up"
    assert trade.pnl > 0


def test_late_leader_strategy_applies_execution_delay_and_polygon_cost() -> None:
    end_time = datetime(2026, 3, 30, 20, 5, tzinfo=timezone.utc)
    t0 = int((end_time - timedelta(minutes=2)).timestamp())
    market = Market(
        condition_id="0xlate-delay",
        question="Bitcoin Up or Down - March 30, 4:00PM-4:05PM ET",
        slug="btc-updown-5m-1774900800",
        end_time=end_time,
        liquidity=7000,
        volume=10000,
        enable_order_book=True,
        closed=True,
        outcomes=(
            OutcomeMarket("Up", "up-token", 0.7, settlement_price=1.0),
            OutcomeMarket("Down", "down-token", 0.3, settlement_price=0.0),
        ),
    )
    bundle = HistoricalMarketBundle(
        market=market,
        price_history={
            "up-token": [
                {"t": t0, "p": 0.52},
                {"t": t0 + 60, "p": 0.70},
                {"t": t0 + 70, "p": 0.80},
            ],
            "down-token": [
                {"t": t0, "p": 0.48},
                {"t": t0 + 60, "p": 0.30},
                {"t": t0 + 70, "p": 0.20},
            ],
        },
        trades=[],
    )

    trade = simulate_market(
        bundle,
        config=make_config(),
        strategy_name="late_leader_60s",
        execution_delay_seconds=5,
        polygon_gas_cost_usdc=0.02,
    )

    assert trade is not None
    assert trade.entry_price == 0.80
    assert trade.observed_entry_price == 0.80
    assert trade.polygon_cost_paid == 0.02
    assert trade.pnl < ((25.0 - trade.fees_paid) / 0.70) - 25.0


def test_simulate_market_applies_slippage_to_fill_price() -> None:
    end_time = datetime(2026, 3, 30, 20, 5, tzinfo=timezone.utc)
    t0 = int((end_time - timedelta(minutes=2)).timestamp())
    market = Market(
        condition_id="0xlate-slip",
        question="Bitcoin Up or Down - March 30, 4:00PM-4:05PM ET",
        slug="btc-updown-5m-1774900800",
        end_time=end_time,
        liquidity=7000,
        volume=10000,
        enable_order_book=True,
        closed=True,
        outcomes=(
            OutcomeMarket("Up", "up-token", 0.7, settlement_price=1.0),
            OutcomeMarket("Down", "down-token", 0.3, settlement_price=0.0),
        ),
    )
    bundle = HistoricalMarketBundle(
        market=market,
        price_history={
            "up-token": [
                {"t": t0, "p": 0.52},
                {"t": t0 + 60, "p": 0.70},
            ],
            "down-token": [
                {"t": t0, "p": 0.48},
                {"t": t0 + 60, "p": 0.30},
            ],
        },
        trades=[],
    )

    no_slippage = simulate_market(
        bundle,
        config=make_config(),
        strategy_name="late_leader_60s",
    )
    slippage = simulate_market(
        bundle,
        config=make_config(),
        strategy_name="late_leader_60s",
        slippage_cents=2.0,
    )

    assert no_slippage is not None
    assert slippage is not None
    assert slippage.observed_entry_price == 0.70
    assert slippage.entry_price == 0.72
    assert slippage.pnl < no_slippage.pnl


def test_list_backtest_strategy_names_includes_late_leader_variants() -> None:
    assert list_backtest_strategy_names()[-4:] == [
        "late_leader_60s",
        "late_leader_30s",
        "late_leader_15s",
        "late_leader_5s",
    ]


def test_run_execution_scenarios_builds_delay_slippage_grid(tmp_path: Path) -> None:
    end_time = datetime(2026, 3, 30, 20, 5, tzinfo=timezone.utc)
    market = Market(
        condition_id="0xgrid",
        question="Bitcoin Up or Down - March 30, 4:00PM-4:05PM ET",
        slug="btc-updown-5m-1774900800",
        end_time=end_time,
        liquidity=7000,
        volume=10000,
        enable_order_book=True,
        closed=True,
        outcomes=(
            OutcomeMarket("Up", "up-token", 0.7, settlement_price=1.0),
            OutcomeMarket("Down", "down-token", 0.3, settlement_price=0.0),
        ),
    )
    t0 = int((end_time - timedelta(minutes=2)).timestamp())
    bundle = HistoricalMarketBundle(
        market=market,
        price_history={
            "up-token": [
                {"t": t0, "p": 0.52},
                {"t": t0 + 60, "p": 0.70},
                {"t": t0 + 70, "p": 0.80},
            ],
            "down-token": [
                {"t": t0, "p": 0.48},
                {"t": t0 + 60, "p": 0.30},
                {"t": t0 + 70, "p": 0.20},
            ],
        },
        trades=[],
    )
    HistoricalDatasetStore(tmp_path).save_bundle(bundle)

    scenarios = run_execution_scenarios(
        tmp_path,
        config=make_config(),
        strategy_name="late_leader_60s",
        execution_delay_seconds=[0, 5],
        slippage_cents=[0.0, 1.0],
    )

    assert [(scenario.execution_delay_seconds, scenario.slippage_cents) for scenario in scenarios] == [
        (0, 0.0),
        (0, 1.0),
        (5, 0.0),
        (5, 1.0),
    ]
    assert all(scenario.summary.markets_seen == 1 for scenario in scenarios)
    assert scenarios[1].summary.total_pnl < scenarios[0].summary.total_pnl


def test_calculate_buy_fee_usdc_matches_crypto_fee_formula() -> None:
    fee = calculate_buy_fee_usdc(
        amount=25.0,
        entry_price=0.50,
        fee_schedule=FeeSchedule(rate=0.072, exponent=1.0),
    )

    assert fee == 0.45
