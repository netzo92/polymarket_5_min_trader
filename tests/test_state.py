from decimal import Decimal
from datetime import datetime, timedelta, timezone
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from polymarket_5_min_trader.cli import StrategyGuardTriggered, execute_cycle
from polymarket_5_min_trader.clob import NoMatchAvailableError
from polymarket_5_min_trader.config import BotConfig
from polymarket_5_min_trader.models import Market, OutcomeMarket, TradeSignal
from polymarket_5_min_trader.state import BotStateStore


def make_config(state_path: Path) -> BotConfig:
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
        strategy_name="late_leader",
        late_leader_horizon_seconds=39,
        late_leader_window_seconds=5,
        order_amount=5.0,
        recent_trade_cooldown_minutes=20,
        state_path=state_path,
        lock_path=state_path.with_name("live.lock"),
        dry_run=False,
        live_enabled=True,
        private_key="0xabc",
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


def test_state_store_persists_recent_trades_across_reload(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    store = BotStateStore(state_path)
    state = store.load()
    placed_at = datetime(2026, 3, 30, 20, 0, tzinfo=timezone.utc)

    store.remember_trade(
        state,
        condition_id="0xmarket",
        token_id="up-token",
        placed_at=placed_at,
        mode="live",
    )
    store.save(state)

    reloaded = store.load()
    assert len(reloaded.recent_trades) == 1
    assert reloaded.recent_trades[0]["mode"] == "live"
    assert store.traded_recently(
        reloaded,
        condition_id="0xmarket",
        now=placed_at + timedelta(minutes=5),
        cooldown_minutes=20,
    )


def test_state_store_marks_trade_as_settled(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    store = BotStateStore(state_path)
    state = store.load()
    placed_at = datetime(2026, 3, 30, 20, 0, tzinfo=timezone.utc)
    settled_at = placed_at + timedelta(minutes=5)

    store.remember_trade(
        state,
        condition_id="0xmarket",
        token_id="up-token",
        placed_at=placed_at,
        mode="live",
        question="Bitcoin Up or Down - March 30, 4:00PM-4:05PM ET",
        outcome="Up",
        strategy_name="late_leader",
    )
    assert store.settle_trade(
        state,
        condition_id="0xmarket",
        token_id="up-token",
        placed_at=placed_at,
        settled_at=settled_at,
        result="loss",
        settlement_price=0.0,
    )
    store.save(state)

    reloaded = store.load()
    assert reloaded.recent_trades[0]["result"] == "loss"
    assert reloaded.recent_trades[0]["settlement_price"] == 0.0
    assert reloaded.recent_trades[0]["settled_at"] == settled_at.isoformat()


def test_execute_cycle_persists_live_pending_trade_before_order_returns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_path = tmp_path / "state.json"
    config = make_config(state_path)
    now = datetime(2026, 3, 30, 20, 0, tzinfo=timezone.utc)
    market = Market(
        condition_id="0xmarket",
        question="Bitcoin Up or Down - March 30, 4:00PM-4:05PM ET",
        slug="btc-updown-5m-1774900800",
        end_time=now + timedelta(minutes=5),
        liquidity=7000,
        volume=10000,
        enable_order_book=True,
        outcomes=(
            OutcomeMarket("Up", "up-token", 0.75),
            OutcomeMarket("Down", "down-token", 0.25),
        ),
    )
    signal = TradeSignal(
        strategy_name="late_leader_39s",
        condition_id=market.condition_id,
        token_id="up-token",
        question=market.question,
        outcome="Up",
        price=0.75,
        spread=0.01,
        minutes_to_expiry=5.0,
        momentum_delta=0.20,
        liquidity=market.liquidity,
        score=1.0,
        reason="leader near expiry",
    )

    class FakeGammaClient:
        def __init__(self, gamma_url: str) -> None:
            self.gamma_url = gamma_url

        def fetch_btc_updown_5m_markets(self, *, limit: int, now: datetime) -> list[Market]:
            return [market]

    class FakePublicClobClient:
        def __init__(self, host: str, chain_id: int) -> None:
            self.host = host
            self.chain_id = chain_id

    class FakeTradingClobClient:
        def __init__(self, config: BotConfig) -> None:
            self.config = config

        def place_market_buy(self, token_id: str, amount: float) -> dict:
            raise RuntimeError("network went away after we decided to trade")

    monkeypatch.setattr("polymarket_5_min_trader.cli.datetime", type("FakeDateTime", (), {"now": staticmethod(lambda tz=None: now)}))
    monkeypatch.setattr("polymarket_5_min_trader.cli.GammaClient", FakeGammaClient)
    monkeypatch.setattr("polymarket_5_min_trader.cli.PublicClobClient", FakePublicClobClient)
    monkeypatch.setattr(
        "polymarket_5_min_trader.cli.filter_tradeable_markets",
        lambda markets, *, config, now, strategy_name=None: markets,
    )
    monkeypatch.setattr(
        "polymarket_5_min_trader.cli.collect_token_metrics",
        lambda markets, public_client: {},
    )
    monkeypatch.setattr(
        "polymarket_5_min_trader.cli.build_trade_signals",
        lambda markets, token_metrics, state, *, config, now, strategy_name=None: [signal],
    )
    monkeypatch.setattr("polymarket_5_min_trader.cli.TradingClobClient", FakeTradingClobClient)

    with pytest.raises(RuntimeError):
        execute_cycle(config)

    reloaded = BotStateStore(state_path).load()
    assert len(reloaded.recent_trades) == 1
    assert reloaded.recent_trades[0]["condition_id"] == market.condition_id
    assert reloaded.recent_trades[0]["mode"] == "live-pending"
    assert BotStateStore(state_path).traded_recently(
        reloaded,
        condition_id=market.condition_id,
        now=now + timedelta(minutes=1),
        cooldown_minutes=config.recent_trade_cooldown_minutes,
    )


def test_execute_cycle_uses_dynamic_order_size_pct_for_live_orders(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_path = tmp_path / "state.json"
    config = BotConfig(
        **{
            **make_config(state_path).__dict__,
            "order_size_pct": 1.0,
        }
    )
    now = datetime(2026, 3, 30, 20, 0, tzinfo=timezone.utc)
    market = Market(
        condition_id="0xmarket",
        question="Bitcoin Up or Down - March 30, 4:00PM-4:05PM ET",
        slug="btc-updown-5m-1774900800",
        end_time=now + timedelta(minutes=5),
        liquidity=7000,
        volume=10000,
        enable_order_book=True,
        outcomes=(
            OutcomeMarket("Up", "up-token", 0.75),
            OutcomeMarket("Down", "down-token", 0.25),
        ),
    )
    signal = TradeSignal(
        strategy_name="late_leader",
        condition_id=market.condition_id,
        token_id="up-token",
        question=market.question,
        outcome="Up",
        price=0.75,
        spread=0.01,
        minutes_to_expiry=5.0,
        momentum_delta=0.20,
        liquidity=market.liquidity,
        score=1.0,
        reason="leader near expiry",
    )
    placed: dict[str, object] = {}

    class FakeGammaClient:
        def __init__(self, gamma_url: str) -> None:
            self.gamma_url = gamma_url

        def fetch_btc_updown_5m_markets(self, *, limit: int, now: datetime) -> list[Market]:
            return [market]

    class FakePublicClobClient:
        def __init__(self, host: str, chain_id: int) -> None:
            self.host = host
            self.chain_id = chain_id

    class FakeTradingClobClient:
        def __init__(self, config: BotConfig) -> None:
            self.config = config

        def get_collateral_balance_allowance(self) -> dict[str, object]:
            return {
                "balance": Decimal("123.456"),
                "balance_raw": "123.456",
                "allowance": Decimal("999.00"),
                "allowance_raw": "999.00",
            }

        def place_market_buy(self, token_id: str, amount: float) -> dict:
            placed["token_id"] = token_id
            placed["amount"] = amount
            return {"ok": True}

    monkeypatch.setattr(
        "polymarket_5_min_trader.cli.datetime",
        type(
            "FakeDateTime",
            (),
            {
                "now": staticmethod(lambda tz=None: now),
                "fromisoformat": staticmethod(datetime.fromisoformat),
            },
        ),
    )
    monkeypatch.setattr("polymarket_5_min_trader.cli.GammaClient", FakeGammaClient)
    monkeypatch.setattr("polymarket_5_min_trader.cli.PublicClobClient", FakePublicClobClient)
    monkeypatch.setattr(
        "polymarket_5_min_trader.cli.filter_tradeable_markets",
        lambda markets, *, config, now, strategy_name=None: markets,
    )
    monkeypatch.setattr(
        "polymarket_5_min_trader.cli.collect_token_metrics",
        lambda markets, public_client: {},
    )
    monkeypatch.setattr(
        "polymarket_5_min_trader.cli.build_trade_signals",
        lambda markets, token_metrics, state, *, config, now, strategy_name=None: [signal],
    )
    monkeypatch.setattr("polymarket_5_min_trader.cli.TradingClobClient", FakeTradingClobClient)

    result = execute_cycle(config)

    assert result == 0
    assert placed["token_id"] == "up-token"
    assert placed["amount"] == 1.23


def test_execute_cycle_submits_claim_for_settled_live_winner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_path = tmp_path / "state.json"
    config = BotConfig(
        **{
            **make_config(state_path).__dict__,
            "auto_claim_winners": True,
            "signature_type": 2,
            "funder_address": "0xproxy",
        }
    )
    now = datetime(2026, 3, 30, 20, 10, tzinfo=timezone.utc)
    placed_at = now - timedelta(minutes=7)
    store = BotStateStore(state_path)
    state = store.load()
    store.remember_trade(
        state,
        condition_id="0x" + ("ab" * 32),
        token_id="up-token",
        placed_at=placed_at,
        mode="live",
        question="Bitcoin Up or Down - March 30, 4:00PM-4:05PM ET",
        outcome="Up",
        strategy_name="late_leader",
    )
    store.settle_trade(
        state,
        condition_id="0x" + ("ab" * 32),
        token_id="up-token",
        placed_at=placed_at,
        settled_at=now - timedelta(minutes=2),
        result="win",
        settlement_price=1.0,
    )
    store.save(state)

    class FakeRedeemer:
        def __init__(self, config: BotConfig) -> None:
            self.config = config

        def get_transaction_status(self, transaction_id: str):
            return None

        def submit_redeem(self, *, condition_id: str, metadata: str | None = None):
            return SimpleNamespace(
                transaction_id="claim-1",
                transaction_hash="0xclaimhash",
            )

    class FakeGammaClient:
        def __init__(self, gamma_url: str) -> None:
            self.gamma_url = gamma_url

        def fetch_btc_updown_5m_markets(self, *, limit: int, now: datetime) -> list[Market]:
            return []

    class FakePublicClobClient:
        def __init__(self, host: str, chain_id: int) -> None:
            self.host = host
            self.chain_id = chain_id

    monkeypatch.setattr(
        "polymarket_5_min_trader.cli.datetime",
        type(
            "FakeDateTime",
            (),
            {
                "now": staticmethod(lambda tz=None: now),
                "fromisoformat": staticmethod(datetime.fromisoformat),
            },
        ),
    )
    monkeypatch.setattr("polymarket_5_min_trader.cli.SafeMarketRedeemer", FakeRedeemer)
    monkeypatch.setattr("polymarket_5_min_trader.cli.GammaClient", FakeGammaClient)
    monkeypatch.setattr("polymarket_5_min_trader.cli.PublicClobClient", FakePublicClobClient)

    result = execute_cycle(config)

    assert result == 0
    reloaded = BotStateStore(state_path).load()
    trade = reloaded.recent_trades[0]
    assert trade["claim_state"] == "submitted"
    assert trade["claim_transaction_id"] == "claim-1"
    assert trade["claim_transaction_hash"] == "0xclaimhash"


def test_execute_cycle_refreshes_settlements_even_without_loss_guard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_path = tmp_path / "state.json"
    config = BotConfig(
        **{
            **make_config(state_path).__dict__,
            "auto_claim_winners": True,
            "signature_type": 2,
            "funder_address": "0xproxy",
            "max_consecutive_losses": 0,
        }
    )
    now = datetime(2026, 3, 30, 20, 10, tzinfo=timezone.utc)
    placed_at = now - timedelta(minutes=7)
    condition_id = "0x" + ("ef" * 32)
    store = BotStateStore(state_path)
    state = store.load()
    store.remember_trade(
        state,
        condition_id=condition_id,
        token_id="down-token",
        placed_at=placed_at,
        mode="live",
        question="Bitcoin Up or Down - March 30, 4:00PM-4:05PM ET",
        outcome="Down",
        strategy_name="late_leader",
    )
    store.save(state)

    closed_market = Market(
        condition_id=condition_id,
        question="Bitcoin Up or Down - March 30, 4:00PM-4:05PM ET",
        slug="btc-updown-5m-1774900800",
        end_time=now - timedelta(minutes=2),
        liquidity=7000,
        volume=10000,
        enable_order_book=True,
        closed=True,
        outcomes=(
            OutcomeMarket("Up", "up-token", 0.0, settlement_price=0.0),
            OutcomeMarket("Down", "down-token", 1.0, settlement_price=1.0),
        ),
    )

    class FakeRedeemer:
        def __init__(self, config: BotConfig) -> None:
            self.config = config

        def get_transaction_status(self, transaction_id: str):
            return None

        def submit_redeem(self, *, condition_id: str, metadata: str | None = None):
            assert condition_id == "0x" + ("ef" * 32)
            return SimpleNamespace(
                transaction_id="claim-refresh",
                transaction_hash="0xrefreshhash",
            )

    class FakeGammaClient:
        def __init__(self, gamma_url: str) -> None:
            self.gamma_url = gamma_url

        def fetch_recent_closed_btc_updown_5m_markets(
            self,
            *,
            limit: int,
            days: int,
            now: datetime | None = None,
        ) -> list[Market]:
            return [closed_market]

        def fetch_btc_updown_5m_markets(self, *, limit: int, now: datetime) -> list[Market]:
            return []

    class FakePublicClobClient:
        def __init__(self, host: str, chain_id: int) -> None:
            self.host = host
            self.chain_id = chain_id

    monkeypatch.setattr(
        "polymarket_5_min_trader.cli.datetime",
        type(
            "FakeDateTime",
            (),
            {
                "now": staticmethod(lambda tz=None: now),
                "fromisoformat": staticmethod(datetime.fromisoformat),
            },
        ),
    )
    monkeypatch.setattr("polymarket_5_min_trader.cli.SafeMarketRedeemer", FakeRedeemer)
    monkeypatch.setattr("polymarket_5_min_trader.cli.GammaClient", FakeGammaClient)
    monkeypatch.setattr("polymarket_5_min_trader.cli.PublicClobClient", FakePublicClobClient)

    result = execute_cycle(config)

    assert result == 0
    reloaded = BotStateStore(state_path).load()
    trade = reloaded.recent_trades[0]
    assert trade["result"] == "win"
    assert trade["settlement_price"] == 1.0
    assert trade["claim_state"] == "submitted"
    assert trade["claim_transaction_id"] == "claim-refresh"
    assert trade["claim_transaction_hash"] == "0xrefreshhash"


def test_execute_cycle_confirms_submitted_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_path = tmp_path / "state.json"
    config = BotConfig(
        **{
            **make_config(state_path).__dict__,
            "auto_claim_winners": True,
            "signature_type": 2,
            "funder_address": "0xproxy",
        }
    )
    now = datetime(2026, 3, 30, 20, 10, tzinfo=timezone.utc)
    placed_at = now - timedelta(minutes=7)
    store = BotStateStore(state_path)
    state = store.load()
    store.remember_trade(
        state,
        condition_id="0x" + ("cd" * 32),
        token_id="up-token",
        placed_at=placed_at,
        mode="live",
        question="Bitcoin Up or Down - March 30, 4:00PM-4:05PM ET",
        outcome="Up",
        strategy_name="late_leader",
    )
    store.settle_trade(
        state,
        condition_id="0x" + ("cd" * 32),
        token_id="up-token",
        placed_at=placed_at,
        settled_at=now - timedelta(minutes=2),
        result="win",
        settlement_price=1.0,
    )
    store.update_trade_fields(
        state,
        condition_id="0x" + ("cd" * 32),
        token_id="up-token",
        placed_at=placed_at,
        claim_state="submitted",
        claim_transaction_id="claim-2",
        claim_transaction_hash="0xpending",
    )
    store.save(state)

    class FakeRedeemer:
        def __init__(self, config: BotConfig) -> None:
            self.config = config

        def get_transaction_status(self, transaction_id: str):
            assert transaction_id == "claim-2"
            return {"state": "STATE_CONFIRMED", "transactionHash": "0xconfirmed"}

        def submit_redeem(self, *, condition_id: str, metadata: str | None = None):
            raise AssertionError("submit_redeem should not be called for an already-submitted claim")

    class FakeGammaClient:
        def __init__(self, gamma_url: str) -> None:
            self.gamma_url = gamma_url

        def fetch_btc_updown_5m_markets(self, *, limit: int, now: datetime) -> list[Market]:
            return []

    class FakePublicClobClient:
        def __init__(self, host: str, chain_id: int) -> None:
            self.host = host
            self.chain_id = chain_id

    monkeypatch.setattr(
        "polymarket_5_min_trader.cli.datetime",
        type(
            "FakeDateTime",
            (),
            {
                "now": staticmethod(lambda tz=None: now),
                "fromisoformat": staticmethod(datetime.fromisoformat),
            },
        ),
    )
    monkeypatch.setattr("polymarket_5_min_trader.cli.SafeMarketRedeemer", FakeRedeemer)
    monkeypatch.setattr("polymarket_5_min_trader.cli.GammaClient", FakeGammaClient)
    monkeypatch.setattr("polymarket_5_min_trader.cli.PublicClobClient", FakePublicClobClient)

    result = execute_cycle(config)

    assert result == 0
    reloaded = BotStateStore(state_path).load()
    trade = reloaded.recent_trades[0]
    assert trade["claim_state"] == "confirmed"
    assert trade["claim_transaction_hash"] == "0xconfirmed"
    assert trade["claimed_at"] == now.isoformat()


def test_execute_cycle_treats_no_match_as_warning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    state_path = tmp_path / "state.json"
    config = make_config(state_path)
    now = datetime(2026, 3, 30, 20, 0, tzinfo=timezone.utc)
    market = Market(
        condition_id="0xmarket",
        question="Bitcoin Up or Down - March 30, 4:00PM-4:05PM ET",
        slug="btc-updown-5m-1774900800",
        end_time=now + timedelta(minutes=5),
        liquidity=7000,
        volume=10000,
        enable_order_book=True,
        outcomes=(
            OutcomeMarket("Up", "up-token", 0.75),
            OutcomeMarket("Down", "down-token", 0.25),
        ),
    )
    signal = TradeSignal(
        strategy_name="late_leader",
        condition_id=market.condition_id,
        token_id="up-token",
        question=market.question,
        outcome="Up",
        price=0.75,
        spread=0.01,
        minutes_to_expiry=5.0,
        momentum_delta=0.20,
        liquidity=market.liquidity,
        score=1.0,
        reason="leader near expiry",
    )

    class FakeGammaClient:
        def __init__(self, gamma_url: str) -> None:
            self.gamma_url = gamma_url

        def fetch_btc_updown_5m_markets(self, *, limit: int, now: datetime) -> list[Market]:
            return [market]

    class FakePublicClobClient:
        def __init__(self, host: str, chain_id: int) -> None:
            self.host = host
            self.chain_id = chain_id

    class FakeTradingClobClient:
        def __init__(self, config: BotConfig) -> None:
            self.config = config

        def place_market_buy(self, token_id: str, amount: float) -> dict:
            raise NoMatchAvailableError("No executable liquidity was available for this market order.")

    monkeypatch.setattr(
        "polymarket_5_min_trader.cli.datetime",
        type("FakeDateTime", (), {"now": staticmethod(lambda tz=None: now)}),
    )
    monkeypatch.setattr("polymarket_5_min_trader.cli.GammaClient", FakeGammaClient)
    monkeypatch.setattr("polymarket_5_min_trader.cli.PublicClobClient", FakePublicClobClient)
    monkeypatch.setattr(
        "polymarket_5_min_trader.cli.filter_tradeable_markets",
        lambda markets, *, config, now, strategy_name=None: markets,
    )
    monkeypatch.setattr(
        "polymarket_5_min_trader.cli.collect_token_metrics",
        lambda markets, public_client: {},
    )
    monkeypatch.setattr(
        "polymarket_5_min_trader.cli.build_trade_signals",
        lambda markets, token_metrics, state, *, config, now, strategy_name=None: [signal],
    )
    monkeypatch.setattr("polymarket_5_min_trader.cli.TradingClobClient", FakeTradingClobClient)

    with caplog.at_level(logging.WARNING):
        result = execute_cycle(config)

    assert result == 0
    assert "Live order skipped" in caplog.text
    reloaded = BotStateStore(state_path).load()
    assert len(reloaded.recent_trades) == 1
    assert reloaded.recent_trades[0]["mode"] == "live-no-match"


def test_execute_cycle_stops_after_configured_settled_loss_streak(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_path = tmp_path / "state.json"
    config = BotConfig(
        **{
            **make_config(state_path).__dict__,
            "max_consecutive_losses": 1,
        }
    )
    now = datetime(2026, 3, 30, 20, 10, tzinfo=timezone.utc)
    previous_trade_at = now - timedelta(minutes=7)
    store = BotStateStore(state_path)
    state = store.load()
    store.remember_trade(
        state,
        condition_id="0xclosed",
        token_id="up-token",
        placed_at=previous_trade_at,
        mode="live",
        question="Bitcoin Up or Down - March 30, 4:00PM-4:05PM ET",
        outcome="Up",
        strategy_name="late_leader",
    )
    store.save(state)

    closed_market = Market(
        condition_id="0xclosed",
        question="Bitcoin Up or Down - March 30, 4:00PM-4:05PM ET",
        slug="btc-updown-5m-1774900800",
        end_time=now - timedelta(minutes=2),
        liquidity=7000,
        volume=10000,
        enable_order_book=True,
        closed=True,
        outcomes=(
            OutcomeMarket("Up", "up-token", 0.0, settlement_price=0.0),
            OutcomeMarket("Down", "down-token", 1.0, settlement_price=1.0),
        ),
    )

    class FakeDateTime:
        @staticmethod
        def now(tz=None):
            return now

        @staticmethod
        def fromisoformat(value: str):
            return datetime.fromisoformat(value)

    class FakeGammaClient:
        def __init__(self, gamma_url: str) -> None:
            self.gamma_url = gamma_url

        def fetch_recent_closed_btc_updown_5m_markets(
            self,
            *,
            limit: int,
            days: int,
            now: datetime | None = None,
        ) -> list[Market]:
            return [closed_market]

        def fetch_btc_updown_5m_markets(self, *, limit: int, now: datetime) -> list[Market]:
            raise AssertionError("guard should stop the cycle before scanning new markets")

    class FakePublicClobClient:
        def __init__(self, host: str, chain_id: int) -> None:
            self.host = host
            self.chain_id = chain_id

    monkeypatch.setattr("polymarket_5_min_trader.cli.datetime", FakeDateTime)
    monkeypatch.setattr("polymarket_5_min_trader.cli.GammaClient", FakeGammaClient)
    monkeypatch.setattr("polymarket_5_min_trader.cli.PublicClobClient", FakePublicClobClient)

    with pytest.raises(StrategyGuardTriggered):
        execute_cycle(config)

    reloaded = BotStateStore(state_path).load()
    assert reloaded.recent_trades[0]["result"] == "loss"
    assert reloaded.recent_trades[0]["settlement_price"] == 0.0
