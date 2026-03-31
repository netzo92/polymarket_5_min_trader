from pathlib import Path
from decimal import Decimal
from types import SimpleNamespace

import pytest

import polymarket_5_min_trader.clob as clob_module
from polymarket_5_min_trader.clob import (
    LiveTradingSetupError,
    NoMatchAvailableError,
    TradingClobClient,
)
from polymarket_5_min_trader.config import BotConfig


def make_config(
    *,
    signature_type: int = 0,
    funder_address: str | None = None,
    api_key: str | None = None,
    api_secret: str | None = None,
    api_passphrase: str | None = None,
) -> BotConfig:
    return BotConfig(
        host="https://clob.polymarket.com",
        gamma_url="https://gamma-api.polymarket.com",
        chain_id=137,
        market_limit=500,
        scan_interval_seconds=5,
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
        state_path=Path("data/state.json"),
        lock_path=Path("data/live.lock"),
        dry_run=False,
        live_enabled=True,
        private_key="0x" + ("11" * 32),
        funder_address=funder_address,
        signature_type=signature_type,
        api_key=api_key,
        api_secret=api_secret,
        api_passphrase=api_passphrase,
        builder_api_key=None,
        builder_secret=None,
        builder_passphrase=None,
        relayer_url="https://relayer-v2.polymarket.com",
        auto_claim_winners=False,
    )


@pytest.fixture(autouse=True)
def clear_derived_api_creds_cache() -> None:
    clob_module._DERIVED_API_CREDS_CACHE.clear()
    yield
    clob_module._DERIVED_API_CREDS_CACHE.clear()


def test_trading_client_explains_proxy_wallet_initialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeClobClient:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        def create_or_derive_api_creds(self) -> object:
            raise RuntimeError("Invalid Funder Address")

    monkeypatch.setattr("polymarket_5_min_trader.clob.ClobClient", FakeClobClient)

    with pytest.raises(LiveTradingSetupError) as excinfo:
        TradingClobClient(make_config(signature_type=2, funder_address="0xproxy"))

    message = str(excinfo.value)
    assert "Invalid Funder Address" in message
    assert "Log in to Polymarket.com once" in message
    assert "POLYMARKET_FUNDER_ADDRESS" in message


def test_trading_client_explains_eoa_mode_when_account_error_mentions_account(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeClobClient:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        def create_or_derive_api_creds(self) -> object:
            raise RuntimeError("account required before trading")

    monkeypatch.setattr("polymarket_5_min_trader.clob.ClobClient", FakeClobClient)

    with pytest.raises(LiveTradingSetupError) as excinfo:
        TradingClobClient(make_config())

    message = str(excinfo.value)
    assert "account required before trading" in message
    assert "does not require a Polymarket.com account" in message
    assert "POLYMARKET_SIGNATURE_TYPE=0" in message


def test_trading_client_derives_api_creds_with_signature_type_and_funder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_calls: list[dict] = []

    class FakeClobClient:
        def __init__(self, **kwargs) -> None:
            init_calls.append(kwargs)

        def create_or_derive_api_creds(self) -> object:
            return object()

    monkeypatch.setattr("polymarket_5_min_trader.clob.ClobClient", FakeClobClient)

    TradingClobClient(make_config(signature_type=2, funder_address="0xproxy"))

    assert len(init_calls) == 2
    assert init_calls[0]["signature_type"] == 2
    assert init_calls[0]["funder"] == "0xproxy"
    assert init_calls[1]["signature_type"] == 2
    assert init_calls[1]["funder"] == "0xproxy"


def test_trading_client_fetches_collateral_balance_and_allowance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_log: list[tuple[str, object]] = []

    class FakeClobClient:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        def create_or_derive_api_creds(self) -> object:
            return object()

        def update_balance_allowance(self, params) -> None:
            call_log.append(("update", params))

        def get_balance_allowance(self, params):
            call_log.append(("get", params))
            return {"balance": "7500000", "allowance": "10000000"}

        def get_collateral_address(self) -> str:
            return "0x2791"

    monkeypatch.setattr("polymarket_5_min_trader.clob.ClobClient", FakeClobClient)

    trader = TradingClobClient(make_config(signature_type=2, funder_address="0xproxy"))
    status = trader.get_collateral_balance_allowance()

    assert [entry[0] for entry in call_log] == ["update", "get"]
    assert call_log[0][1].asset_type == "COLLATERAL"
    assert call_log[0][1].signature_type == 2
    assert status["collateral_address"] == "0x2791"
    assert status["balance_raw"] == "7500000"
    assert status["allowance_raw"] == "10000000"
    assert status["balance"] == Decimal("7.5")
    assert status["allowance"] == Decimal("10.0")


def test_trading_client_prefers_wallet_derived_api_creds_over_stored_triplet(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_modes: list[str] = []

    class FakeClobClient:
        def __init__(self, **kwargs) -> None:
            creds = kwargs.get("creds")
            api_key = getattr(creds, "api_key", None)
            if creds is None:
                self.mode = "derive"
            elif api_key == "stored-key":
                self.mode = "stored"
            else:
                self.mode = "fresh"
            init_modes.append(self.mode)

        def create_or_derive_api_creds(self) -> object:
            return SimpleNamespace(
                api_key="fresh-key",
                api_secret="fresh-secret",
                api_passphrase="fresh-passphrase",
            )

        def update_balance_allowance(self, params) -> None:
            pass

        def get_balance_allowance(self, params):
            return {"balance": "7500000", "allowance": "10000000"}

        def get_collateral_address(self) -> str:
            return "0x2791"

    monkeypatch.setattr("polymarket_5_min_trader.clob.ClobClient", FakeClobClient)

    trader = TradingClobClient(
        make_config(
            signature_type=2,
            funder_address="0xproxy",
            api_key="stored-key",
            api_secret="stored-secret",
            api_passphrase="stored-passphrase",
        )
    )
    status = trader.get_collateral_balance_allowance()

    assert init_modes == ["derive", "fresh"]
    assert status["balance"] == Decimal("7.5")
    assert status["allowance"] == Decimal("10.0")


def test_trading_client_falls_back_to_stored_api_creds_when_derivation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_modes: list[str] = []

    class FakeClobClient:
        def __init__(self, **kwargs) -> None:
            creds = kwargs.get("creds")
            api_key = getattr(creds, "api_key", None)
            if creds is None:
                self.mode = "derive"
            elif api_key == "stored-key":
                self.mode = "stored"
            else:
                self.mode = "fresh"
            init_modes.append(self.mode)

        def create_or_derive_api_creds(self) -> object:
            raise RuntimeError("account required before trading")

        def update_balance_allowance(self, params) -> None:
            pass

        def get_balance_allowance(self, params):
            return {"balance": "7500000", "allowance": "10000000"}

        def get_collateral_address(self) -> str:
            return "0x2791"

    monkeypatch.setattr("polymarket_5_min_trader.clob.ClobClient", FakeClobClient)

    trader = TradingClobClient(
        make_config(
            signature_type=2,
            funder_address="0xproxy",
            api_key="stored-key",
            api_secret="stored-secret",
            api_passphrase="stored-passphrase",
        )
    )
    status = trader.get_collateral_balance_allowance()

    assert init_modes == ["derive", "stored"]
    assert status["balance"] == Decimal("7.5")


def test_trading_client_refreshes_invalid_api_creds_for_balance_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_modes: list[str] = []
    update_modes: list[str] = []
    derived_calls = 0

    class FakeClobClient:
        def __init__(self, **kwargs) -> None:
            creds = kwargs.get("creds")
            api_key = getattr(creds, "api_key", None)
            if creds is None:
                self.mode = "derive"
            elif api_key == "stale-derived":
                self.mode = "stale"
            else:
                self.mode = "fresh"
            init_modes.append(self.mode)

        def create_or_derive_api_creds(self) -> object:
            nonlocal derived_calls
            derived_calls += 1
            if derived_calls == 1:
                return SimpleNamespace(
                    api_key="stale-derived",
                    api_secret="stale-secret",
                    api_passphrase="stale-passphrase",
                )
            return SimpleNamespace(
                api_key="fresh-derived",
                api_secret="fresh-secret",
                api_passphrase="fresh-passphrase",
            )

        def update_balance_allowance(self, params) -> None:
            update_modes.append(self.mode)
            if self.mode == "stale":
                raise RuntimeError("Unauthorized/Invalid api key")

        def get_balance_allowance(self, params):
            return {"balance": "7500000", "allowance": "10000000"}

        def get_collateral_address(self) -> str:
            return "0x2791"

    monkeypatch.setattr("polymarket_5_min_trader.clob.ClobClient", FakeClobClient)

    trader = TradingClobClient(make_config(signature_type=2, funder_address="0xproxy"))
    status = trader.get_collateral_balance_allowance()

    assert init_modes == ["derive", "stale", "derive", "fresh"]
    assert update_modes == ["stale", "fresh"]
    assert derived_calls == 2
    assert status["balance"] == Decimal("7.5")
    assert status["allowance"] == Decimal("10.0")


def test_trading_client_refreshes_invalid_api_creds_for_order_submission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_modes: list[str] = []
    post_modes: list[str] = []
    derived_calls = 0

    class FakeClobClient:
        def __init__(self, **kwargs) -> None:
            creds = kwargs.get("creds")
            api_key = getattr(creds, "api_key", None)
            if creds is None:
                self.mode = "derive"
            elif api_key == "stale-derived":
                self.mode = "stale"
            else:
                self.mode = "fresh"
            init_modes.append(self.mode)

        def create_or_derive_api_creds(self) -> object:
            nonlocal derived_calls
            derived_calls += 1
            if derived_calls == 1:
                return SimpleNamespace(
                    api_key="stale-derived",
                    api_secret="stale-secret",
                    api_passphrase="stale-passphrase",
                )
            return SimpleNamespace(
                api_key="fresh-derived",
                api_secret="fresh-secret",
                api_passphrase="fresh-passphrase",
            )

        def create_market_order(self, args):
            return {"token_id": args.token_id, "amount": args.amount}

        def post_order(self, order, order_type):
            post_modes.append(self.mode)
            if self.mode == "stale":
                raise RuntimeError("Unauthorized/Invalid api key")
            return {"status": "ok", "order": order}

    monkeypatch.setattr("polymarket_5_min_trader.clob.ClobClient", FakeClobClient)

    trader = TradingClobClient(make_config(signature_type=2, funder_address="0xproxy"))
    response = trader.place_market_buy("token-1", 3.25)

    assert init_modes == ["derive", "stale", "derive", "fresh"]
    assert post_modes == ["stale", "fresh"]
    assert derived_calls == 2
    assert response["status"] == "ok"


def test_trading_client_raises_no_match_for_unfilled_market_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeClobClient:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        def create_or_derive_api_creds(self) -> object:
            return object()

        def create_market_order(self, args):
            raise Exception("no match")

    monkeypatch.setattr("polymarket_5_min_trader.clob.ClobClient", FakeClobClient)

    trader = TradingClobClient(make_config(signature_type=2, funder_address="0xproxy"))

    with pytest.raises(NoMatchAvailableError):
        trader.place_market_buy("token-1", 3.25)


def test_trading_client_raises_no_match_for_fok_killed_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeClobClient:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        def create_or_derive_api_creds(self) -> object:
            return object()

        def create_market_order(self, args):
            return {"token_id": args.token_id, "amount": args.amount}

        def post_order(self, order, order_type):
            raise RuntimeError(
                "PolyApiException[status_code=400, error_message={'error': \"order couldn't be fully filled. FOK orders are fully filled or killed.\"}]"
            )

    monkeypatch.setattr("polymarket_5_min_trader.clob.ClobClient", FakeClobClient)

    trader = TradingClobClient(make_config(signature_type=2, funder_address="0xproxy"))

    with pytest.raises(NoMatchAvailableError):
        trader.place_market_buy("token-1", 3.25)
