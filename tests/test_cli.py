import gzip
from decimal import Decimal
from pathlib import Path

import pytest

from polymarket_5_min_trader.cli import (
    build_parser,
    _collateral_funding_messages,
    _compressed_log_name,
    _gzip_rotated_log,
    _resolve_order_amount,
    run_autoresearch_command,
    validate_config,
)
from polymarket_5_min_trader.config import BotConfig


def make_config() -> BotConfig:
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


def test_validate_config_rejects_backtest_only_runtime_strategy() -> None:
    config = make_config()
    config = BotConfig(**{**config.__dict__, "strategy_name": "late_leader_60s"})

    errors, warnings = validate_config(config)

    assert errors
    assert "backtest-only" in errors[0]
    assert warnings == []


def test_validate_config_allows_late_leader_scan_interval_to_define_entry_window() -> None:
    config = make_config()
    config = BotConfig(
        **{
            **config.__dict__,
            "strategy_name": "late_leader",
            "scan_interval_seconds": 1,
            "late_leader_window_seconds": 5,
        }
    )

    errors, warnings = validate_config(config)

    assert errors == []
    assert warnings == []


def test_validate_config_rejects_negative_loss_streak_guard() -> None:
    config = make_config()
    config = BotConfig(**{**config.__dict__, "max_consecutive_losses": -1})

    errors, warnings = validate_config(config)

    assert any("MAX_CONSECUTIVE_LOSSES" in error for error in errors)
    assert warnings == []


def test_validate_config_rejects_invalid_dynamic_order_size_pct() -> None:
    config = make_config()
    config = BotConfig(**{**config.__dict__, "order_size_pct": 101})

    errors, warnings = validate_config(config)

    assert any("ORDER_SIZE_PCT" in error for error in errors)
    assert warnings == []


def test_validate_config_rejects_invalid_log_rotation_settings() -> None:
    config = make_config()
    config = BotConfig(
        **{
            **config.__dict__,
            "log_backup_count": 0,
        }
    )

    errors, warnings = validate_config(config)

    assert any("LOG_BACKUP_COUNT" in error for error in errors)
    assert warnings == []


def test_resolve_order_amount_uses_collateral_percentage() -> None:
    config = BotConfig(**{**make_config().__dict__, "order_size_pct": 1.0})

    amount = _resolve_order_amount(
        config,
        {
            "balance": Decimal("123.456"),
        },
    )

    assert amount == Decimal("1.23")


def test_build_parser_works_without_optional_research_module() -> None:
    parser = build_parser()

    args = parser.parse_args([])

    assert args.command == "run-once"


def test_run_autoresearch_command_requires_optional_module(tmp_path: Path) -> None:
    config = make_config()

    with pytest.raises(RuntimeError) as excinfo:
        run_autoresearch_command(
            config,
            dataset_dir=tmp_path / "research",
            days=1,
            limit=1,
            lookback_minutes=10,
            fidelity=1,
            execution_delay_seconds=0,
            slippage_cents=0.0,
            polygon_gas_cost_usdc=0.0,
            skip_download=True,
            output_dir=None,
        )

    assert "autoresearch" in str(excinfo.value)


def test_gzip_rotated_log_compresses_archives(tmp_path: Path) -> None:
    source = tmp_path / "bot.log"
    dest = tmp_path / "bot.log.2026-03-29"
    source.write_text("hello log\n", encoding="utf-8")

    _gzip_rotated_log(str(source), str(dest))

    archive = Path(_compressed_log_name(str(dest)))
    assert not source.exists()
    assert archive.exists()
    with gzip.open(archive, "rt", encoding="utf-8") as handle:
        assert handle.read() == "hello log\n"


def test_collateral_funding_messages_flags_low_balance_and_allowance() -> None:
    config = make_config()

    messages = _collateral_funding_messages(
        config,
        {
            "balance": Decimal("4.99"),
            "allowance": Decimal("2.00"),
        },
    )

    assert any("Collateral balance" in message for message in messages)
    assert any("Collateral allowance" in message for message in messages)
