from __future__ import annotations

import argparse
import fcntl
import gzip
import logging
import os
import shutil
import sys
import time
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
from decimal import ROUND_DOWN, Decimal
from logging.handlers import TimedRotatingFileHandler
from math import ceil
from pathlib import Path

from polymarket_5_min_trader.backtest import (
    BACKTEST_ONLY_STRATEGIES,
    list_backtest_strategy_names,
    run_backtest,
    run_execution_scenarios,
    run_strategy_backtests,
)
from polymarket_5_min_trader.history import HistoricalDatasetStore, HistoricalMarketBundle, HistoryClient
from polymarket_5_min_trader.clob import (
    NoMatchAvailableError,
    PublicClobClient,
    TradingClobClient,
)
from polymarket_5_min_trader.config import BotConfig
from polymarket_5_min_trader.gamma import GammaClient
from polymarket_5_min_trader.models import Market, TokenMetrics, TradeSignal
from polymarket_5_min_trader.relayer import RelayerSetupError, SafeMarketRedeemer
from polymarket_5_min_trader.state import BotStateStore
from polymarket_5_min_trader.strategy import (
    DEFAULT_STRATEGY_NAME,
    build_trade_signals,
    filter_tradeable_markets,
    late_leader_expiry_bounds_seconds,
    make_observation,
)

LOGGER = logging.getLogger(__name__)
NOISY_LOGGERS = ("urllib3", "httpcore", "httpx", "hpack")
USD_CENT = Decimal("0.01")
PERCENT_DENOMINATOR = Decimal("100")
DEFAULT_RESEARCH_DIR = Path("data/research")
DEFAULT_RESEARCH_DAYS = 3
DEFAULT_RESEARCH_LIMIT = 25
DEFAULT_RESEARCH_LOOKBACK_MINUTES = 90
DEFAULT_RESEARCH_FIDELITY = 1
DEFAULT_EXEC_DELAY = 0
DEFAULT_SLIPPAGE_CENTS = 0.0
DEFAULT_GAS_COST = 0.0


class StrategyGuardTriggered(RuntimeError):
    """Raised when recent settled trades violate the configured guardrail."""


def _load_research_tools():
    try:
        from polymarket_5_min_trader.research import (
            format_report,
            run_autoresearch,
            save_report,
        )
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The optional autoresearch module is not installed in this checkout. "
            "The trading commands still work, but `autoresearch` is unavailable."
        ) from exc
    return run_autoresearch, format_report, save_report


def _format_decimal(value: object, *, places: int = 6) -> str:
    if not isinstance(value, Decimal):
        return str(value)
    text = f"{value:.{places}f}"
    return text.rstrip("0").rstrip(".")


def _compressed_log_name(default_name: str) -> str:
    return f"{default_name}.gz"


def _gzip_rotated_log(source: str, dest: str) -> None:
    archive_path = _compressed_log_name(dest)
    with open(source, "rb") as source_handle, gzip.open(archive_path, "wb") as archive_handle:
        shutil.copyfileobj(source_handle, archive_handle)
    os.remove(source)


def configure_logging(
    verbose: bool,
    *,
    log_path: Path,
    log_backup_count: int,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = TimedRotatingFileHandler(
        log_path,
        when="midnight",
        interval=1,
        backupCount=max(1, log_backup_count),
        encoding="utf-8",
    )
    file_handler.suffix = "%Y-%m-%d"
    file_handler.namer = _compressed_log_name
    file_handler.rotator = _gzip_rotated_log
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            file_handler,
        ],
        force=True,
    )
    for logger_name in NOISY_LOGGERS:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def collect_token_metrics(
    markets: list[Market],
    public_client: PublicClobClient,
) -> dict[str, TokenMetrics]:
    metrics: dict[str, TokenMetrics] = {}
    for market in markets:
        for outcome in market.outcomes:
            if outcome.token_id not in metrics:
                metrics[outcome.token_id] = public_client.get_token_metrics(outcome.token_id)
    return metrics


@contextmanager
def execution_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another bot process is already using lock {path}.") from exc
        handle.seek(0)
        handle.truncate()
        handle.write(f"{os.getpid()}\n")
        handle.flush()
        try:
            yield
        finally:
            handle.seek(0)
            handle.truncate()
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def validate_config(config: BotConfig) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    if config.strategy_name not in list_backtest_strategy_names():
        errors.append(f"Unknown strategy '{config.strategy_name}'.")
    elif config.strategy_name in BACKTEST_ONLY_STRATEGIES:
        errors.append(
            f"Strategy '{config.strategy_name}' is backtest-only and cannot be used for live runtime commands."
        )
    if config.order_amount <= 0:
        errors.append("POLYMARKET_ORDER_AMOUNT must be greater than 0.")
    if config.order_size_pct < 0 or config.order_size_pct > 100:
        errors.append("POLYMARKET_ORDER_SIZE_PCT must be between 0 and 100.")
    if config.auto_claim_winners and config.signature_type != 2:
        warnings.append(
            "POLYMARKET_AUTO_CLAIM_WINNERS currently supports only POLYMARKET_SIGNATURE_TYPE=2 (GNOSIS_SAFE)."
        )
    if config.scan_interval_seconds <= 0:
        errors.append("POLYMARKET_SCAN_INTERVAL_SECONDS must be greater than 0.")
    if config.recent_trade_cooldown_minutes <= 0:
        errors.append("POLYMARKET_RECENT_TRADE_COOLDOWN_MINUTES must be greater than 0.")
    if config.max_spread <= 0:
        errors.append("POLYMARKET_MAX_SPREAD must be greater than 0.")
    if config.max_consecutive_losses < 0:
        errors.append("POLYMARKET_MAX_CONSECUTIVE_LOSSES must be 0 or greater.")
    if config.order_size_pct > 0 and not config.private_key:
        errors.append(
            "POLYMARKET_PRIVATE_KEY is required when POLYMARKET_ORDER_SIZE_PCT is greater than 0."
        )
    if config.order_size_pct > 0 and config.signature_type != 0 and not config.funder_address:
        errors.append(
            "POLYMARKET_FUNDER_ADDRESS is required for POLYMARKET_ORDER_SIZE_PCT with proxy/safe wallets."
        )
    if config.log_backup_count <= 0:
        errors.append("POLYMARKET_LOG_BACKUP_COUNT must be greater than 0.")

    if config.strategy_name == "late_leader":
        if config.late_leader_horizon_seconds <= 0:
            errors.append("POLYMARKET_LATE_LEADER_HORIZON_SECONDS must be greater than 0.")
        if config.late_leader_window_seconds <= 0:
            warnings.append(
                "POLYMARKET_LATE_LEADER_WINDOW_SECONDS is ignored and the live late-leader "
                "entry window now follows POLYMARKET_SCAN_INTERVAL_SECONDS."
            )

    if config.live_enabled and not config.dry_run:
        if not config.private_key:
            errors.append("POLYMARKET_PRIVATE_KEY is required for live trading.")
        if config.signature_type != 0 and not config.funder_address:
            errors.append(
                "POLYMARKET_FUNDER_ADDRESS is required for live proxy/safe trading."
            )
    else:
        if not config.live_enabled and not config.dry_run:
            warnings.append(
                "POLYMARKET_DRY_RUN=false but POLYMARKET_LIVE_ENABLED=false, so orders will still not be placed."
            )
        if config.live_enabled and config.dry_run:
            warnings.append(
                "POLYMARKET_LIVE_ENABLED=true while POLYMARKET_DRY_RUN=true, so live orders remain disabled."
            )
        if config.auto_claim_winners and not (
            config.builder_api_key
            and config.builder_secret
            and config.builder_passphrase
        ):
            warnings.append(
                "POLYMARKET_AUTO_CLAIM_WINNERS=true but no Builder credentials are configured. "
                "Set POLYMARKET_BUILDER_* or reuse POLYMARKET_API_* if those are Builder keys."
            )

    return errors, warnings


def run_doctor_command(config: BotConfig) -> int:
    now = datetime.now(timezone.utc)
    errors, warnings = validate_config(config)
    for warning in warnings:
        LOGGER.warning("Doctor warning: %s", warning)
    for error in errors:
        LOGGER.error("Doctor error: %s", error)

    LOGGER.info(
        "Doctor config | strategy=%s window=%s scan_interval=%ss order_amount=%s state=%s lock=%s log=%s",
        config.strategy_name,
        _describe_strategy_window(config),
        config.scan_interval_seconds,
        _describe_order_sizing(config),
        config.state_path,
        config.lock_path,
        config.log_path,
    )

    markets: list[Market] = []
    try:
        gamma_client = GammaClient(config.gamma_url)
        markets = gamma_client.fetch_btc_updown_5m_markets(limit=min(config.market_limit, 25), now=now)
        LOGGER.info("Doctor Gamma check: loaded %s active BTC 5-minute markets.", len(markets))
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Doctor Gamma check failed: %s", exc)
        errors.append("Gamma API connectivity failed.")

    try:
        public_client = PublicClobClient(config.host, config.chain_id)
        if markets:
            checked_token_id: str | None = None
            checked_metrics: TokenMetrics | None = None
            for market in markets:
                for outcome in market.outcomes:
                    metrics = public_client.get_token_metrics(outcome.token_id)
                    if metrics.midpoint is None and metrics.spread is None:
                        continue
                    checked_token_id = outcome.token_id
                    checked_metrics = metrics
                    break
                if checked_metrics is not None:
                    break
            if checked_metrics is not None and checked_token_id is not None:
                LOGGER.info(
                    "Doctor CLOB check: midpoint=%s spread=%s for %s",
                    checked_metrics.midpoint,
                    checked_metrics.spread,
                    checked_token_id,
                )
            else:
                warnings.append(
                    "CLOB responded, but none of the sampled active outcomes returned an orderbook midpoint or spread."
                )
                LOGGER.warning(
                    "Doctor CLOB check: no sampled active outcome returned an orderbook midpoint or spread."
                )
        else:
            LOGGER.info("Doctor CLOB check skipped midpoint lookup because no active markets were returned.")
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Doctor CLOB check failed: %s", exc)
        errors.append("CLOB connectivity failed.")

    try:
        with execution_lock(config.lock_path):
            LOGGER.info("Doctor lock check: acquired %s", config.lock_path)
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Doctor lock check failed: %s", exc)
        errors.append("Unable to acquire execution lock.")

    try:
        state_store = BotStateStore(config.state_path)
        state = state_store.load()
        state_store.save(state)
        LOGGER.info("Doctor state check: %s is readable and writable.", config.state_path)
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Doctor state check failed: %s", exc)
        errors.append("State store is not writable.")

    _run_doctor_wallet_checks(config=config, errors=errors, warnings=warnings)

    if errors:
        LOGGER.error("Doctor finished with %s error(s) and %s warning(s).", len(errors), len(warnings))
        return 1

    LOGGER.info("Doctor finished cleanly with %s warning(s).", len(warnings))
    return 0


def _run_doctor_wallet_checks(
    *,
    config: BotConfig,
    errors: list[str],
    warnings: list[str],
) -> None:
    if not config.private_key:
        LOGGER.info("Doctor wallet check skipped: no POLYMARKET_PRIVATE_KEY configured.")
        return
    if config.signature_type != 0 and not config.funder_address:
        LOGGER.info("Doctor wallet check skipped: POLYMARKET_FUNDER_ADDRESS is required for this signature type.")
        return

    live_mode = config.live_enabled and not config.dry_run
    try:
        trader = TradingClobClient(config)
        mode_label = "live auth" if live_mode else "wallet auth"
        LOGGER.info("Doctor %s check: trading client initialized successfully.", mode_label)
        collateral_status = trader.get_collateral_balance_allowance()
        LOGGER.info(
            "Doctor balance check: collateral=%s balance=%s allowance=%s raw_balance=%s raw_allowance=%s",
            collateral_status.get("collateral_address"),
            _format_decimal(collateral_status.get("balance")),
            _format_decimal(collateral_status.get("allowance")),
            collateral_status.get("balance_raw"),
            collateral_status.get("allowance_raw"),
        )
        resolved_amount = _resolve_order_amount(config, collateral_status)
        if resolved_amount is not None and config.order_size_pct > 0:
            LOGGER.info(
                "Doctor dynamic size check: target=%.2f%% of collateral -> order_amount=$%.2f",
                config.order_size_pct,
                float(resolved_amount),
            )
        for message in _collateral_funding_messages(config, collateral_status):
            if live_mode:
                errors.append(message)
                LOGGER.error("Doctor balance error: %s", message)
            else:
                warnings.append(message)
                LOGGER.warning("Doctor balance warning: %s", message)
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Doctor wallet check failed: %s", exc)
        message = "Trading wallet balance check failed."
        if live_mode:
            errors.append(message)
            LOGGER.error("Doctor wallet error: %s", message)
        else:
            warnings.append(message)
            LOGGER.warning("Doctor wallet warning: %s", message)


def _collateral_funding_messages(
    config: BotConfig,
    collateral_status: dict[str, object],
) -> list[str]:
    messages: list[str] = []
    required_amount = _resolve_order_amount(config, collateral_status)
    balance = collateral_status.get("balance")
    allowance = collateral_status.get("allowance")

    if required_amount is None:
        if config.order_size_pct > 0:
            messages.append(
                "Dynamic order sizing is enabled, but the bot could not compute an order amount "
                f"from POLYMARKET_ORDER_SIZE_PCT={config.order_size_pct:.2f}."
            )
        return messages

    if isinstance(balance, Decimal) and balance < required_amount:
        messages.append(
            f"Collateral balance {balance} is below POLYMARKET_ORDER_AMOUNT {required_amount}."
        )
    if isinstance(allowance, Decimal) and allowance < required_amount:
        messages.append(
            f"Collateral allowance {allowance} is below POLYMARKET_ORDER_AMOUNT {required_amount}."
        )
    return messages


def _resolve_order_amount(
    config: BotConfig,
    collateral_status: dict[str, object] | None = None,
) -> Decimal | None:
    if config.order_size_pct <= 0:
        return Decimal(str(config.order_amount))
    if collateral_status is None:
        return None

    balance = collateral_status.get("balance")
    if not isinstance(balance, Decimal):
        return None

    target_amount = (
        balance * Decimal(str(config.order_size_pct)) / PERCENT_DENOMINATOR
    ).quantize(USD_CENT, rounding=ROUND_DOWN)
    if target_amount <= 0:
        return None
    return target_amount


def _describe_order_sizing(config: BotConfig) -> str:
    if config.order_size_pct > 0:
        return f"{config.order_size_pct:.2f}% of collateral"
    return f"${config.order_amount:.2f}"


def execute_cycle(config: BotConfig) -> int:
    now = datetime.now(timezone.utc)
    gamma_client = GammaClient(config.gamma_url)
    state_store = BotStateStore(config.state_path)
    state = state_store.load()

    _refresh_settled_trades(
        config=config,
        gamma_client=gamma_client,
        state_store=state_store,
        state=state,
        now=now,
    )
    _process_claims(
        config=config,
        state_store=state_store,
        state=state,
        now=now,
    )
    _raise_if_loss_streak_triggered(config=config, state=state)

    public_client = PublicClobClient(config.host, config.chain_id)
    markets = gamma_client.fetch_btc_updown_5m_markets(limit=config.market_limit, now=now)
    tradeable_markets = filter_tradeable_markets(
        markets,
        config=config,
        now=now,
        strategy_name=config.strategy_name,
    )
    LOGGER.info(
        "Loaded %s active BTC up/down 5m markets, %s are inside the current %s window.",
        len(markets),
        len(tradeable_markets),
        _describe_strategy_window(config),
    )

    if not tradeable_markets:
        state_store.save(state)
        return 0

    token_metrics = collect_token_metrics(tradeable_markets, public_client)
    for token_id, metrics in token_metrics.items():
        observation = make_observation(now, metrics)
        if observation is not None:
            state_store.append_observation(state, token_id, observation)

    signals = build_trade_signals(
        tradeable_markets,
        token_metrics,
        state,
        config=config,
        now=now,
        strategy_name=config.strategy_name,
    )

    chosen_signal = next(
        (
            signal
            for signal in signals
            if not state_store.traded_recently(
                state,
                condition_id=signal.condition_id,
                now=now,
                cooldown_minutes=config.recent_trade_cooldown_minutes,
            )
        ),
        None,
    )

    if chosen_signal is None:
        LOGGER.info("No eligible trade signal this cycle.")
        state_store.save(state)
        return 0

    order_amount = Decimal(str(config.order_amount))
    trader: TradingClobClient | None = None
    if config.order_size_pct > 0:
        trader = TradingClobClient(config)
        collateral_status = trader.get_collateral_balance_allowance()
        resolved_amount = _resolve_order_amount(config, collateral_status)
        if resolved_amount is None:
            raise RuntimeError(
                "Dynamic order sizing could not compute a valid spend amount from the current "
                f"collateral balance for POLYMARKET_ORDER_SIZE_PCT={config.order_size_pct:.2f}."
            )
        order_amount = resolved_amount
        LOGGER.info(
            "Order sizing | target=%.2f%% of collateral balance=%s -> amount=$%.2f",
            config.order_size_pct,
            _format_decimal(collateral_status.get("balance")),
            float(order_amount),
        )

    _log_signal(chosen_signal, config, order_amount=float(order_amount))

    if config.dry_run or not config.live_enabled:
        LOGGER.info(
            "Dry run only. Would buy $%.2f of %s (%s).",
            float(order_amount),
            chosen_signal.outcome,
            chosen_signal.question,
        )
        state_store.remember_trade(
            state,
            condition_id=chosen_signal.condition_id,
            token_id=chosen_signal.token_id,
            placed_at=now,
            mode="dry-run",
            question=chosen_signal.question,
            outcome=chosen_signal.outcome,
            strategy_name=chosen_signal.strategy_name,
        )
        state_store.save(state)
        return 0

    if trader is None:
        trader = TradingClobClient(config)
    state_store.remember_trade(
        state,
        condition_id=chosen_signal.condition_id,
        token_id=chosen_signal.token_id,
        placed_at=now,
        mode="live-pending",
        question=chosen_signal.question,
        outcome=chosen_signal.outcome,
        strategy_name=chosen_signal.strategy_name,
    )
    state_store.save(state)
    try:
        response = trader.place_market_buy(chosen_signal.token_id, float(order_amount))
    except NoMatchAvailableError:
        state_store.update_trade_mode(
            state,
            condition_id=chosen_signal.condition_id,
            token_id=chosen_signal.token_id,
            placed_at=now,
            mode="live-no-match",
        )
        state_store.save(state)
        LOGGER.warning(
            "Live order skipped: no executable liquidity for $%.2f of %s (%s) at the time of submission.",
            float(order_amount),
            chosen_signal.outcome,
            chosen_signal.question,
        )
        return 0
    LOGGER.info("Live order response: %s", response)
    state_store.update_trade_mode(
        state,
        condition_id=chosen_signal.condition_id,
        token_id=chosen_signal.token_id,
        placed_at=now,
        mode="live",
    )
    state_store.save(state)
    return 0


def _refresh_settled_trades(
    *,
    config: BotConfig,
    gamma_client: GammaClient,
    state_store: BotStateStore,
    state,
    now: datetime,
) -> None:
    unsettled = [
        trade
        for trade in state.recent_trades
        if trade.get("mode") in {"live", "dry-run"} and not trade.get("settled_at")
    ]
    if not unsettled:
        return

    oldest_placed_at = min(
        datetime.fromisoformat(str(trade["placed_at"])).astimezone(timezone.utc)
        for trade in unsettled
    )
    age_seconds = max(0.0, (now - oldest_placed_at).total_seconds())
    lookback_days = max(1, ceil(age_seconds / 86400) + 1)
    limit = max(config.market_limit, len(unsettled) * 2, 50)
    closed_markets = gamma_client.fetch_recent_closed_btc_updown_5m_markets(
        limit=limit,
        days=lookback_days,
        now=now,
    )
    markets_by_condition = {
        market.condition_id: market
        for market in closed_markets
    }
    updated = False

    for trade in unsettled:
        market = markets_by_condition.get(str(trade.get("condition_id") or ""))
        if market is None:
            continue
        settlement_price = next(
            (
                outcome.settlement_price
                for outcome in market.outcomes
                if outcome.token_id == trade.get("token_id")
                and outcome.settlement_price is not None
            ),
            None,
        )
        if settlement_price is None:
            continue
        placed_at = datetime.fromisoformat(str(trade["placed_at"])).astimezone(timezone.utc)
        result = "win" if settlement_price > 0 else "loss"
        changed = state_store.settle_trade(
            state,
            condition_id=str(trade.get("condition_id") or ""),
            token_id=str(trade.get("token_id") or ""),
            placed_at=placed_at,
            settled_at=market.end_time,
            result=result,
            settlement_price=settlement_price,
        )
        updated = changed or updated
        if changed:
            LOGGER.info(
                "Settled prior %s trade | [%s] %s / %s | result=%s",
                trade.get("mode"),
                trade.get("strategy_name", "unknown"),
                trade.get("question", trade.get("condition_id")),
                trade.get("outcome", trade.get("token_id")),
                result,
            )

    if updated:
        state_store.save(state)


def _process_claims(
    *,
    config: BotConfig,
    state_store: BotStateStore,
    state,
    now: datetime,
) -> None:
    if not config.auto_claim_winners:
        return

    submitted_claims = [
        trade
        for trade in state.recent_trades
        if trade.get("claim_state") == "submitted"
        and trade.get("claim_transaction_id")
        and trade.get("mode") == "live"
    ]
    claimable_wins = [
        trade
        for trade in state.recent_trades
        if trade.get("mode") == "live"
        and trade.get("result") == "win"
        and trade.get("settled_at")
        and not trade.get("claim_state")
    ]
    if not submitted_claims and not claimable_wins:
        return

    try:
        redeemer = SafeMarketRedeemer(config)
    except RelayerSetupError as exc:
        LOGGER.warning("Auto-claim skipped: %s", exc)
        return
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Auto-claim setup failed: %s", exc)
        return

    updated = False

    for trade in submitted_claims:
        transaction_id = str(trade.get("claim_transaction_id") or "")
        if not transaction_id:
            continue
        try:
            payload = redeemer.get_transaction_status(transaction_id)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning(
                "Auto-claim status check failed for %s / %s: %s",
                trade.get("question", trade.get("condition_id")),
                trade.get("outcome", trade.get("token_id")),
                exc,
            )
            continue
        if not payload:
            continue
        state_value = str(payload.get("state") or "")
        tx_hash = payload.get("transactionHash") or trade.get("claim_transaction_hash")
        if state_value in {"STATE_MINED", "STATE_CONFIRMED"}:
            changed = state_store.update_trade_fields(
                state,
                condition_id=str(trade.get("condition_id") or ""),
                token_id=str(trade.get("token_id") or ""),
                placed_at=datetime.fromisoformat(str(trade["placed_at"])).astimezone(timezone.utc),
                claim_state="confirmed",
                claim_transaction_hash=tx_hash,
                claimed_at=now.isoformat(),
            )
            updated = changed or updated
            if changed:
                LOGGER.info(
                    "Auto-claim confirmed | %s / %s | tx=%s",
                    trade.get("question", trade.get("condition_id")),
                    trade.get("outcome", trade.get("token_id")),
                    tx_hash,
                )
        elif state_value in {"STATE_FAILED", "STATE_INVALID"}:
            changed = state_store.update_trade_fields(
                state,
                condition_id=str(trade.get("condition_id") or ""),
                token_id=str(trade.get("token_id") or ""),
                placed_at=datetime.fromisoformat(str(trade["placed_at"])).astimezone(timezone.utc),
                claim_state="failed",
                claim_transaction_hash=tx_hash,
                claim_error=state_value,
            )
            updated = changed or updated
            if changed:
                LOGGER.warning(
                    "Auto-claim failed | %s / %s | state=%s tx=%s",
                    trade.get("question", trade.get("condition_id")),
                    trade.get("outcome", trade.get("token_id")),
                    state_value,
                    tx_hash,
                )

    for trade in claimable_wins:
        try:
            submission = redeemer.submit_redeem(
                condition_id=str(trade.get("condition_id") or ""),
                metadata=f"Redeem winning tokens for {trade.get('question', trade.get('condition_id'))}",
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning(
                "Auto-claim submission failed | %s / %s | %s",
                trade.get("question", trade.get("condition_id")),
                trade.get("outcome", trade.get("token_id")),
                exc,
            )
            changed = state_store.update_trade_fields(
                state,
                condition_id=str(trade.get("condition_id") or ""),
                token_id=str(trade.get("token_id") or ""),
                placed_at=datetime.fromisoformat(str(trade["placed_at"])).astimezone(timezone.utc),
                claim_state="failed",
                claim_error=str(exc),
            )
            updated = changed or updated
            continue

        changed = state_store.update_trade_fields(
            state,
            condition_id=str(trade.get("condition_id") or ""),
            token_id=str(trade.get("token_id") or ""),
            placed_at=datetime.fromisoformat(str(trade["placed_at"])).astimezone(timezone.utc),
            claim_state="submitted",
            claim_requested_at=now.isoformat(),
            claim_transaction_id=submission.transaction_id,
            claim_transaction_hash=submission.transaction_hash,
            claim_error=None,
        )
        updated = changed or updated
        if changed:
            LOGGER.info(
                "Auto-claim submitted | %s / %s | transaction_id=%s tx=%s",
                trade.get("question", trade.get("condition_id")),
                trade.get("outcome", trade.get("token_id")),
                submission.transaction_id,
                submission.transaction_hash,
            )

    if updated:
        state_store.save(state)


def _raise_if_loss_streak_triggered(*, config: BotConfig, state) -> None:
    if config.max_consecutive_losses <= 0:
        return

    settled_trades = sorted(
        (
            trade
            for trade in state.recent_trades
            if trade.get("mode") in {"live", "dry-run"}
            and trade.get("result") in {"win", "loss"}
        ),
        key=lambda trade: str(trade.get("placed_at") or ""),
    )
    if not settled_trades:
        return

    loss_streak: list[dict[str, object]] = []
    for trade in reversed(settled_trades):
        if trade.get("result") != "loss":
            break
        loss_streak.append(trade)

    if len(loss_streak) < config.max_consecutive_losses:
        return

    recent_losses = " | ".join(
        f"{trade.get('outcome', trade.get('token_id'))} @ {trade.get('question', trade.get('condition_id'))}"
        for trade in reversed(loss_streak[:3])
    )
    raise StrategyGuardTriggered(
        "Strategy guard triggered: "
        f"{len(loss_streak)} consecutive settled losses reached the configured limit of "
        f"{config.max_consecutive_losses}. "
        f"Most recent losses: {recent_losses}"
    )


def _log_signal(signal: TradeSignal, config: BotConfig, *, order_amount: float) -> None:
    LOGGER.info(
        "Selected [%s] %s / %s | score=%.3f price=%.3f spread=%.3f expiry=%.2fm liquidity=%.2f | %s",
        signal.strategy_name,
        signal.question,
        signal.outcome,
        signal.score,
        signal.price,
        signal.spread,
        signal.minutes_to_expiry,
        signal.liquidity,
        signal.reason,
    )
    LOGGER.info(
        "Mode: strategy=%s dry_run=%s live_enabled=%s amount=$%.2f sizing=%s",
        config.strategy_name,
        config.dry_run,
        config.live_enabled,
        order_amount,
        _describe_order_sizing(config),
    )


def _describe_strategy_window(config: BotConfig) -> str:
    if config.strategy_name == "late_leader":
        lower, upper = late_leader_expiry_bounds_seconds(config)
        return f"{lower}-{upper}s to expiry"
    return f"{config.min_minutes_to_expiry}-{config.max_minutes_to_expiry}m to expiry"


def download_history(
    config: BotConfig,
    *,
    days: int,
    limit: int,
    lookback_minutes: int,
    fidelity: int,
    dataset_dir: Path,
) -> int:
    client = HistoryClient(gamma_url=config.gamma_url, clob_host=config.host)
    store = HistoricalDatasetStore(dataset_dir)
    markets = client.fetch_closed_markets_for_backtest(days=days, limit=limit)
    LOGGER.info("Found %s closed markets in the last %s day(s).", len(markets), days)

    saved = 0
    for market in markets:
        if not market.enable_order_book:
            continue

        end_ts = int(market.end_time.timestamp())
        start_ts = end_ts - (lookback_minutes * 60)
        trades = client.fetch_market_trades(market.condition_id)
        trade_price_history: dict[str, list[dict[str, float]]] = {}
        for trade in trades:
            token_id = trade.get("asset")
            timestamp = trade.get("timestamp")
            price = trade.get("price")
            if token_id is None or timestamp is None or price is None:
                continue
            if int(timestamp) < start_ts or int(timestamp) > end_ts:
                continue
            trade_price_history.setdefault(str(token_id), []).append(
                {"t": int(timestamp), "p": float(price)}
            )

        price_history = {
            outcome.token_id: client.fetch_price_history(
                outcome.token_id,
                start_ts=start_ts,
                end_ts=end_ts,
                fidelity=fidelity,
            )
            for outcome in market.outcomes
        }
        for token_id, rows in trade_price_history.items():
            rows.sort(key=lambda row: row["t"])
            if not price_history.get(token_id):
                price_history[token_id] = rows
        if not any(price_history.values()):
            continue

        bundle = HistoricalMarketBundle(
            market=market,
            price_history=price_history,
            trades=trades,
        )
        path = store.save_bundle(bundle)
        saved += 1
        LOGGER.info("Saved %s", path)

    LOGGER.info("Downloaded %s historical market bundle(s) into %s.", saved, dataset_dir)
    return 0


def run_backtest_command(
    config: BotConfig,
    *,
    dataset_dir: Path,
    limit: int | None,
    strategy_name: str,
    execution_delay_seconds: int,
    slippage_cents: float,
    polygon_gas_cost_usdc: float,
) -> int:
    summary = run_backtest(
        dataset_dir,
        config=config,
        limit=limit,
        strategy_name=strategy_name,
        execution_delay_seconds=execution_delay_seconds,
        slippage_cents=slippage_cents,
        polygon_gas_cost_usdc=polygon_gas_cost_usdc,
    )
    LOGGER.info(
        "Backtest summary | strategy=%s markets=%s trades=%s delay=%ss slippage=%.2fc total_pnl=%.2f fees=%.2f polygon=%.2f win_rate=%.2f%% avg_roi=%.2f%%",
        summary.strategy_name,
        summary.markets_seen,
        len(summary.trades),
        execution_delay_seconds,
        slippage_cents,
        summary.total_pnl,
        summary.total_fees,
        summary.total_polygon_costs,
        summary.win_rate * 100,
        summary.average_roi * 100,
    )
    for trade in summary.trades[:10]:
        LOGGER.info(
            "Trade | %s | %s | %s | entry=%s observed=%.3f fill=%.3f settle=%.1f fee=%.4f polygon=%.4f pnl=%.2f roi=%.2f%%",
            trade.strategy_name,
            trade.question,
            trade.outcome,
            trade.entry_time.isoformat(),
            trade.observed_entry_price,
            trade.entry_price,
            trade.settlement_price,
            trade.fees_paid,
            trade.polygon_cost_paid,
            trade.pnl,
            trade.roi * 100,
        )
    return 0


def run_compare_strategies_command(
    config: BotConfig,
    *,
    dataset_dir: Path,
    limit: int | None,
    execution_delay_seconds: int,
    slippage_cents: float,
    polygon_gas_cost_usdc: float,
) -> int:
    summaries = run_strategy_backtests(
        dataset_dir,
        config=config,
        limit=limit,
        execution_delay_seconds=execution_delay_seconds,
        slippage_cents=slippage_cents,
        polygon_gas_cost_usdc=polygon_gas_cost_usdc,
    )
    markets_seen = summaries[0].markets_seen if summaries else 0
    LOGGER.info(
        "Strategy comparison | markets=%s dataset=%s delay=%ss slippage=%.2fc polygon=%.4f",
        markets_seen,
        dataset_dir,
        execution_delay_seconds,
        slippage_cents,
        polygon_gas_cost_usdc,
    )

    for index, summary in enumerate(summaries, start=1):
        LOGGER.info(
            "%s. %s | trades=%s total_pnl=%.2f fees=%.2f polygon=%.2f win_rate=%.2f%% avg_roi=%.2f%%",
            index,
            summary.strategy_name,
            len(summary.trades),
            summary.total_pnl,
            summary.total_fees,
            summary.total_polygon_costs,
            summary.win_rate * 100,
            summary.average_roi * 100,
        )
    return 0


def run_execution_grid_command(
    config: BotConfig,
    *,
    dataset_dir: Path,
    limit: int | None,
    strategy_name: str,
    delay_grid: str,
    slippage_grid_cents: str,
    polygon_gas_cost_usdc: float,
) -> int:
    delay_values = _parse_int_grid(delay_grid)
    slippage_values = _parse_float_grid(slippage_grid_cents)
    scenarios = run_execution_scenarios(
        dataset_dir,
        config=config,
        strategy_name=strategy_name,
        limit=limit,
        execution_delay_seconds=delay_values,
        slippage_cents=slippage_values,
        polygon_gas_cost_usdc=polygon_gas_cost_usdc,
    )
    markets_seen = scenarios[0].summary.markets_seen if scenarios else 0
    LOGGER.info(
        "Execution grid | strategy=%s markets=%s dataset=%s delays=%s slippage=%s polygon=%.4f",
        strategy_name,
        markets_seen,
        dataset_dir,
        delay_values,
        slippage_values,
        polygon_gas_cost_usdc,
    )
    for index, scenario in enumerate(scenarios, start=1):
        summary = scenario.summary
        LOGGER.info(
            "%s. delay=%ss slippage=%.2fc | trades=%s total_pnl=%.2f fees=%.2f polygon=%.2f win_rate=%.2f%% avg_roi=%.2f%%",
            index,
            scenario.execution_delay_seconds,
            scenario.slippage_cents,
            len(summary.trades),
            summary.total_pnl,
            summary.total_fees,
            summary.total_polygon_costs,
            summary.win_rate * 100,
            summary.average_roi * 100,
        )
    if scenarios:
        best = max(
            scenarios,
            key=lambda scenario: (
                scenario.summary.total_pnl,
                scenario.summary.win_rate,
                len(scenario.summary.trades),
            ),
        )
        LOGGER.info(
            "Best scenario | delay=%ss slippage=%.2fc total_pnl=%.2f trades=%s win_rate=%.2f%% avg_roi=%.2f%%",
            best.execution_delay_seconds,
            best.slippage_cents,
            best.summary.total_pnl,
            len(best.summary.trades),
            best.summary.win_rate * 100,
            best.summary.average_roi * 100,
        )
    return 0


def run_autoresearch_command(
    config: BotConfig,
    *,
    dataset_dir: Path,
    days: int,
    limit: int,
    lookback_minutes: int,
    fidelity: int,
    execution_delay_seconds: int,
    slippage_cents: float,
    polygon_gas_cost_usdc: float,
    skip_download: bool,
    output_dir: Path | None,
) -> int:
    run_autoresearch, format_report, save_report = _load_research_tools()
    report = run_autoresearch(
        config,
        dataset_dir=dataset_dir,
        days=days,
        limit=limit,
        lookback_minutes=lookback_minutes,
        fidelity=fidelity,
        execution_delay_seconds=execution_delay_seconds,
        slippage_cents=slippage_cents,
        polygon_gas_cost_usdc=polygon_gas_cost_usdc,
        skip_download=skip_download,
    )
    text = format_report(report)
    LOGGER.info("\n%s", text)

    if output_dir is not None:
        saved_path = save_report(report, output_dir)
        LOGGER.info("Autoresearch report saved to %s", saved_path)

    return 0


def _parse_int_grid(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _parse_float_grid(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Polymarket 5-minute trader.")
    parser.add_argument(
        "command",
        choices=[
            "run",
            "run-once",
            "doctor",
            "download-history",
            "backtest",
            "compare-strategies",
            "execution-grid",
            "autoresearch",
        ],
        nargs="?",
        default="run-once",
    )
    parser.add_argument("--days", type=int, default=3)
    parser.add_argument("--history-limit", type=int, default=25)
    parser.add_argument("--history-lookback-minutes", type=int, default=90)
    parser.add_argument("--history-fidelity", type=int, default=1)
    parser.add_argument("--dataset-dir", default="data/historical")
    parser.add_argument("--backtest-limit", type=int, default=None)
    parser.add_argument("--execution-delay-seconds", type=int, default=0)
    parser.add_argument("--slippage-cents", type=float, default=0.0)
    parser.add_argument("--delay-grid", default="0,2,5,10")
    parser.add_argument("--slippage-grid-cents", default="0,0.5,1,2")
    parser.add_argument("--polygon-gas-cost-usdc", type=float, default=0.0)
    parser.add_argument("--late-leader-seconds", type=int, default=None)
    parser.add_argument("--late-leader-window-seconds", type=int, default=None)
    parser.add_argument(
        "--strategy",
        default=None,
        choices=list_backtest_strategy_names(),
    )
    parser.add_argument("--verbose", action="store_true")
    # Autoresearch arguments
    parser.add_argument("--research-dir", default=str(DEFAULT_RESEARCH_DIR))
    parser.add_argument("--research-days", type=int, default=DEFAULT_RESEARCH_DAYS)
    parser.add_argument("--research-limit", type=int, default=DEFAULT_RESEARCH_LIMIT)
    parser.add_argument("--research-lookback-minutes", type=int, default=DEFAULT_RESEARCH_LOOKBACK_MINUTES)
    parser.add_argument("--research-fidelity", type=int, default=DEFAULT_RESEARCH_FIDELITY)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--report-output-dir", default=None)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    config = BotConfig.from_env()
    if args.strategy is not None:
        if args.command in {"run", "run-once", "doctor"} and args.strategy in BACKTEST_ONLY_STRATEGIES:
            parser.error(
                f"Strategy '{args.strategy}' is backtest-only. Use 'late_leader' or another live strategy for runtime commands."
            )
        config = replace(config, strategy_name=args.strategy)
    if args.late_leader_seconds is not None:
        config = replace(config, late_leader_horizon_seconds=args.late_leader_seconds)
    if args.late_leader_window_seconds is not None:
        config = replace(config, late_leader_window_seconds=args.late_leader_window_seconds)
    configure_logging(
        args.verbose,
        log_path=config.log_path,
        log_backup_count=config.log_backup_count,
    )
    selected_strategy = config.strategy_name

    errors, warnings = validate_config(config)
    for warning in warnings:
        LOGGER.warning("Config warning: %s", warning)
    if args.command in {"run", "run-once"} and errors:
        for error in errors:
            LOGGER.error("Config error: %s", error)
        return 1

    if args.command == "run-once":
        try:
            with execution_lock(config.lock_path):
                return execute_cycle(config)
        except StrategyGuardTriggered as exc:
            LOGGER.error("%s", exc)
            return 1
    if args.command == "doctor":
        return run_doctor_command(config)
    if args.command == "download-history":
        return download_history(
            config,
            days=args.days,
            limit=args.history_limit,
            lookback_minutes=args.history_lookback_minutes,
            fidelity=args.history_fidelity,
            dataset_dir=Path(args.dataset_dir),
        )
    if args.command == "backtest":
        return run_backtest_command(
            config,
            dataset_dir=Path(args.dataset_dir),
            limit=args.backtest_limit,
            strategy_name=selected_strategy,
            execution_delay_seconds=args.execution_delay_seconds,
            slippage_cents=args.slippage_cents,
            polygon_gas_cost_usdc=args.polygon_gas_cost_usdc,
        )
    if args.command == "compare-strategies":
        return run_compare_strategies_command(
            config,
            dataset_dir=Path(args.dataset_dir),
            limit=args.backtest_limit,
            execution_delay_seconds=args.execution_delay_seconds,
            slippage_cents=args.slippage_cents,
            polygon_gas_cost_usdc=args.polygon_gas_cost_usdc,
        )
    if args.command == "execution-grid":
        return run_execution_grid_command(
            config,
            dataset_dir=Path(args.dataset_dir),
            limit=args.backtest_limit,
            strategy_name=selected_strategy,
            delay_grid=args.delay_grid,
            slippage_grid_cents=args.slippage_grid_cents,
            polygon_gas_cost_usdc=args.polygon_gas_cost_usdc,
        )
    if args.command == "autoresearch":
        return run_autoresearch_command(
            config,
            dataset_dir=Path(args.research_dir),
            days=args.research_days,
            limit=args.research_limit,
            lookback_minutes=args.research_lookback_minutes,
            fidelity=args.research_fidelity,
            execution_delay_seconds=args.execution_delay_seconds or DEFAULT_EXEC_DELAY,
            slippage_cents=args.slippage_cents or DEFAULT_SLIPPAGE_CENTS,
            polygon_gas_cost_usdc=args.polygon_gas_cost_usdc or DEFAULT_GAS_COST,
            skip_download=args.skip_download,
            output_dir=Path(args.report_output_dir) if args.report_output_dir else Path(args.research_dir) / "reports",
        )

    try:
        with execution_lock(config.lock_path):
            while True:
                try:
                    execute_cycle(config)
                    time.sleep(config.scan_interval_seconds)
                except StrategyGuardTriggered as exc:
                    LOGGER.error("%s", exc)
                    return 1
                except KeyboardInterrupt:
                    LOGGER.info("Stopping bot loop.")
                    return 0
                except Exception as exc:  # noqa: BLE001
                    LOGGER.exception("Cycle failed: %s", exc)
                    time.sleep(config.scan_interval_seconds)
    except RuntimeError as exc:
        LOGGER.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
