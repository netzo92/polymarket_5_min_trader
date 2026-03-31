from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

from polymarket_5_min_trader.config import BotConfig
from polymarket_5_min_trader.history import HistoricalDatasetStore, HistoricalMarketBundle
from polymarket_5_min_trader.models import FeeSchedule, Market, TokenMetrics, TradeSignal
from polymarket_5_min_trader.state import BotState, BotStateStore
from polymarket_5_min_trader.strategy import (
    DEFAULT_STRATEGY_NAME,
    build_trade_signals,
    filter_tradeable_markets,
    list_strategy_names,
    make_observation,
)

BACKTEST_ONLY_STRATEGIES = {
    "late_leader_60s": 60,
    "late_leader_30s": 30,
    "late_leader_15s": 15,
    "late_leader_5s": 5,
}


@dataclass(frozen=True)
class SimulatedTrade:
    strategy_name: str
    condition_id: str
    question: str
    outcome: str
    entry_time: datetime
    observed_entry_price: float
    entry_price: float
    settlement_price: float
    amount: float
    fees_paid: float
    polygon_cost_paid: float
    pnl: float
    roi: float
    score: float


@dataclass(frozen=True)
class BacktestSummary:
    strategy_name: str
    markets_seen: int
    trades: list[SimulatedTrade]

    @property
    def total_pnl(self) -> float:
        return sum(trade.pnl for trade in self.trades)

    @property
    def total_fees(self) -> float:
        return sum(trade.fees_paid for trade in self.trades)

    @property
    def total_polygon_costs(self) -> float:
        return sum(trade.polygon_cost_paid for trade in self.trades)

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        winners = sum(1 for trade in self.trades if trade.pnl > 0)
        return winners / len(self.trades)

    @property
    def average_roi(self) -> float:
        if not self.trades:
            return 0.0
        return sum(trade.roi for trade in self.trades) / len(self.trades)


@dataclass(frozen=True)
class ExecutionScenarioSummary:
    strategy_name: str
    execution_delay_seconds: int
    slippage_cents: float
    polygon_gas_cost_usdc: float
    summary: BacktestSummary


def list_backtest_strategy_names() -> list[str]:
    return list_strategy_names() + list(BACKTEST_ONLY_STRATEGIES)


def run_backtest(
    dataset_dir: Path,
    *,
    config: BotConfig,
    limit: int | None = None,
    strategy_name: str = DEFAULT_STRATEGY_NAME,
    execution_delay_seconds: int = 0,
    slippage_cents: float = 0.0,
    polygon_gas_cost_usdc: float = 0.0,
) -> BacktestSummary:
    bundles = load_backtest_bundles(dataset_dir, limit=limit)
    return run_backtest_on_bundles(
        bundles,
        config=config,
        strategy_name=strategy_name,
        execution_delay_seconds=execution_delay_seconds,
        slippage_cents=slippage_cents,
        polygon_gas_cost_usdc=polygon_gas_cost_usdc,
    )


def run_strategy_backtests(
    dataset_dir: Path,
    *,
    config: BotConfig,
    limit: int | None = None,
    strategy_names: list[str] | None = None,
    execution_delay_seconds: int = 0,
    slippage_cents: float = 0.0,
    polygon_gas_cost_usdc: float = 0.0,
) -> list[BacktestSummary]:
    bundles = load_backtest_bundles(dataset_dir, limit=limit)
    names = strategy_names or list_backtest_strategy_names()
    summaries = [
        run_backtest_on_bundles(
            bundles,
            config=config,
            strategy_name=strategy_name,
            execution_delay_seconds=execution_delay_seconds,
            slippage_cents=slippage_cents,
            polygon_gas_cost_usdc=polygon_gas_cost_usdc,
        )
        for strategy_name in names
    ]
    return sorted(
        summaries,
        key=lambda summary: (summary.total_pnl, summary.win_rate, len(summary.trades)),
        reverse=True,
    )


def run_execution_scenarios(
    dataset_dir: Path,
    *,
    config: BotConfig,
    strategy_name: str,
    limit: int | None = None,
    execution_delay_seconds: list[int] | None = None,
    slippage_cents: list[float] | None = None,
    polygon_gas_cost_usdc: float = 0.0,
) -> list[ExecutionScenarioSummary]:
    bundles = load_backtest_bundles(dataset_dir, limit=limit)
    delay_values = execution_delay_seconds or [0, 2, 5, 10]
    slippage_values = slippage_cents or [0.0, 0.5, 1.0, 2.0]
    scenarios: list[ExecutionScenarioSummary] = []
    for delay_seconds in delay_values:
        for slippage_value in slippage_values:
            summary = run_backtest_on_bundles(
                bundles,
                config=config,
                strategy_name=strategy_name,
                execution_delay_seconds=delay_seconds,
                slippage_cents=slippage_value,
                polygon_gas_cost_usdc=polygon_gas_cost_usdc,
            )
            scenarios.append(
                ExecutionScenarioSummary(
                    strategy_name=strategy_name,
                    execution_delay_seconds=delay_seconds,
                    slippage_cents=slippage_value,
                    polygon_gas_cost_usdc=polygon_gas_cost_usdc,
                    summary=summary,
                )
            )
    return scenarios


def load_backtest_bundles(
    dataset_dir: Path,
    *,
    limit: int | None = None,
) -> list[HistoricalMarketBundle]:
    store = HistoricalDatasetStore(dataset_dir)
    return store.load_bundles(limit=limit)


def run_backtest_on_bundles(
    bundles: list[HistoricalMarketBundle],
    *,
    config: BotConfig,
    strategy_name: str,
    execution_delay_seconds: int,
    slippage_cents: float,
    polygon_gas_cost_usdc: float,
) -> BacktestSummary:
    trades: list[SimulatedTrade] = []
    for bundle in bundles:
        trade = simulate_market(
            bundle,
            config=config,
            strategy_name=strategy_name,
            execution_delay_seconds=execution_delay_seconds,
            slippage_cents=slippage_cents,
            polygon_gas_cost_usdc=polygon_gas_cost_usdc,
        )
        if trade is not None:
            trades.append(trade)
    return BacktestSummary(strategy_name=strategy_name, markets_seen=len(bundles), trades=trades)


def simulate_market(
    bundle: HistoricalMarketBundle,
    *,
    config: BotConfig,
    strategy_name: str = DEFAULT_STRATEGY_NAME,
    execution_delay_seconds: int = 0,
    slippage_cents: float = 0.0,
    polygon_gas_cost_usdc: float = 0.0,
) -> SimulatedTrade | None:
    if strategy_name in BACKTEST_ONLY_STRATEGIES:
        return _simulate_late_leader_market(
            bundle,
            config=config,
            strategy_name=strategy_name,
            horizon_seconds=BACKTEST_ONLY_STRATEGIES[strategy_name],
            execution_delay_seconds=execution_delay_seconds,
            slippage_cents=slippage_cents,
            polygon_gas_cost_usdc=polygon_gas_cost_usdc,
        )

    market = bundle.market
    if not market.closed:
        return None

    settlement_by_token = {
        outcome.token_id: outcome.settlement_price
        for outcome in market.outcomes
        if outcome.settlement_price is not None
    }
    if not settlement_by_token:
        return None

    events: list[tuple[int, str, float]] = []
    for token_id, rows in bundle.price_history.items():
        for row in rows:
            price = row.get("p")
            timestamp = row.get("t")
            if price is None or timestamp is None:
                continue
            events.append((int(timestamp), token_id, float(price)))

    if not events:
        return None

    events.sort(key=lambda item: item[0])
    state = BotState()
    state_store = BotStateStore(Path("data/_backtest_state.json"))
    latest_metrics: dict[str, TokenMetrics] = {}
    traded_signal: TradeSignal | None = None
    pending_signal: TradeSignal | None = None
    pending_signal_at: datetime | None = None
    traded_at: datetime | None = None
    traded_observed_entry_price: float | None = None
    traded_entry_price: float | None = None

    for timestamp, token_id, price in events:
        now = datetime.fromtimestamp(timestamp, tz=timezone.utc)
        latest_metrics[token_id] = TokenMetrics(midpoint=price, spread=0.0)
        observation = make_observation(now, latest_metrics[token_id])
        if observation is not None:
            state_store.append_observation(state, token_id, observation)

        if pending_signal is not None and pending_signal_at is not None:
            eligible_timestamp = int(pending_signal_at.timestamp()) + execution_delay_seconds
            if timestamp >= eligible_timestamp and token_id == pending_signal.token_id:
                traded_signal = pending_signal
                traded_at = now
                traded_observed_entry_price = price
                traded_entry_price = _apply_buy_slippage(price, slippage_cents=slippage_cents)
                break

        if traded_signal is not None:
            continue

        if pending_signal is not None:
            continue

        tradeable_markets = filter_tradeable_markets([market], config=config, now=now)
        if not tradeable_markets:
            continue

        signals = build_trade_signals(
            tradeable_markets,
            latest_metrics,
            state,
            config=config,
            now=now,
            strategy_name=strategy_name,
        )
        if not signals:
            continue

        chosen_signal = signals[0]
        if execution_delay_seconds == 0:
            traded_signal = chosen_signal
            traded_at = now
            traded_observed_entry_price = latest_metrics[chosen_signal.token_id].midpoint
            traded_entry_price = _apply_buy_slippage(
                traded_observed_entry_price,
                slippage_cents=slippage_cents,
            )
            break
        pending_signal = chosen_signal
        pending_signal_at = now

    if (
        traded_signal is None
        or traded_at is None
        or traded_observed_entry_price is None
        or traded_entry_price is None
    ):
        return None

    settlement_price = settlement_by_token.get(traded_signal.token_id)
    if settlement_price is None or traded_entry_price <= 0:
        return None

    return _settle_trade(
        market=market,
        strategy_name=strategy_name,
        condition_id=market.condition_id,
        question=market.question,
        outcome=traded_signal.outcome,
        entry_time=traded_at,
        observed_entry_price=traded_observed_entry_price,
        entry_price=traded_entry_price,
        settlement_price=settlement_price,
        amount=config.order_amount,
        polygon_gas_cost_usdc=polygon_gas_cost_usdc,
        score=traded_signal.score,
    )


def _simulate_late_leader_market(
    bundle: HistoricalMarketBundle,
    *,
    config: BotConfig,
    strategy_name: str,
    horizon_seconds: int,
    execution_delay_seconds: int,
    slippage_cents: float,
    polygon_gas_cost_usdc: float,
) -> SimulatedTrade | None:
    market = bundle.market
    if not market.closed:
        return None

    settlement_by_token = {
        outcome.token_id: outcome.settlement_price
        for outcome in market.outcomes
        if outcome.settlement_price is not None
    }
    if not settlement_by_token:
        return None

    end_ts = int(market.end_time.timestamp())
    cutoff_ts = end_ts - horizon_seconds
    latest_by_token: dict[str, tuple[int, float]] = {}
    for token_id, rows in bundle.price_history.items():
        latest: tuple[int, float] | None = None
        for row in rows:
            price = row.get("p")
            timestamp = row.get("t")
            if price is None or timestamp is None:
                continue
            ts = int(timestamp)
            if ts > cutoff_ts:
                continue
            candidate = (ts, float(price))
            if latest is None or ts >= latest[0]:
                latest = candidate
        if latest is not None:
            latest_by_token[token_id] = latest

    if len(latest_by_token) < 2:
        return None

    ranked = sorted(
        latest_by_token.items(),
        key=lambda item: (item[1][1], item[1][0]),
        reverse=True,
    )
    token_id, (entry_ts, observed_entry_price) = ranked[0]
    runner_up_price = ranked[1][1][1]
    price_gap = observed_entry_price - runner_up_price
    if price_gap < 0.20:
        return None
    if observed_entry_price <= 0:
        return None

    if execution_delay_seconds > 0:
        fill = _find_first_fill_after(
            bundle.price_history.get(token_id, []),
            earliest_timestamp=cutoff_ts + execution_delay_seconds,
        )
        if fill is None:
            return None
        entry_ts, observed_entry_price = fill
    entry_price = _apply_buy_slippage(observed_entry_price, slippage_cents=slippage_cents)

    outcome_name = next(
        (
            outcome.outcome
            for outcome in market.outcomes
            if outcome.token_id == token_id
        ),
        token_id,
    )
    settlement_price = settlement_by_token.get(token_id)
    if settlement_price is None:
        return None

    return _settle_trade(
        market=market,
        strategy_name=strategy_name,
        condition_id=market.condition_id,
        question=market.question,
        outcome=outcome_name,
        entry_time=datetime.fromtimestamp(entry_ts, tz=timezone.utc),
        observed_entry_price=observed_entry_price,
        entry_price=entry_price,
        settlement_price=settlement_price,
        amount=config.order_amount,
        polygon_gas_cost_usdc=polygon_gas_cost_usdc,
        score=entry_price,
    )


def _settle_trade(
    *,
    market: Market,
    strategy_name: str,
    condition_id: str,
    question: str,
    outcome: str,
    entry_time: datetime,
    observed_entry_price: float,
    entry_price: float,
    settlement_price: float,
    amount: float,
    polygon_gas_cost_usdc: float,
    score: float,
) -> SimulatedTrade:
    fee_schedule = resolve_fee_schedule(market)
    fee_usdc = calculate_buy_fee_usdc(
        amount=amount,
        entry_price=entry_price,
        fee_schedule=fee_schedule,
    )
    shares = max(0.0, (amount - fee_usdc) / entry_price)
    exit_value = shares * settlement_price
    pnl = exit_value - amount - polygon_gas_cost_usdc
    roi = pnl / amount
    return SimulatedTrade(
        strategy_name=strategy_name,
        condition_id=condition_id,
        question=question,
        outcome=outcome,
        entry_time=entry_time,
        observed_entry_price=observed_entry_price,
        entry_price=entry_price,
        settlement_price=settlement_price,
        amount=amount,
        fees_paid=fee_usdc,
        polygon_cost_paid=polygon_gas_cost_usdc,
        pnl=pnl,
        roi=roi,
        score=score,
    )


def resolve_fee_schedule(market: Market) -> FeeSchedule | None:
    if market.fees_enabled is False:
        return None
    if market.fee_schedule is not None:
        return market.fee_schedule
    if market.fee_type == "crypto_fees_v2" or market.question.lower().startswith("bitcoin up or down -"):
        return FeeSchedule(rate=0.072, exponent=1.0, taker_only=True, rebate_rate=0.2)
    return None


def calculate_buy_fee_usdc(
    *,
    amount: float,
    entry_price: float,
    fee_schedule: FeeSchedule | None,
) -> float:
    if fee_schedule is None or entry_price <= 0:
        return 0.0
    fee_rate = Decimal(str(fee_schedule.rate))
    price = Decimal(str(entry_price))
    trade_value = Decimal(str(amount))
    exponent = Decimal(str(fee_schedule.exponent))
    raw_fee = trade_value * fee_rate * (price * (Decimal("1") - price)) ** exponent
    rounded_fee = raw_fee.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
    return float(rounded_fee)


def _apply_buy_slippage(
    observed_entry_price: float,
    *,
    slippage_cents: float,
) -> float:
    adjusted_price = observed_entry_price + (slippage_cents / 100.0)
    return min(0.9999, max(0.0001, adjusted_price))


def _find_first_fill_after(
    rows: list[dict[str, float]],
    *,
    earliest_timestamp: int,
) -> tuple[int, float] | None:
    candidates = sorted(
        (
            (int(row["t"]), float(row["p"]))
            for row in rows
            if row.get("t") is not None and row.get("p") is not None and int(row["t"]) >= earliest_timestamp
        ),
        key=lambda item: item[0],
    )
    return candidates[0] if candidates else None
