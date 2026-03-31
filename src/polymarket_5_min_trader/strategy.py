from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from polymarket_5_min_trader.config import BotConfig
from polymarket_5_min_trader.models import (
    Market,
    Observation,
    OutcomeMarket,
    TokenMetrics,
    TradeSignal,
)
from polymarket_5_min_trader.state import BotState

DEFAULT_STRATEGY_NAME = "momentum_follow"
LATE_LEADER_STRATEGY_NAME = "late_leader"


@dataclass(frozen=True)
class OutcomeFeatures:
    market: Market
    outcome: OutcomeMarket
    metrics: TokenMetrics
    history: tuple[Observation, ...]
    price_points: tuple[float, ...]
    baseline_price: float
    delta: float
    last_step_delta: float
    prior_step_delta: float
    minutes_to_expiry: float
    expiry_weight: float
    liquidity_weight: float
    spread_penalty: float
    peak_price: float
    trough_price: float
    range_size: float
    range_position: float
    rebound_from_trough: float
    pullback_from_peak: float
    positive_step_ratio: float
    opposite_outcome: OutcomeMarket | None
    opposite_metrics: TokenMetrics | None
    opposite_delta: float | None
    opposite_last_step_delta: float | None


@dataclass(frozen=True)
class StrategyDefinition:
    name: str
    description: str


def market_depth(market: Market) -> float:
    return market.liquidity if market.liquidity > 0 else market.volume


def filter_tradeable_markets(
    markets: list[Market],
    *,
    config: BotConfig,
    now: datetime,
    strategy_name: str = DEFAULT_STRATEGY_NAME,
) -> list[Market]:
    filtered: list[Market] = []
    for market in markets:
        seconds_to_expiry = (market.end_time - now).total_seconds()
        if strategy_name == LATE_LEADER_STRATEGY_NAME:
            min_seconds, max_seconds = late_leader_expiry_bounds_seconds(config)
            if seconds_to_expiry < min_seconds:
                continue
            if seconds_to_expiry > max_seconds:
                continue
        else:
            minutes_to_expiry = seconds_to_expiry / 60
            if minutes_to_expiry < config.min_minutes_to_expiry:
                continue
            if minutes_to_expiry > config.max_minutes_to_expiry:
                continue
        if market_depth(market) < config.min_liquidity:
            continue
        if not market.enable_order_book:
            continue
        filtered.append(market)
    return filtered


def build_trade_signals(
    markets: list[Market],
    token_metrics: dict[str, TokenMetrics],
    state: BotState,
    *,
    config: BotConfig,
    now: datetime,
    strategy_name: str = DEFAULT_STRATEGY_NAME,
) -> list[TradeSignal]:
    if strategy_name == LATE_LEADER_STRATEGY_NAME:
        return _build_late_leader_signals(
            markets,
            token_metrics,
            config=config,
            now=now,
        )
    if strategy_name not in _STRATEGY_BUILDERS:
        available = ", ".join(list_strategy_names())
        raise ValueError(f"Unknown strategy '{strategy_name}'. Available strategies: {available}")

    builder = _STRATEGY_BUILDERS[strategy_name]
    signals: list[TradeSignal] = []
    history_cutoff = now - timedelta(minutes=config.momentum_window_minutes)

    for market in markets:
        for outcome in market.outcomes:
            features = _build_outcome_features(
                market,
                outcome,
                token_metrics,
                state,
                config=config,
                now=now,
                history_cutoff=history_cutoff,
            )
            if features is None:
                continue

            signal = builder(features, config)
            if signal is not None:
                signals.append(signal)

    return sorted(signals, key=lambda signal: signal.score, reverse=True)


def list_strategy_names() -> list[str]:
    return [definition.name for definition in STRATEGY_DEFINITIONS]


def strategy_definitions() -> list[StrategyDefinition]:
    return list(STRATEGY_DEFINITIONS)


def effective_late_leader_window_seconds(config: BotConfig) -> int:
    return max(1, config.scan_interval_seconds)


def late_leader_expiry_bounds_seconds(config: BotConfig) -> tuple[int, int]:
    window = effective_late_leader_window_seconds(config)
    lower = max(0, config.late_leader_horizon_seconds - window)
    upper = config.late_leader_horizon_seconds + window
    return lower, upper


def make_observation(
    now: datetime,
    metrics: TokenMetrics,
) -> Observation | None:
    if metrics.midpoint is None:
        return None
    return Observation(
        recorded_at=now.astimezone(timezone.utc),
        price=metrics.midpoint,
        spread=metrics.spread,
    )


def _build_outcome_features(
    market: Market,
    outcome: OutcomeMarket,
    token_metrics: dict[str, TokenMetrics],
    state: BotState,
    *,
    config: BotConfig,
    now: datetime,
    history_cutoff: datetime,
) -> OutcomeFeatures | None:
    metrics = token_metrics.get(outcome.token_id)
    if metrics is None or metrics.midpoint is None or metrics.spread is None:
        return None
    if metrics.midpoint < config.min_midpoint or metrics.midpoint > config.max_midpoint:
        return None
    if metrics.spread > config.max_spread:
        return None

    history = tuple(
        item
        for item in state.observations.get(outcome.token_id, [])
        if item.recorded_at >= history_cutoff
    )
    if len(history) < config.min_observations:
        return None

    minutes_to_expiry = (market.end_time - now).total_seconds() / 60
    target_distance = abs(minutes_to_expiry - config.target_minutes_to_expiry)
    expiry_weight = max(
        0.1,
        1 - (target_distance / max(1, config.max_minutes_to_expiry - config.min_minutes_to_expiry)),
    )
    liquidity_weight = min(2.0, max(0.5, market_depth(market) / max(config.min_liquidity, 1.0)))
    spread_penalty = max(0.1, 1 - (metrics.spread / config.max_spread))

    price_points = _price_points(history, current_price=metrics.midpoint, now=now)
    previous_price = price_points[-2] if len(price_points) >= 2 else price_points[-1]
    prior_reference = price_points[-3] if len(price_points) >= 3 else previous_price
    delta = price_points[-1] - price_points[0]
    last_step_delta = price_points[-1] - previous_price
    prior_step_delta = previous_price - prior_reference
    peak_price = max(price_points)
    trough_price = min(price_points)
    range_size = peak_price - trough_price
    range_position = (
        (price_points[-1] - trough_price) / range_size
        if range_size > 0
        else 0.5
    )
    rebound_from_trough = price_points[-1] - trough_price
    pullback_from_peak = peak_price - price_points[-1]
    positive_step_ratio = _positive_step_ratio(price_points)

    opposite_outcome = next(
        (candidate for candidate in market.outcomes if candidate.token_id != outcome.token_id),
        None,
    )
    opposite_metrics: TokenMetrics | None = None
    opposite_delta: float | None = None
    opposite_last_step_delta: float | None = None
    if opposite_outcome is not None:
        opposite_metrics = token_metrics.get(opposite_outcome.token_id)
        opposite_history = tuple(
            item
            for item in state.observations.get(opposite_outcome.token_id, [])
            if item.recorded_at >= history_cutoff
        )
        if (
            opposite_metrics is not None
            and opposite_metrics.midpoint is not None
            and len(opposite_history) >= config.min_observations
        ):
            opposite_points = _price_points(
                opposite_history,
                current_price=opposite_metrics.midpoint,
                now=now,
            )
            opposite_previous = opposite_points[-2] if len(opposite_points) >= 2 else opposite_points[-1]
            opposite_delta = opposite_points[-1] - opposite_points[0]
            opposite_last_step_delta = opposite_points[-1] - opposite_previous

    return OutcomeFeatures(
        market=market,
        outcome=outcome,
        metrics=metrics,
        history=history,
        price_points=price_points,
        baseline_price=history[0].price,
        delta=delta,
        last_step_delta=last_step_delta,
        prior_step_delta=prior_step_delta,
        minutes_to_expiry=minutes_to_expiry,
        expiry_weight=expiry_weight,
        liquidity_weight=liquidity_weight,
        spread_penalty=spread_penalty,
        peak_price=peak_price,
        trough_price=trough_price,
        range_size=range_size,
        range_position=range_position,
        rebound_from_trough=rebound_from_trough,
        pullback_from_peak=pullback_from_peak,
        positive_step_ratio=positive_step_ratio,
        opposite_outcome=opposite_outcome,
        opposite_metrics=opposite_metrics,
        opposite_delta=opposite_delta,
        opposite_last_step_delta=opposite_last_step_delta,
    )


def _base_score_multiplier(features: OutcomeFeatures) -> float:
    return features.expiry_weight * features.spread_penalty * features.liquidity_weight


def _price_points(
    history: tuple[Observation, ...],
    *,
    current_price: float,
    now: datetime,
) -> tuple[float, ...]:
    points = tuple(item.price for item in history)
    if not points:
        return (current_price,)
    if history[-1].recorded_at == now.astimezone(timezone.utc) and history[-1].price == current_price:
        return points
    return points + (current_price,)


def _positive_step_ratio(price_points: tuple[float, ...]) -> float:
    if len(price_points) < 2:
        return 0.0
    positive_steps = sum(
        1
        for previous, current in zip(price_points, price_points[1:], strict=False)
        if current > previous
    )
    return positive_steps / (len(price_points) - 1)


def _make_signal(
    *,
    strategy_name: str,
    features: OutcomeFeatures,
    score: float,
    reason: str,
) -> TradeSignal:
    return TradeSignal(
        strategy_name=strategy_name,
        condition_id=features.market.condition_id,
        token_id=features.outcome.token_id,
        question=features.market.question,
        outcome=features.outcome.outcome,
        price=features.metrics.midpoint or 0.0,
        spread=features.metrics.spread or 0.0,
        minutes_to_expiry=features.minutes_to_expiry,
        momentum_delta=features.delta,
        liquidity=market_depth(features.market),
        score=score,
        reason=reason,
    )


def _build_momentum_follow_signal(
    features: OutcomeFeatures,
    config: BotConfig,
) -> TradeSignal | None:
    if features.delta < config.min_edge:
        return None
    score = features.delta * 100 * _base_score_multiplier(features)
    reason = (
        f"Momentum +{features.delta:.3f} over {config.momentum_window_minutes}m "
        f"with {features.metrics.spread:.3f} spread."
    )
    return _make_signal(
        strategy_name="momentum_follow",
        features=features,
        score=score,
        reason=reason,
    )


def _build_relative_strength_signal(
    features: OutcomeFeatures,
    config: BotConfig,
) -> TradeSignal | None:
    if features.opposite_delta is None:
        return None

    strength_gap = features.delta - features.opposite_delta
    if features.delta < config.min_edge:
        return None
    if strength_gap < config.min_edge * 2:
        return None
    if features.last_step_delta <= 0:
        return None
    if features.metrics.midpoint > 0.72:
        return None

    payout_weight = max(0.25, 1 - features.metrics.midpoint)
    score = strength_gap * 110 * _base_score_multiplier(features) * payout_weight
    reason = (
        f"Relative strength gap {strength_gap:.3f}: {features.outcome.outcome} kept leading "
        f"while {features.opposite_outcome.outcome if features.opposite_outcome else 'peer'} lagged."
    )
    return _make_signal(
        strategy_name="relative_strength",
        features=features,
        score=score,
        reason=reason,
    )


def _build_cheap_momentum_signal(
    features: OutcomeFeatures,
    config: BotConfig,
) -> TradeSignal | None:
    if features.delta < config.min_edge:
        return None
    if features.last_step_delta <= 0:
        return None
    if features.metrics.midpoint > 0.58:
        return None

    payout_weight = max(0.35, 1 - features.metrics.midpoint)
    score = features.delta * 100 * _base_score_multiplier(features) * payout_weight * 1.5
    reason = (
        f"Cheap momentum +{features.delta:.3f} with entry {features.metrics.midpoint:.3f} "
        f"and positive last step {features.last_step_delta:.3f}."
    )
    return _make_signal(
        strategy_name="cheap_momentum",
        features=features,
        score=score,
        reason=reason,
    )


def _build_mean_reversion_signal(
    features: OutcomeFeatures,
    config: BotConfig,
) -> TradeSignal | None:
    if features.opposite_delta is None or features.opposite_last_step_delta is None:
        return None
    if features.delta > -config.min_edge:
        return None
    if features.opposite_delta < config.min_edge:
        return None
    if features.last_step_delta <= 0:
        return None
    if features.opposite_last_step_delta >= 0:
        return None
    if features.metrics.midpoint > 0.60:
        return None

    reversion_gap = features.opposite_delta - features.delta
    if reversion_gap < config.min_edge * 2:
        return None

    payout_weight = max(0.35, 1 - features.metrics.midpoint)
    score = reversion_gap * 85 * _base_score_multiplier(features) * payout_weight
    reason = (
        f"Mean reversion setup: {features.outcome.outcome} is still {features.delta:.3f} off the window "
        f"open but just bounced {features.last_step_delta:.3f} while the other side rolled over."
    )
    return _make_signal(
        strategy_name="mean_reversion",
        features=features,
        score=score,
        reason=reason,
    )


def _build_short_term_flip_signal(
    features: OutcomeFeatures,
    config: BotConfig,
) -> TradeSignal | None:
    if features.opposite_last_step_delta is None:
        return None
    if features.last_step_delta <= config.min_edge / 3:
        return None
    if features.opposite_last_step_delta >= -(config.min_edge / 3):
        return None
    if features.metrics.midpoint < 0.30 or features.metrics.midpoint > 0.70:
        return None

    flip_gap = features.last_step_delta - features.opposite_last_step_delta
    if flip_gap < config.min_edge:
        return None

    payout_weight = max(0.3, 1 - features.metrics.midpoint)
    score = flip_gap * 175 * _base_score_multiplier(features) * payout_weight
    reason = (
        f"Short-term flip {flip_gap:.3f}: {features.outcome.outcome} turned up "
        f"{features.last_step_delta:.3f} as the other side faded."
    )
    return _make_signal(
        strategy_name="short_term_flip",
        features=features,
        score=score,
        reason=reason,
    )


def _build_steady_trend_signal(
    features: OutcomeFeatures,
    config: BotConfig,
) -> TradeSignal | None:
    if features.delta < config.min_edge:
        return None
    if features.last_step_delta <= 0:
        return None
    if features.prior_step_delta < 0:
        return None
    if features.positive_step_ratio < 0.6:
        return None
    if features.metrics.midpoint > 0.72:
        return None

    payout_weight = max(0.25, 1 - features.metrics.midpoint)
    score = (
        (features.delta + features.last_step_delta)
        * 95
        * _base_score_multiplier(features)
        * payout_weight
        * (0.5 + features.positive_step_ratio)
    )
    reason = (
        f"Steady trend with {features.positive_step_ratio:.0%} positive steps and "
        f"fresh push of {features.last_step_delta:.3f}."
    )
    return _make_signal(
        strategy_name="steady_trend",
        features=features,
        score=score,
        reason=reason,
    )


def _build_breakout_continuation_signal(
    features: OutcomeFeatures,
    config: BotConfig,
) -> TradeSignal | None:
    if features.delta < config.min_edge:
        return None
    if features.last_step_delta <= config.min_edge / 3:
        return None
    if features.range_size < config.min_edge * 3:
        return None
    if features.range_position < 0.78:
        return None
    if features.pullback_from_peak > max(0.03, features.range_size * 0.25):
        return None
    if features.metrics.midpoint > 0.72:
        return None
    if features.opposite_last_step_delta is not None and features.opposite_last_step_delta > 0:
        return None

    payout_weight = max(0.25, 1 - features.metrics.midpoint)
    score = (
        (features.delta + features.range_size * 0.5 + features.last_step_delta)
        * 100
        * _base_score_multiplier(features)
        * payout_weight
    )
    reason = (
        f"Breakout continuation: {features.outcome.outcome} is in the top "
        f"{features.range_position:.0%} of its range and still pushing higher."
    )
    return _make_signal(
        strategy_name="breakout_continuation",
        features=features,
        score=score,
        reason=reason,
    )


def _build_pullback_reclaim_signal(
    features: OutcomeFeatures,
    config: BotConfig,
) -> TradeSignal | None:
    if features.delta < config.min_edge:
        return None
    if features.prior_step_delta >= 0:
        return None
    if features.last_step_delta <= 0:
        return None
    if features.pullback_from_peak < config.min_edge / 2:
        return None
    if features.pullback_from_peak > 0.12:
        return None
    if features.range_position < 0.45 or features.range_position > 0.85:
        return None
    if features.range_size < config.min_edge * 3:
        return None
    if features.metrics.midpoint > 0.68:
        return None

    payout_weight = max(0.25, 1 - features.metrics.midpoint)
    score = (
        (features.delta + features.last_step_delta + features.pullback_from_peak)
        * 92
        * _base_score_multiplier(features)
        * payout_weight
    )
    reason = (
        f"Pullback reclaim: {features.outcome.outcome} dipped {features.pullback_from_peak:.3f} "
        f"off the high and just turned back up."
    )
    return _make_signal(
        strategy_name="pullback_reclaim",
        features=features,
        score=score,
        reason=reason,
    )


def _build_underdog_reclaim_signal(
    features: OutcomeFeatures,
    config: BotConfig,
) -> TradeSignal | None:
    if features.delta > config.min_edge:
        return None
    if features.last_step_delta <= config.min_edge / 3:
        return None
    if features.rebound_from_trough < config.min_edge * 2:
        return None
    if features.range_size < config.min_edge * 4:
        return None
    if features.range_position < 0.38:
        return None
    if features.metrics.midpoint > 0.48:
        return None
    if features.opposite_last_step_delta is not None and features.opposite_last_step_delta >= 0:
        return None

    payout_weight = max(0.35, 1 - features.metrics.midpoint)
    score = (
        (features.rebound_from_trough + max(0.0, -features.delta))
        * 88
        * _base_score_multiplier(features)
        * payout_weight
    )
    reason = (
        f"Underdog reclaim: {features.outcome.outcome} bounced {features.rebound_from_trough:.3f} "
        f"off the low while still priced at {features.metrics.midpoint:.3f}."
    )
    return _make_signal(
        strategy_name="underdog_reclaim",
        features=features,
        score=score,
        reason=reason,
    )


def _build_late_leader_signals(
    markets: list[Market],
    token_metrics: dict[str, TokenMetrics],
    *,
    config: BotConfig,
    now: datetime,
) -> list[TradeSignal]:
    signals: list[TradeSignal] = []
    min_seconds, max_seconds = late_leader_expiry_bounds_seconds(config)
    for market in markets:
        seconds_to_expiry = (market.end_time - now).total_seconds()
        if seconds_to_expiry < min_seconds or seconds_to_expiry > max_seconds:
            continue

        ranked_outcomes: list[tuple[OutcomeMarket, TokenMetrics]] = []
        for outcome in market.outcomes:
            metrics = token_metrics.get(outcome.token_id)
            if metrics is None or metrics.midpoint is None or metrics.spread is None:
                continue
            if metrics.spread > config.max_spread:
                continue
            ranked_outcomes.append((outcome, metrics))

        if len(ranked_outcomes) < 2:
            continue

        ranked_outcomes.sort(
            key=lambda item: (item[1].midpoint or 0.0, -(item[1].spread or 0.0)),
            reverse=True,
        )
        leader_outcome, leader_metrics = ranked_outcomes[0]
        runner_up_outcome, runner_up_metrics = ranked_outcomes[1]
        leader_price = leader_metrics.midpoint or 0.0
        runner_up_price = runner_up_metrics.midpoint or 0.0
        price_gap = leader_price - runner_up_price
        if price_gap <= 0:
            continue

        timing_error = abs(seconds_to_expiry - config.late_leader_horizon_seconds)
        timing_window = effective_late_leader_window_seconds(config)
        timing_weight = max(0.25, 1 - (timing_error / timing_window))
        spread_penalty = max(0.1, 1 - ((leader_metrics.spread or 0.0) / config.max_spread))
        liquidity_weight = min(2.0, max(0.5, market_depth(market) / max(config.min_liquidity, 1.0)))
        score = price_gap * 100 * timing_weight * spread_penalty * liquidity_weight
        reason = (
            f"Late leader by {price_gap:.3f} with {seconds_to_expiry:.0f}s to expiry "
            f"(target {config.late_leader_horizon_seconds}s)."
        )
        signals.append(
            TradeSignal(
                strategy_name=LATE_LEADER_STRATEGY_NAME,
                condition_id=market.condition_id,
                token_id=leader_outcome.token_id,
                question=market.question,
                outcome=leader_outcome.outcome,
                price=leader_price,
                spread=leader_metrics.spread or 0.0,
                minutes_to_expiry=seconds_to_expiry / 60,
                momentum_delta=price_gap,
                liquidity=market_depth(market),
                score=score,
                reason=reason,
            )
        )

    return sorted(signals, key=lambda signal: signal.score, reverse=True)


STRATEGY_DEFINITIONS = (
    StrategyDefinition(
        name=LATE_LEADER_STRATEGY_NAME,
        description="Buy the currently leading side very close to expiry.",
    ),
    StrategyDefinition(
        name="momentum_follow",
        description="Buy the side with positive full-window momentum.",
    ),
    StrategyDefinition(
        name="relative_strength",
        description="Buy the leading side only when it is beating the opposite token decisively.",
    ),
    StrategyDefinition(
        name="cheap_momentum",
        description="Buy momentum only when the entry still offers better upside.",
    ),
    StrategyDefinition(
        name="mean_reversion",
        description="Fade the early move once the losing side starts bouncing back.",
    ),
    StrategyDefinition(
        name="short_term_flip",
        description="Trade the most recent micro-reversal near the middle of the price range.",
    ),
    StrategyDefinition(
        name="steady_trend",
        description="Buy the side that has climbed consistently instead of spiking once.",
    ),
    StrategyDefinition(
        name="breakout_continuation",
        description="Buy a side pressing the top of its 5-minute range into the close.",
    ),
    StrategyDefinition(
        name="pullback_reclaim",
        description="Buy a trend that dips and then turns back up before expiry.",
    ),
    StrategyDefinition(
        name="underdog_reclaim",
        description="Buy the cheap side when it springs off the session low late.",
    ),
)

_STRATEGY_BUILDERS = {
    "momentum_follow": _build_momentum_follow_signal,
    "relative_strength": _build_relative_strength_signal,
    "cheap_momentum": _build_cheap_momentum_signal,
    "mean_reversion": _build_mean_reversion_signal,
    "short_term_flip": _build_short_term_flip_signal,
    "steady_trend": _build_steady_trend_signal,
    "breakout_continuation": _build_breakout_continuation_signal,
    "pullback_reclaim": _build_pullback_reclaim_signal,
    "underdog_reclaim": _build_underdog_reclaim_signal,
}
