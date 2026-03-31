from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from polymarket_5_min_trader.models import Observation


@dataclass
class BotState:
    observations: dict[str, list[Observation]] = field(default_factory=dict)
    recent_trades: list[dict[str, object]] = field(default_factory=list)


class BotStateStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> BotState:
        if not self.path.exists():
            return BotState()
        payload = json.loads(self.path.read_text())
        observations = {
            token_id: [
                Observation(
                    recorded_at=datetime.fromisoformat(item["recorded_at"]).astimezone(timezone.utc),
                    price=float(item["price"]),
                    spread=float(item["spread"]) if item["spread"] is not None else None,
                )
                for item in items
            ]
            for token_id, items in payload.get("observations", {}).items()
        }
        recent_trades = list(payload.get("recent_trades", []))
        return BotState(observations=observations, recent_trades=recent_trades)

    def save(self, state: BotState) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "observations": {
                token_id: [
                    {
                        "recorded_at": observation.recorded_at.isoformat(),
                        "price": observation.price,
                        "spread": observation.spread,
                    }
                    for observation in observations
                ]
                for token_id, observations in state.observations.items()
            },
            "recent_trades": state.recent_trades,
        }
        temp_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
        temp_path.write_text(json.dumps(payload, indent=2))
        temp_path.replace(self.path)

    def append_observation(
        self,
        state: BotState,
        token_id: str,
        observation: Observation,
        retention_minutes: int = 90,
    ) -> None:
        horizon = observation.recorded_at - timedelta(minutes=retention_minutes)
        series = state.observations.setdefault(token_id, [])
        series.append(observation)
        state.observations[token_id] = [item for item in series if item.recorded_at >= horizon][-200:]

    def remember_trade(
        self,
        state: BotState,
        *,
        condition_id: str,
        token_id: str,
        placed_at: datetime,
        mode: str,
        question: str | None = None,
        outcome: str | None = None,
        strategy_name: str | None = None,
    ) -> None:
        trade: dict[str, object] = {
            "condition_id": condition_id,
            "token_id": token_id,
            "placed_at": placed_at.isoformat(),
            "mode": mode,
        }
        if question:
            trade["question"] = question
        if outcome:
            trade["outcome"] = outcome
        if strategy_name:
            trade["strategy_name"] = strategy_name
        state.recent_trades.append(trade)
        cutoff = placed_at - timedelta(hours=12)
        state.recent_trades = [
            trade
            for trade in state.recent_trades
            if datetime.fromisoformat(trade["placed_at"]).astimezone(timezone.utc) >= cutoff
        ]

    def update_trade_mode(
        self,
        state: BotState,
        *,
        condition_id: str,
        token_id: str,
        placed_at: datetime,
        mode: str,
    ) -> bool:
        return self.update_trade_fields(
            state,
            condition_id=condition_id,
            token_id=token_id,
            placed_at=placed_at,
            mode=mode,
        )

    def update_trade_fields(
        self,
        state: BotState,
        *,
        condition_id: str,
        token_id: str,
        placed_at: datetime,
        **updates: object,
    ) -> bool:
        placed_at_iso = placed_at.isoformat()
        for trade in reversed(state.recent_trades):
            if trade.get("condition_id") != condition_id:
                continue
            if trade.get("token_id") != token_id:
                continue
            if trade.get("placed_at") != placed_at_iso:
                continue
            trade.update(updates)
            return True
        return False

    def settle_trade(
        self,
        state: BotState,
        *,
        condition_id: str,
        token_id: str,
        placed_at: datetime,
        settled_at: datetime,
        result: str,
        settlement_price: float,
    ) -> bool:
        placed_at_iso = placed_at.isoformat()
        for trade in reversed(state.recent_trades):
            if trade.get("condition_id") != condition_id:
                continue
            if trade.get("token_id") != token_id:
                continue
            if trade.get("placed_at") != placed_at_iso:
                continue
            trade["settled_at"] = settled_at.isoformat()
            trade["result"] = result
            trade["settlement_price"] = settlement_price
            return True
        return False

    def traded_recently(
        self,
        state: BotState,
        *,
        condition_id: str,
        now: datetime,
        cooldown_minutes: int,
    ) -> bool:
        cutoff = now - timedelta(minutes=cooldown_minutes)
        for trade in state.recent_trades:
            if trade.get("condition_id") != condition_id:
                continue
            placed_at = datetime.fromisoformat(trade["placed_at"]).astimezone(timezone.utc)
            if placed_at >= cutoff:
                return True
        return False
