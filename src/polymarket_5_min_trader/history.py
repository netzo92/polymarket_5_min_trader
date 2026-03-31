from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

from polymarket_5_min_trader.gamma import GammaClient
from polymarket_5_min_trader.models import FeeSchedule, Market, OutcomeMarket


@dataclass(frozen=True)
class HistoricalMarketBundle:
    market: Market
    price_history: dict[str, list[dict[str, float]]]
    trades: list[dict]


class HistoryClient:
    def __init__(
        self,
        *,
        gamma_url: str,
        clob_host: str,
        data_api_url: str = "https://data-api.polymarket.com",
        timeout: float = 15.0,
    ) -> None:
        self.gamma = GammaClient(gamma_url, timeout=timeout)
        self.clob_host = clob_host.rstrip("/")
        self.data_api_url = data_api_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "polymarket-5m-bot/0.1"})
        self.timeout = timeout

    def fetch_closed_markets_for_backtest(
        self,
        *,
        days: int,
        limit: int,
    ) -> list[Market]:
        return self.gamma.fetch_recent_closed_btc_updown_5m_markets(limit=limit, days=days)

    def fetch_price_history(
        self,
        token_id: str,
        *,
        start_ts: int,
        end_ts: int,
        fidelity: int,
    ) -> list[dict[str, float]]:
        response = self.session.get(
            f"{self.clob_host}/prices-history",
            params={
                "market": token_id,
                "startTs": start_ts,
                "endTs": end_ts,
                "interval": "max",
                "fidelity": fidelity,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        history = payload.get("history", [])
        return history if isinstance(history, list) else []

    def fetch_market_trades(self, condition_id: str, *, limit: int = 10000) -> list[dict]:
        response = self.session.get(
            f"{self.data_api_url}/trades",
            params={"market": condition_id, "limit": limit, "offset": 0},
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        return payload if isinstance(payload, list) else []


class HistoricalDatasetStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def save_bundle(self, bundle: HistoricalMarketBundle) -> Path:
        path = self.root / f"{bundle.market.condition_id}.json"
        payload = {
            "market": _market_to_dict(bundle.market),
            "price_history": bundle.price_history,
            "trades": bundle.trades,
        }
        path.write_text(json.dumps(payload, indent=2))
        return path

    def load_bundles(self, *, limit: int | None = None) -> list[HistoricalMarketBundle]:
        bundles: list[HistoricalMarketBundle] = []
        paths = sorted(self.root.glob("*.json"))
        if limit is not None:
            paths = paths[:limit]
        for path in paths:
            payload = json.loads(path.read_text())
            bundles.append(
                HistoricalMarketBundle(
                    market=_market_from_dict(payload["market"]),
                    price_history=payload.get("price_history", {}),
                    trades=payload.get("trades", []),
                )
            )
        return bundles


def _market_to_dict(market: Market) -> dict:
    return {
        "condition_id": market.condition_id,
        "question": market.question,
        "slug": market.slug,
        "end_time": market.end_time.isoformat(),
        "liquidity": market.liquidity,
        "volume": market.volume,
        "enable_order_book": market.enable_order_book,
        "closed": market.closed,
        "fees_enabled": market.fees_enabled,
        "fee_type": market.fee_type,
        "fee_schedule": _fee_schedule_to_dict(market.fee_schedule),
        "outcomes": [
            {
                "outcome": outcome.outcome,
                "token_id": outcome.token_id,
                "last_traded_price": outcome.last_traded_price,
                "settlement_price": outcome.settlement_price,
            }
            for outcome in market.outcomes
        ],
    }


def _market_from_dict(payload: dict) -> Market:
    return Market(
        condition_id=payload["condition_id"],
        question=payload["question"],
        slug=payload["slug"],
        end_time=datetime.fromisoformat(payload["end_time"]).astimezone(timezone.utc),
        liquidity=float(payload["liquidity"]),
        volume=float(payload["volume"]),
        enable_order_book=bool(payload["enable_order_book"]),
        closed=bool(payload.get("closed", False)),
        fees_enabled=(
            bool(payload["fees_enabled"])
            if payload.get("fees_enabled") is not None
            else None
        ),
        fee_type=str(payload.get("fee_type") or "").strip() or None,
        fee_schedule=_fee_schedule_from_dict(payload.get("fee_schedule")),
        outcomes=tuple(
            OutcomeMarket(
                outcome=item["outcome"],
                token_id=item["token_id"],
                last_traded_price=float(item["last_traded_price"])
                if item.get("last_traded_price") is not None
                else None,
                settlement_price=float(item["settlement_price"])
                if item.get("settlement_price") is not None
                else None,
            )
            for item in payload["outcomes"]
        ),
    )


def _fee_schedule_to_dict(fee_schedule: FeeSchedule | None) -> dict | None:
    if fee_schedule is None:
        return None
    return {
        "rate": fee_schedule.rate,
        "exponent": fee_schedule.exponent,
        "taker_only": fee_schedule.taker_only,
        "rebate_rate": fee_schedule.rebate_rate,
    }


def _fee_schedule_from_dict(payload: dict | None) -> FeeSchedule | None:
    if not isinstance(payload, dict):
        return None
    try:
        rate = float(payload["rate"])
        exponent = float(payload["exponent"])
    except (KeyError, TypeError, ValueError):
        return None
    rebate_rate_raw = payload.get("rebate_rate")
    try:
        rebate_rate = float(rebate_rate_raw) if rebate_rate_raw is not None else None
    except (TypeError, ValueError):
        rebate_rate = None
    return FeeSchedule(
        rate=rate,
        exponent=exponent,
        taker_only=bool(payload.get("taker_only", True)),
        rebate_rate=rebate_rate,
    )
