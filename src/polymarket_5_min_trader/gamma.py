from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import requests

from polymarket_5_min_trader.models import FeeSchedule, Market, OutcomeMarket

BTC_UPDOWN_5M_TAG_ID = "102892"


def _decode_list_field(value: object) -> list[object]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return [value]
        return parsed if isinstance(parsed, list) else [parsed]
    return [value]


class GammaClient:
    def __init__(self, base_url: str, timeout: float = 10.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "polymarket-5m-bot/0.1"})
        self.timeout = timeout

    def fetch_active_markets(self, limit: int = 500, page_size: int = 200) -> list[Market]:
        return self.fetch_markets(
            limit=limit,
            page_size=page_size,
            filters={"active": "true", "closed": "false"},
        )

    def fetch_bitcoin_markets(
        self,
        *,
        include_closed: bool,
        limit: int = 500,
        queries: tuple[str, ...] = ("Bitcoin price on", "Bitcoin above"),
    ) -> list[Market]:
        seen_slugs: set[str] = set()
        event_slugs: list[str] = []
        for query in queries:
            for item in self.search_events(query):
                slug = str(item.get("slug") or "").strip()
                if not slug or slug in seen_slugs:
                    continue
                title = str(item.get("title") or "")
                if "Bitcoin" not in title:
                    continue
                is_closed = bool(item.get("closed"))
                if not include_closed and is_closed:
                    continue
                event_slugs.append(slug)
                seen_slugs.add(slug)

        markets: list[Market] = []
        seen_conditions: set[str] = set()
        for slug in event_slugs:
            for item in self.fetch_event_markets(slug):
                market = self._normalize_market(item)
                if market is None:
                    continue
                if market.condition_id in seen_conditions:
                    continue
                if not include_closed and market.closed:
                    continue
                if include_closed and not market.closed:
                    continue
                if not _is_bitcoin_price_market(market.question):
                    continue
                markets.append(market)
                seen_conditions.add(market.condition_id)
                if len(markets) >= limit:
                    return markets
        return markets

    def fetch_btc_updown_5m_markets(
        self,
        *,
        limit: int,
        now: datetime | None = None,
        look_back_intervals: int = 2,
        look_ahead_intervals: int = 12,
    ) -> list[Market]:
        anchor = _align_to_five_minutes(now or datetime.now(timezone.utc))
        candidate_times = [
            anchor + timedelta(minutes=5 * offset)
            for offset in range(-look_back_intervals, look_ahead_intervals + 1)
        ]
        markets: list[Market] = []
        seen_conditions: set[str] = set()
        for start_time in candidate_times:
            slug = _btc_updown_slug(start_time)
            for item in self.fetch_event_markets(slug):
                market = self._normalize_market(item)
                if market is None or market.closed:
                    continue
                if market.condition_id in seen_conditions:
                    continue
                if not _is_btc_updown_5m_market(market.question):
                    continue
                markets.append(market)
                seen_conditions.add(market.condition_id)
                if len(markets) >= limit:
                    return markets
        return markets

    def fetch_recent_closed_btc_updown_5m_markets(
        self,
        *,
        limit: int,
        days: int,
        now: datetime | None = None,
    ) -> list[Market]:
        now_utc = now or datetime.now(timezone.utc)
        cutoff = now_utc - timedelta(days=days)
        markets = self._fetch_recent_closed_btc_updown_5m_by_tag(limit=limit, cutoff=cutoff)
        if len(markets) >= limit or markets:
            return markets[:limit]

        cursor = _align_to_five_minutes(now_utc) - timedelta(minutes=5)
        seen_conditions: set[str] = set()
        seen_conditions.update(market.condition_id for market in markets)
        while cursor >= cutoff and len(markets) < limit:
            slug = _btc_updown_slug(cursor)
            for item in self.fetch_event_markets(slug):
                market = self._normalize_market(item)
                if market is None or not market.closed:
                    continue
                if market.condition_id in seen_conditions:
                    continue
                if not _is_btc_updown_5m_market(market.question):
                    continue
                markets.append(market)
                seen_conditions.add(market.condition_id)
                if len(markets) >= limit:
                    break
            cursor -= timedelta(minutes=5)
        return markets

    def _fetch_recent_closed_btc_updown_5m_by_tag(
        self,
        *,
        limit: int,
        cutoff: datetime,
        page_size: int = 200,
    ) -> list[Market]:
        markets: list[Market] = []
        seen_conditions: set[str] = set()
        offset = 0

        while len(markets) < limit:
            batch_limit = min(page_size, max(1, limit - len(markets)))
            response = self.session.get(
                f"{self.base_url}/markets",
                params={
                    "tag_id": BTC_UPDOWN_5M_TAG_ID,
                    "closed": "true",
                    "order": "endDate",
                    "ascending": "false",
                    "limit": batch_limit,
                    "offset": offset,
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, list) or not payload:
                break

            reached_cutoff = False
            for item in payload:
                market = self._normalize_market(item)
                if market is None:
                    continue
                if market.end_time < cutoff:
                    reached_cutoff = True
                    break
                if not market.closed:
                    continue
                if market.condition_id in seen_conditions:
                    continue
                if not _is_btc_updown_5m_market(market.question):
                    continue
                markets.append(market)
                seen_conditions.add(market.condition_id)
                if len(markets) >= limit:
                    break

            if reached_cutoff or len(payload) < batch_limit:
                break
            offset += len(payload)

        return markets

    def fetch_closed_markets(
        self,
        *,
        limit: int = 500,
        page_size: int = 200,
        end_date_min: str | None = None,
        end_date_max: str | None = None,
        order: str = "endDate",
        ascending: bool = False,
    ) -> list[Market]:
        filters: dict[str, str] = {
            "closed": "true",
            "order": order,
            "ascending": str(ascending).lower(),
        }
        if end_date_min:
            filters["end_date_min"] = end_date_min
        if end_date_max:
            filters["end_date_max"] = end_date_max
        return self.fetch_markets(limit=limit, page_size=page_size, filters=filters)

    def fetch_markets(
        self,
        *,
        limit: int,
        page_size: int,
        filters: dict[str, str],
    ) -> list[Market]:
        markets: list[Market] = []
        offset = 0
        while len(markets) < limit:
            batch_limit = min(page_size, limit - len(markets))
            params = {
                **filters,
                "limit": batch_limit,
                "offset": offset,
            }
            response = self.session.get(
                f"{self.base_url}/markets",
                params=params,
                timeout=self.timeout,
            )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, list) or not payload:
                break
            normalized = [self._normalize_market(item) for item in payload]
            markets.extend([market for market in normalized if market is not None])
            if len(payload) < batch_limit:
                break
            offset += len(payload)
        return markets

    def search_events(self, query: str) -> list[dict[str, object]]:
        response = self.session.get(
            f"{self.base_url}/public-search",
            params={"q": query},
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        events = payload.get("events", [])
        return events if isinstance(events, list) else []

    def fetch_event_markets(self, slug: str) -> list[dict[str, object]]:
        response = self.session.get(
            f"{self.base_url}/events",
            params={"slug": slug, "limit": 10},
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list):
            return []
        for event in payload:
            if str(event.get("slug") or "") == slug:
                markets = event.get("markets", [])
                return markets if isinstance(markets, list) else []
        return []

    def _normalize_market(self, item: dict[str, object]) -> Market | None:
        condition_id = str(item.get("conditionId") or "").strip()
        question = str(item.get("question") or "").strip()
        slug = str(item.get("slug") or "").strip()
        end_date = str(item.get("endDate") or "").strip()
        if not condition_id or not question or not end_date:
            return None

        try:
            end_time = datetime.fromisoformat(end_date.replace("Z", "+00:00")).astimezone(timezone.utc)
        except ValueError:
            return None

        token_ids = [str(value) for value in _decode_list_field(item.get("clobTokenIds"))]
        outcomes = [str(value) for value in _decode_list_field(item.get("outcomes"))]
        prices = _decode_list_field(item.get("outcomePrices"))
        is_closed = bool(item.get("closed"))

        if len(token_ids) != len(outcomes) or not token_ids:
            return None

        parsed_outcomes: list[OutcomeMarket] = []
        for index, token_id in enumerate(token_ids):
            raw_price = prices[index] if index < len(prices) else None
            try:
                last_price = float(raw_price) if raw_price is not None else None
            except (TypeError, ValueError):
                last_price = None
            settlement_price = None
            if is_closed and last_price in {0.0, 1.0}:
                settlement_price = last_price
            parsed_outcomes.append(
                OutcomeMarket(
                    outcome=outcomes[index],
                    token_id=token_id,
                    last_traded_price=last_price,
                    settlement_price=settlement_price,
                )
            )

        try:
            liquidity = float(item.get("liquidity") or 0.0)
        except (TypeError, ValueError):
            liquidity = 0.0
        try:
            volume = float(item.get("volume") or 0.0)
        except (TypeError, ValueError):
            volume = 0.0
        fee_schedule = _parse_fee_schedule(item.get("feeSchedule"))
        fee_type = str(item.get("feeType") or "").strip() or None
        fees_enabled = item.get("feesEnabled")

        return Market(
            condition_id=condition_id,
            question=question,
            slug=slug,
            end_time=end_time,
            liquidity=liquidity,
            volume=volume,
            enable_order_book=bool(item.get("enableOrderBook")),
            closed=is_closed,
            outcomes=tuple(parsed_outcomes),
            fees_enabled=bool(fees_enabled) if fees_enabled is not None else None,
            fee_schedule=fee_schedule,
            fee_type=fee_type,
        )


def _is_bitcoin_price_market(question: str) -> bool:
    lowered = question.lower()
    return lowered.startswith("will the price of bitcoin be between") or lowered.startswith(
        "will bitcoin be above"
    )


def _is_btc_updown_5m_market(question: str) -> bool:
    return question.lower().startswith("bitcoin up or down -")


def _align_to_five_minutes(value: datetime) -> datetime:
    value = value.astimezone(timezone.utc).replace(second=0, microsecond=0)
    aligned_minute = value.minute - (value.minute % 5)
    return value.replace(minute=aligned_minute)


def _btc_updown_slug(start_time: datetime) -> str:
    timestamp = int(start_time.astimezone(timezone.utc).timestamp())
    return f"btc-updown-5m-{timestamp}"


def _parse_fee_schedule(value: object) -> FeeSchedule | None:
    if not isinstance(value, dict):
        return None
    try:
        rate = float(value["rate"])
        exponent = float(value["exponent"])
    except (KeyError, TypeError, ValueError):
        return None
    rebate_rate_raw = value.get("rebateRate")
    try:
        rebate_rate = float(rebate_rate_raw) if rebate_rate_raw is not None else None
    except (TypeError, ValueError):
        rebate_rate = None
    return FeeSchedule(
        rate=rate,
        exponent=exponent,
        taker_only=bool(value.get("takerOnly", True)),
        rebate_rate=rebate_rate,
    )
