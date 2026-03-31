from datetime import datetime, timezone

from polymarket_5_min_trader.gamma import (
    _align_to_five_minutes,
    _btc_updown_slug,
    _decode_list_field,
    GammaClient,
)


def test_decode_list_field_handles_json_strings() -> None:
    assert _decode_list_field('["Yes", "No"]') == ["Yes", "No"]
    assert _decode_list_field(["a", "b"]) == ["a", "b"]
    assert _decode_list_field(None) == []


def test_normalize_market_decodes_stringified_arrays() -> None:
    client = GammaClient("https://gamma-api.polymarket.com")
    market = client._normalize_market(
        {
            "conditionId": "0xabc",
            "question": "Will BTC finish above 85k?",
            "slug": "btc-above-85k",
            "endDate": "2026-03-30T20:05:00Z",
            "liquidity": "1234.5",
            "volume": "99.8",
            "enableOrderBook": True,
            "clobTokenIds": '["1","2"]',
            "outcomes": '["Yes","No"]',
            "outcomePrices": '["0.41","0.59"]',
        }
    )
    assert market is not None
    assert market.condition_id == "0xabc"
    assert market.end_time == datetime(2026, 3, 30, 20, 5, tzinfo=timezone.utc)
    assert market.outcomes[0].token_id == "1"
    assert market.outcomes[1].outcome == "No"


def test_btc_updown_slug_uses_five_minute_alignment() -> None:
    instant = datetime(2026, 3, 30, 20, 8, 44, tzinfo=timezone.utc)
    aligned = _align_to_five_minutes(instant)
    assert aligned == datetime(2026, 3, 30, 20, 5, tzinfo=timezone.utc)
    assert _btc_updown_slug(aligned) == "btc-updown-5m-1774901100"


def test_fetch_recent_closed_btc_updown_5m_markets_uses_5m_tag_stream() -> None:
    class FakeResponse:
        def __init__(self, payload: list[dict]) -> None:
            self._payload = payload

        def raise_for_status(self) -> None:
            return None

        def json(self) -> list[dict]:
            return self._payload

    class FakeSession:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def get(self, url: str, params: dict, timeout: float) -> FakeResponse:
            self.calls.append({"url": url, "params": params, "timeout": timeout})
            return FakeResponse(
                [
                    {
                        "conditionId": "0xbtc",
                        "question": "Bitcoin Up or Down - March 30, 4:00PM-4:05PM ET",
                        "slug": "btc-updown-5m-1774900800",
                        "endDate": "2026-03-30T20:05:00Z",
                        "closed": True,
                        "enableOrderBook": True,
                        "clobTokenIds": '["1","2"]',
                        "outcomes": '["Up","Down"]',
                        "outcomePrices": '["1","0"]',
                    },
                    {
                        "conditionId": "0xeth",
                        "question": "Ethereum Up or Down - March 30, 4:00PM-4:05PM ET",
                        "slug": "eth-updown-5m-1774900800",
                        "endDate": "2026-03-30T20:05:00Z",
                        "closed": True,
                        "enableOrderBook": True,
                        "clobTokenIds": '["3","4"]',
                        "outcomes": '["Up","Down"]',
                        "outcomePrices": '["1","0"]',
                    },
                    {
                        "conditionId": "0xold",
                        "question": "Bitcoin Up or Down - March 20, 4:00PM-4:05PM ET",
                        "slug": "btc-updown-5m-1774036800",
                        "endDate": "2026-03-20T20:05:00Z",
                        "closed": True,
                        "enableOrderBook": True,
                        "clobTokenIds": '["5","6"]',
                        "outcomes": '["Up","Down"]',
                        "outcomePrices": '["1","0"]',
                    },
                ]
            )

    client = GammaClient("https://gamma-api.polymarket.com")
    client.session = FakeSession()

    markets = client.fetch_recent_closed_btc_updown_5m_markets(
        limit=10,
        days=3,
        now=datetime(2026, 3, 30, 20, 10, tzinfo=timezone.utc),
    )

    assert len(markets) == 1
    assert markets[0].condition_id == "0xbtc"
    assert client.session.calls[0]["params"]["tag_id"] == "102892"
