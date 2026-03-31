from __future__ import annotations

import logging
from decimal import Decimal, InvalidOperation

from eth_account import Account
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import (
    ApiCreds,
    AssetType,
    BalanceAllowanceParams,
    MarketOrderArgs,
    OrderType,
)

from polymarket_5_min_trader.config import BotConfig
from polymarket_5_min_trader.models import TokenMetrics

LOGGER = logging.getLogger(__name__)
COLLATERAL_TOKEN_DECIMALS = 6
COLLATERAL_TOKEN_SCALE = Decimal("1").scaleb(COLLATERAL_TOKEN_DECIMALS)
_DERIVED_API_CREDS_CACHE: dict[tuple[str, int, int, str], ApiCreds] = {}


class LiveTradingSetupError(RuntimeError):
    """Raised when Polymarket rejects the wallet or credential setup."""


class NoMatchAvailableError(RuntimeError):
    """Raised when the order book cannot satisfy a market order right now."""


def _wallet_setup_guidance(config: BotConfig) -> str:
    if config.signature_type == 0:
        return (
            "This bot is configured for a standalone EOA wallet "
            "(POLYMARKET_SIGNATURE_TYPE=0). That flow does not require a "
            "Polymarket.com account, but the same wallet must hold the trading "
            "funds and Polygon gas."
        )
    return (
        "This bot is configured for a Polymarket proxy wallet "
        f"(POLYMARKET_SIGNATURE_TYPE={config.signature_type}). Log in to "
        "Polymarket.com once to deploy that proxy wallet before deriving API "
        "credentials, then set POLYMARKET_FUNDER_ADDRESS to the proxy wallet "
        "shown in the Polymarket profile or settings UI."
    )


def _region_guidance() -> str:
    return (
        "If the wallet setup is correct and live orders are still rejected, "
        "check whether Polymarket blocks trading from your current region."
    )


def _approval_guidance() -> str:
    return (
        "On a first live trade, Polymarket may also require token approval for "
        "the Exchange contract. The docs say this is usually handled once in the "
        "Polymarket UI, or with the CTF contract's setApprovalForAll() flow."
    )


def _looks_like_setup_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "account",
            "api key",
            "approval",
            "allowance",
            "credential",
            "funder",
            "proxy",
            "region",
            "restricted",
            "blocked",
            "unauthorized",
            "forbidden",
            "signature",
        )
    )


def _looks_like_invalid_api_key(exc: Exception) -> bool:
    message = str(exc).lower()
    return "invalid api key" in message or (
        "unauthorized" in message and "api key" in message
    )


def _looks_like_no_match(exc: Exception) -> bool:
    message = str(exc).strip().lower()
    return message == "no match" or "fully filled or killed" in message


def _live_setup_error(config: BotConfig, *, phase: str, exc: Exception) -> LiveTradingSetupError:
    detail = str(exc).strip() or exc.__class__.__name__
    guidance_parts = [_wallet_setup_guidance(config)]
    lowered = detail.lower()
    if "approval" in lowered or "allowance" in lowered:
        guidance_parts.append(_approval_guidance())
    guidance_parts.append(_region_guidance())
    return LiveTradingSetupError(
        f"Polymarket {phase} failed: {detail} {' '.join(guidance_parts)}"
    )


class PublicClobClient:
    def __init__(self, host: str, chain_id: int) -> None:
        self.client = ClobClient(host=host, chain_id=chain_id)

    def get_token_metrics(self, token_id: str) -> TokenMetrics:
        midpoint: float | None = None
        spread: float | None = None

        try:
            midpoint_response = self.client.get_midpoint(token_id)
            midpoint = float(midpoint_response["mid"]) if midpoint_response and midpoint_response.get("mid") else None
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("midpoint lookup failed for %s: %s", token_id, exc)

        try:
            spread_response = self.client.get_spread(token_id)
            spread = float(spread_response["spread"]) if spread_response and spread_response.get("spread") else None
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("spread lookup failed for %s: %s", token_id, exc)

        return TokenMetrics(midpoint=midpoint, spread=spread)


class TradingClobClient:
    def __init__(self, config: BotConfig) -> None:
        self.config = config
        if not config.private_key:
            raise ValueError("POLYMARKET_PRIVATE_KEY is required for live trading.")

        funder = config.funder_address
        if not funder and config.signature_type == 0:
            funder = Account.from_key(config.private_key).address
        if not funder:
            raise ValueError("POLYMARKET_FUNDER_ADDRESS is required for non-EOA wallets.")

        self.funder = funder
        self._cache_key = (
            self.config.host,
            self.config.chain_id,
            self.config.signature_type,
            self.funder.lower(),
        )

        api_creds = self._load_initial_api_creds()

        self.client = self._build_authenticated_client(api_creds)

    def _configured_api_creds(self) -> ApiCreds | None:
        if not (
            self.config.api_key and self.config.api_secret and self.config.api_passphrase
        ):
            return None
        return ApiCreds(
            api_key=self.config.api_key,
            api_secret=self.config.api_secret,
            api_passphrase=self.config.api_passphrase,
        )

    def _load_initial_api_creds(self) -> ApiCreds:
        cached_api_creds = _DERIVED_API_CREDS_CACHE.get(self._cache_key)
        if cached_api_creds is not None:
            return cached_api_creds

        configured_api_creds = self._configured_api_creds()
        try:
            api_creds = self._create_or_derive_api_creds()
        except LiveTradingSetupError:
            if configured_api_creds is None:
                raise
            LOGGER.warning(
                "Wallet-based trading credential derivation failed; falling back to "
                "stored POLYMARKET_API_* credentials."
            )
            return configured_api_creds

        _DERIVED_API_CREDS_CACHE[self._cache_key] = api_creds
        return api_creds

    def _build_authenticated_client(self, api_creds: ApiCreds) -> ClobClient:
        return ClobClient(
            host=self.config.host,
            chain_id=self.config.chain_id,
            key=self.config.private_key,
            creds=api_creds,
            signature_type=self.config.signature_type,
            funder=self.funder,
        )

    def _create_or_derive_api_creds(self) -> ApiCreds:
        temp_client = ClobClient(
            host=self.config.host,
            chain_id=self.config.chain_id,
            key=self.config.private_key,
            signature_type=self.config.signature_type,
            funder=self.funder,
        )
        try:
            return temp_client.create_or_derive_api_creds()
        except Exception as exc:  # noqa: BLE001
            raise _live_setup_error(
                self.config,
                phase="API credential setup",
                exc=exc,
            ) from exc

    def _refresh_api_creds(self) -> None:
        LOGGER.warning(
            "Trading API credentials were rejected by Polymarket; deriving a fresh "
            "credential set from the wallet and retrying once."
        )
        api_creds = self._create_or_derive_api_creds()
        _DERIVED_API_CREDS_CACHE[self._cache_key] = api_creds
        self.client = self._build_authenticated_client(api_creds)

    def _call_with_auth_refresh(self, func, *, phase: str):
        try:
            return func()
        except Exception as exc:  # noqa: BLE001
            if _looks_like_invalid_api_key(exc):
                self._refresh_api_creds()
                try:
                    return func()
                except Exception as retry_exc:  # noqa: BLE001
                    exc = retry_exc
            if _looks_like_setup_error(exc):
                raise _live_setup_error(
                    self.config,
                    phase=phase,
                    exc=exc,
                ) from exc
            raise

    def get_collateral_balance_allowance(self) -> dict[str, str | Decimal | None]:
        params = BalanceAllowanceParams(
            asset_type=AssetType.COLLATERAL,
            signature_type=self.config.signature_type,
        )

        def _fetch() -> dict:
            self.client.update_balance_allowance(params)
            return self.client.get_balance_allowance(params)

        payload = self._call_with_auth_refresh(
            _fetch,
            phase="balance/allowance lookup",
        )
        balance_raw = str(payload.get("balance", "")).strip() if isinstance(payload, dict) else ""
        allowance_raw = str(payload.get("allowance", "")).strip() if isinstance(payload, dict) else ""
        return {
            "collateral_address": self.client.get_collateral_address(),
            "balance_raw": balance_raw,
            "allowance_raw": allowance_raw,
            "balance": _as_collateral_amount(balance_raw),
            "allowance": _as_collateral_amount(allowance_raw),
        }

    def place_market_buy(self, token_id: str, amount: float) -> dict:
        def _submit() -> dict:
            order = self.client.create_market_order(
                MarketOrderArgs(
                    token_id=token_id,
                    amount=amount,
                    side="BUY",
                )
            )
            return self.client.post_order(order, OrderType.FOK)

        try:
            return self._call_with_auth_refresh(
                _submit,
                phase="order submission",
            )
        except Exception as exc:  # noqa: BLE001
            if _looks_like_no_match(exc):
                raise NoMatchAvailableError(
                    "No executable liquidity was available for this market order."
                ) from exc
            raise


def _as_decimal(value: str) -> Decimal | None:
    if not value:
        return None
    try:
        return Decimal(value)
    except (InvalidOperation, ValueError):
        return None


def _as_collateral_amount(value: str) -> Decimal | None:
    amount = _as_decimal(value)
    if amount is None:
        return None
    if "." in value:
        return amount
    return amount / COLLATERAL_TOKEN_SCALE
