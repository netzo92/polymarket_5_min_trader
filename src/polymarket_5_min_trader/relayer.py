from __future__ import annotations

from dataclasses import dataclass

from eth_abi import encode
from eth_utils import keccak
from py_builder_relayer_client.client import RelayClient
from py_builder_relayer_client.models import OperationType, SafeTransaction
from py_builder_signing_sdk.config import BuilderConfig
from py_builder_signing_sdk.sdk_types import BuilderApiKeyCreds

from polymarket_5_min_trader.config import BotConfig

CTF_ADDRESS = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"
USDC_E_ADDRESS = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
ROOT_COLLECTION_ID = bytes(32)
DEFAULT_REDEEM_INDEX_SETS = [1, 2]
REDEEM_POSITIONS_SIGNATURE = "redeemPositions(address,bytes32,bytes32,uint256[])"


class RelayerSetupError(RuntimeError):
    """Raised when the relayer or builder configuration is incomplete."""


@dataclass(frozen=True)
class ClaimSubmission:
    transaction_id: str | None
    transaction_hash: str | None


class SafeMarketRedeemer:
    """Redeems resolved winning positions through the Polymarket relayer."""

    def __init__(self, config: BotConfig) -> None:
        if config.signature_type != 2:
            raise RelayerSetupError(
                "Auto-claim currently supports only POLYMARKET_SIGNATURE_TYPE=2 (GNOSIS_SAFE)."
            )
        if not config.private_key:
            raise RelayerSetupError(
                "POLYMARKET_PRIVATE_KEY is required for auto-claim."
            )
        if not config.funder_address:
            raise RelayerSetupError(
                "POLYMARKET_FUNDER_ADDRESS is required for auto-claim."
            )
        if not (
            config.builder_api_key
            and config.builder_secret
            and config.builder_passphrase
        ):
            raise RelayerSetupError(
                "Builder credentials are required for auto-claim. "
                "Set POLYMARKET_BUILDER_API_KEY / POLYMARKET_BUILDER_SECRET / "
                "POLYMARKET_BUILDER_PASSPHRASE, or reuse POLYMARKET_API_* if "
                "those are your Builder keys."
            )

        builder_config = BuilderConfig(
            local_builder_creds=BuilderApiKeyCreds(
                key=config.builder_api_key,
                secret=config.builder_secret,
                passphrase=config.builder_passphrase,
            )
        )
        self.client = RelayClient(
            config.relayer_url,
            config.chain_id,
            private_key=config.private_key,
            builder_config=builder_config,
        )
        self.safe_address = self.client.get_expected_safe()
        if self.safe_address.lower() != config.funder_address.lower():
            raise RelayerSetupError(
                "Configured POLYMARKET_FUNDER_ADDRESS does not match the relayer-derived "
                f"Safe address for this signer. expected={self.safe_address} "
                f"configured={config.funder_address}"
            )

    def submit_redeem(
        self,
        *,
        condition_id: str,
        metadata: str | None = None,
    ) -> ClaimSubmission:
        transaction = SafeTransaction(
            to=CTF_ADDRESS,
            operation=OperationType.Call,
            data=_encode_redeem_positions(condition_id),
            value="0",
        )
        response = self.client.execute([transaction], metadata)
        return ClaimSubmission(
            transaction_id=response.transaction_id,
            transaction_hash=response.transaction_hash,
        )

    def get_transaction_status(self, transaction_id: str) -> dict[str, object] | None:
        transactions = self.client.get_transaction(transaction_id)
        if not transactions:
            return None
        if isinstance(transactions, list):
            return transactions[0] if transactions else None
        if isinstance(transactions, dict):
            return transactions
        return None


def _encode_redeem_positions(condition_id: str) -> str:
    normalized = condition_id.removeprefix("0x")
    if len(normalized) != 64:
        raise ValueError(f"Invalid condition_id: {condition_id}")
    selector = keccak(text=REDEEM_POSITIONS_SIGNATURE)[:4]
    encoded_args = encode(
        ["address", "bytes32", "bytes32", "uint256[]"],
        [
            USDC_E_ADDRESS,
            ROOT_COLLECTION_ID,
            bytes.fromhex(normalized),
            DEFAULT_REDEEM_INDEX_SETS,
        ],
    )
    return "0x" + (selector + encoded_args).hex()
