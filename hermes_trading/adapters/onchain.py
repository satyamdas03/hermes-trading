"""On-chain adapter — Glassnode API (free tier with key) or fallback to public data."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

import httpx

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1.0"


async def fetch(symbol: str = "BTC") -> dict:
    """Fetch on-chain metrics via Glassnode API if key present, else return stub.

    Args:
        symbol: Asset symbol (BTC, ETH, etc.).

    Returns:
        Dict with schema_version, timestamp, and available on-chain metrics.
    """
    api_key = os.getenv("GLASSNODE_API_KEY", "")

    if not api_key:
        logger.debug("No GLASSNODE_API_KEY — returning stub on-chain data")
        return {
            "schema_version": SCHEMA_VERSION,
            "symbol": symbol,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "source": "stub",
            "metrics": {},
            "warning": "No Glassnode API key configured",
        }

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            headers = {"X-API-KEY": api_key}
            base_url = "https://api.glassnode.com/v1/metrics"

            metrics = {}
            endpoints = [
                ("addresses/active_count", "active_addresses"),
                ("transactions/count", "tx_count"),
                ("market/price_usd_close", "price_usd"),
            ]

            for path, name in endpoints:
                try:
                    resp = await client.get(
                        f"{base_url}/{path}",
                        params={"a": symbol, "i": "24h"},
                        headers=headers,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        if data:
                            metrics[name] = data[-1].get("v")
                except Exception as e:
                    logger.debug(f"Glassnode {path} failed: {e}")

            return {
                "schema_version": SCHEMA_VERSION,
                "symbol": symbol,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "source": "glassnode",
                "metrics": metrics,
            }
    except Exception as e:
        logger.warning(f"Glassnode fetch failed: {e}")
        return {
            "schema_version": SCHEMA_VERSION,
            "symbol": symbol,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "source": "error",
            "metrics": {},
            "error": str(e),
        }
