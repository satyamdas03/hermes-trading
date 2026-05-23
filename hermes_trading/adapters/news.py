"""News adapter — fetches crypto headlines via free RSS/CryptoPanic public API."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

import httpx

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1.0"


async def fetch(symbol: str = "BTC") -> dict:
    """Fetch recent news headlines for a crypto asset.

    Uses CryptoPanic free public API (no key needed for basic tier).
    Falls back to empty if network fails.

    Args:
        symbol: Asset symbol (BTC, ETH, etc.).

    Returns:
        Dict with schema_version, timestamp, and list of headlines.
    """
    api_key = os.getenv("NEWS_API_KEY", "")

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            if api_key:
                url = "https://cryptopanic.com/api/v1/posts/"
                params = {"auth_token": api_key, "currencies": symbol, "public": "true"}
            else:
                # Free public endpoint — rate-limited but works without key
                url = "https://cryptopanic.com/api/v1/posts/"
                params = {"currencies": symbol, "public": "true", "filter": "hot"}

            resp = await client.get(url, params=params)
            if resp.status_code == 200:
                data = resp.json()
                headlines = []
                for post in data.get("results", [])[:10]:
                    headlines.append({
                        "title": post.get("title"),
                        "published_at": post.get("published_at"),
                        "source": post.get("domain"),
                        "sentiment": post.get("votes", {}).get("positive", 0) - post.get("votes", {}).get("negative", 0),
                    })

                return {
                    "schema_version": SCHEMA_VERSION,
                    "symbol": symbol,
                    "fetched_at": datetime.now(timezone.utc).isoformat(),
                    "headlines": headlines,
                    "headline_count": len(headlines),
                }
    except Exception as e:
        logger.debug(f"News fetch failed: {e}")

    return {
        "schema_version": SCHEMA_VERSION,
        "symbol": symbol,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "headlines": [],
        "headline_count": 0,
    }
