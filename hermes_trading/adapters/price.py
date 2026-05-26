"""Price adapter — fetches OHLCV data via ccxt (free tier, no API key needed)."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1.0"


async def fetch(symbol: str = "BTC/USDT", exchange_id: str = "coinbase") -> dict:
    """Fetch latest OHLCV and ticker data for a symbol.

    Args:
        symbol: CCXT ticker pair (e.g. BTC/USDT).
        exchange_id: Exchange name (default: binance — works without API key for public data).

    Returns:
        Dict with schema_version, timestamp, last price, OHLCV candles, volume.
    """
    import ccxt.async_support as ccxt_async

    api_key = os.getenv("EXCHANGE_API_KEY", "")
    api_secret = os.getenv("EXCHANGE_API_SECRET", "")

    exchange = getattr(ccxt_async, exchange_id)({
        "apiKey": api_key,
        "secret": api_secret,
        "enableRateLimit": True,
    })

    try:
        ticker = await exchange.fetch_ticker(symbol)
        ohlcv = await exchange.fetch_ohlcv(symbol, timeframe="1m", limit=90)

        candles = [
            {
                "timestamp": c[0],
                "open": c[1],
                "high": c[2],
                "low": c[3],
                "close": c[4],
                "volume": c[5],
            }
            for c in ohlcv
        ]

        return {
            "schema_version": SCHEMA_VERSION,
            "symbol": symbol,
            "exchange": exchange_id,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "last": ticker.get("last"),
            "bid": ticker.get("bid"),
            "ask": ticker.get("ask"),
            "change_24h_pct": ticker.get("percentage"),
            "volume_24h": ticker.get("baseVolume"),
            "candles_1m": candles,
        }
    finally:
        await exchange.close()
