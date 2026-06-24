"""Price adapter — fetches OHLCV data via ccxt (free tier, no API key needed).

Resilience:
  - Multi-exchange fallback chain. If the primary exchange (Kraken) returns an
    error / is down, we transparently fall back to other public exchanges so the
    loop never goes blind on open positions.
  - Exchange instances are cached module-level and markets are loaded ONCE per
    instance. Previously a fresh exchange was built on every fetch, which reloaded
    markets (/0/public/Assets, /0/public/AssetPairs) on every call — hammering the
    exchange and amplifying rate-limit/instability.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1.0"

# Primary first, then fallbacks. All support USDT spot pairs with public OHLCV.
FALLBACK_EXCHANGES = ["kraken", "kucoin", "okx", "coinbase"]

# Cache of live ccxt async exchange instances (markets preloaded once each).
_EXCHANGE_CACHE: dict[str, object] = {}


async def _get_exchange(exchange_id: str):
    """Return a cached ccxt async exchange with markets preloaded.

    Builds + loads markets exactly once per exchange id. Raises on failure so the
    caller can fall back to the next exchange in the chain.
    """
    ex = _EXCHANGE_CACHE.get(exchange_id)
    if ex is not None:
        return ex

    import ccxt.async_support as ccxt_async

    api_key = os.getenv("EXCHANGE_API_KEY", "")
    api_secret = os.getenv("EXCHANGE_API_SECRET", "")

    ex = getattr(ccxt_async, exchange_id)({
        "apiKey": api_key,
        "secret": api_secret,
        "enableRateLimit": True,
    })
    try:
        await ex.load_markets()
    except Exception:
        # Failed to init — close and don't cache so we retry cleanly next time.
        try:
            await ex.close()
        except Exception:
            pass
        raise
    _EXCHANGE_CACHE[exchange_id] = ex
    return ex


async def fetch(symbol: str = "BTC/USDT", exchange_id: str = "kraken") -> dict:
    """Fetch latest OHLCV and ticker data for a symbol, with exchange fallback.

    Args:
        symbol: CCXT ticker pair (e.g. BTC/USDT).
        exchange_id: Preferred exchange. Falls back to FALLBACK_EXCHANGES on error.

    Returns:
        Dict with schema_version, timestamp, last price, OHLCV candles, volume.

    Raises:
        RuntimeError if every exchange in the chain fails.
    """
    chain = [exchange_id] + [e for e in FALLBACK_EXCHANGES if e != exchange_id]
    errors: list[str] = []

    for ex_id in chain:
        try:
            exchange = await _get_exchange(ex_id)
            ticker = await exchange.fetch_ticker(symbol)
            ohlcv = await exchange.fetch_ohlcv(symbol, timeframe="1m", limit=60)

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

            if ex_id != exchange_id:
                logger.info(f"price:{symbol} served by fallback exchange '{ex_id}' (primary '{exchange_id}' failed)")

            return {
                "schema_version": SCHEMA_VERSION,
                "symbol": symbol,
                "exchange": ex_id,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "last": ticker.get("last"),
                "bid": ticker.get("bid"),
                "ask": ticker.get("ask"),
                "change_24h_pct": ticker.get("percentage"),
                "volume_24h": ticker.get("baseVolume"),
                "candles_1m": candles,
            }
        except Exception as e:
            errors.append(f"{ex_id}:{type(e).__name__}")
            continue

    raise RuntimeError(f"all exchanges failed for {symbol} — {', '.join(errors)}")
