"""Macro adapter — fetches VIX, SPX, DXY via yfinance (free)."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import yfinance as yf

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1.0"


async def fetch() -> dict:
    """Fetch macro indicators: VIX, SPX, DXY.

    Returns:
        Dict with schema_version, timestamp, and macro indicator values.
    """
    try:
        # Fetch individually for reliability (multi-ticker MultiIndex fragile)
        vix = _fetch_ticker_last("^VIX")
        spx = _fetch_ticker_last("^GSPC")
        spx_20d_ago = _fetch_ticker_n_days_ago("^GSPC", 20)
        dxy = _fetch_ticker_last("DX-Y.NYB")

        # Compute SPX 20-day return
        spx_20d = None
        if spx is not None and spx_20d_ago is not None and spx_20d_ago > 0:
            spx_20d = (spx - spx_20d_ago) / spx_20d_ago

        return {
            "schema_version": SCHEMA_VERSION,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "vix": vix,
            "spx": spx,
            "dxy": dxy,
            "spx_return_20d": spx_20d,
            "regime": _classify_regime(vix, spx_20d),
        }
    except Exception as e:
        logger.warning(f"Macro fetch failed: {e}")
        return {
            "schema_version": SCHEMA_VERSION,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "vix": None,
            "spx": None,
            "dxy": None,
            "spx_return_20d": None,
            "regime": "unknown",
            "error": str(e),
        }


def _fetch_ticker_last(symbol: str) -> float | None:
    """Fetch last close price for a single ticker."""
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="5d")
        if hist.empty:
            return None
        return float(hist["Close"].iloc[-1])
    except Exception:
        return None


def _fetch_ticker_n_days_ago(symbol: str, days: int) -> float | None:
    """Fetch close price from N days ago."""
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=f"{days + 5}d")
        if len(hist) < days:
            return None
        return float(hist["Close"].iloc[-days])
    except Exception:
        return None


def _classify_regime(vix: float | None, spx_20d: float | None) -> str:
    """Simple regime classifier based on VIX and SPX momentum."""
    if vix is None or spx_20d is None:
        return "unknown"

    if vix > 30:
        return "fear" if spx_20d < 0 else "volatile_bull"
    elif vix > 20:
        return "cautious" if spx_20d < 0 else "neutral"
    else:
        return "bear" if spx_20d < -0.02 else "bull"
