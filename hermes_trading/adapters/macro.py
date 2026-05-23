"""Macro adapter — fetches VIX, SPX, DXY via yfinance (free)."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1.0"


async def fetch() -> dict:
    """Fetch macro indicators: VIX, SPX, DXY.

    Returns:
        Dict with schema_version, timestamp, and macro indicator values.
    """
    try:
        tickers = yf.download(["^VIX", "^GSPC", "DX-Y.NYB"], period="5d", progress=False)

        vix = _safe_last(tickers, "Close", "^VIX")
        spx = _safe_last(tickers, "Close", "^GSPC")
        dxy = _safe_last(tickers, "Close", "DX-Y.NYB")

        # Compute SPX 20-day return for regime classification
        spx_20d = _safe_return(tickers, "Close", "^GSPC", 20)

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


def _safe_last(df, col: str, ticker: str) -> float | None:
    """Safely extract last value from multi-level yfinance output."""
    try:
        if isinstance(df.columns, pd.MultiIndex):
            val = df[(col, ticker)].dropna().iloc[-1]
        else:
            val = df[col].dropna().iloc[-1]
        return float(val)
    except Exception:
        return None


def _safe_return(df, col: str, ticker: str, days: int) -> float | None:
    """Compute rolling return over N days."""
    try:
        if isinstance(df.columns, pd.MultiIndex):
            series = df[(col, ticker)].dropna()
        else:
            series = df[col].dropna()
        if len(series) < days:
            return None
        return float((series.iloc[-1] - series.iloc[-min(days, len(series))]) / series.iloc[-min(days, len(series))])
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
