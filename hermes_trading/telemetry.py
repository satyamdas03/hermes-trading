"""Reflection telemetry — the diagnostics a human trader would demand.

Pure functions over closed-trade records. Fed into the reflection prompt so the
brain can reason about direction and fee drag, not just entry depth.
"""
from __future__ import annotations

from datetime import datetime


def _wr(trades: list[dict]) -> float:
    if not trades:
        return 0.0
    wins = sum(1 for t in trades if (t.get("pnl_usd", 0.0) or 0.0) > 0)
    return wins / len(trades)


def _avg(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def compute_telemetry(closed_trades: list[dict], goal: dict) -> dict:
    closed = [t for t in closed_trades if t.get("status") == "closed"]
    n = len(closed)
    longs = [t for t in closed if t.get("direction") == "long"]
    shorts = [t for t in closed if t.get("direction") == "short"]

    wins = [t for t in closed if (t.get("pnl_usd", 0.0) or 0.0) > 0]
    losses = [t for t in closed if (t.get("pnl_usd", 0.0) or 0.0) <= 0]
    avg_win = _avg([t["pnl_usd"] for t in wins])
    avg_loss_mag = abs(_avg([t["pnl_usd"] for t in losses]))

    # Breakeven win rate from realised payoff asymmetry (net of fees).
    if avg_win + avg_loss_mag > 0:
        breakeven = avg_loss_mag / (avg_win + avg_loss_mag)
    else:
        breakeven = 0.0
    actual_wr = _wr(closed)

    # Regime win rates
    by_regime: dict[str, float] = {}
    regimes = {t.get("regime", "unknown") for t in closed}
    for r in regimes:
        rows = [t for t in closed if t.get("regime", "unknown") == r]
        by_regime[r] = _wr(rows)

    total_fees = sum(t.get("fee_usd", 0.0) or 0.0 for t in closed)
    gross_mag = sum(abs(t.get("pnl_usd_gross", t.get("pnl_usd", 0.0)) or 0.0) for t in closed)
    fees_pct_gross = (total_fees / gross_mag) if gross_mag > 0 else 0.0

    # Trade frequency
    stamps = []
    for t in closed:
        for key in ("exit_time", "entry_time"):
            ts = t.get(key)
            if ts:
                try:
                    stamps.append(datetime.fromisoformat(ts))
                    break
                except (ValueError, TypeError):
                    pass
    if len(stamps) >= 2:
        span_days = (max(stamps) - min(stamps)).total_seconds() / 86400.0
        trades_per_day = (n / span_days) if span_days > 0 else 0.0
    else:
        trades_per_day = 0.0

    return {
        "n_closed": n,
        "win_rate": round(actual_wr, 4),
        "win_rate_long": round(_wr(longs), 4),
        "win_rate_short": round(_wr(shorts), 4),
        "n_long": len(longs),
        "n_short": len(shorts),
        "win_rate_by_regime": {k: round(v, 4) for k, v in by_regime.items()},
        "avg_net_pnl_usd": round(_avg([t.get("pnl_usd", 0.0) or 0.0 for t in closed]), 2),
        "avg_win_usd": round(avg_win, 2),
        "avg_loss_usd": round(-avg_loss_mag, 2),
        "total_fees_usd": round(total_fees, 2),
        "fees_pct_of_gross": round(fees_pct_gross, 4),
        "breakeven_win_rate": round(breakeven, 4),
        "wr_minus_breakeven": round(actual_wr - breakeven, 4),
        "trades_per_day": round(trades_per_day, 2),
    }


def format_telemetry(tel: dict) -> str:
    reg = ", ".join(f"{k}={v:.0%}" for k, v in tel.get("win_rate_by_regime", {}).items())
    gap = tel.get("wr_minus_breakeven", 0.0)
    verdict = "PROFITABLE" if gap > 0 else "LOSING (win rate below breakeven)"
    return (
        "PERFORMANCE TELEMETRY (net of fees):\n"
        f"- closed trades: {tel.get('n_closed')}\n"
        f"- overall win_rate: {tel.get('win_rate', 0):.1%}\n"
        f"- win_rate by direction: long {tel.get('win_rate_long', 0):.1%} "
        f"(n={tel.get('n_long')}), short {tel.get('win_rate_short', 0):.1%} (n={tel.get('n_short')})\n"
        f"- win_rate by regime: {reg or 'n/a'}\n"
        f"- avg net P&L/trade: ${tel.get('avg_net_pnl_usd')} "
        f"(avg win ${tel.get('avg_win_usd')}, avg loss ${tel.get('avg_loss_usd')})\n"
        f"- fees: ${tel.get('total_fees_usd')} = {tel.get('fees_pct_of_gross', 0):.1%} of gross P&L\n"
        f"- breakeven win_rate: {tel.get('breakeven_win_rate', 0):.1%} "
        f"-> actual minus breakeven: {gap:+.1%} => {verdict}\n"
        f"- trade frequency: {tel.get('trades_per_day')}/day\n"
    )
