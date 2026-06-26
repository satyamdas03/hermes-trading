"""Read-only Hermes performance health check.

Pulls the public /hermes API and reports whether the self-learning loop is
healthy: a live score gradient (not the -0.04 flatline), an active revert-guard,
sane version churn, and a fee/P&L ratio that isn't bleeding. Exit 0 = healthy.

Run a baseline BEFORE the Phase A deploy/reseed, then again AFTER to prove recovery.
"""
from __future__ import annotations

import statistics
import sys
from datetime import datetime

BASE = "https://neuralquant.onrender.com/hermes"


def score_dispersion(reflections: list[dict], window: int = 8) -> float:
    """Stdev of score_before over the most RECENT `window` reflections.

    Kept as a secondary signal. The primary blind-optimizer detector is
    floor_pin_ratio — dispersion alone is fooled by the transition period where
    scores were sliding (0.30 -> 0.17 -> 0.036) into the -0.04 floor.
    Reflections are returned oldest->newest, so the tail is the most recent.
    """
    scores = [r.get("score_before") for r in reflections if r.get("score_before") is not None]
    scores = scores[-window:]
    if len(scores) < 2:
        return 0.0
    return float(statistics.pstdev(scores))


def floor_pin_ratio(reflections: list[dict], k: int = 6, floor: float = -0.04,
                    eps: float = 1e-6) -> float:
    """Fraction of the most recent k reflections whose score_before sits exactly
    on the failure floor. This IS the blind-optimizer signature: once underwater,
    the old clamped score pinned every strategy at -0.04, so the optimizer saw a
    flat landscape. A high ratio = the loop is flying blind. Post-fix the unclamped
    composite won't land exactly on -0.04, so the ratio collapses toward 0.
    """
    scores = [r.get("score_before") for r in reflections if r.get("score_before") is not None]
    scores = scores[-k:]
    if not scores:
        return 0.0
    pinned = sum(1 for s in scores if abs(s - floor) <= eps)
    return pinned / len(scores)


def has_revert_guard(reflections: list[dict]) -> bool:
    return any(r.get("reflector") == "revert-guard" for r in reflections)


def version_churn_per_day(reflections: list[dict]) -> float:
    stamps = []
    for r in reflections:
        ts = r.get("timestamp")
        if ts:
            try:
                stamps.append(datetime.fromisoformat(ts))
            except ValueError:
                pass
    if len(stamps) < 2:
        return 0.0
    span_hours = (max(stamps) - min(stamps)).total_seconds() / 3600.0
    if span_hours <= 0:
        return 0.0
    # intervals = len-1 changes across the span
    return (len(stamps) - 1) / (span_hours / 24.0)


def fee_to_pnl_ratio(status: dict) -> float:
    agg = status.get("aggregates", {})
    hb = status.get("heartbeat", {})
    fees = hb.get("cumulative_fees_usd", 0.0) or 0.0
    pnl = agg.get("total_pnl_usd", 0.0) or 0.0
    return fees / max(abs(pnl), 1.0)


def health_report(status: dict, reflections: list[dict]) -> dict:
    pin = floor_pin_ratio(reflections)
    guard = has_revert_guard(reflections)
    churn = version_churn_per_day(reflections)
    fee_ratio = fee_to_pnl_ratio(status)

    checks = {
        "score_gradient_alive": {
            "value": round(pin, 2), "target": "< 0.6 of recent scores pinned at -0.04 floor",
            "pass": pin < 0.6},
        "revert_guard_active": {
            "value": guard, "target": "at least one revert-guard entry", "pass": guard},
        "version_churn_sane": {
            "value": round(churn, 2), "target": "<= 8 changes/day", "pass": churn <= 8.0},
        "fee_bleed_controlled": {
            "value": round(fee_ratio, 2), "target": "fees < 5x net P&L", "pass": fee_ratio < 5.0},
    }
    return {"checks": checks, "healthy": all(c["pass"] for c in checks.values())}


def main() -> int:
    import httpx

    with httpx.Client(timeout=30) as client:
        status = client.get(f"{BASE}/status").json()
        reflections = client.get(f"{BASE}/reflections?n=30").json().get("reflections", [])

    rep = health_report(status, reflections)
    print(f"Strategy v{status.get('strategy', {}).get('version', '?')} | "
          f"WR {status.get('aggregates', {}).get('win_rate_pct', '?')}% | "
          f"net ${status.get('aggregates', {}).get('total_pnl_usd', '?')} | "
          f"fees ${status.get('heartbeat', {}).get('cumulative_fees_usd', '?')}")
    print("-" * 60)
    for name, c in rep["checks"].items():
        flag = "PASS" if c["pass"] else "FAIL"
        print(f"[{flag}] {name:24} = {c['value']!s:10} (target: {c['target']})")
    print("-" * 60)
    print("HEALTHY" if rep["healthy"] else "UNHEALTHY")
    return 0 if rep["healthy"] else 1


if __name__ == "__main__":
    sys.exit(main())
