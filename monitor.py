#!/usr/bin/env python3
"""Hermes Trading Monitor — pull Railway state, compute real-time P&L, alert on anomalies.

Usage:
    python monitor.py          # one-shot status report
    python monitor.py --watch  # loop every 5 minutes
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

_RAILWAY_BIN = shutil.which("railway") or "railway"

# Local persistence for cross-run comparisons
_PROJECT_ROOT = Path(__file__).resolve().parent
_STATE_DIR = _PROJECT_ROOT / "state"
_MONITOR_STATE_PATH = _STATE_DIR / "monitor_state.json"


def _load_monitor_state() -> dict:
    """Load persisted state from previous check."""
    if _MONITOR_STATE_PATH.exists():
        try:
            with open(_MONITOR_STATE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_monitor_state(state: dict) -> None:
    """Persist current check state for next run."""
    _STATE_DIR.mkdir(parents=True, exist_ok=True)
    with open(_MONITOR_STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def ssh_read(service: str, remote_path: str) -> str | None:
    """Read a file from Railway container via SSH."""
    try:
        result = subprocess.run(
            [_RAILWAY_BIN, "ssh", "--service", service, "--", f"cat {remote_path}"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            return None
        lines = result.stdout.split("\n")
        return "\n".join(l for l in lines if not l.startswith("Using SSH key")).strip()
    except Exception:
        return None


def parse_jsonl(raw: str) -> list[dict]:
    """Parse newline-delimited JSON."""
    out = []
    for line in raw.strip().split("\n"):
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def dedup_trades(trades: list[dict]) -> list[dict]:
    """Deduplicate by (trade_id, symbol), keeping last."""
    by_key = {}
    for t in trades:
        key = (t.get("trade_id"), t.get("symbol"))
        by_key[key] = t
    return list(by_key.values())


def compute_metrics(trades: list[dict], prices: dict[str, float]) -> dict:
    """Compute comprehensive performance metrics with fee awareness."""
    deduped = dedup_trades(trades)
    closed = [t for t in deduped if t.get("status") == "closed"]
    open_trades = [t for t in deduped if t.get("status") == "open"]

    total_net = sum(t.get("pnl_usd", 0) or 0 for t in closed)
    total_gross = sum(t.get("pnl_usd_gross", t.get("pnl_usd", 0)) or 0 for t in closed)
    total_fees = sum(t.get("fee_usd", 0) or 0 for t in closed)
    wins = sum(1 for t in closed if (t.get("pnl_usd", 0) or 0) > 0)
    win_rate = wins / len(closed) * 100 if closed else 0

    # Unrealized P&L
    unrealized = 0.0
    for t in open_trades:
        sym = t.get("symbol", "")
        entry = t.get("entry_price", 0) or 0
        qty = t.get("qty", 0) or 0
        direction = t.get("direction", "long")
        current = prices.get(sym, entry)
        if direction == "short":
            unrealized += (entry - current) * qty
        else:
            unrealized += (current - entry) * qty

    # Equity curve + drawdown
    equity = 10000.0
    peak = equity
    max_dd_pct = 0.0
    closed_sorted = sorted(closed, key=lambda x: x.get("exit_time", "") or "")
    for t in closed_sorted:
        equity += t.get("pnl_usd", 0) or 0
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak if peak > 0 else 0
        if dd > max_dd_pct:
            max_dd_pct = dd

    # Sharpe (requires numpy)
    sharpe = 0.0
    if len(closed) >= 3:
        try:
            import numpy as np
            rets = [(t.get("pnl_pct", 0) or 0) / 100.0 for t in closed]
            rets = [r for r in rets if abs(r) < 1.0]
            if len(rets) >= 3:
                mean_r = np.mean(rets)
                std_r = np.std(rets, ddof=1)
                sharpe = float(mean_r / std_r * np.sqrt(len(rets))) if std_r > 0 else 0
        except ImportError:
            pass

    # Version stats
    ver_stats = defaultdict(lambda: {"count": 0, "wins": 0, "net": 0})
    for t in closed:
        v = t.get("strategy_version", "?")
        ver_stats[v]["count"] += 1
        if (t.get("pnl_usd", 0) or 0) > 0:
            ver_stats[v]["wins"] += 1
        ver_stats[v]["net"] += t.get("pnl_usd", 0) or 0

    return {
        "total_closed": len(closed),
        "total_open": len(open_trades),
        "wins": wins,
        "win_rate": win_rate,
        "gross_pnl": total_gross,
        "fees": total_fees,
        "net_pnl": total_net,
        "unrealized": unrealized,
        "combined": total_net + unrealized,
        "equity": 10000.0 + total_net,
        "peak_equity": peak,
        "max_dd_pct": max_dd_pct * 100,
        "sharpe": sharpe,
        "version_stats": dict(ver_stats),
        "open_trades": open_trades,
        "prices": prices,
    }


def run_check(service: str = "zonal-curiosity") -> dict:
    """Pull Railway state, compute metrics, compare to previous check, and alert."""
    hb_raw = ssh_read(service, "//app/state/heartbeat.json")
    trades_raw = ssh_read(service, "//app/state/trades.jsonl")

    if not hb_raw:
        return {"error": "Failed to read heartbeat from Railway"}
    if not trades_raw:
        return {"error": "Failed to read trades from Railway"}

    hb = json.loads(hb_raw)
    trades = parse_jsonl(trades_raw)
    prices = hb.get("prices", {})

    metrics = compute_metrics(trades, prices)
    prev = _load_monitor_state()

    # Current values for comparison
    curr_version = str(hb.get("strategy_version", "?"))
    curr_fees = float(hb.get("cumulative_fees_usd", metrics["fees"]) or 0)
    curr_pos_count = int(hb.get("position_count", metrics["total_open"]))
    curr_closed = metrics["total_closed"]
    curr_time = datetime.now(timezone.utc).isoformat()

    # Alerts
    alerts = []

    # 1. Position count != 4
    if curr_pos_count != 4:
        alerts.append(f"[CRIT] position_count={curr_pos_count} (expected 4)")

    # 2. Consecutive failures > 0
    if hb.get("consecutive_failures", 0) > 0:
        alerts.append(f"[CRIT] consecutive_failures={hb['consecutive_failures']}")

    # 3. Strategy version changed unexpectedly
    prev_version = prev.get("last_strategy_version")
    if prev_version is not None and curr_version != prev_version:
        alerts.append(f"[WARN] Strategy version changed: {prev_version} -> {curr_version}")

    # 4. Cumulative fees not increasing (only if new trades closed since last check)
    prev_fees = prev.get("last_cumulative_fees")
    prev_closed = prev.get("last_closed_count", 0)
    if prev_fees is not None and curr_fees <= prev_fees and curr_closed > prev_closed:
        alerts.append(
            f"[WARN] Cumulative fees stuck at ${curr_fees:.2f} "
            f"({curr_closed - prev_closed} new closed trades)"
        )

    # 5. Heartbeat vs trades open count mismatch
    if hb.get("position_count", 0) != metrics["total_open"]:
        alerts.append(
            f"[WARN] heartbeat says {hb['position_count']} positions but trades.jsonl has {metrics['total_open']}"
        )

    # 6. Fee modeling may be broken (zero cumulative fees with many trades)
    if curr_fees == 0 and metrics["total_closed"] > 10:
        alerts.append("[WARN] cumulative_fees_usd is 0 — fee modeling may be broken")

    # 7. Drawdown threshold
    if metrics["max_dd_pct"] > 10:
        alerts.append(f"[CRIT] Max drawdown {metrics['max_dd_pct']:.1f}% exceeds 10% threshold")

    # 8. Win rate threshold
    if metrics["win_rate"] < 40 and metrics["total_closed"] >= 20:
        alerts.append(f"[WARN] Win rate {metrics['win_rate']:.1f}% below 40%")

    # Persist state for next run
    _save_monitor_state({
        "last_check_iso": curr_time,
        "last_strategy_version": curr_version,
        "last_cumulative_fees": curr_fees,
        "last_position_count": curr_pos_count,
        "last_closed_count": curr_closed,
    })

    return {
        "hb": hb,
        "metrics": metrics,
        "alerts": alerts,
        "timestamp": curr_time,
        "prev": prev,
    }


def print_report(result: dict):
    """Pretty-print the monitoring report."""
    if "error" in result:
        print(f"ERROR: {result['error']}")
        return

    hb = result["hb"]
    m = result["metrics"]
    alerts = result["alerts"]
    prev = result.get("prev", {})
    prices = hb.get("prices", {})

    print(f"\n{'='*70}")
    print(f"  HERMES TRADING STATUS — {result['timestamp'][:19]} UTC")
    print(f"{'='*70}")

    print(f"\n  Worker:      {hb.get('mode', '?')} mode | v{hb.get('strategy_version', '?')} | {hb.get('regime', '?')} regime")
    print(f"  Tick time:   {hb.get('tick_duration_ms', 0):.1f}ms")
    print(f"  Failures:    {hb.get('consecutive_failures', 0)} consecutive")
    print(f"  Cum. fees:   ${float(hb.get('cumulative_fees_usd', m['fees']) or 0):.2f}")

    print(f"\n  {'TRADES':<20} {'VALUE':>15}")
    print(f"  {'-'*36}")
    print(f"  {'Closed trades':<20} {m['total_closed']:>15}")
    print(f"  {'Open trades':<20} {m['total_open']:>15}")
    _wr = f"{m['wins']}/{m['total_closed']} ({m['win_rate']:.1f}%)"
    print(f"  {'Win rate':<20} {_wr:>15}")
    print(f"  {'Gross P&L':<20} ${m['gross_pnl']:>+14.2f}")
    print(f"  {'Fees paid':<20} ${m['fees']:>14.2f}")
    print(f"  {'Net P&L (realized)':<20} ${m['net_pnl']:>+14.2f}")
    print(f"  {'Unrealized P&L':<20} ${m['unrealized']:>+14.2f}")
    print(f"  {'Combined P&L':<20} ${m['combined']:>+14.2f}")
    print(f"  {'Equity':<20} ${m['equity']:>14.2f}")
    print(f"  {'Peak equity':<20} ${m['peak_equity']:>14.2f}")
    _dd = f"{m['max_dd_pct']:.2f}%"
    print(f"  {'Max drawdown':<20} {_dd:>15}")
    print(f"  {'Sharpe ratio':<20} {m['sharpe']:>15.2f}")

    # Fee-as-% context
    if m["gross_pnl"] != 0:
        fee_pct = abs(m["fees"] / m["gross_pnl"]) * 100
        print(f"  {'Fees / Gross':<20} {fee_pct:>14.1f}%")

    if m["open_trades"]:
        print(f"\n  OPEN POSITIONS:")
        print(f"  {'Symbol':<12} {'Dir':>5} {'Entry':>12} {'Current':>12} {'Qty':>10} {'Version':>4}")
        for t in m["open_trades"]:
            sym = t.get("symbol", "?")
            entry = t.get("entry_price", 0) or 0
            qty = t.get("qty", 0) or 0
            current = m["prices"].get(sym, entry)
            print(
                f"  {sym:<12} {t.get('direction','long'):>5} "
                f"${entry:>11.2f} ${current:>11.2f} {qty:>10.4f} {t.get('strategy_version','?'):>4}"
            )

    # Version breakdown
    if m["version_stats"]:
        print(f"\n  VERSION PERFORMANCE:")
        print(f"  {'Version':<8} {'Trades':>8} {'Wins':>8} {'Net P&L':>12}")
        for v, s in sorted(m["version_stats"].items()):
            print(f"  {v:<8} {s['count']:>8} {s['wins']:>8} ${s['net']:>+11.2f}")

    # Comparison with previous check
    if prev:
        print(f"\n  DELTA (vs last check {prev.get('last_check_iso',''):19}):")
        prev_net = 0.0  # We don't store previous net P&L, but we can show version/fee/pos changes
        prev_version = prev.get("last_strategy_version", "?")
        prev_fees = prev.get("last_cumulative_fees", 0)
        prev_pos = prev.get("last_position_count", 0)
        prev_closed = prev.get("last_closed_count", 0)
        new_closed = m["total_closed"] - prev_closed
        print(f"    New closed trades: {new_closed}")
        print(f"    Version:           {prev_version} -> {hb.get('strategy_version', '?')}")
        print(f"    Cum. fees:         ${prev_fees:.2f} -> ${float(hb.get('cumulative_fees_usd', m['fees']) or 0):.2f}")
        print(f"    Positions:         {prev_pos} -> {m['total_open']}")

    if alerts:
        print(f"\n  [!] ALERTS:")
        for a in alerts:
            print(f"    {a}")
    else:
        print(f"\n  [OK] All checks passed")

    print()


def main():
    parser = argparse.ArgumentParser(description="Hermes Trading Monitor")
    parser.add_argument("--watch", action="store_true", help="Loop every 5 minutes")
    parser.add_argument("--service", default="zonal-curiosity", help="Railway service name")
    args = parser.parse_args()

    if args.watch:
        import time
        while True:
            result = run_check(args.service)
            print_report(result)
            time.sleep(300)
    else:
        result = run_check(args.service)
        print_report(result)


if __name__ == "__main__":
    main()
