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

_RAILWAY_BIN = shutil.which("railway") or "railway"


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
    """Compute comprehensive performance metrics."""
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
        entry = t.get("entry_price", 0)
        qty = t.get("qty", 0)
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
    closed_sorted = sorted(closed, key=lambda x: x.get("exit_time", ""))
    for t in closed_sorted:
        equity += t.get("pnl_usd", 0) or 0
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak if peak > 0 else 0
        if dd > max_dd_pct:
            max_dd_pct = dd

    # Sharpe
    sharpe = 0.0
    if len(closed) >= 3:
        try:
            import numpy as np
            rets = [(t.get("pnl_pct", 0) or 0) / 100.0 for t in closed]
            rets = [r for r in rets if abs(r) < 1.0]
            if len(rets) >= 3:
                mean_r = np.mean(rets)
                std_r = np.std(rets)
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
    }


def run_check(service: str = "zonal-curiosity") -> dict:
    """Pull Railway state and compute metrics."""
    hb_raw = ssh_read(service, "/app/state/heartbeat.json")
    trades_raw = ssh_read(service, "/app/state/trades.jsonl")
    strategy_raw = ssh_read(service, "/app/state/strategy.yaml")

    if not hb_raw:
        return {"error": "Failed to read heartbeat from Railway"}
    if not trades_raw:
        return {"error": "Failed to read trades from Railway"}

    hb = json.loads(hb_raw)
    trades = parse_jsonl(trades_raw)
    prices = hb.get("prices", {})

    metrics = compute_metrics(trades, prices)

    # Alerts
    alerts = []
    if hb.get("consecutive_failures", 0) > 0:
        alerts.append(f"[CRIT] consecutive_failures={hb['consecutive_failures']}")
    if hb.get("position_count", 0) != len(metrics["open_trades"]):
        alerts.append(f"[WARN] heartbeat says {hb['position_count']} positions but trades.jsonl has {metrics['total_open']}")
    if hb.get("cumulative_fees_usd", 0) == 0 and metrics["total_closed"] > 10:
        alerts.append("[WARN] cumulative_fees_usd is 0 — fee modeling may be broken")
    if metrics["max_dd_pct"] > 10:
        alerts.append(f"[CRIT] Max drawdown {metrics['max_dd_pct']:.1f}% exceeds 10% threshold")
    if metrics["win_rate"] < 40 and metrics["total_closed"] >= 20:
        alerts.append(f"[WARN] Win rate {metrics['win_rate']:.1f}% below 40%")

    return {
        "hb": hb,
        "metrics": metrics,
        "strategy_raw": strategy_raw,
        "alerts": alerts,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def print_report(result: dict):
    """Pretty-print the monitoring report."""
    if "error" in result:
        print(f"ERROR: {result['error']}")
        return

    hb = result["hb"]
    m = result["metrics"]
    alerts = result["alerts"]

    print(f"\n{'='*70}")
    print(f"  HERMES TRADING STATUS — {result['timestamp'][:19]} UTC")
    print(f"{'='*70}")

    print(f"\n  Worker:      {hb.get('mode', '?')} mode | v{hb.get('strategy_version', '?')} | {hb.get('regime', '?')} regime")
    print(f"  Tick time:   {hb.get('tick_duration_ms', 0):.1f}ms")
    print(f"  Failures:    {hb.get('consecutive_failures', 0)} consecutive")
    print(f"  Cum. fees:   ${hb.get('cumulative_fees_usd', 0):.2f}")

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

    if m["open_trades"]:
        print(f"\n  OPEN POSITIONS:")
        print(f"  {'Symbol':<12} {'Dir':>5} {'Entry':>12} {'Current':>12} {'Version':>4}")
        for t in m["open_trades"]:
            sym = t.get("symbol", "?")
            entry = t.get("entry_price", 0)
            current = hb.get("prices", {}).get(sym, entry)
            print(f"  {sym:<12} {t.get('direction','long'):>5} ${entry:>11.2f} ${current:>11.2f} {t.get('strategy_version','?'):>4}")

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
