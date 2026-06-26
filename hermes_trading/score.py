"""Score trades against goal.yaml — returns a float in [-1, +1]."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import yaml

logger = logging.getLogger(__name__)

STATE_DIR = Path(__file__).resolve().parent.parent / "state"
GOAL_PATH = STATE_DIR / "goal.yaml"
TRADES_PATH = STATE_DIR / "trades.jsonl"


def load_goal() -> dict:
    """Load the goal configuration."""
    if not GOAL_PATH.exists():
        raise FileNotFoundError(f"goal.yaml not found at {GOAL_PATH}")
    return yaml.safe_load(GOAL_PATH.read_text())


def load_trades() -> list[dict]:
    """Load all trades from trades.jsonl."""
    if not TRADES_PATH.exists():
        return []
    trades = []
    for line in TRADES_PATH.read_text().strip().split("\n"):
        if line.strip():
            trades.append(json.loads(line))
    return trades


def _net_pnl(trade: dict) -> float:
    """Get fee-adjusted P&L from a trade record.

    Uses pnl_usd (which is net-of-fees for trades recorded after fee modeling
    was added). For legacy trades without fee_usd field, pnl_usd is already
    the only number available, so it's used as-is.

    This ensures the score function optimizes for real (fee-adjusted) profit
    rather than gross profit that ignores exchange costs.
    """
    return trade.get("pnl_usd", 0.0) or 0.0


def compute_realised_return(trades: list[dict]) -> float:
    """Compute fee-adjusted realised return from closed trades."""
    if not trades:
        return 0.0
    total_pnl = sum(_net_pnl(t) for t in trades if t.get("status") == "closed")
    initial_capital = 10000.0  # paper default
    return total_pnl / initial_capital


def compute_max_drawdown(trades: list[dict]) -> float:
    """Compute maximum drawdown from fee-adjusted equity curve."""
    if not trades:
        return 0.0
    equity = 10000.0
    peak = equity
    max_dd = 0.0
    for t in trades:
        if t.get("status") == "closed":
            equity += _net_pnl(t)
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak if peak > 0 else 0.0
            if dd > max_dd:
                max_dd = dd
    return max_dd


def compute_sharpe(trades: list[dict]) -> float:
    """Compute Sharpe ratio from fee-adjusted trade returns.

    Uses pnl_pct which is net-of-fees for trades recorded after fee modeling.
    Legacy trades without fee fields use their original pnl_pct (gross).
    """
    closed = [t for t in trades if t.get("status") == "closed"]
    if len(closed) < 3:
        return 0.0
    returns = [(t.get("pnl_pct", 0.0) or 0.0) / 100.0 for t in closed]
    returns = [r for r in returns if abs(r) < 1.0]
    if len(returns) < 3:
        return 0.0
    mean_ret = np.mean(returns)
    std_ret = np.std(returns)
    if std_ret == 0:
        return 0.0
    return float(mean_ret / std_ret * np.sqrt(len(returns)))


def score(trades: list[dict] | None = None, goal: dict | None = None) -> float:
    """Score trades against goal. Returns float in [-1, +1].

    Composite of:
      - Realised return vs target
      - Drawdown vs max allowed
      - Sharpe vs min required
    """
    if trades is None:
        trades = load_trades()
    if goal is None:
        goal = load_goal()

    target_return = goal.get("target_return_30d", 0.05)
    max_dd = goal.get("max_drawdown", 0.08)
    min_sharpe = goal.get("min_sharpe", 1.2)
    failure_below = goal.get("failure_below", -0.04)

    realised = compute_realised_return(trades)
    drawdown = compute_max_drawdown(trades)
    sharpe_ratio = compute_sharpe(trades)

    # Return score: 0 to 1 based on how close to target
    return_score = min(1.0, max(0.0, realised / target_return)) if target_return > 0 else 0.5

    # Drawdown score: 1 at 0% DD, 0 at max_dd, negative beyond
    dd_score = 1.0 - (drawdown / max_dd) if max_dd > 0 else 1.0

    # Sharpe score: 0 to 1
    sharpe_score = min(1.0, max(0.0, sharpe_ratio / min_sharpe)) if min_sharpe > 0 else 1.0

    # Composite: equally weighted
    composite = (return_score + dd_score + sharpe_score) / 3.0

    # Fee summary for logging
    closed_trades = [t for t in trades if t.get("status") == "closed"]
    total_fees = sum(t.get("fee_usd", 0.0) or 0.0 for t in closed_trades)
    fee_aware_count = sum(1 for t in closed_trades if "fee_usd" in t)

    logger.info(
        f"Score: {composite:.3f} | return={realised:+.2%} "
        f"(target={target_return:+.0%}) | DD={drawdown:.1%} "
        f"(max={max_dd:.0%}) | Sharpe={sharpe_ratio:.2f} "
        f"(min={min_sharpe}) | fees=${total_fees:.2f} "
        f"({fee_aware_count}/{len(closed_trades)} fee-modeled trades)"
    )

    # Clamp to the documented [-1, +1] range. NOTE: do NOT floor at
    # goal.failure_below — that is a *goal threshold* (what return counts as
    # "failing"), not a score floor. Flooring here pinned every underwater
    # strategy at -0.04, erasing the gradient the reflection optimizer and the
    # revert-guard rely on (they read this returned value). See score.py:136 —
    # the real composite was logged but discarded one line later.
    return max(-1.0, min(1.0, composite))
