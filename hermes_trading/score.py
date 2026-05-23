"""Score trades against goal.yaml — returns a float in [-1, +1]."""

from __future__ import annotations

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
            trades.append(yaml.safe_load(line))
    return trades


def compute_realised_return(trades: list[dict]) -> float:
    """Compute realised return from closed trades."""
    if not trades:
        return 0.0
    total_pnl = sum(t.get("pnl_usd", 0.0) or 0.0 for t in trades if t.get("status") == "closed")
    initial_capital = 10000.0  # paper default
    return total_pnl / initial_capital


def compute_max_drawdown(trades: list[dict]) -> float:
    """Compute maximum drawdown from equity curve."""
    if not trades:
        return 0.0
    equity = 10000.0
    peak = equity
    max_dd = 0.0
    for t in trades:
        if t.get("status") == "closed":
            equity += t.get("pnl_usd", 0.0) or 0.0
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak if peak > 0 else 0.0
            if dd > max_dd:
                max_dd = dd
    return max_dd


def compute_sharpe(trades: list[dict]) -> float:
    """Compute Sharpe ratio from trade returns."""
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

    logger.info(
        f"Score: {composite:.3f} | return={realised:+.2%} "
        f"(target={target_return:+.0%}) | DD={drawdown:.1%} "
        f"(max={max_dd:.0%}) | Sharpe={sharpe_ratio:.2f} "
        f"(min={min_sharpe})"
    )

    return max(failure_below, min(1.0, composite))
