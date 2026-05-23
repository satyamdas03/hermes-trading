"""Reflection cycle — evaluates outcomes and proposes ONE variable change.

Two modes:
  --fallback : Deterministic rule-based reflection (Phase 5, pre-Hermes).
  --hermes   : Production mode (Phase 7). Calls hermes subprocess with prompt.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from hermes_trading.score import score as score_func

logger = logging.getLogger("hermes-trading.reflect")

STATE_DIR = Path(__file__).resolve().parent.parent / "state"
STRATEGY_PATH = STATE_DIR / "strategy.yaml"
TRADES_PATH = STATE_DIR / "trades.jsonl"
HYPOTHESES_PATH = STATE_DIR / "hypotheses.jsonl"
HISTORY_DIR = STATE_DIR / "history"
GOAL_PATH = STATE_DIR / "goal.yaml"


def load_current_state() -> tuple[dict | None, list[dict], dict | None]:
    """Load current strategy, trades, and goal."""
    strategy = yaml.safe_load(STRATEGY_PATH.read_text()) if STRATEGY_PATH.exists() else None
    goal = yaml.safe_load(GOAL_PATH.read_text()) if GOAL_PATH.exists() else None

    trades = []
    if TRADES_PATH.exists():
        raw = TRADES_PATH.read_text(encoding="utf-8-sig")
        for line in raw.strip().split("\n"):
            if line.strip():
                trades.append(json.loads(line))

    return strategy, trades, goal


def reflect_fallback(strategy: dict, trades: list[dict], goal: dict) -> dict | None:
    """Deterministic rule-based reflection. Changes exactly ONE variable.

    Rules:
      - If realised return < target: loosen entry.threshold by 2
      - If drawdown > max allowed: tighten stop_loss_pct by 0.2
    """
    if not strategy or not goal:
        logger.error("Missing strategy or goal — cannot reflect")
        return None

    closed_trades = [t for t in trades if t.get("status") == "closed"]
    if len(closed_trades) < 3:
        logger.info(f"Not enough closed trades ({len(closed_trades)}) for reflection")
        return None

    # Compute current metrics
    score_val = score_func(trades, goal)
    target_return = goal.get("target_return_30d", 0.05)
    max_drawdown = goal.get("max_drawdown", 0.08)

    # Compute realised return and drawdown
    total_pnl = sum(t.get("pnl_usd", 0.0) or 0.0 for t in closed_trades)
    capital = 10000.0
    realised_return = total_pnl / capital

    equity = capital
    peak = equity
    dd = 0.0
    for t in closed_trades:
        equity += t.get("pnl_usd", 0.0) or 0.0
        if equity > peak:
            peak = equity
        dd = max(dd, (peak - equity) / peak if peak > 0 else 0.0)

    old_strategy = dict(strategy)
    version = int(strategy.get("version", "1"))
    new_version = f"{version + 1:02d}"
    variable_changed = None
    reason = None

    # Rule: return below target -> loosen entry threshold
    if realised_return < target_return:
        old_threshold = strategy["entry"]["threshold"]
        new_threshold = old_threshold - 2  # Loosen: e.g. 30 -> 28 (enter earlier)
        strategy["entry"]["threshold"] = max(10, new_threshold)
        variable_changed = f"entry.threshold: {old_threshold} -> {strategy['entry']['threshold']}"
        reason = f"Realised return {realised_return:+.2%} below target {target_return:+.0%} — loosened entry"
    # Rule: drawdown too high -> tighten stop loss
    elif dd > max_drawdown:
        old_sl = strategy["stop_loss_pct"]
        strategy["stop_loss_pct"] = round(old_sl - 0.2, 1)
        strategy["stop_loss_pct"] = max(0.5, strategy["stop_loss_pct"])
        variable_changed = f"stop_loss_pct: {old_sl} -> {strategy['stop_loss_pct']}"
        reason = f"Drawdown {dd:.1%} above max {max_drawdown:.0%} — tightened stop loss"
    else:
        # Everything on track — minor optimization
        old_sl = strategy["stop_loss_pct"]
        strategy["stop_loss_pct"] = round(old_sl + 0.1, 1)
        variable_changed = f"stop_loss_pct: {old_sl} -> {strategy['stop_loss_pct']}"
        reason = f"On track (score={score_val:.3f}) — slight stop-loss relaxation"

    strategy["version"] = new_version

    # Save prior version to history
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    old_version = old_strategy.get("version", "01")
    history_path = HISTORY_DIR / f"v{old_version}.yaml"
    shutil.copy(STRATEGY_PATH, history_path)
    logger.info(f"Prior strategy saved to {history_path}")

    # Write new strategy
    STRATEGY_PATH.write_text(yaml.dump(strategy, default_flow_style=False, sort_keys=False))
    logger.info(f"Strategy updated: v{old_version} -> v{new_version} — {variable_changed}")

    # Log hypothesis
    hypothesis = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "reflector": "fallback",
        "from_version": old_version,
        "to_version": new_version,
        "variable_changed": variable_changed,
        "reason": reason,
        "score_before": round(score_val, 4),
        "num_trades_considered": len(closed_trades),
    }
    _append_hypothesis(hypothesis)

    return strategy


def reflect_hermes(strategy: dict, trades: list[dict], goal: dict) -> dict | None:
    """Production reflection — calls Anthropic API directly with prompt.

    Claude reads the last 25 trades and current strategy, proposes one variable change,
    and returns a hypothesis.
    """
    import httpx

    if not strategy or not goal:
        logger.error("Missing strategy or goal")
        return None

    closed_trades = [t for t in trades if t.get("status") == "closed"]
    if len(closed_trades) < 3:
        logger.info(f"Not enough closed trades ({len(closed_trades)}) for reflection")
        return None

    last_25 = closed_trades[-25:]

    prompt = f"""You are a trading strategy optimizer. Analyze these trades and propose
exactly ONE variable to change in the strategy.

Goal: {json.dumps(goal, indent=2)}

Current strategy:
{yaml.dump(strategy, default_flow_style=False, sort_keys=False)}

Last {len(last_25)} trades:
{yaml.dump(last_25, default_flow_style=False, sort_keys=False)}

Respond with ONLY a JSON object:
{{
  "variable": "entry.threshold",
  "old_value": 30,
  "new_value": 28,
  "reason": "one sentence explaining why"
}}
"""

    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        logger.error("ANTHROPIC_API_KEY not set")
        return None

    try:
        with httpx.Client(timeout=120) as client:
            resp = client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": "claude-sonnet-4-6-20250514",
                    "max_tokens": 1024,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            if resp.status_code != 200:
                logger.error(f"Anthropic API error {resp.status_code}: {resp.text[:300]}")
                return None

            data = resp.json()
            response = data["content"][0]["text"]

        # Parse JSON from response
        if "{" in response and "}" in response:
            start = response.index("{")
            end = response.rindex("}") + 1
            plan = json.loads(response[start:end])
        else:
            logger.error(f"Response not JSON: {response[:200]}")
            return None

        variable = plan.get("variable", "")
        new_value = plan.get("new_value")
        reason = plan.get("reason", "No reason provided")

        # Apply the change
        old_strategy = dict(strategy)
        version = int(strategy.get("version", "1"))
        new_version = f"{version + 1:02d}"

        # Navigate the field path (e.g. "entry.threshold")
        parts = variable.split(".")
        target = strategy
        for part in parts[:-1]:
            target = target[part]
        old_value = target[parts[-1]]
        target[parts[-1]] = new_value

        strategy["version"] = new_version

        # Save prior to history
        HISTORY_DIR.mkdir(parents=True, exist_ok=True)
        old_version_stamp = old_strategy.get("version", "01")
        history_path = HISTORY_DIR / f"v{old_version_stamp}.yaml"
        shutil.copy(STRATEGY_PATH, history_path)
        logger.info(f"Prior strategy saved to {history_path}")

        # Write new strategy
        STRATEGY_PATH.write_text(yaml.dump(strategy, default_flow_style=False, sort_keys=False))
        logger.info(f"Strategy updated via Claude: v{old_version_stamp} -> v{new_version} — {variable}: {old_value} -> {new_value}")

        # Log hypothesis
        score_before = score_func(trades, goal)
        hypothesis = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "reflector": "claude",
            "from_version": old_version_stamp,
            "to_version": new_version,
            "variable_changed": f"{variable}: {old_value} -> {new_value}",
            "reason": reason,
            "score_before": round(score_before, 4),
            "num_trades_considered": len(last_25),
        }
        _append_hypothesis(hypothesis)

        return strategy

    except Exception as e:
        logger.exception(f"Claude reflection failed: {e}")
        return None


def _append_hypothesis(hypothesis: dict):
    """Append a hypothesis to hypotheses.jsonl."""
    HYPOTHESES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(HYPOTHESES_PATH, "a") as f:
        f.write(json.dumps(hypothesis, default=str) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Hermes Trading Reflection")
    parser.add_argument(
        "--fallback",
        action="store_true",
        help="Use deterministic fallback reflection (pre-Hermes)",
    )
    parser.add_argument(
        "--hermes",
        action="store_true",
        help="Use Hermes AI for reflection (production)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stdout,
    )

    strategy, trades, goal = load_current_state()

    if not strategy:
        logger.error("No strategy found — nothing to reflect on")
        sys.exit(1)
    if not goal:
        logger.error("No goal found — cannot score")
        sys.exit(1)

    logger.info(f"Reflecting on v{strategy.get('version', '?')} with {len(trades)} trades")

    if args.hermes:
        result = reflect_hermes(strategy, trades, goal)
    else:
        result = reflect_fallback(strategy, trades, goal)

    if result:
        logger.info(f"Reflection complete — strategy now v{result['version']}")
    else:
        logger.info("No changes made — conditions not met for reflection")
        sys.exit(1)


if __name__ == "__main__":
    main()
