"""Reflection cycle — evaluates outcomes and proposes strategy improvements.

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


def _filter_trades(trades: list[dict]) -> list[dict]:
    """Filter out admin/emergency closes for clean reflection analysis."""
    return [
        t for t in trades
        if not (t.get("exit_reason") or "").startswith("admin_")
    ]


def reflect_fallback(strategy: dict, trades: list[dict], goal: dict) -> dict | None:
    """Deterministic rule-based reflection. Changes exactly ONE variable.

    Rules:
      - If realised return < target: loosen entry.threshold by 2
      - If drawdown > max allowed: tighten stop_loss_pct by 0.2
    """
    if not strategy or not goal:
        logger.error("Missing strategy or goal — cannot reflect")
        return None

    valid_trades = _filter_trades(trades)
    closed_trades = [t for t in valid_trades if t.get("status") == "closed"]
    if len(closed_trades) < 3:
        logger.info(f"Not enough closed trades ({len(closed_trades)}) for reflection")
        return None

    # Compute current metrics
    score_val = score_func(valid_trades, goal)
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
    # Use explicit keys if present, else fall back to legacy threshold
    if realised_return < target_return:
        entry_cfg = strategy.get("entry", {})
        if "threshold_long" in entry_cfg:
            old_val = entry_cfg["threshold_long"]
            entry_cfg["threshold_long"] = max(10, old_val - 2)
            variable_changed = f"entry.threshold_long: {old_val} -> {entry_cfg['threshold_long']}"
        else:
            old_val = entry_cfg["threshold"]
            entry_cfg["threshold"] = max(10, old_val - 2)
            variable_changed = f"entry.threshold: {old_val} -> {entry_cfg['threshold']}"
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

    # --- Integrity checks & atomic write ---
    old_version = old_strategy.get("version", "01")

    # 1. Verify disk state
    disk_strategy = yaml.safe_load(STRATEGY_PATH.read_text()) if STRATEGY_PATH.exists() else {}
    if str(disk_strategy.get("version", "01")) != old_version:
        logger.error("Version mismatch on disk — aborting fallback reflection")
        return None

    # 2. Save prior to history
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    history_path = HISTORY_DIR / f"v{old_version}.yaml"
    shutil.copy(STRATEGY_PATH, history_path)
    logger.info(f"Prior strategy saved to {history_path}")

    # 3. Atomic write
    temp_path = STRATEGY_PATH.with_suffix(".yaml.tmp")
    temp_path.write_text(yaml.dump(strategy, default_flow_style=False, sort_keys=False))
    temp_path.replace(STRATEGY_PATH)

    # 4. Verify new version on disk
    written = yaml.safe_load(STRATEGY_PATH.read_text())
    if str(written.get("version", "01")) != new_version:
        logger.error("Write verification failed for fallback reflection")
        return None

    logger.info(f"Strategy updated: v{old_version} -> v{new_version} — {variable_changed}")

    # 5. Log hypothesis (best-effort)
    try:
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
    except Exception as e:
        logger.warning(f"Hypothesis logging failed: {e}")

    return strategy


def reflect_hermes_cli(strategy: dict, trades: list[dict], goal: dict) -> dict | None:
    """Production reflection via Hermes CLI subprocess (master prompt architecture).

    Calls `hermes -z` with the reflection prompt, parses JSON from stdout.
    Requires Hermes CLI installed and ANTHROPIC_API_KEY set.
    """
    import subprocess

    if not strategy or not goal:
        logger.error("Missing strategy or goal")
        return None

    valid_trades = _filter_trades(trades)
    closed_trades = [t for t in valid_trades if t.get("status") == "closed"]
    if len(closed_trades) < 3:
        logger.info(f"Not enough closed trades ({len(closed_trades)}) for reflection")
        return None

    last_25 = closed_trades[-25:]

    prompt = f"""You are a trading strategy optimizer. Analyze these trades and propose
ONE change to the strategy to improve performance. You MUST change exactly ONE variable.

Goal: {json.dumps(goal, indent=2)}

Current strategy:
{yaml.dump(strategy, default_flow_style=False, sort_keys=False)}

Last {len(last_25)} trades:
{yaml.dump(last_25, default_flow_style=False, sort_keys=False)}

THRESHOLD SEMANTICS — READ CAREFULLY:
- For LONG entries: trigger when RSI < entry.threshold_long
- For SHORT entries: trigger when RSI > entry.threshold_short
- If the strategy only has a legacy "entry.threshold" key (no _long/_short), the code derives:
    long trigger  = RSI < threshold
    short trigger = RSI > (100 - threshold)
- This means setting threshold=75 for shorts actually triggers at RSI>25 (nearly always true),
  which causes overtrading. To avoid this, ALWAYS use the explicit keys:
    entry.threshold_long  = RSI level below which longs trigger
    entry.threshold_short = RSI level above which shorts trigger

Respond with ONLY a JSON object with a "changes" array containing exactly ONE change:
{{
  "changes": [
    {{"variable": "entry.threshold_short", "old_value": 70, "new_value": 75, "reason": "Raise short threshold to 75 so shorts only fire on strong overbought RSI>75, reducing false signals in bull regime"}}
  ],
  "summary": "one sentence summary of what changed and why"
}}"""

    hermes_bin = os.getenv("HERMES_BIN", "hermes")
    model = os.getenv("HERMES_MODEL", "claude-haiku-4-5-20251001")

    # Build clean env for subprocess — inherit only system essentials.
    # The parent process may have ANTHROPIC_* env vars pointing to Ollama etc.
    subprocess_env = {
        k: v for k, v in os.environ.items()
        if k in ("SYSTEMROOT", "USERPROFILE", "HOMEDRIVE", "HOMEPATH", "TEMP", "TMP",
                 "PATH", "COMSPEC", "PATHEXT", "WINDIR", "ProgramFiles", "CommonProgramFiles")
    }
    subprocess_env["ANTHROPIC_BASE_URL"] = "https://api.anthropic.com"
    if api_key := os.getenv("ANTHROPIC_API_KEY"):
        subprocess_env["ANTHROPIC_API_KEY"] = api_key

    try:
        result = subprocess.run(
            [hermes_bin, "-m", model, "-z", prompt],
            capture_output=True,
            text=True,
            timeout=120,
            env=subprocess_env,
        )

        if result.returncode != 0:
            stderr = result.stderr[:300] if result.stderr else "(no stderr)"
            logger.error(f"Hermes CLI failed (exit {result.returncode}): {stderr}")
            return None

        response = result.stdout.strip()
        if not response:
            logger.error("Hermes CLI returned empty response")
            return None

    except FileNotFoundError:
        logger.error(f"Hermes CLI not found at '{hermes_bin}' — install hermes-agent")
        return None
    except subprocess.TimeoutExpired:
        logger.error("Hermes CLI timed out after 120s")
        return None
    except Exception as e:
        logger.exception(f"Hermes CLI call failed: {e}")
        return None

    return _apply_claude_reflection(strategy, trades, goal, response)


def reflect_hermes(strategy: dict, trades: list[dict], goal: dict) -> dict | None:
    """Production reflection — calls Anthropic API directly with prompt.

    Claude reads the last 25 trades and current strategy, proposes changes,
    and returns hypotheses.
    """
    import httpx

    if not strategy or not goal:
        logger.error("Missing strategy or goal")
        return None

    valid_trades = _filter_trades(trades)
    closed_trades = [t for t in valid_trades if t.get("status") == "closed"]
    if len(closed_trades) < 3:
        logger.info(f"Not enough closed trades ({len(closed_trades)}) for reflection")
        return None

    last_25 = closed_trades[-25:]

    prompt = f"""You are a trading strategy optimizer. Analyze these trades and propose
ONE change to the strategy to improve performance. You MUST change exactly ONE variable.

Goal: {json.dumps(goal, indent=2)}

Current strategy:
{yaml.dump(strategy, default_flow_style=False, sort_keys=False)}

Last {len(last_25)} trades:
{yaml.dump(last_25, default_flow_style=False, sort_keys=False)}

THRESHOLD SEMANTICS — READ CAREFULLY:
- For LONG entries: trigger when RSI < entry.threshold_long
- For SHORT entries: trigger when RSI > entry.threshold_short
- If the strategy only has a legacy "entry.threshold" key (no _long/_short), the code derives:
    long trigger  = RSI < threshold
    short trigger = RSI > (100 - threshold)
- This means setting threshold=75 for shorts actually triggers at RSI>25 (nearly always true),
  which causes overtrading. To avoid this, ALWAYS use the explicit keys:
    entry.threshold_long  = RSI level below which longs trigger
    entry.threshold_short = RSI level above which shorts trigger

Respond with ONLY a JSON object with a "changes" array containing exactly ONE change:
{{
  "changes": [
    {"variable": "entry.threshold_short", "old_value": 70, "new_value": 75, "reason": "Raise short threshold to 75 so shorts only fire on strong overbought RSI>75, reducing false signals in bull regime"}
  ],
  "summary": "one sentence summary of what changed and why"
}}"""

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
                    "model": "claude-sonnet-4-6",
                    "max_tokens": 1024,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            if resp.status_code != 200:
                logger.error(f"Anthropic API error {resp.status_code}: {resp.text[:300]}")
                return None

            data = resp.json()
            response = data["content"][0]["text"]

        return _apply_claude_reflection(strategy, trades, goal, response)

    except Exception as e:
        logger.exception(f"Claude reflection failed: {e}")
        return None


def _apply_claude_reflection(
    strategy: dict, trades: list[dict], goal: dict, response: str
) -> dict | None:
    """Parse Claude's JSON response and apply strategy changes."""
    # Parse JSON from response
    if "{" in response and "}" in response:
        start = response.index("{")
        end = response.rindex("}") + 1
        plan = json.loads(response[start:end])
    else:
        logger.error(f"Response not JSON: {response[:200]}")
        return None

    changes = plan.get("changes", [])
    if not changes:
        logger.error("No changes array in response")
        return None

    # Enforce one_variable_only guardrail — scientific method
    if goal.get("one_variable_only", True) and len(changes) > 1:
        logger.warning(
            f"Claude proposed {len(changes)} changes but one_variable_only=true. "
            f"Only applying first: {changes[0].get('variable')}"
        )
        changes = changes[:1]

    summary = plan.get("summary", "No summary provided")

    old_strategy = dict(strategy)
    version = int(strategy.get("version", "1"))
    new_version = f"{version + 1:02d}"
    change_descriptions = []

    for change in changes:
        variable = change.get("variable", "")
        new_value = change.get("new_value")
        old_value = change.get("old_value")
        reason = change.get("reason", "")

        # Navigate the field path (e.g. "entry.threshold")
        parts = variable.split(".")
        target = strategy
        for part in parts[:-1]:
            if part not in target:
                target[part] = {}
            target = target[part]
        if old_value is None:
            old_value = target.get(parts[-1])
        target[parts[-1]] = new_value

        change_descriptions.append(f"{variable}: {old_value} -> {new_value}")

    strategy["version"] = new_version

    # --- Integrity checks & atomic write ---
    old_version_stamp = old_strategy.get("version", "01")

    # 1. Verify disk state matches what we loaded (detect races / manual edits)
    disk_strategy = yaml.safe_load(STRATEGY_PATH.read_text()) if STRATEGY_PATH.exists() else {}
    disk_version = str(disk_strategy.get("version", "01"))
    if disk_version != old_version_stamp:
        logger.error(
            f"Version mismatch: disk has v{disk_version} but expected v{old_version_stamp}. "
            f"Aborting reflection to prevent history corruption."
        )
        return None

    # 2. Save old version to history BEFORE writing new strategy
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    history_path = HISTORY_DIR / f"v{old_version_stamp}.yaml"
    shutil.copy(STRATEGY_PATH, history_path)
    logger.info(f"Prior strategy saved to {history_path}")

    # 3. Atomic write: temp file + rename
    temp_path = STRATEGY_PATH.with_suffix(".yaml.tmp")
    temp_path.write_text(yaml.dump(strategy, default_flow_style=False, sort_keys=False))
    temp_path.replace(STRATEGY_PATH)

    # 4. Verify new strategy on disk
    written_strategy = yaml.safe_load(STRATEGY_PATH.read_text())
    written_version = str(written_strategy.get("version", "01"))
    if written_version != new_version:
        logger.error(
            f"Write verification failed: expected v{new_version} but disk shows v{written_version}. "
            f"Reflection state is inconsistent."
        )
        return None

    change_str = ", ".join(change_descriptions)
    logger.info(f"Strategy updated via Claude: v{old_version_stamp} -> v{new_version} — {change_str}")

    # 5. Log hypothesis (best-effort; don't crash if this fails)
    try:
        score_before = score_func(trades, goal)
        hypothesis = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "reflector": "claude",
            "from_version": old_version_stamp,
            "to_version": new_version,
            "variable_changed": change_str,
            "reason": summary,
            "score_before": round(score_before, 4),
            "num_trades_considered": len([t for t in trades if t.get("status") == "closed"]),
        }
        _append_hypothesis(hypothesis)
    except Exception as e:
        logger.warning(f"Hypothesis logging failed (strategy already saved): {e}")

    return strategy


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
        help="Use Hermes AI for reflection (direct API call)",
    )
    parser.add_argument(
        "--hermes-cli",
        action="store_true",
        help="Use Hermes CLI subprocess for reflection (master prompt architecture)",
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

    if args.hermes_cli:
        result = reflect_hermes_cli(strategy, trades, goal)
    elif args.hermes:
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
