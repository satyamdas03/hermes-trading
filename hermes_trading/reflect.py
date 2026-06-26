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
from hermes_trading.telemetry import compute_telemetry, format_telemetry

logger = logging.getLogger("hermes-trading.reflect")

STATE_DIR = Path(__file__).resolve().parent.parent / "state"
STRATEGY_PATH = STATE_DIR / "strategy.yaml"
TRADES_PATH = STATE_DIR / "trades.jsonl"
HYPOTHESES_PATH = STATE_DIR / "hypotheses.jsonl"
HISTORY_DIR = STATE_DIR / "history"
GOAL_PATH = STATE_DIR / "goal.yaml"
SCORE_STATE_PATH = STATE_DIR / "reflect_score.json"

# Revert guard: if a strategy change makes the recent-window score worse by more
# than this margin, roll the change back instead of mutating further. This closes
# the feedback loop that was missing — previously reflection never checked whether
# its last change helped, causing an endless random walk.
REVERT_MARGIN = 0.05
SCORE_WINDOW = 40  # number of recent closed trades the guard scores on

# Hard bounds for the widened action space. The brain may tune any of these
# (one per cycle); values outside the range are rejected, never applied.
BOUNDS = {
    "stop_loss_pct": (1.0, 5.0),
    "take_profit_pct": (2.0, 10.0),
    "position_size_r": (0.1, 0.5),
    "max_position_age_hours": (6, 168),
    "entry.threshold_long": (10, 45),
    "entry.threshold_short": (55, 90),
    "entry.trend_ema": (10, 100),
}
_DIRECTIONS = {"long", "short", "both"}


def validate_change(variable: str, new_value) -> tuple[bool, str]:
    """Return (ok, reason). Rejects unknown variables and out-of-range values."""
    if variable == "entry.direction":
        if new_value in _DIRECTIONS:
            return True, "ok"
        return False, f"entry.direction must be one of {sorted(_DIRECTIONS)}, got {new_value!r}"
    if variable not in BOUNDS:
        return False, f"unknown or non-tunable variable: {variable}"
    lo, hi = BOUNDS[variable]
    try:
        v = float(new_value)
    except (TypeError, ValueError):
        return False, f"{variable} value not numeric: {new_value!r}"
    if lo <= v <= hi:
        return True, "ok"
    return False, f"{variable}={v} out of bounds [{lo}, {hi}]"


def _telemetry_block(trades: list[dict], goal: dict) -> str:
    closed = [t for t in _filter_trades(trades) if t.get("status") == "closed"]
    return format_telemetry(compute_telemetry(closed, goal))


def _bounds_block() -> str:
    lines = [f"- {k}: allowed range [{lo}, {hi}]" for k, (lo, hi) in BOUNDS.items()]
    lines.append("- entry.direction: one of long | short | both")
    return "TUNABLE VARIABLES (change exactly ONE per cycle, must stay in range):\n" + "\n".join(lines)


def _build_reflection_prompt(strategy: dict, last_trades: list[dict], goal: dict,
                             telemetry_block: str) -> str:
    return f"""You are a trading strategy optimizer. Analyze the telemetry and trades and propose
ONE change to improve net-of-fees performance. You MUST change exactly ONE variable.

Goal: {json.dumps(goal, indent=2)}

{telemetry_block}

Current strategy:
{yaml.dump(strategy, default_flow_style=False, sort_keys=False)}

Last {len(last_trades)} trades:
{yaml.dump(last_trades, default_flow_style=False, sort_keys=False)}

{_bounds_block()}

GUIDANCE:
- If win_rate by direction shows one side losing badly in this regime, consider entry.direction
  or the trend filter rather than only nudging a threshold.
- If fees are a large % of gross and trades/day is high, fewer/higher-quality trades help: widen
  take_profit_pct vs stop_loss_pct, raise position_size_r, or tighten entry thresholds.
- take_profit_pct should stay greater than stop_loss_pct.

THRESHOLD SEMANTICS: long triggers when RSI < entry.threshold_long; short when RSI > entry.threshold_short.

Respond with ONLY a JSON object:
{{
  "changes": [
    {{"variable": "take_profit_pct", "old_value": 4.0, "new_value": 5.0, "reason": "..."}}
  ],
  "summary": "one sentence"
}}"""


def _recent_score(valid_trades: list[dict], goal: dict, window: int = SCORE_WINDOW) -> float:
    """Score only the most recent `window` closed trades — responsive to the last change."""
    closed = [t for t in valid_trades if t.get("status") == "closed"]
    return score_func(closed[-window:], goal)


def _load_score_state() -> dict | None:
    """Load {version, predecessor, score} recorded at the last applied change."""
    if not SCORE_STATE_PATH.exists():
        return None
    try:
        return json.loads(SCORE_STATE_PATH.read_text())
    except Exception:
        return None


def _save_score_state(version: str, predecessor: str | None, score: float) -> None:
    """Record the score of the strategy we just left, so the next reflection can
    judge whether the change we just applied helped or hurt."""
    try:
        SCORE_STATE_PATH.write_text(json.dumps({
            "version": str(version),
            "predecessor": str(predecessor) if predecessor is not None else None,
            "score": round(float(score), 4),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }, indent=2))
    except Exception as e:
        logger.warning(f"Failed to save score state: {e}")


def _maybe_revert(valid_trades: list[dict], goal: dict) -> dict | None:
    """If the last applied change hurt the recent-window score, roll it back.

    Returns the reverted strategy dict if a revert happened, else None (proceed
    with normal reflection). Best-effort: never raises into the loop.
    """
    try:
        st = _load_score_state()
        if not st:
            return None
        prev_score = st.get("score")
        predecessor = st.get("predecessor")  # version to roll back TO
        applied = st.get("version")          # version currently on disk (being judged)
        if prev_score is None or predecessor is None or applied is None:
            return None

        disk = yaml.safe_load(STRATEGY_PATH.read_text()) if STRATEGY_PATH.exists() else {}
        if str(disk.get("version")) != str(applied):
            return None  # disk changed out from under us (manual edit / reseed) — skip

        current_score = _recent_score(valid_trades, goal)
        if current_score >= prev_score - REVERT_MARGIN:
            return None  # change held up (or improved) — keep it

        hist = HISTORY_DIR / f"v{predecessor}.yaml"
        if not hist.exists():
            return None
        reverted = yaml.safe_load(hist.read_text())
        new_version = f"{int(applied) + 1:02d}"
        reverted["version"] = new_version

        # Save current (the losing version) to history, then atomic-write the revert.
        HISTORY_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy(STRATEGY_PATH, HISTORY_DIR / f"v{applied}.yaml")
        temp_path = STRATEGY_PATH.with_suffix(".yaml.tmp")
        temp_path.write_text(yaml.dump(reverted, default_flow_style=False, sort_keys=False))
        temp_path.replace(STRATEGY_PATH)

        logger.info(
            f"REVERT GUARD: v{applied} recent score {current_score:.3f} < prior "
            f"v{predecessor} {prev_score:.3f} (margin {REVERT_MARGIN}) — rolled back "
            f"to v{predecessor} config as v{new_version}"
        )
        try:
            _append_hypothesis({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "reflector": "revert-guard",
                "from_version": applied,
                "to_version": new_version,
                "variable_changed": f"rollback to v{predecessor} config",
                "reason": f"v{applied} recent score {current_score:.3f} fell below v{predecessor} {prev_score:.3f}",
                "score_before": round(current_score, 4),
                "num_trades_considered": len([t for t in valid_trades if t.get("status") == "closed"]),
            })
        except Exception:
            pass
        # The revert is itself a change we can re-evaluate next cycle.
        _save_score_state(new_version, applied, current_score)
        return reverted
    except Exception as e:
        logger.warning(f"Revert guard failed (continuing with normal reflection): {e}")
        return None


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

    # Closed-loop guard: roll back the last change if it hurt the recent window.
    reverted = _maybe_revert(valid_trades, goal)
    if reverted is not None:
        return reverted

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

    # Record the score of the strategy we just left so the next reflection can
    # judge whether this change helped (revert guard).
    _save_score_state(new_version, old_version, _recent_score(valid_trades, goal))

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

    # Closed-loop guard: roll back the last change if it hurt (skip the AI call).
    reverted = _maybe_revert(valid_trades, goal)
    if reverted is not None:
        return reverted

    last_25 = closed_trades[-25:]

    prompt = _build_reflection_prompt(strategy, last_25, goal, _telemetry_block(trades, goal))

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

    # Closed-loop guard: roll back the last change if it hurt (skip the AI call).
    reverted = _maybe_revert(valid_trades, goal)
    if reverted is not None:
        return reverted

    last_25 = closed_trades[-25:]

    prompt = _build_reflection_prompt(strategy, last_25, goal, _telemetry_block(trades, goal))

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

        ok, why = validate_change(variable, new_value)
        if not ok:
            logger.warning(f"Rejected out-of-bounds change {variable}={new_value}: {why}")
            continue

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

    if not change_descriptions:
        logger.warning("All proposed changes were rejected by bounds — no mutation applied")
        return None

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

    # Record the score of the strategy we just left so the next reflection's revert
    # guard can judge whether this change helped.
    try:
        _save_score_state(new_version, old_version_stamp, _recent_score(_filter_trades(trades), goal))
    except Exception as e:
        logger.warning(f"Failed to record score state: {e}")

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
