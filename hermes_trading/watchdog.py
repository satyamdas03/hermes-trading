"""Hermes Watchdog — separate brain daemon watching Railway worker.

Master prompt architecture: runs on Windows, SSHs into Railway every N minutes,
pulls trade state, runs Hermes CLI reflection, pushes strategy back.

Run as daemon:
  uv run python -m hermes_trading.watchdog

Run once (manual trigger):
  uv run python -m hermes_trading.watchdog --once
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

logger = logging.getLogger("hermes-trading.watchdog")

PROJECT_DIR = Path(__file__).resolve().parent.parent
STATE_DIR = PROJECT_DIR / "state"
WATCHDOG_STATE_PATH = STATE_DIR / "watchdog_state.json"
WATCHDOG_HEARTBEAT_PATH = STATE_DIR / "watchdog_heartbeat.json"

DEFAULT_INTERVAL = 1800  # 30 minutes
REFLECTION_THRESHOLD = 5  # new closed trades to trigger

# Resolve Railway CLI path — Python 3.13 subprocess can't resolve .cmd files from PATH
_RAILWAY_BIN = shutil.which("railway") or "railway"

# Resolve Hermes CLI path — check common install locations
_HERMES_BIN = shutil.which("hermes") or shutil.which("hermes.exe")
if not _HERMES_BIN:
    _hermes_default = Path.home() / "AppData" / "Local" / "hermes" / "hermes-agent" / "venv" / "Scripts" / "hermes.exe"
    if _hermes_default.exists():
        _HERMES_BIN = str(_hermes_default)

# Load ANTHROPIC_API_KEY from Hermes .env if not already set
_HERMES_ENV = Path.home() / ".hermes" / ".env"
if _HERMES_ENV.exists() and not os.getenv("ANTHROPIC_API_KEY"):
    for _line in _HERMES_ENV.read_text().strip().split("\n"):
        if "=" in _line:
            _key, _val = _line.split("=", 1)
            if _key.strip() == "ANTHROPIC_API_KEY":
                os.environ["ANTHROPIC_API_KEY"] = _val.strip().strip('"').strip("'")
                break


class HermesWatchdog:
    """Separate brain daemon — watches Railway worker, runs Hermes reflection."""

    def __init__(self, interval: int = DEFAULT_INTERVAL):
        self._interval = interval
        self._last_reflected_count: int = 0
        self._restore_state()

    # ── state persistence ──────────────────────────────────────────────

    def _restore_state(self):
        if WATCHDOG_STATE_PATH.exists():
            state = json.loads(WATCHDOG_STATE_PATH.read_text())
            self._last_reflected_count = state.get("last_reflected_count", 0)
            logger.info(f"Restored watchdog state — last_reflected_count={self._last_reflected_count}")

    def _save_state(self):
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        WATCHDOG_STATE_PATH.write_text(json.dumps({
            "last_reflected_count": self._last_reflected_count,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }, indent=2))

    # ── Railway SSH ────────────────────────────────────────────────────

    def _ssh_read(self, remote_path: str) -> str | None:
        """Read a file from Railway container via SSH."""
        try:
            result = subprocess.run(
                [_RAILWAY_BIN, "ssh", "--", f"cat {remote_path}"],
                capture_output=True, text=True, timeout=30,
                cwd=PROJECT_DIR,
            )
            if result.returncode != 0:
                logger.error(f"SSH read {remote_path} failed (rc={result.returncode}): {result.stderr[:200]}")
                return None
            # Strip "Using SSH key: ..." banner line
            lines = result.stdout.split("\n")
            content = "\n".join(l for l in lines if not l.startswith("Using SSH key"))
            return content.strip()
        except subprocess.TimeoutExpired:
            logger.error(f"SSH read {remote_path} timed out")
            return None
        except Exception as e:
            logger.error(f"SSH read {remote_path} error: {e}")
            return None

    def _ssh_write(self, remote_path: str, content: str) -> bool:
        """Write a file to Railway container via SSH stdin."""
        try:
            result = subprocess.run(
                [_RAILWAY_BIN, "ssh", "--", f"cat > {remote_path}"],
                input=content, capture_output=True, text=True, timeout=30,
                cwd=PROJECT_DIR,
            )
            if result.returncode != 0:
                logger.error(f"SSH write {remote_path} failed (rc={result.returncode}): {result.stderr[:200]}")
                return False
            return True
        except subprocess.TimeoutExpired:
            logger.error(f"SSH write {remote_path} timed out")
            return False
        except Exception as e:
            logger.error(f"SSH write {remote_path} error: {e}")
            return False

    # ── state pull / push ──────────────────────────────────────────────

    def _pull_railway_state(self) -> tuple[dict | None, list[dict], dict | None]:
        """Pull strategy, trades, goal from Railway worker."""
        strategy_raw = self._ssh_read("/app/state/strategy.yaml")
        trades_raw = self._ssh_read("/app/state/trades.jsonl")
        goal_raw = self._ssh_read("/app/state/goal.yaml")

        strategy = yaml.safe_load(strategy_raw) if strategy_raw else None
        goal = yaml.safe_load(goal_raw) if goal_raw else None

        trades = []
        if trades_raw:
            for line in trades_raw.strip().split("\n"):
                line = line.strip()
                if line:
                    try:
                        trades.append(json.loads(line))
                    except json.JSONDecodeError:
                        logger.warning(f"Skipping malformed trade line: {line[:80]}")

        return strategy, trades, goal

    def _push_strategy(self, strategy: dict) -> bool:
        """Push updated strategy.yaml to Railway worker."""
        content = yaml.dump(strategy, default_flow_style=False, sort_keys=False)
        return self._ssh_write("/app/state/strategy.yaml", content)

    # ── reflection ─────────────────────────────────────────────────────

    def _stage_railway_state_locally(self, strategy: dict, trades: list[dict], goal: dict):
        """Write pulled Railway state to local files so reflect.py can read them."""
        from hermes_trading.reflect import STRATEGY_PATH, TRADES_PATH, GOAL_PATH

        STRATEGY_PATH.parent.mkdir(parents=True, exist_ok=True)
        STRATEGY_PATH.write_text(yaml.dump(strategy, default_flow_style=False, sort_keys=False))
        GOAL_PATH.write_text(yaml.dump(goal, default_flow_style=False, sort_keys=False))

        # Write trades.jsonl — append only new closed trades to avoid duplicating
        existing_ids = set()
        if TRADES_PATH.exists():
            for line in TRADES_PATH.read_text(encoding="utf-8-sig").strip().split("\n"):
                if line.strip():
                    try:
                        t = json.loads(line)
                        existing_ids.add(t.get("trade_id", ""))
                    except json.JSONDecodeError:
                        pass

        with open(TRADES_PATH, "a") as f:
            for t in trades:
                if t.get("trade_id") not in existing_ids:
                    f.write(json.dumps(t, default=str) + "\n")

    def _run_hermes_reflection(self, strategy: dict, trades: list[dict], goal: dict) -> dict | None:
        """Run Hermes CLI reflection on staged state. Returns updated strategy."""
        from hermes_trading.reflect import reflect_hermes_cli, reflect_hermes

        # Stage Railway state locally so reflect functions can read/write files
        self._stage_railway_state_locally(strategy, trades, goal)

        # Try Hermes CLI first, fall back to direct API
        hermes_bin = os.getenv("HERMES_BIN") or _HERMES_BIN
        if hermes_bin and os.getenv("ANTHROPIC_API_KEY"):
            logger.info(f"Using Hermes CLI ({hermes_bin}) for reflection")
            result = reflect_hermes_cli(strategy, trades, goal)
        elif os.getenv("ANTHROPIC_API_KEY"):
            logger.info("Hermes CLI not found, falling back to direct Anthropic API")
            result = reflect_hermes(strategy, trades, goal)
        else:
            from hermes_trading.reflect import reflect_fallback
            logger.info("No ANTHROPIC_API_KEY, using fallback rules")
            result = reflect_fallback(strategy, trades, goal)

        if result:
            logger.info(f"Reflection complete — strategy now v{result.get('version', '?')}")
        return result

    # ── main tick ──────────────────────────────────────────────────────

    def tick(self) -> bool:
        """One watchdog cycle: pull, check, reflect, push."""
        logger.info("=== Watchdog tick — pulling Railway state ===")

        strategy, trades, goal = self._pull_railway_state()
        if not strategy:
            logger.error("Failed to pull strategy — skipping tick")
            return False
        if not goal:
            logger.error("Failed to pull goal — skipping tick")
            return False

        closed_count = len([t for t in trades if t.get("status") == "closed"])
        open_count = len([t for t in trades if t.get("status") == "open"])
        new_closed = closed_count - self._last_reflected_count

        logger.info(
            f"Railway state: {len(trades)} trades ({open_count} open, {closed_count} closed), "
            f"{new_closed} new closed since last reflect (threshold={REFLECTION_THRESHOLD})"
        )

        if new_closed >= REFLECTION_THRESHOLD:
            logger.info(f"Triggering reflection ({new_closed} >= {REFLECTION_THRESHOLD})")
            result = self._run_hermes_reflection(strategy, trades, goal)
            if result:
                if self._push_strategy(result):
                    self._last_reflected_count = closed_count
                    self._save_state()
                    logger.info(f"Strategy v{result['version']} pushed to Railway")
                    return True
                else:
                    logger.error("Reflection succeeded but push to Railway failed")
                    return False
            else:
                logger.info("Reflection returned no changes")
                return False
        else:
            logger.info(f"Skipping reflection ({new_closed} < {REFLECTION_THRESHOLD})")
            return False

    # ── daemon loop ────────────────────────────────────────────────────

    def run_forever(self):
        """Run watchdog daemon loop until interrupted."""
        logger.info(
            f"Hermes Watchdog started — interval={self._interval}s, "
            f"threshold={REFLECTION_THRESHOLD}, "
            f"last_reflected_count={self._last_reflected_count}"
        )

        while True:
            try:
                self.tick()
            except KeyboardInterrupt:
                logger.info("Watchdog stopped by user")
                return
            except Exception:
                logger.exception("Watchdog tick crashed — will retry after interval")

            self._write_heartbeat()
            self._sleep_with_progress()

    def _sleep_with_progress(self):
        """Sleep for interval, logging progress every 5 minutes."""
        remaining = self._interval
        while remaining > 0:
            chunk = min(300, remaining)  # 5-minute chunks
            time.sleep(chunk)
            remaining -= 300
            if remaining > 0:
                logger.debug(f"Next tick in {remaining // 60}min...")

    def _write_heartbeat(self):
        heartbeat = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "last_reflected_count": self._last_reflected_count,
            "interval_seconds": self._interval,
            "threshold": REFLECTION_THRESHOLD,
        }
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        WATCHDOG_HEARTBEAT_PATH.write_text(json.dumps(heartbeat, indent=2))


# ── CLI ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Hermes Watchdog — brain daemon watching Railway")
    parser.add_argument(
        "--once", action="store_true",
        help="Run one tick and exit (manual trigger)",
    )
    parser.add_argument(
        "--interval", type=int, default=DEFAULT_INTERVAL,
        help=f"Check interval in seconds (default: {DEFAULT_INTERVAL}s = 30min)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stdout,
    )

    wd = HermesWatchdog(interval=args.interval)

    if args.once:
        changed = wd.tick()
        wd._write_heartbeat()
        print(f"\nTick complete. Strategy changed: {changed}")
        sys.exit(0 if changed else 1)
    else:
        wd.run_forever()


if __name__ == "__main__":
    main()
