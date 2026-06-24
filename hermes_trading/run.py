"""Entrypoint for the Hermes Trading worker.

Parses --asset from goal.yaml (override with --asset flag). Starts the loop.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from hermes_trading.loop import TradingLoop

logger = logging.getLogger("hermes-trading")

STATE_DIR = Path(__file__).resolve().parent.parent / "state"
GOAL_PATH = STATE_DIR / "goal.yaml"

# Sound hand-tuned seed used to recover the live strategy from a degraded state.
# Embedded here (not read from the repo file) because Railway's persistent volume
# mounts over state/, shadowing the image's committed strategy.yaml. Applied only
# when RESEED_STRATEGY env is truthy and not already applied (marker-guarded).
RESEED_STRATEGY_CONFIG = {
    "version": "59",
    "entry": {
        "indicator": "rsi",
        "threshold_long": 22,
        "threshold_short": 78,
        "direction": "both",
        "trend_filter": True,
        "trend_ema": 30,
    },
    "stop_loss_pct": 2.0,
    "take_profit_pct": 4.0,
    "position_size_r": 0.35,
    "max_position_age_hours": 72,
}


def _maybe_reseed_strategy() -> None:
    """One-time reset of the live strategy to a sound seed when RESEED_STRATEGY is set.

    Idempotent: writes a marker file so a left-on env flag doesn't keep resetting.
    Also clears the reflection counter and revert-guard baseline so the agent gets
    a fresh start rather than immediately mutating the new seed.
    """
    flag = os.getenv("RESEED_STRATEGY", "").strip().lower()
    if flag not in ("1", "true", "yes", "on"):
        return

    target_version = RESEED_STRATEGY_CONFIG["version"]
    marker = STATE_DIR / f".reseeded_v{target_version}"
    if marker.exists():
        logger.info(f"RESEED_STRATEGY set but already reseeded to v{target_version} — skipping (you can remove the env flag)")
        return

    strategy_path = STATE_DIR / "strategy.yaml"
    history_dir = STATE_DIR / "history"
    history_dir.mkdir(parents=True, exist_ok=True)

    # Back up the existing (degraded) live strategy.
    if strategy_path.exists():
        try:
            old = yaml.safe_load(strategy_path.read_text()) or {}
            old_v = old.get("version", "unknown")
            shutil.copy(strategy_path, history_dir / f"v{old_v}.yaml")
            logger.info(f"RESEED: backed up live strategy v{old_v} to history")
        except Exception as e:
            logger.warning(f"RESEED backup failed: {e}")

    strategy_path.write_text(yaml.dump(RESEED_STRATEGY_CONFIG, default_flow_style=False, sort_keys=False))
    logger.info(f"RESEED: strategy reset to v{target_version} (sound seed: trend filter on, threshold_long 22, sl 2.0 / tp 4.0)")

    # Bump reflection cadence on the live volume's goal.yaml (5 -> 20). The volume
    # shadows the image's committed goal.yaml, so we must patch it here.
    if GOAL_PATH.exists():
        try:
            g = yaml.safe_load(GOAL_PATH.read_text()) or {}
            if int(g.get("reflection_every", 5)) < 20:
                g["reflection_every"] = 20
                GOAL_PATH.write_text(yaml.dump(g, default_flow_style=False, sort_keys=False))
                logger.info("RESEED: goal.reflection_every bumped to 20")
        except Exception as e:
            logger.warning(f"RESEED: failed to patch goal.yaml cadence: {e}")

    # Reset reflection counter to current closed-trade count so it waits a full
    # cadence before mutating the fresh seed.
    trades_path = STATE_DIR / "trades.jsonl"
    closed = 0
    if trades_path.exists():
        for line in trades_path.read_text(encoding="utf-8-sig").strip().split("\n"):
            if not line.strip():
                continue
            try:
                if json.loads(line).get("status") == "closed":
                    closed += 1
            except Exception:
                pass
    try:
        (STATE_DIR / "reflection_state.json").write_text(json.dumps({
            "last_reflected_count": closed,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }, indent=2))
    except Exception as e:
        logger.warning(f"RESEED: failed to reset reflection counter: {e}")

    # Clear revert-guard baseline (fresh lineage).
    score_state = STATE_DIR / "reflect_score.json"
    if score_state.exists():
        try:
            score_state.unlink()
        except Exception:
            pass

    marker.write_text(datetime.now(timezone.utc).isoformat())
    logger.info(f"RESEED complete — reflection counter set to {closed}. You can now remove the RESEED_STRATEGY env flag.")


def setup_logging():
    """Configure logging — INFO to stdout, DEBUG to file."""
    sys.stdout.reconfigure(line_buffering=True)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logging.basicConfig(
        level=logging.INFO,
        handlers=[handler],
        force=True,
    )


def _ensure_state_files():
    """Create default state files if they don't exist (e.g. fresh volume mount)."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    (STATE_DIR / "history").mkdir(parents=True, exist_ok=True)

    if not GOAL_PATH.exists():
        default_goal = {
            "assets": ["BTC/USDT", "ETH/USDT", "SOL/USDT", "AVAX/USDT"],
            "target_return_30d": 0.05,
            "max_drawdown": 0.08,
            "min_sharpe": 1.2,
            "failure_below": -0.04,
            "reflection_every": 5,
            "one_variable_only": True,
        }
        GOAL_PATH.write_text(yaml.dump(default_goal, default_flow_style=False, sort_keys=False))
        logger.info("Initialized default goal.yaml")

    strategy_path = STATE_DIR / "strategy.yaml"
    if not strategy_path.exists():
        default_strategy = {
            "version": "01",
            "entry": {"indicator": "rsi", "threshold": 30, "direction": "both"},
            "stop_loss_pct": 2.0,
            "position_size_r": 0.5,
        }
        strategy_path.write_text(yaml.dump(default_strategy, default_flow_style=False, sort_keys=False))
        logger.info("Initialized default strategy.yaml")

    trades_path = STATE_DIR / "trades.jsonl"
    if not trades_path.exists():
        trades_path.write_text("")
    hyp_path = STATE_DIR / "hypotheses.jsonl"
    if not hyp_path.exists():
        hyp_path.write_text("")


def main():
    parser = argparse.ArgumentParser(description="Hermes Trading Worker")
    parser.add_argument(
        "--asset",
        type=str,
        default=None,
        help="Trading pair (e.g. BTC/USDT). Defaults to goal.yaml value.",
    )
    args = parser.parse_args()

    setup_logging()

    # Ensure state files exist (volume mount may be empty)
    _ensure_state_files()

    # One-time strategy reseed when RESEED_STRATEGY is set (recover degraded live state)
    _maybe_reseed_strategy()

    # Load assets from goal.yaml
    assets = [args.asset] if args.asset else None
    if not assets:
        goal = yaml.safe_load(GOAL_PATH.read_text())
        assets = goal.get("assets", ["BTC/USDT"])

    logger.info(f"Booting hermes-trading worker — assets={assets}")
    logger.info(f"Mode: {__import__('os').getenv('HERMES_TRADING_MODE', 'paper')}")

    # Optional read-only state API (feeds the QuantAlpha /hermes page).
    # Railway sets PORT when the service is exposed over HTTP.
    import os
    api_port = os.getenv("HERMES_API_PORT") or os.getenv("PORT")
    if api_port:
        from hermes_trading.api import start_in_thread
        start_in_thread(int(api_port))

    loop = TradingLoop(assets=assets)
    asyncio.run(loop.run())


if __name__ == "__main__":
    main()
