"""Entrypoint for the Hermes Trading worker.

Parses --asset from goal.yaml (override with --asset flag). Starts the loop.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

import yaml

from hermes_trading.loop import TradingLoop

logger = logging.getLogger("hermes-trading")

STATE_DIR = Path(__file__).resolve().parent.parent / "state"
GOAL_PATH = STATE_DIR / "goal.yaml"


def setup_logging():
    """Configure logging — INFO to stdout, DEBUG to file."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
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
            "entry": {"indicator": "rsi", "threshold": 30, "direction": "long"},
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

    # Load assets from goal.yaml
    assets = [args.asset] if args.asset else None
    if not assets:
        goal = yaml.safe_load(GOAL_PATH.read_text())
        assets = goal.get("assets", ["BTC/USDT"])

    logger.info(f"Booting hermes-trading worker — assets={assets}")
    logger.info(f"Mode: {__import__('os').getenv('HERMES_TRADING_MODE', 'paper')}")

    loop = TradingLoop(assets=assets)
    asyncio.run(loop.run())


if __name__ == "__main__":
    main()
