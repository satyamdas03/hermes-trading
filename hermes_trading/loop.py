"""24/7 reliability loop — pull data, evaluate strategy, paper trade, log outcome.

Every minute:
  1. Pull data via adapters (price, macro, news, onchain)
  2. Evaluate strategy from strategy.yaml
  3. Decide: paper trade if entry condition fires
  4. Log outcome to state/trades.jsonl
  5. Write heartbeat

Retry: 3 attempts per adapter, exponential backoff.
Circuit break: after 5 consecutive failures, halt and require restart.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from hermes_trading.adapters import price as price_adapter
from hermes_trading.adapters import macro as macro_adapter
from hermes_trading.adapters import news as news_adapter
from hermes_trading.adapters import onchain as onchain_adapter
from hermes_trading.reflect import reflect_fallback, reflect_hermes, reflect_hermes_cli, load_current_state

logger = logging.getLogger("hermes-trading.loop")

STATE_DIR = Path(__file__).resolve().parent.parent / "state"
STRATEGY_PATH = STATE_DIR / "strategy.yaml"
TRADES_PATH = STATE_DIR / "trades.jsonl"
HEARTBEAT_PATH = STATE_DIR / "heartbeat.json"
GOAL_PATH = STATE_DIR / "goal.yaml"

MAX_RETRIES = 3
RETRY_BASE_DELAY = 2  # seconds, exponential
CIRCUIT_BREAKER_LIMIT = 5
REFLECTION_INTERVAL = 5  # trigger reflection every N closed trades


class CircuitBreakerOpen(Exception):
    """Raised when circuit breaker trips after consecutive failures."""
    pass


class TradingLoop:
    """Main trading loop — single async runner for the worker."""

    def __init__(self, assets: list[str] | None = None):
        self._assets = assets or ["BTC/USDT"]
        self._mode = os.getenv("HERMES_TRADING_MODE", "paper")
        self._consecutive_failures = 0
        self._open_positions: dict[str, dict] = {}  # symbol -> position
        self._strategy_version: str = "01"
        self._last_reflected_count: int = 0  # closed trades count at last reflection

    async def run(self):
        """Run the main loop — fetch, evaluate, act, heartbeat. Forever."""
        logger.info(f"Trading loop starting — {self._assets} ({self._mode} mode)")
        self._restore_open_positions()

        while True:
            try:
                await self._tick()
                self._consecutive_failures = 0
            except Exception as e:
                self._consecutive_failures += 1
                logger.error(f"Tick failed ({self._consecutive_failures}/{CIRCUIT_BREAKER_LIMIT}): {e}")
                if self._consecutive_failures >= CIRCUIT_BREAKER_LIMIT:
                    logger.critical("CIRCUIT BREAKER OPEN — halting loop")
                    raise CircuitBreakerOpen(
                        f"Circuit breaker tripped after {CIRCUIT_BREAKER_LIMIT} consecutive failures"
                    ) from e

            await asyncio.sleep(60)  # 1 tick per minute

    async def _tick(self):
        """One iteration of the trading loop — check each asset independently."""
        tick_start = time.perf_counter()

        # 1. Shared data (macro/news apply to all assets)
        macro_data = await self._fetch_with_retry(macro_adapter.fetch, name="macro")
        strategy = self._load_strategy()

        # Collect price snapshots for heartbeat
        prices: dict[str, float] = {}

        # 2. Loop over each asset
        for symbol in self._assets:
            try:
                price_data = await self._fetch_with_retry(price_adapter.fetch, symbol, "kraken", name=f"price:{symbol}")
                prices[symbol] = price_data.get("last")
            except Exception as e:
                logger.error(f"Price fetch failed for {symbol}: {e}")
                continue

            has_position = symbol in self._open_positions

            if has_position:
                # Check stop-loss / take-profit
                self._check_position(price_data, symbol)
            else:
                # Evaluate entry
                entry_signal = self._evaluate_entry(price_data, macro_data, strategy)
                if entry_signal and self._mode == "paper":
                    await self._paper_trade(price_data, strategy, macro_data, symbol)

        # 3. Auto-reflection
        self._check_reflection()

        # 4. Write heartbeat
        tick_duration = time.perf_counter() - tick_start
        self._write_heartbeat(tick_duration, prices, macro_data)

    async def _fetch_with_retry(self, fetch_fn, *args, name: str = "unknown") -> dict:
        """Fetch with retries and exponential backoff."""
        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                result = await fetch_fn(*args)
                schema_version = result.get("schema_version")
                if not schema_version:
                    raise ValueError(f"Missing schema_version in {name} response")
                return result
            except Exception as e:
                last_error = e
                delay = RETRY_BASE_DELAY * (2 ** attempt)
                logger.warning(f"{name} fetch attempt {attempt + 1}/{MAX_RETRIES} failed: {e}. Retrying in {delay}s")
                await asyncio.sleep(delay)

        raise RuntimeError(f"{name} fetch failed after {MAX_RETRIES} attempts") from last_error

    def _load_strategy(self) -> dict:
        """Load strategy from YAML file."""
        if STRATEGY_PATH.exists():
            strategy = yaml.safe_load(STRATEGY_PATH.read_text())
            self._strategy_version = strategy.get("version", "01")
            return strategy
        logger.warning("strategy.yaml not found — using defaults")
        return {
            "version": "01",
            "entry": {"indicator": "rsi", "threshold": 30, "direction": "long"},
            "stop_loss_pct": 2.0,
            "position_size_r": 0.5,
        }

    def _evaluate_entry(self, price_data: dict, macro_data: dict, strategy: dict) -> bool:
        """Evaluate entry condition from strategy."""
        candles = price_data.get("candles_1m", [])
        if len(candles) < 30:
            return False

        entry = strategy.get("entry", {})
        indicator = entry.get("indicator", "rsi")
        threshold = entry.get("threshold", 30)
        direction = entry.get("direction", "long")

        closes = [c["close"] for c in candles[-30:] if c.get("close")]

        if indicator == "rsi" and len(closes) >= 14:
            rsi = self._compute_rsi(closes, 14)
            if direction == "long":
                return rsi < threshold
            else:
                return rsi > (100 - threshold)

        return False

    def _compute_rsi(self, closes: list[float], period: int = 14) -> float:
        """Compute RSI from closing prices."""
        if len(closes) < period + 1:
            return 50.0

        gains = []
        losses = []
        for i in range(1, len(closes)):
            diff = closes[i] - closes[i - 1]
            if diff > 0:
                gains.append(diff)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(diff))

        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period

        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    def _restore_open_positions(self):
        """Scan trades.jsonl for open positions lost on restart."""
        if not TRADES_PATH.exists():
            return
        for line in TRADES_PATH.read_text(encoding="utf-8-sig").strip().split("\n"):
            if not line.strip():
                continue
            trade = json.loads(line)
            if trade.get("status") == "open":
                symbol = trade.get("symbol", "BTC/USDT")
                self._open_positions[symbol] = trade
                logger.info(f"Restored open position: {symbol} {trade['trade_id']} @ ${trade['entry_price']:.2f}")

    async def _check_position(self, price_data: dict, symbol: str):
        """Check if open position should be closed (stop-loss hit)."""
        position = self._open_positions.get(symbol)
        if not position:
            return

        last = price_data.get("last")
        if not last:
            return

        entry_price = position["entry_price"]
        stop_loss_pct = position.get("stop_loss_pct", 2.0)

        pnl_pct = (last - entry_price) / entry_price * 100

        if pnl_pct <= -stop_loss_pct:
            logger.info(f"Stop-loss triggered {symbol}: {pnl_pct:.2f}% (limit=-{stop_loss_pct}%)")
            self._close_position(last, "stop_loss", symbol)
        elif pnl_pct >= stop_loss_pct * 1.5:
            logger.info(f"Take-profit triggered {symbol}: {pnl_pct:.2f}%")
            self._close_position(last, "take_profit", symbol)

    async def _paper_trade(self, price_data: dict, strategy: dict, macro_data: dict, symbol: str):
        """Execute a paper trade — log entry to trades.jsonl."""
        last = price_data.get("last")
        if not last:
            return

        position_size_r = strategy.get("position_size_r", 0.5)
        capital = 10000.0  # paper default
        qty = (capital * position_size_r) / last

        position = {
            "trade_id": f"ppr_{int(time.time())}",
            "symbol": symbol,
            "entry_price": last,
            "qty": round(qty, 6),
            "entry_time": datetime.now(timezone.utc).isoformat(),
            "stop_loss_pct": strategy.get("stop_loss_pct", 2.0),
            "strategy_version": strategy.get("version", "01"),
            "regime": macro_data.get("regime", "unknown"),
            "status": "open",
        }
        self._open_positions[symbol] = position

        logger.info(
            f"PAPER TRADE OPEN: {symbol} qty={qty:.6f} "
            f"@ ${last:.2f} | strategy v{strategy.get('version', '01')}"
        )
        self._append_trade(position.copy())

    def _close_position(self, exit_price: float, reason: str, symbol: str):
        """Close the open position and log."""
        position = self._open_positions.get(symbol)
        if not position:
            return

        entry = position["entry_price"]
        qty = position["qty"]
        pnl_usd = (exit_price - entry) * qty
        pnl_pct = (exit_price - entry) / entry * 100

        position["exit_price"] = exit_price
        position["exit_time"] = datetime.now(timezone.utc).isoformat()
        position["exit_reason"] = reason
        position["pnl_usd"] = round(pnl_usd, 2)
        position["pnl_pct"] = round(pnl_pct, 4)
        position["status"] = "closed"

        logger.info(
            f"PAPER TRADE CLOSED {symbol}: {reason} | "
            f"pnl=${pnl_usd:.2f} ({pnl_pct:+.2f}%)"
        )
        self._append_trade(position.copy())
        del self._open_positions[symbol]

    def _check_reflection(self):
        """Trigger reflection if enough new closed trades accumulated."""
        if not TRADES_PATH.exists():
            return

        closed_count = 0
        for line in TRADES_PATH.read_text(encoding="utf-8-sig").strip().split("\n"):
            if not line.strip():
                continue
            trade = json.loads(line)
            if trade.get("status") == "closed":
                closed_count += 1

        new_closed = closed_count - self._last_reflected_count
        if new_closed >= REFLECTION_INTERVAL:
            logger.info(f"Auto-reflection triggered: {new_closed} new closed trades since last reflect")
            try:
                strategy, trades, goal = load_current_state()
                if strategy and goal:
                    if os.getenv("ANTHROPIC_API_KEY"):
                        # Try Hermes CLI first (master prompt architecture), fall back to direct API
                        hermes_bin = os.getenv("HERMES_BIN", "hermes")
                        if shutil.which(hermes_bin):
                            result = reflect_hermes_cli(strategy, trades, goal)
                        else:
                            result = reflect_hermes(strategy, trades, goal)
                    else:
                        result = reflect_fallback(strategy, trades, goal)
                    if result:
                        self._strategy_version = result.get("version", self._strategy_version)
                        logger.info(f"Reflection complete — strategy now v{self._strategy_version}")
            except Exception as e:
                logger.error(f"Reflection failed: {e}")
            finally:
                self._last_reflected_count = closed_count

    def _append_trade(self, trade: dict):
        """Append a trade record to trades.jsonl."""
        TRADES_PATH.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(trade, default=str)
        with open(TRADES_PATH, "a") as f:
            f.write(line + "\n")

    def _write_heartbeat(self, tick_duration: float, prices: dict[str, float], macro_data: dict):
        """Write heartbeat file for monitoring."""
        open_positions = {
            symbol: {"entry_price": pos["entry_price"], "qty": pos["qty"]}
            for symbol, pos in self._open_positions.items()
        }
        heartbeat = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tick_duration_ms": round(tick_duration * 1000, 1),
            "assets": list(prices.keys()),
            "prices": prices,
            "regime": macro_data.get("regime"),
            "vix": macro_data.get("vix"),
            "mode": self._mode,
            "open_positions": open_positions,
            "position_count": len(self._open_positions),
            "strategy_version": self._strategy_version,
            "consecutive_failures": self._consecutive_failures,
        }
        HEARTBEAT_PATH.write_text(json.dumps(heartbeat, indent=2, default=str))
