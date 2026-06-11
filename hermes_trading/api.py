"""Read-only HTTP API over the trading state — feeds the QuantAlpha /hermes page.

Runs in-process with the trading loop (uvicorn in a daemon thread, see run.py)
so the SSE log stream can tap the live logger output directly.

Endpoints (all require X-Hermes-Secret == HERMES_API_SECRET):
    GET /status       heartbeat + strategy + computed trade aggregates
    GET /trades?n=200 tail of trades.jsonl (newest last) + cumulative P&L
    GET /strategy     current strategy + version history
    GET /reflections  tail of hypotheses.jsonl (what changed and why)
    GET /events       SSE stream of live log lines (the Matrix feed)
    GET /health       unauthenticated liveness probe
"""

from __future__ import annotations

import asyncio
import collections
import json
import logging
import os
import threading
import time
from pathlib import Path

import yaml
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

STATE_DIR = Path(__file__).resolve().parent.parent / "state"

# ── Live log ring buffer ─────────────────────────────────────────────────────
# (seq, line) pairs; SSE clients poll the buffer by sequence number, which
# avoids cross-event-loop handoff between the trading loop and uvicorn.
_LOG_LOCK = threading.Lock()
_LOG_BUF: collections.deque[tuple[int, str]] = collections.deque(maxlen=500)
_LOG_SEQ = 0


class RingBufferHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        global _LOG_SEQ
        try:
            line = self.format(record)
        except Exception:
            return
        with _LOG_LOCK:
            _LOG_SEQ += 1
            _LOG_BUF.append((_LOG_SEQ, line))


def install_log_capture() -> None:
    """Attach the ring-buffer handler to the root logger (idempotent)."""
    root = logging.getLogger()
    if any(isinstance(h, RingBufferHandler) for h in root.handlers):
        return
    h = RingBufferHandler(level=logging.INFO)
    h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    root.addHandler(h)


# ── Auth ─────────────────────────────────────────────────────────────────────

def _require_secret(request: Request) -> None:
    secret = os.getenv("HERMES_API_SECRET", "")
    if not secret:
        raise HTTPException(503, "HERMES_API_SECRET not configured")
    if request.headers.get("x-hermes-secret") != secret:
        raise HTTPException(403, "invalid secret")


# ── State readers ────────────────────────────────────────────────────────────

def _read_yaml(name: str) -> dict:
    p = STATE_DIR / name
    if not p.exists():
        return {}
    try:
        return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _read_json(name: str) -> dict:
    p = STATE_DIR / name
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _tail_jsonl(name: str, n: int) -> list[dict]:
    p = STATE_DIR / name
    if not p.exists():
        return []
    rows: list[dict] = []
    try:
        lines = p.read_text(encoding="utf-8").splitlines()
        for line in lines[-n:]:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    except Exception:
        pass
    return rows


def _trade_aggregates(trades: list[dict]) -> dict:
    closed = [t for t in trades if t.get("status") == "closed"]
    open_ = [t for t in trades if t.get("status") == "open"]
    pnls = [t.get("pnl_usd") or 0.0 for t in closed]
    wins = [p for p in pnls if p > 0]
    return {
        "total_trades": len(trades),
        "closed_trades": len(closed),
        "open_positions": open_,
        "total_pnl_usd": round(sum(pnls), 2),
        "win_rate_pct": round(len(wins) / len(closed) * 100, 1) if closed else None,
        "avg_pnl_pct": round(sum(t.get("pnl_pct") or 0.0 for t in closed) / len(closed), 3) if closed else None,
        "best_trade_usd": round(max(pnls), 2) if pnls else None,
        "worst_trade_usd": round(min(pnls), 2) if pnls else None,
    }


# ── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(title="hermes-trading state API", docs_url=None, redoc_url=None)


@app.get("/health")
def health() -> dict:
    hb = _read_json("heartbeat.json")
    return {"status": "ok", "heartbeat_at": hb.get("timestamp") or hb.get("updated_at")}


@app.get("/status", dependencies=[Depends(_require_secret)])
def status() -> dict:
    heartbeat = _read_json("heartbeat.json")
    strategy = _read_yaml("strategy.yaml")
    goal = _read_yaml("goal.yaml")
    trades = _tail_jsonl("trades.jsonl", 10_000)
    return {
        "heartbeat": heartbeat,
        "strategy": strategy,
        "goal": goal,
        "aggregates": _trade_aggregates(trades),
        "server_time": time.time(),
    }


@app.get("/trades", dependencies=[Depends(_require_secret)])
def trades(n: int = Query(default=200, ge=1, le=2000)) -> dict:
    rows = _tail_jsonl("trades.jsonl", n)
    # Cumulative P&L over the returned window (closed trades, entry order)
    cum = 0.0
    curve = []
    for t in rows:
        if t.get("status") == "closed":
            cum += t.get("pnl_usd") or 0.0
            curve.append({"time": t.get("exit_time"), "cum_pnl_usd": round(cum, 2)})
    return {"trades": rows, "pnl_curve": curve}


@app.get("/strategy", dependencies=[Depends(_require_secret)])
def strategy() -> dict:
    current = _read_yaml("strategy.yaml")
    history = []
    hist_dir = STATE_DIR / "history"
    if hist_dir.exists():
        for f in sorted(hist_dir.iterdir()):
            if f.suffix in (".yaml", ".yml"):
                try:
                    history.append({"file": f.name, "strategy": yaml.safe_load(f.read_text(encoding="utf-8"))})
                except Exception:
                    continue
    return {"current": current, "history": history}


@app.get("/reflections", dependencies=[Depends(_require_secret)])
def reflections(n: int = Query(default=50, ge=1, le=500)) -> dict:
    return {"reflections": _tail_jsonl("hypotheses.jsonl", n)}


@app.get("/events", dependencies=[Depends(_require_secret)])
async def events() -> StreamingResponse:
    """SSE stream of live log lines. Replays the last 50 on connect."""

    async def gen():
        with _LOG_LOCK:
            backlog = list(_LOG_BUF)[-50:]
            last_seq = backlog[-1][0] if backlog else _LOG_SEQ
        for _, line in backlog:
            yield f"data: {json.dumps({'line': line})}\n\n"
        while True:
            await asyncio.sleep(1.0)
            with _LOG_LOCK:
                fresh = [(s, l) for s, l in _LOG_BUF if s > last_seq]
            for s, line in fresh:
                last_seq = s
                yield f"data: {json.dumps({'line': line})}\n\n"
            if not fresh:
                yield ": keepalive\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def start_in_thread(port: int) -> threading.Thread:
    """Launch uvicorn in a daemon thread next to the trading loop."""
    import uvicorn

    install_log_capture()

    def _serve():
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")

    t = threading.Thread(target=_serve, name="hermes-api", daemon=True)
    t.start()
    logging.getLogger("hermes-trading.api").info(f"State API listening on :{port}")
    return t
