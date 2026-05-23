# Hermes Trading

Self-improving paper trading agent running 24/7 on Railway. BTC/USDT, minute ticks, RSI-based entry with deterministic + AI-powered reflection cycles.

## Architecture

```
hermes-trading/
├── hermes_trading/
│   ├── run.py              # Entrypoint — starts loop
│   ├── loop.py             # Main trading loop (fetch → evaluate → act → heartbeat)
│   ├── reflect.py          # Reflection engine (--fallback / --hermes)
│   ├── score.py            # Composite score [-1, +1] from returns, DD, Sharpe
│   └── adapters/
│       ├── price.py        # ccxt/Kraken OHLCV + ticker
│       ├── macro.py        # yfinance VIX/SPX/DXY
│       ├── news.py         # CryptoPanic headlines
│       └── onchain.py      # Glassnode (stub without key)
├── state/
│   ├── goal.yaml           # Strategy targets (+5%/30d, 8% DD, 1.2 Sharpe)
│   ├── strategy.yaml       # Current active strategy (self-modifying)
│   ├── trades.jsonl        # Trade log (appended in real time)
│   ├── hypotheses.jsonl    # Reflection change log
│   ├── heartbeat.json       # Live status for monitoring
│   └── history/            # Archived strategy versions
├── pyproject.toml
├── Dockerfile
└── uv.lock
```

## How It Works

1. **Every minute:** Pull price (Kraken), macro (VIX/SPX/DXY), news, on-chain data
2. **Evaluate:** RSI < threshold → paper trade entry (long BTC/USDT)
3. **Manage:** Stop-loss at -2%, take-profit at +3%
4. **Reflect:** Every 5 closed trades, evaluate performance vs goal
5. **Self-improve:** Change exactly 1 strategy variable (threshold, stop-loss%, position size)
6. **Heartbeat:** Write status file for monitoring

## Strategy

| Version | Entry | Stop-Loss | Position Size | Notes |
|---------|-------|-----------|---------------|-------|
| v01 | RSI < 30 | 2.0% | 0.5R | Initial |
| v02 | RSI < 28 | 2.0% | 0.5R | Loosened entry (return below target) |

## Reflection Modes

### Fallback (`--fallback`)
Deterministic rules, no API needed:
- Return < target → loosen entry threshold by 2
- Drawdown > max → tighten stop-loss by 0.2%
- On track → relax stop-loss by 0.1%

### Claude (`--hermes`)
Calls Anthropic API (Claude Sonnet 4.6) with last 25 trades + current strategy. Claude proposes exactly one variable change with reasoning.

```bash
uv run python -m hermes_trading.reflect --fallback
uv run python -m hermes_trading.reflect --hermes
```

## Deployment

Runs on Railway with persistent volume at `/app/state`.

```bash
railway up
```

Environment variables:
- `HERMES_TRADING_MODE=paper`
- `ANTHROPIC_API_KEY` (for --hermes reflection)
- `EXCHANGE_API_KEY` / `EXCHANGE_API_SECRET` (optional, Kraken public data works without)

## Local

```bash
uv sync
uv run python -m hermes_trading.run --asset BTC/USDT
```
