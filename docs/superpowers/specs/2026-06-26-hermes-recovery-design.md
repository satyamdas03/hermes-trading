# Hermes Recovery — Design Spec

**Date:** 2026-06-26
**Repo:** `github.com/satyamdas03/hermes-trading` (branch `main`, deployed on Railway)
**Status:** Approved design — ready for implementation plan
**Author:** Session 101 (continuation of Session 100 deep-dive)

---

## 1. Problem

The Hermes autonomous paper-trading agent (the `/hermes` "Matrix" page on neuralquant.co)
has been bleeding. As of 2026-06-26: **202 closed trades, 38.1% win rate, +$32.8 net P&L,
$1,851 cumulative fees (56× the net profit), max drawdown 23.6% (cap 8%), Sharpe −0.20,
strategy random-walked to v66.**

Session 100 diagnosed knife-catching, a noisy reflection cadence, and a single-exchange
SPOF, shipped fixes in commit `a5394a7`, and left deploy + reseed as pending operator work.
**That fix never deployed** — Railway still runs pre-fix code — and, more importantly, the
deep-dive for this spec found a deeper root cause Session 100 missed: **the self-learning
optimizer is blind.**

### Goal (from the operator)

> "First a profitable paper strategy that self-improves, then an actual agent that can trade
> with real money reliably and make profit."

So: preserve full autonomy and the self-learning loop; make the learning **converge** to a
positive, stable strategy; build toward real-money reliability. Truth over story.

---

## 2. Root-cause analysis (from line-by-line code read)

Ranked by leverage.

### 2.1 THE BIG ONE — the optimizer reads a blinded score (`score.py:136` vs `:144`)

`score()` **logs** the real composite but **returns** a clamped one:

```python
logger.info(f"Score: {composite:.3f} ...")        # line 136 → logs e.g. -0.648 (real gradient)
return max(failure_below, min(1.0, composite))     # line 144 → returns -0.04 (clamped, flat)
```

`failure_below = -0.04` is a **goal threshold** ("this return means we're failing") that was
mis-used as the score's **lower clamp**. Once Hermes went underwater, the *returned* score
pinned at **−0.04 for every strategy**. The reflection tape proves it: `score_before: -0.04`
appears twelve times in a row.

`reflect.py` reads the *returned* value in two places:
- `_recent_score()` → the **revert-guard** comparison (`current >= prev − 0.05`).
- `score_before` logging in the hypothesis record.

With a flat −0.04 landscape the optimizer has **zero gradient** — it cannot tell a good change
from a bad one, so it thrashes `threshold_long` (35→28→22→28…) forever. The gradient it needs
**is computed and discarded one line later.**

This also **defeats Session 100's revert-guard**: `-0.04 >= -0.04 − 0.05` (i.e. `≥ −0.09`) is
always true → the guard would never fire, even if deployed. The S100 fix was neutered before
it ran.

**Verified safe to change:** `api.py` and the frontend do **not** import `score`. The clamped
return is read only inside `reflect.py`. Unclamping affects exactly the two intended consumers.

### 2.2 Structural negative expectancy (fees + asymmetry)

TP +4% / SL −2% gross; fees 0.42% round-trip; 1-minute tick interval slips real stops to
−2.4%→−3.2%. Approximate breakeven win rate ≈ **45%**; actual **38.1%**. Result: a slow bleed
where fees ($1,851) dwarf net P&L (+$32.8).

### 2.3 Wrong direction in a bear regime

Live code is effectively long-only with **no trend filter** → it buys oversold dips that keep
falling (falling knives). A trend filter exists in `a5394a7` but is not deployed.

### 2.4 Wrong optimization lever

The brain only ever tunes `threshold_long` — which cannot fix direction or fee drag. It
optimizes a variable that cannot reach profitability.

### 2.5 Noise cadence

Reflection fires every **5** closed trades (`goal.reflection_every: 5` on the Railway volume
shadows the committed fix). Five crypto trades is statistical noise.

### 2.6 Never deployed (ops)

`a5394a7` is on GitHub; Railway runs pre-fix code. Confirmed live: no `trend_filter` keys,
`direction: long`, zero `revert-guard` reflection entries.

### 2.7 No pre-validation

Every change applies to the **live** strategy untested. Good ideas can't be confirmed; bad
ideas ship instantly. This is the architectural gap blocking real, convergent self-improvement.

---

## 3. Design principle

**Give the autonomous brain working eyes and better instruments — never take the wheel.** The
self-learning loop stays autonomous; every change either restores a signal it should already
have had, enriches the information it reasons over, or validates its proposals before they go
live. We do not freeze the strategy or hard-code decisions.

---

## 4. Phased design

Each phase is independently shippable and verifiable on `/hermes`.

### Phase A — Restore eyesight + deploy (surgical, ship first)

**A1. Score gradient fix** — `score.py:144`:
`return max(failure_below, min(1.0, composite))` → `return max(-1.0, min(1.0, composite))`.
The docstring already promises `[-1, +1]`. `failure_below` remains a goal field for its real
purpose (failure detection), no longer a score clamp. This single line restores the gradient
to the optimizer and makes the revert-guard functional.

**A2. Deploy `a5394a7`** to Railway — activates: EMA trend filter (default on), multi-exchange
price fallback (Kraken→KuCoin→OKX→Coinbase, cached instances), and reflection-cadence-from-goal.

**A3. Reseed once** — set `RESEED_STRATEGY=true`, redeploy/restart once, then remove the flag.
Seeds strategy v59 (`direction: both`, `trend_filter: true`, `trend_ema: 30`,
`threshold_long: 22`, `threshold_short: 78`, `stop_loss_pct: 2.0`, `take_profit_pct: 4.0`,
`position_size_r: 0.35`, `max_position_age_hours: 72`), bumps `goal.reflection_every: 20`,
resets reflection + score state.

**A4. Bundle** A1 into a new commit on top of `a5394a7` so Phase A deploys as one unit.

**Verification (live, after some hours):** `score_before` varies across reflections (not a flat
−0.04); `revert-guard` entries appear in `/hermes/reflections`; shorts fire in the bear regime;
fewer trades per hour; slower strategy-version churn; lower fee accrual rate.

**Risk:** low. One-line logic change plus already-written, offline-verified `a5394a7`.

### Phase B — Smarter instruments (richer inputs + wider lever)

**B1. Reflection telemetry.** Compute and inject into the reflection prompt: per-direction win
rate (long vs short), win rate by regime, average net expectancy per trade, fees as % of gross
P&L, trade frequency, and actual-vs-breakeven win rate. Give the brain the diagnostics a human
trader would demand, so it can reason about *direction and fees*, not just entry depth.

**B2. Widen action space (bounded, still one variable per reflection).** Allow the brain to
change any of: `threshold_long`, `threshold_short`, `take_profit_pct`, `stop_loss_pct`,
`trend_ema`, `position_size_r`, `max_position_age_hours`, `direction`. Add bounds validation in
`_apply_claude_reflection` that rejects out-of-range or self-destructive values
(e.g. `stop_loss_pct ∈ [1, 5]`, `take_profit_pct ∈ [2, 10]`, `take_profit_pct > stop_loss_pct`,
`position_size_r ∈ [0.1, 0.5]`, `trend_ema ∈ [10, 100]`, `threshold_long ∈ [10, 45]`,
`threshold_short ∈ [55, 90]`). On a rejected value, log and skip the change rather than apply
a bad one. `one_variable_only` stays — good science; now the variable can be the right one.

**B3. Guard hardening.** Require a minimum sample (≈15 closed trades in the score window) before
the revert-guard judges a change, so it doesn't react to a too-small window.

### Phase C — Backtest-gated learning (real-money grade)

**C1. Refactor to a shared strategy engine.** Extract the pure, side-effect-free entry/exit/fee
logic out of `TradingLoop` into a new `strategy_engine.py` (functions like
`evaluate_entry(price_data, macro, strategy)`, `evaluate_exit(position, last, strategy)`,
`apply_fees(...)`). The live loop and the backtester both import this one implementation — no
logic drift between live and simulated behavior. This is the key architectural improvement and
is independently testable.

**C2. Candle store.** New adapter pulls historical OHLCV via ccxt `fetch_ohlcv(since=...)` for
the four assets at 1m (and/or a higher timeframe), appended incrementally to `state/candles/`
(format chosen at plan time — jsonl or parquet). Includes a small fetch/backfill routine.

**C3. Backtest engine** (`backtest.py`). Replays a strategy dict over stored candles using the
shared `strategy_engine`, returns score + the same metrics the live system tracks (net return,
drawdown, Sharpe, win rate, fee total, trade count).

**C4. Gated reflection.** The reflection cycle becomes: propose change → backtest **both**
current and proposed strategy on a **held-out** window (walk-forward: optimize on older candles,
validate on newer) → deploy the proposed change **only if** its out-of-sample score beats current
by a margin; otherwise keep current, log the rejected hypothesis. This is the real convergence
mechanism and the trust foundation for live capital.

### Phase D — Real-money readiness (after C proves out; scoped later)

P&L kill-switch (max daily loss / drawdown halt), concurrent-position caps, then a live-exchange
order adapter behind a hard `LIVE_TRADING` env flag (default off). Flagged now, designed when C
is proven.

---

## 5. Sequencing & boundaries

- **A** ships standalone (hours). It is the unblock and makes the loop honest.
- **B** depends on A (needs a working gradient before richer inputs help).
- **C** depends on A and benefits from B; the `strategy_engine` refactor (C1) is the gate for
  the backtester (C3) and gated reflection (C4).
- **D** depends on C proving convergence; out of scope until then.
- Every phase is verifiable on `/hermes` before the next begins.

## 6. Files touched (by phase)

- **A:** `hermes_trading/score.py` (1 line); deploy/reseed are operator ops on Railway.
- **B:** `hermes_trading/reflect.py` (telemetry builder, prompt, bounds validation, guard min-sample).
- **C:** new `hermes_trading/strategy_engine.py`, new `hermes_trading/backtest.py`, new
  `hermes_trading/adapters/history.py` (candle store), refactor `hermes_trading/loop.py` to use
  `strategy_engine`, and `hermes_trading/reflect.py` to gate on backtest.
- **D:** new risk/kill-switch module + live-exchange adapter (later).

## 7. Verification per phase

- **A:** reflection `score_before` varies; `revert-guard` entries appear; shorts fire in bear;
  trade frequency and version churn drop; fee accrual rate falls.
- **B:** reflections change variables beyond `threshold_long`; out-of-bounds proposals are
  rejected in logs; prompt contains the new telemetry.
- **C:** `strategy_engine` unit tests (entry/exit/fee parity with prior live behavior); backtest
  reproduces known historical trades; reflections show rejected proposals (gated) and only
  out-of-sample winners deploy.

## 8. Out of scope

- Replacing RSI mean-reversion with a different strategy family (deferred; the trend filter +
  working optimizer + gated learning may make mean-reversion viable; revisit if not).
- Crypto-native regime classifier (`macro.py` still uses SPX/VIX; the per-asset EMA trend filter
  now effectively gates direction, so low priority).
- Sub-minute exit checks to reduce stop slippage (inherent to the tick interval; revisit later).

## 9. Key references

- Session 100 memory: `session100_hermes_strategy_fix.md` (infra facts, `a5394a7` contents,
  reseed mechanics, pending operator steps).
- Live data (no secret): `https://neuralquant.onrender.com/hermes/{status,trades,reflections}`.
- Deployed commit baseline: `a5394a7` on `main`.
