# Hermes Phase B — Smarter Instruments — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Give the autonomous reflection brain (a) richer diagnostics to reason over and (b) a wider, bounded action space — so it can fix *direction and fees*, not just nudge one RSI threshold — while keeping one-variable-per-cycle scientific discipline.

**Architecture:** A new pure `telemetry.py` computes the diagnostics a human trader would demand (per-direction win rate, win rate by regime, expectancy, fee drag, breakeven gap). `reflect.py` injects that telemetry into the reflection prompt, advertises the full set of tunable variables with hard bounds, and validates every proposed change against those bounds (reject out-of-range rather than apply a self-destructive value). The revert-guard gains a minimum-sample gate. All offline-testable; no dependency on the Railway deploy.

**Tech Stack:** Python 3.11, numpy, pyyaml, httpx; pytest.

## Global Constraints

- Preserve autonomy: the brain still chooses what to change and why. Bounds only *reject* destructive values; they never pick the change.
- `one_variable_only` stays enforced.
- Depends on Phase A's restored score gradient (commit `26bc1be`) already on `main`.
- Run tests with `python -m pytest tests/ -q`.
- One concern per commit.

---

### Task 1: Telemetry module — the diagnostics the brain reasons over

**Files:**
- Create: `hermes_trading/telemetry.py`
- Test: `tests/test_telemetry.py`

**Interfaces:**
- Produces:
  - `compute_telemetry(closed_trades: list[dict], goal: dict) -> dict` with keys:
    `n_closed`, `win_rate`, `win_rate_long`, `win_rate_short`, `win_rate_by_regime` (dict),
    `avg_net_pnl_usd`, `avg_win_usd`, `avg_loss_usd`, `total_fees_usd`,
    `fees_pct_of_gross`, `breakeven_win_rate`, `wr_minus_breakeven`, `trades_per_day`.
  - `format_telemetry(tel: dict) -> str` — a compact human-readable block for prompt injection.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_telemetry.py
from hermes_trading.telemetry import compute_telemetry, format_telemetry

GOAL = {"target_return_30d": 0.05, "max_drawdown": 0.08, "min_sharpe": 1.2}


def _t(direction, pnl_usd, regime="bear", gross=None, fee=14.5,
       et="2026-06-25T12:00:00+00:00", xt="2026-06-25T13:00:00+00:00"):
    g = pnl_usd + fee if gross is None else gross
    return {"status": "closed", "direction": direction, "pnl_usd": pnl_usd,
            "pnl_usd_gross": g, "fee_usd": fee, "regime": regime,
            "entry_time": et, "exit_time": xt}


def test_per_direction_win_rate():
    trades = [_t("long", 100), _t("long", -90), _t("short", 100), _t("short", 100)]
    tel = compute_telemetry(trades, GOAL)
    assert tel["win_rate"] == 0.75
    assert tel["win_rate_long"] == 0.5
    assert tel["win_rate_short"] == 1.0


def test_win_rate_by_regime():
    trades = [_t("long", 100, regime="bull"), _t("long", -90, regime="bear"),
              _t("long", -90, regime="bear")]
    tel = compute_telemetry(trades, GOAL)
    assert tel["win_rate_by_regime"]["bull"] == 1.0
    assert tel["win_rate_by_regime"]["bear"] == 0.0


def test_breakeven_and_gap():
    # avg win 100, avg loss 100 -> breakeven 0.5; actual wr 0.5 -> gap 0.0
    trades = [_t("long", 100), _t("long", -100)]
    tel = compute_telemetry(trades, GOAL)
    assert abs(tel["breakeven_win_rate"] - 0.5) < 1e-6
    assert abs(tel["wr_minus_breakeven"] - 0.0) < 1e-6


def test_fees_pct_of_gross():
    # gross profit magnitude 100+? use gross fields; fees 14.5*2=29 vs gross 200
    trades = [_t("long", 100, gross=114.5), _t("long", 100, gross=114.5)]
    tel = compute_telemetry(trades, GOAL)
    assert tel["total_fees_usd"] == 29.0
    assert tel["fees_pct_of_gross"] > 0.0


def test_format_is_string_with_key_numbers():
    trades = [_t("long", 100), _t("short", -90)]
    s = format_telemetry(compute_telemetry(trades, GOAL))
    assert "win_rate" in s.lower() or "win rate" in s.lower()
    assert isinstance(s, str) and len(s) > 0


def test_empty_trades_safe():
    tel = compute_telemetry([], GOAL)
    assert tel["n_closed"] == 0
    assert tel["win_rate"] == 0.0
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_telemetry.py -v`
Expected: `ModuleNotFoundError: hermes_trading.telemetry`.

- [ ] **Step 3: Implement**

```python
# hermes_trading/telemetry.py
"""Reflection telemetry — the diagnostics a human trader would demand.

Pure functions over closed-trade records. Fed into the reflection prompt so the
brain can reason about direction and fee drag, not just entry depth.
"""
from __future__ import annotations

from datetime import datetime


def _wr(trades: list[dict]) -> float:
    if not trades:
        return 0.0
    wins = sum(1 for t in trades if (t.get("pnl_usd", 0.0) or 0.0) > 0)
    return wins / len(trades)


def _avg(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def compute_telemetry(closed_trades: list[dict], goal: dict) -> dict:
    closed = [t for t in closed_trades if t.get("status") == "closed"]
    n = len(closed)
    longs = [t for t in closed if t.get("direction") == "long"]
    shorts = [t for t in closed if t.get("direction") == "short"]

    wins = [t for t in closed if (t.get("pnl_usd", 0.0) or 0.0) > 0]
    losses = [t for t in closed if (t.get("pnl_usd", 0.0) or 0.0) <= 0]
    avg_win = _avg([t["pnl_usd"] for t in wins])
    avg_loss_mag = abs(_avg([t["pnl_usd"] for t in losses]))

    # Breakeven win rate from realised payoff asymmetry (net of fees).
    if avg_win + avg_loss_mag > 0:
        breakeven = avg_loss_mag / (avg_win + avg_loss_mag)
    else:
        breakeven = 0.0
    actual_wr = _wr(closed)

    # Regime win rates
    by_regime: dict[str, float] = {}
    regimes = {t.get("regime", "unknown") for t in closed}
    for r in regimes:
        rows = [t for t in closed if t.get("regime", "unknown") == r]
        by_regime[r] = _wr(rows)

    total_fees = sum(t.get("fee_usd", 0.0) or 0.0 for t in closed)
    gross_mag = sum(abs(t.get("pnl_usd_gross", t.get("pnl_usd", 0.0)) or 0.0) for t in closed)
    fees_pct_gross = (total_fees / gross_mag) if gross_mag > 0 else 0.0

    # Trade frequency
    stamps = []
    for t in closed:
        for key in ("exit_time", "entry_time"):
            ts = t.get(key)
            if ts:
                try:
                    stamps.append(datetime.fromisoformat(ts))
                    break
                except (ValueError, TypeError):
                    pass
    if len(stamps) >= 2:
        span_days = (max(stamps) - min(stamps)).total_seconds() / 86400.0
        trades_per_day = (n / span_days) if span_days > 0 else 0.0
    else:
        trades_per_day = 0.0

    return {
        "n_closed": n,
        "win_rate": round(actual_wr, 4),
        "win_rate_long": round(_wr(longs), 4),
        "win_rate_short": round(_wr(shorts), 4),
        "n_long": len(longs),
        "n_short": len(shorts),
        "win_rate_by_regime": {k: round(v, 4) for k, v in by_regime.items()},
        "avg_net_pnl_usd": round(_avg([t.get("pnl_usd", 0.0) or 0.0 for t in closed]), 2),
        "avg_win_usd": round(avg_win, 2),
        "avg_loss_usd": round(-avg_loss_mag, 2),
        "total_fees_usd": round(total_fees, 2),
        "fees_pct_of_gross": round(fees_pct_gross, 4),
        "breakeven_win_rate": round(breakeven, 4),
        "wr_minus_breakeven": round(actual_wr - breakeven, 4),
        "trades_per_day": round(trades_per_day, 2),
    }


def format_telemetry(tel: dict) -> str:
    reg = ", ".join(f"{k}={v:.0%}" for k, v in tel.get("win_rate_by_regime", {}).items())
    gap = tel.get("wr_minus_breakeven", 0.0)
    verdict = "PROFITABLE" if gap > 0 else "LOSING (win rate below breakeven)"
    return (
        "PERFORMANCE TELEMETRY (net of fees):\n"
        f"- closed trades: {tel.get('n_closed')}\n"
        f"- overall win_rate: {tel.get('win_rate', 0):.1%}\n"
        f"- win_rate by direction: long {tel.get('win_rate_long', 0):.1%} "
        f"(n={tel.get('n_long')}), short {tel.get('win_rate_short', 0):.1%} (n={tel.get('n_short')})\n"
        f"- win_rate by regime: {reg or 'n/a'}\n"
        f"- avg net P&L/trade: ${tel.get('avg_net_pnl_usd')} "
        f"(avg win ${tel.get('avg_win_usd')}, avg loss ${tel.get('avg_loss_usd')})\n"
        f"- fees: ${tel.get('total_fees_usd')} = {tel.get('fees_pct_of_gross', 0):.1%} of gross P&L\n"
        f"- breakeven win_rate: {tel.get('breakeven_win_rate', 0):.1%} "
        f"-> actual minus breakeven: {gap:+.1%} => {verdict}\n"
        f"- trade frequency: {tel.get('trades_per_day')}/day\n"
    )
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_telemetry.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add hermes_trading/telemetry.py tests/test_telemetry.py
git commit -m "feat(telemetry): per-direction/regime WR, expectancy, fee drag, breakeven gap

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Bounded action-space validation

**Files:**
- Modify: `hermes_trading/reflect.py` (add `BOUNDS`, `validate_change`; call it in `_apply_claude_reflection`)
- Test: `tests/test_action_bounds.py`

**Interfaces:**
- Produces: `reflect.validate_change(variable: str, new_value) -> tuple[bool, str]` — `(ok, reason)`.
- Changes `_apply_claude_reflection` to skip (log + drop) any change that fails validation; if the single allowed change is rejected, make no mutation and return None.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_action_bounds.py
import hermes_trading.reflect as reflect


def test_accepts_in_range():
    ok, _ = reflect.validate_change("stop_loss_pct", 2.0)
    assert ok
    ok, _ = reflect.validate_change("entry.threshold_long", 22)
    assert ok
    ok, _ = reflect.validate_change("entry.direction", "both")
    assert ok


def test_rejects_out_of_range():
    ok, reason = reflect.validate_change("stop_loss_pct", 0.2)
    assert not ok and "stop_loss_pct" in reason
    ok, _ = reflect.validate_change("position_size_r", 0.95)
    assert not ok
    ok, _ = reflect.validate_change("entry.threshold_long", 80)
    assert not ok


def test_rejects_bad_direction():
    ok, _ = reflect.validate_change("entry.direction", "sideways")
    assert not ok


def test_unknown_variable_rejected():
    ok, reason = reflect.validate_change("entry.magic", 5)
    assert not ok and "unknown" in reason.lower()
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_action_bounds.py -v`
Expected: FAIL — `validate_change` does not exist.

- [ ] **Step 3: Implement**

Add near the top of `reflect.py` (after the existing constants):

```python
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
```

In `_apply_claude_reflection`, inside the `for change in changes:` loop, validate before applying. Replace the apply block:

```python
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
```

After the loop, guard the no-op case (insert immediately before `strategy["version"] = new_version`):

```python
    if not change_descriptions:
        logger.warning("All proposed changes were rejected by bounds — no mutation applied")
        return None
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_action_bounds.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add hermes_trading/reflect.py tests/test_action_bounds.py
git commit -m "feat(reflect): bounded action-space validation — reject self-destructive changes

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Inject telemetry + advertise the wider action space into the prompt

**Files:**
- Modify: `hermes_trading/reflect.py` (both `reflect_hermes` and `reflect_hermes_cli` prompts)
- Test: `tests/test_prompt_enrichment.py`

**Interfaces:**
- Consumes: `telemetry.compute_telemetry`, `telemetry.format_telemetry`, `reflect.BOUNDS`.
- Produces: a `_build_reflection_prompt(strategy, last_trades, goal, telemetry_block) -> str` helper used by both reflectors so the prompt text is defined once (DRY).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_prompt_enrichment.py
import hermes_trading.reflect as reflect


def test_prompt_includes_telemetry_and_bounds():
    strategy = {"version": "59", "entry": {"indicator": "rsi", "threshold_long": 22,
                "threshold_short": 78, "direction": "both"}, "stop_loss_pct": 2.0,
                "take_profit_pct": 4.0}
    goal = {"target_return_30d": 0.05, "one_variable_only": True}
    trades = [{"status": "closed", "direction": "long", "pnl_usd": -90, "pnl_usd_gross": -75,
               "fee_usd": 14.5, "regime": "bear", "entry_time": "2026-06-25T12:00:00+00:00",
               "exit_time": "2026-06-25T13:00:00+00:00"}]
    p = reflect._build_reflection_prompt(strategy, trades, goal, reflect._telemetry_block(trades, goal))
    assert "TELEMETRY" in p
    assert "stop_loss_pct" in p  # advertises a tunable beyond threshold
    assert "take_profit_pct" in p
    assert "one variable" in p.lower() or "one_variable" in p.lower()
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_prompt_enrichment.py -v`
Expected: FAIL — `_build_reflection_prompt` / `_telemetry_block` do not exist.

- [ ] **Step 3: Implement**

Add helpers to `reflect.py`:

```python
from hermes_trading.telemetry import compute_telemetry, format_telemetry


def _telemetry_block(trades: list[dict], goal: dict) -> str:
    closed = [t for t in _filter_trades(trades) if t.get("status") == "closed"]
    return format_telemetry(compute_telemetry(closed, goal))


def _bounds_block() -> str:
    lines = [f"- {k}: allowed range [{lo}, {hi}]" for k, (lo, hi) in BOUNDS.items()]
    lines.append("- entry.direction: one of long | short | both")
    return "TUNABLE VARIABLES (change exactly ONE per cycle, must stay in range):\n" + "\n".join(lines)


def _build_reflection_prompt(strategy: dict, last_trades: list[dict], goal: dict,
                             telemetry_block: str) -> str:
    import json as _json
    return f"""You are a trading strategy optimizer. Analyze the telemetry and trades and propose
ONE change to improve net-of-fees performance. You MUST change exactly ONE variable.

Goal: {_json.dumps(goal, indent=2)}

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
```

Then in `reflect_hermes` and `reflect_hermes_cli`, replace the inline `prompt = f"""..."""` assignment with:

```python
    prompt = _build_reflection_prompt(strategy, last_25, goal, _telemetry_block(trades, goal))
```

(Leave the surrounding logic — `last_25 = closed_trades[-25:]`, the API/subprocess calls, and `_apply_claude_reflection` — unchanged.)

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_prompt_enrichment.py -v`
Expected: PASS.

- [ ] **Step 5: Run the full suite + commit**

```bash
python -m pytest tests/ -q
git add hermes_trading/reflect.py tests/test_prompt_enrichment.py
git commit -m "feat(reflect): inject performance telemetry + advertise bounded action space in prompt

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Revert-guard minimum-sample gate

**Files:**
- Modify: `hermes_trading/reflect.py` (`_maybe_revert`)
- Test: `tests/test_guard_min_sample.py`

**Interfaces:**
- Adds module constant `GUARD_MIN_SAMPLE = 15`. `_maybe_revert` returns None (no revert) when the closed-trade window is smaller than this.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_guard_min_sample.py
import json
import yaml
import hermes_trading.reflect as reflect

GOAL = {"target_return_30d": 0.05, "max_drawdown": 0.08, "min_sharpe": 1.2, "failure_below": -0.04}


def _setup(tmp_path, monkeypatch):
    hist = tmp_path / "history"; hist.mkdir()
    monkeypatch.setattr(reflect, "STRATEGY_PATH", tmp_path / "strategy.yaml")
    monkeypatch.setattr(reflect, "HISTORY_DIR", hist)
    monkeypatch.setattr(reflect, "SCORE_STATE_PATH", tmp_path / "reflect_score.json")
    monkeypatch.setattr(reflect, "HYPOTHESES_PATH", tmp_path / "hypotheses.jsonl")
    (hist / "v40.yaml").write_text(yaml.dump({"version": "40", "entry": {"threshold_long": 30}, "stop_loss_pct": 2.0}))
    (tmp_path / "strategy.yaml").write_text(yaml.dump({"version": "41", "entry": {"threshold_long": 18}, "stop_loss_pct": 2.0}))
    (tmp_path / "reflect_score.json").write_text(json.dumps({"version": "41", "predecessor": "40", "score": -0.10}))


def _losers(n):
    return [{"status": "closed", "pnl_usd": -300.0, "pnl_pct": -3.0, "fee_usd": 14.5} for _ in range(n)]


def test_no_revert_below_min_sample(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    # Only 10 closed (< GUARD_MIN_SAMPLE) even though score collapsed -> do not revert yet.
    assert reflect._maybe_revert(_losers(10), GOAL) is None


def test_revert_at_or_above_min_sample(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    assert reflect._maybe_revert(_losers(20), GOAL) is not None
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_guard_min_sample.py -v`
Expected: `test_no_revert_below_min_sample` FAILS — current guard reverts on 10 trades.

- [ ] **Step 3: Implement**

Add constant near `REVERT_MARGIN`:

```python
GUARD_MIN_SAMPLE = 15  # don't judge a change until the window has enough closed trades
```

In `_maybe_revert`, after loading `st` and before computing `current_score`, add:

```python
        closed_n = len([t for t in valid_trades if t.get("status") == "closed"])
        if closed_n < GUARD_MIN_SAMPLE:
            return None  # too few trades to judge the last change yet
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_guard_min_sample.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Full suite + commit**

```bash
python -m pytest tests/ -q
git add hermes_trading/reflect.py tests/test_guard_min_sample.py
git commit -m "feat(reflect): revert-guard requires min 15-trade sample before judging a change

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage (Phase B section):**
- B1 reflection telemetry (per-direction WR, regime WR, expectancy, fee%, frequency, breakeven) → Task 1 + injected in Task 3. ✓
- B2 widen action space, bounded, one-var → Task 2 (validation) + Task 3 (advertise in prompt). ✓
- B3 guard min-sample → Task 4. ✓

**Placeholder scan:** complete code in every step; exact commands + expected output. ✓

**Type consistency:** `compute_telemetry(closed_trades, goal) -> dict` / `format_telemetry(tel) -> str` consistent across telemetry.py, its test, and reflect helpers. `validate_change(variable, new_value) -> (bool, str)` consistent in source + test. `_build_reflection_prompt` / `_telemetry_block` names match test. `BOUNDS` keys use dotted paths (`entry.threshold_long`) matching the variable strings the brain emits and `_apply_claude_reflection` parses. ✓

**Cross-field note for executor:** the `take_profit_pct > stop_loss_pct` relationship is advised in the prompt (Task 3 GUIDANCE) rather than hard-enforced, because `validate_change` sees one field at a time; enforcing the cross-field invariant would require reading the merged strategy. Acceptable for Phase B — revisit if the brain violates it.
