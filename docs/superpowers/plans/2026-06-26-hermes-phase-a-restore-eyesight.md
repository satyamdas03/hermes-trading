# Hermes Phase A — Restore Eyesight + Deploy — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore the discarded fitness gradient to Hermes's self-learning optimizer (the −0.04 score-floor bug), deploy the already-written `a5394a7` fixes, reseed the degraded live strategy, and ship a performance-verification harness that proves recovery on live data.

**Architecture:** Phase A is surgical. One logic line in `score.py` un-blinds the optimizer and the revert-guard. The trend filter / multi-exchange / cadence code already exists in committed `a5394a7` and activates on deploy. A reseed env-flag resets the live strategy to a sound v59. A new read-only `verify_performance.py` harness pulls the public `/hermes` API and reports PASS/FAIL health signals so we can measure before vs after.

**Tech Stack:** Python 3.11, numpy, pyyaml, httpx, ccxt; uv for deps; pytest for tests; Railway for deploy.

## Global Constraints

- Python `>=3.11` (from `pyproject.toml`); ccxt `>=4.4.0`.
- Deployed repo: `github.com/satyamdas03/hermes-trading`, branch `main`, on **Railway** (start cmd `uv run python -m hermes_trading.run`).
- Live strategy/goal YAML live on Railway's **persistent volume**, which shadows committed `state/*.yaml`. Code changes alone never mutate live strategy — the reseed flag does.
- Public live data (no secret): `https://neuralquant.onrender.com/hermes/{status,trades,reflections}`.
- Preserve autonomy: do NOT hard-code trade decisions or freeze the strategy. Phase A only restores a signal and deploys existing self-mod code.
- Run tests with `uv run pytest` (deps numpy/pyyaml/httpx are not guaranteed in the bare local interpreter).
- One concern per commit; commit after each task.

---

### Task 1: Restore the score gradient (the −0.04 floor bug)

**Files:**
- Modify: `hermes_trading/score.py:144`
- Test: `tests/test_score_gradient.py`

**Interfaces:**
- Consumes: `hermes_trading.score.score(trades: list[dict], goal: dict) -> float`
- Produces: `score()` returns the real composite clamped to `[-1.0, 1.0]` (no longer floored at `goal.failure_below`), so two differently-bad strategies return different scores.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_score_gradient.py
"""Phase A — prove the optimizer sees a real gradient, not a flat -0.04 floor."""
from hermes_trading.score import score

GOAL = {
    "target_return_30d": 0.05,
    "max_drawdown": 0.08,
    "min_sharpe": 1.2,
    "failure_below": -0.04,
}

def _losers(n: int, pnl_usd: float, pnl_pct: float) -> list[dict]:
    return [
        {"status": "closed", "pnl_usd": pnl_usd, "pnl_pct": pnl_pct, "fee_usd": 14.5}
        for _ in range(n)
    ]

def test_gradient_exists_between_two_bad_strategies():
    # Mild bleed: cumulative -1200 -> ~12% drawdown.
    mild = _losers(10, -120.0, -3.0)
    # Severe bleed: cumulative -3000 -> ~30% drawdown.
    severe = _losers(10, -300.0, -3.0)
    s_mild = score(mild, GOAL)
    s_severe = score(severe, GOAL)
    # The whole point: a worse strategy must score strictly lower (gradient),
    # instead of both pinning at the -0.04 floor.
    assert s_severe < s_mild, f"no gradient: severe={s_severe} mild={s_mild}"

def test_score_floored_at_negative_one_not_failure_below():
    severe = _losers(10, -300.0, -3.0)  # composite well below -1
    s = score(severe, GOAL)
    assert s >= -1.0
    assert s < -0.04, f"still clamped at failure_below: {s}"

def test_good_strategy_still_positive():
    winners = [
        {"status": "closed", "pnl_usd": 120.0, "pnl_pct": 3.5, "fee_usd": 14.5}
        for _ in range(10)
    ]
    assert score(winners, GOAL) > 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_score_gradient.py -v`
Expected: `test_gradient_exists_between_two_bad_strategies` and `test_score_floored_at_negative_one_not_failure_below` FAIL — both bad strategies return `-0.04` (current clamp), so `severe < mild` is false and `s < -0.04` is false.

- [ ] **Step 3: Write minimal implementation**

In `hermes_trading/score.py`, change the final return (line 144) from:

```python
    return max(failure_below, min(1.0, composite))
```

to:

```python
    # Clamp to the documented [-1, +1] range. NOTE: do NOT floor at
    # goal.failure_below — that is a *goal threshold* (what return counts as
    # "failing"), not a score floor. Flooring here pinned every underwater
    # strategy at -0.04, erasing the gradient the reflection optimizer and the
    # revert-guard rely on (they read this returned value). See score.py:136 —
    # the real composite was logged but discarded one line later.
    return max(-1.0, min(1.0, composite))
```

Leave `failure_below = goal.get("failure_below", -0.04)` in place (line 113) — it stays a goal field; it is simply no longer used as the clamp.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_score_gradient.py -v`
Expected: PASS (all three tests).

- [ ] **Step 5: Commit**

```bash
git add hermes_trading/score.py tests/test_score_gradient.py
git commit -m "fix(score): stop flooring score at failure_below — restore optimizer gradient

The optimizer + revert-guard read score()'s return value, which was clamped to
-0.04 once underwater, erasing all gradient and causing the strategy random-walk.
The real composite was logged (score.py:136) then discarded at the return.
Clamp to the documented [-1,1] instead.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Prove the revert-guard now actually fires

**Files:**
- Test: `tests/test_revert_guard_gradient.py`
- (No source change — this locks in the behavioral payoff of Task 1.)

**Interfaces:**
- Consumes: `hermes_trading.reflect._maybe_revert(valid_trades: list[dict], goal: dict) -> dict | None`, plus module path constants `reflect.STRATEGY_PATH`, `reflect.HISTORY_DIR`, `reflect.SCORE_STATE_PATH`.
- Produces: regression guarantee that a degrading change rolls back (with the unclamped score) and a holding change does not.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_revert_guard_gradient.py
"""With the gradient restored (Task 1), the revert-guard must actually roll back
a degrading change and leave a holding change alone."""
import json
import yaml
import hermes_trading.reflect as reflect

GOAL = {
    "target_return_30d": 0.05,
    "max_drawdown": 0.08,
    "min_sharpe": 1.2,
    "failure_below": -0.04,
    "one_variable_only": True,
}

def _point_state_at(tmp_path, monkeypatch):
    """Redirect reflect's module-level paths into a temp dir."""
    state = tmp_path
    hist = state / "history"
    hist.mkdir()
    monkeypatch.setattr(reflect, "STATE_DIR", state)
    monkeypatch.setattr(reflect, "STRATEGY_PATH", state / "strategy.yaml")
    monkeypatch.setattr(reflect, "HISTORY_DIR", hist)
    monkeypatch.setattr(reflect, "SCORE_STATE_PATH", state / "reflect_score.json")
    monkeypatch.setattr(reflect, "HYPOTHESES_PATH", state / "hypotheses.jsonl")
    return state, hist

def _losers(n, pnl_usd):
    return [{"status": "closed", "pnl_usd": pnl_usd, "pnl_pct": -3.0, "fee_usd": 14.5} for _ in range(n)]

def test_degrading_change_is_reverted(tmp_path, monkeypatch):
    state, hist = _point_state_at(tmp_path, monkeypatch)
    # Predecessor config (the good one we should roll back TO)
    (hist / "v40.yaml").write_text(yaml.dump(
        {"version": "40", "entry": {"indicator": "rsi", "threshold_long": 30}, "stop_loss_pct": 2.0}))
    # Current on-disk strategy is v41 (the change being judged)
    (state / "strategy.yaml").write_text(yaml.dump(
        {"version": "41", "entry": {"indicator": "rsi", "threshold_long": 18}, "stop_loss_pct": 2.0}))
    # Score state: v41 applied after leaving v40 whose recent score was a mild -0.17.
    (state / "reflect_score.json").write_text(json.dumps(
        {"version": "41", "predecessor": "40", "score": -0.17}))
    # Recent window is now much worse than -0.17 (severe drawdown) -> should revert.
    reverted = reflect._maybe_revert(_losers(20, -300.0), GOAL)
    assert reverted is not None, "guard failed to fire on a degrading change"
    assert reverted["entry"]["threshold_long"] == 30  # rolled back to v40 config
    assert reverted["version"] == "42"               # bumped, not mutated in place
    on_disk = yaml.safe_load((state / "strategy.yaml").read_text())
    assert on_disk["version"] == "42"

def test_holding_change_is_not_reverted(tmp_path, monkeypatch):
    state, hist = _point_state_at(tmp_path, monkeypatch)
    (hist / "v40.yaml").write_text(yaml.dump(
        {"version": "40", "entry": {"indicator": "rsi", "threshold_long": 30}, "stop_loss_pct": 2.0}))
    (state / "strategy.yaml").write_text(yaml.dump(
        {"version": "41", "entry": {"indicator": "rsi", "threshold_long": 18}, "stop_loss_pct": 2.0}))
    # Prior score was a severe -0.90; current window is only a mild bleed -> change held up.
    (state / "reflect_score.json").write_text(json.dumps(
        {"version": "41", "predecessor": "40", "score": -0.90}))
    reverted = reflect._maybe_revert(_losers(20, -120.0), GOAL)
    assert reverted is None, "guard reverted a change that actually improved the score"
```

- [ ] **Step 2: Run test to verify it fails (or errors) on the pre-fix clamp**

Run: `uv run pytest tests/test_revert_guard_gradient.py -v`
Expected (if run against pre-Task-1 code): `test_degrading_change_is_reverted` FAILS — with the −0.04 clamp, `current_score (−0.04) >= prev (−0.17) − 0.05` is true, so the guard never fires and `reverted is None`. After Task 1 it passes. (If running after Task 1 is already committed, write the test first and confirm it captures the behavior, then proceed.)

- [ ] **Step 3: Implementation**

No source change required — Task 1 already restored the gradient. If a path constant referenced in the test does not exist in `reflect.py`, reconcile the test to the actual constant names rather than renaming source.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_revert_guard_gradient.py -v`
Expected: PASS (both tests).

- [ ] **Step 5: Commit**

```bash
git add tests/test_revert_guard_gradient.py
git commit -m "test(reflect): lock in revert-guard firing now that the gradient is restored

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Performance-verification harness (the before/after health check)

**Files:**
- Create: `scripts/verify_performance.py`
- Test: `tests/test_verify_performance.py`

**Interfaces:**
- Consumes: live JSON from `https://neuralquant.onrender.com/hermes/{status,reflections,trades}`.
- Produces, all pure (testable on fixtures, no network):
  - `score_dispersion(reflections: list[dict]) -> float` — stdev of `score_before` over the records (0.0 ⇒ flat/blind).
  - `has_revert_guard(reflections: list[dict]) -> bool` — any record with `reflector == "revert-guard"`.
  - `version_churn_per_day(reflections: list[dict]) -> float` — reflections per 24h over the spanned window.
  - `fee_to_pnl_ratio(status: dict) -> float` — `cumulative_fees_usd / max(|total_pnl_usd|, 1)`.
  - `health_report(status: dict, reflections: list[dict]) -> dict` — `{checks: {name: {value, pass: bool, target}}, healthy: bool}`.
  - `main()` — fetch live, print the report, exit 0 if healthy else 1.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_verify_performance.py
from scripts.verify_performance import (
    score_dispersion, has_revert_guard, version_churn_per_day,
    fee_to_pnl_ratio, health_report,
)

SICK_REFLECTIONS = [
    {"timestamp": "2026-06-25T13:41:43+00:00", "reflector": "claude", "score_before": -0.04},
    {"timestamp": "2026-06-25T13:56:09+00:00", "reflector": "claude", "score_before": -0.04},
    {"timestamp": "2026-06-25T23:31:22+00:00", "reflector": "claude", "score_before": -0.04},
]
HEALTHY_REFLECTIONS = [
    {"timestamp": "2026-06-26T02:00:00+00:00", "reflector": "claude", "score_before": -0.31},
    {"timestamp": "2026-06-26T10:00:00+00:00", "reflector": "revert-guard", "score_before": -0.52},
    {"timestamp": "2026-06-26T20:00:00+00:00", "reflector": "claude", "score_before": -0.18},
]
SICK_STATUS = {"aggregates": {"total_pnl_usd": 32.8}, "heartbeat": {"cumulative_fees_usd": 1851.67}}

def test_score_dispersion_zero_when_flat():
    assert score_dispersion(SICK_REFLECTIONS) == 0.0

def test_score_dispersion_positive_when_varied():
    assert score_dispersion(HEALTHY_REFLECTIONS) > 0.0

def test_has_revert_guard():
    assert has_revert_guard(SICK_REFLECTIONS) is False
    assert has_revert_guard(HEALTHY_REFLECTIONS) is True

def test_fee_to_pnl_ratio_flags_bleed():
    assert fee_to_pnl_ratio(SICK_STATUS) > 50.0  # fees 56x the net P&L

def test_version_churn_per_day():
    # 3 reflections spanning 18h -> ~4/day
    churn = version_churn_per_day(HEALTHY_REFLECTIONS)
    assert 3.0 <= churn <= 5.0

def test_health_report_sick_fails():
    rep = health_report(SICK_STATUS, SICK_REFLECTIONS)
    assert rep["healthy"] is False
    assert rep["checks"]["score_gradient_alive"]["pass"] is False

def test_health_report_healthy_passes():
    healthy_status = {"aggregates": {"total_pnl_usd": 300.0}, "heartbeat": {"cumulative_fees_usd": 200.0}}
    rep = health_report(healthy_status, HEALTHY_REFLECTIONS)
    assert rep["checks"]["score_gradient_alive"]["pass"] is True
    assert rep["checks"]["revert_guard_active"]["pass"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_verify_performance.py -v`
Expected: FAIL — `ModuleNotFoundError: scripts.verify_performance`.

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/verify_performance.py
"""Read-only Hermes performance health check.

Pulls the public /hermes API and reports whether the self-learning loop is
healthy: a live score gradient (not the -0.04 flatline), an active revert-guard,
sane version churn, and a fee/P&L ratio that isn't bleeding. Exit 0 = healthy.

Run a baseline BEFORE the Phase A deploy/reseed, then again AFTER to prove recovery.
"""
from __future__ import annotations

import statistics
import sys
from datetime import datetime

BASE = "https://neuralquant.onrender.com/hermes"


def score_dispersion(reflections: list[dict]) -> float:
    scores = [r.get("score_before") for r in reflections if r.get("score_before") is not None]
    if len(scores) < 2:
        return 0.0
    return float(statistics.pstdev(scores))


def has_revert_guard(reflections: list[dict]) -> bool:
    return any(r.get("reflector") == "revert-guard" for r in reflections)


def version_churn_per_day(reflections: list[dict]) -> float:
    stamps = []
    for r in reflections:
        ts = r.get("timestamp")
        if ts:
            try:
                stamps.append(datetime.fromisoformat(ts))
            except ValueError:
                pass
    if len(stamps) < 2:
        return 0.0
    span_hours = (max(stamps) - min(stamps)).total_seconds() / 3600.0
    if span_hours <= 0:
        return 0.0
    # intervals = len-1 changes across the span
    return (len(stamps) - 1) / (span_hours / 24.0)


def fee_to_pnl_ratio(status: dict) -> float:
    agg = status.get("aggregates", {})
    hb = status.get("heartbeat", {})
    fees = hb.get("cumulative_fees_usd", 0.0) or 0.0
    pnl = agg.get("total_pnl_usd", 0.0) or 0.0
    return fees / max(abs(pnl), 1.0)


def health_report(status: dict, reflections: list[dict]) -> dict:
    disp = score_dispersion(reflections)
    guard = has_revert_guard(reflections)
    churn = version_churn_per_day(reflections)
    fee_ratio = fee_to_pnl_ratio(status)

    checks = {
        "score_gradient_alive": {
            "value": round(disp, 4), "target": "> 0 (not a flat -0.04)", "pass": disp > 0.0},
        "revert_guard_active": {
            "value": guard, "target": "at least one revert-guard entry", "pass": guard},
        "version_churn_sane": {
            "value": round(churn, 2), "target": "<= 8 changes/day", "pass": churn <= 8.0},
        "fee_bleed_controlled": {
            "value": round(fee_ratio, 2), "target": "fees < 5x net P&L", "pass": fee_ratio < 5.0},
    }
    return {"checks": checks, "healthy": all(c["pass"] for c in checks.values())}


def main() -> int:
    import httpx

    with httpx.Client(timeout=30) as client:
        status = client.get(f"{BASE}/status").json()
        reflections = client.get(f"{BASE}/reflections?n=30").json().get("reflections", [])

    rep = health_report(status, reflections)
    print(f"Strategy v{status.get('strategy', {}).get('version', '?')} | "
          f"WR {status.get('aggregates', {}).get('win_rate_pct', '?')}% | "
          f"net ${status.get('aggregates', {}).get('total_pnl_usd', '?')} | "
          f"fees ${status.get('heartbeat', {}).get('cumulative_fees_usd', '?')}")
    print("-" * 60)
    for name, c in rep["checks"].items():
        flag = "PASS" if c["pass"] else "FAIL"
        print(f"[{flag}] {name:24} = {c['value']!s:10} (target: {c['target']})")
    print("-" * 60)
    print("HEALTHY" if rep["healthy"] else "UNHEALTHY")
    return 0 if rep["healthy"] else 1


if __name__ == "__main__":
    sys.exit(main())
```

Add an empty `scripts/__init__.py` so `from scripts.verify_performance import ...` resolves under pytest:

```bash
touch scripts/__init__.py
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_verify_performance.py -v`
Expected: PASS (all tests).

- [ ] **Step 5: Capture the BASELINE (sick state) and commit**

```bash
uv run python -m scripts.verify_performance | tee docs/superpowers/plans/phase-a-baseline.txt
git add scripts/verify_performance.py scripts/__init__.py tests/test_verify_performance.py docs/superpowers/plans/phase-a-baseline.txt
git commit -m "feat(verify): read-only Hermes performance health harness + sick baseline

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```
Expected baseline output: `UNHEALTHY` — `score_gradient_alive` FAIL (flat -0.04), `revert_guard_active` FAIL, `fee_bleed_controlled` FAIL (~56x).

---

### Task 4: Push + deploy + reseed (operator runbook)

**Files:** none (operations on Railway). This task is a checklist with exact commands and expected log lines.

**Interfaces:**
- Consumes: commits from Tasks 1–3 on top of `a5394a7`.
- Produces: live Railway service running the gradient fix + trend filter + multi-exchange + slow cadence, with strategy reset to v59.

- [ ] **Step 1: Push the Phase A commits**

```bash
git push origin main
git log --oneline -5   # confirm score fix + tests + harness sit on top of a5394a7
```

- [ ] **Step 2: Confirm Railway picked up the deploy**

In the Railway dashboard for the hermes service, confirm a new deployment built from the latest `main` commit. If auto-deploy is off, trigger a manual deploy. Expected boot logs include `Trading loop starting` and (within a few ticks) no per-call market reloads.

- [ ] **Step 3: Reseed the degraded live strategy (one-shot)**

Set env var `RESEED_STRATEGY=true` on the Railway service, then restart/redeploy ONCE. Expected boot logs:
```
RESEED: backed up live strategy v66 -> state/history/v66.yaml
RESEED: strategy reset to v59 ...
RESEED: goal.reflection_every bumped to 20
RESEED complete
```
(The marker `.reseeded_v59` makes it idempotent — it will not re-fire on later restarts.)

- [ ] **Step 4: Remove the flag**

Delete the `RESEED_STRATEGY` env var (cleanliness; marker already guards re-fire). No restart needed.

- [ ] **Step 5: Smoke-confirm the live strategy shape**

```bash
curl -s https://neuralquant.onrender.com/hermes/status | python -c "import sys,json; d=json.load(sys.stdin); s=d['strategy']; print('version', s['version']); print('direction', s['entry'].get('direction')); print('trend_filter', s['entry'].get('trend_filter')); print('reflection_every', d['goal'].get('reflection_every'))"
```
Expected: `version 59`, `direction both`, `trend_filter True`, `reflection_every 20`.

No commit (operations only).

---

### Task 5: Verify recovery on live data (the performance test)

**Files:** none (re-run of the Task 3 harness after the loop has traded post-reseed).

**Interfaces:**
- Consumes: `scripts/verify_performance.py`, live `/hermes` data accumulated for ≥ ~12–24h after reseed.

- [ ] **Step 1: Re-run the harness after the loop has traded post-reseed**

Run (≥12h after Task 4, so enough closed trades + reflections accumulate):
```bash
uv run python -m scripts.verify_performance | tee docs/superpowers/plans/phase-a-after.txt
```

- [ ] **Step 2: Confirm the recovery signals**

Compare `phase-a-after.txt` against `phase-a-baseline.txt`. Expected improvements:
- `score_gradient_alive` → **PASS** (`score_before` now varies; reflection tape no longer a flat `-0.04`).
- `revert_guard_active` → **PASS** (at least one `revert-guard` entry in `/hermes/reflections`).
- `version_churn_sane` → **PASS** (cadence 20 ⇒ far fewer version bumps/day).
- Spot-check `/hermes/reflections`: shorts now appear in the bear regime; falling-knife long entries drop.

- [ ] **Step 3: Record the result and decide**

```bash
git add docs/superpowers/plans/phase-a-after.txt
git commit -m "docs(verify): Phase A post-reseed recovery snapshot

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
git push origin main
```
If healthy → proceed to the Phase B plan. If `fee_bleed_controlled` is still failing after recovery, that confirms the structural fee/expectancy issue and is the explicit trigger to prioritize Phase B's wider action space (TP/SL tuning) — note it for the Phase B plan.

---

## Self-Review

**Spec coverage (Phase A section of the design spec):**
- A1 score gradient fix → Task 1. ✓
- A2 deploy `a5394a7` → Task 4 Steps 1–2. ✓
- A3 reseed v59 + reflection_every 20 → Task 4 Steps 3–5. ✓
- A4 bundle score fix on top of a5394a7 as one deploy → Tasks 1–3 commits then Task 4 push. ✓
- A "verify live" criteria (score varies, revert-guard appears, shorts fire, fewer trades, slower churn, lower fees) → Tasks 3 + 5 (harness encodes them). ✓
- Revert-guard un-neutering (the spec's stated payoff of the score fix) → Task 2. ✓

**Placeholder scan:** No TBD/TODO; every code step shows complete code; operator steps give exact commands + expected output. ✓

**Type consistency:** `score(trades, goal) -> float` used consistently. `_maybe_revert(valid_trades, goal) -> dict | None` matches `reflect.py`. Harness function names (`score_dispersion`, `has_revert_guard`, `version_churn_per_day`, `fee_to_pnl_ratio`, `health_report`) identical between `verify_performance.py` and its test. `health_report` check key `score_gradient_alive` referenced identically in source and test. ✓

**Note for executor:** Task 2's path constants (`reflect.STRATEGY_PATH`, `reflect.HISTORY_DIR`, `reflect.SCORE_STATE_PATH`, `reflect.HYPOTHESES_PATH`) were confirmed present in `reflect.py` during spec research. If any differs, adjust the test's monkeypatch targets to the real names — do not rename source.
