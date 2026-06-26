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
    return [
        {"status": "closed", "pnl_usd": pnl_usd, "pnl_pct": -3.0, "fee_usd": 14.5}
        for _ in range(n)
    ]


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
