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
