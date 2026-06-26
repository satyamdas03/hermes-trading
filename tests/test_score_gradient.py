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
