from scripts.verify_performance import (
    score_dispersion, floor_pin_ratio, has_revert_guard, version_churn_per_day,
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


def test_floor_pin_ratio_detects_blind_loop():
    # All recent scores pinned at the -0.04 floor -> blind.
    assert floor_pin_ratio(SICK_REFLECTIONS) == 1.0
    # Varied scores -> not pinned.
    assert floor_pin_ratio(HEALTHY_REFLECTIONS) == 0.0


def test_floor_pin_ratio_transition_not_fooled():
    # Slide into the floor then 5 pinned: dispersion would look "alive" but the
    # loop IS blind. Last-6 -> 5/6 pinned -> flagged.
    transition = [
        {"score_before": 0.30}, {"score_before": 0.17}, {"score_before": 0.036},
        {"score_before": -0.04}, {"score_before": -0.04}, {"score_before": -0.04},
        {"score_before": -0.04}, {"score_before": -0.04},
    ]
    assert floor_pin_ratio(transition) >= 0.6


def test_has_revert_guard():
    assert has_revert_guard(SICK_REFLECTIONS) is False
    assert has_revert_guard(HEALTHY_REFLECTIONS) is True


def test_fee_to_pnl_ratio_flags_bleed():
    assert fee_to_pnl_ratio(SICK_STATUS) > 50.0  # fees 56x the net P&L


def test_version_churn_per_day():
    # 2 changes (3 reflections) spanning 18h -> 2/(18/24) = 2.67/day
    churn = version_churn_per_day(HEALTHY_REFLECTIONS)
    assert 2.0 <= churn <= 3.5


def test_health_report_sick_fails():
    rep = health_report(SICK_STATUS, SICK_REFLECTIONS)
    assert rep["healthy"] is False
    assert rep["checks"]["score_gradient_alive"]["pass"] is False


def test_health_report_healthy_passes():
    healthy_status = {"aggregates": {"total_pnl_usd": 300.0}, "heartbeat": {"cumulative_fees_usd": 200.0}}
    rep = health_report(healthy_status, HEALTHY_REFLECTIONS)
    assert rep["checks"]["score_gradient_alive"]["pass"] is True
    assert rep["checks"]["revert_guard_active"]["pass"] is True
