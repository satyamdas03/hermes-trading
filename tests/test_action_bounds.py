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
