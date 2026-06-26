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
