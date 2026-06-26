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
    # gross profit magnitude 114.5*2; fees 14.5*2=29
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
