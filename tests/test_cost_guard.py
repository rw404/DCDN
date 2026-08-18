from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from scripts.cost_guard import (
    ABSOLUTE_STOP_UTC,
    EFFECTIVE_STOP_UTC,
    ELAPSED_STOP_UTC,
    HARD_CAP_USD,
    LEDGER_EPOCH_UTC,
    RESERVE_START_UTC,
    STORAGE_RESERVE_USD,
    WARNING_USD,
    estimate_spend_usd,
    evaluate_cost,
    parse_datetime,
)


def test_frozen_epoch_and_effective_stop() -> None:
    assert datetime(2026, 8, 18, 9, 13, 18, tzinfo=UTC) == LEDGER_EPOCH_UTC
    assert datetime(2026, 8, 23, 9, 13, 18, tzinfo=UTC) == ELAPSED_STOP_UTC
    assert min(ELAPSED_STOP_UTC, ABSOLUTE_STOP_UTC) == EFFECTIVE_STOP_UTC
    assert EFFECTIVE_STOP_UTC - timedelta(hours=8) == RESERVE_START_UTC


def test_estimate_uses_elapsed_pod_time_plus_storage_reserve() -> None:
    assert estimate_spend_usd(LEDGER_EPOCH_UTC) == STORAGE_RESERVE_USD
    assert estimate_spend_usd(LEDGER_EPOCH_UTC + timedelta(hours=2)) == Decimal("31.58")


def test_time_before_epoch_is_rejected() -> None:
    with pytest.raises(ValueError, match="precedes immutable"):
        estimate_spend_usd(LEDGER_EPOCH_UTC - timedelta(microseconds=1))


def test_naive_estimate_time_is_rejected() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        estimate_spend_usd(datetime(2026, 8, 18, 9, 13, 18))


def test_compute_eta_may_not_enter_final_reserve() -> None:
    snapshot = evaluate_cost(
        RESERVE_START_UTC - timedelta(hours=1), eta_hours=Decimal("1.01")
    )
    assert not snapshot.launch_allowed
    assert any("aggregation reserve" in reason for reason in snapshot.reasons)


def test_aggregation_is_allowed_inside_reserve_before_hard_stop() -> None:
    snapshot = evaluate_cost(
        RESERVE_START_UTC + timedelta(hours=1),
        eta_hours=Decimal("1"),
        phase="aggregation",
    )
    assert snapshot.status == "aggregation_reserve"
    assert snapshot.launch_allowed


def test_optional_work_stops_at_warning_threshold() -> None:
    now = LEDGER_EPOCH_UTC + timedelta(hours=105)
    assert estimate_spend_usd(now) >= WARNING_USD
    optional = evaluate_cost(now, work_class="optional")
    mandatory = evaluate_cost(now, work_class="mandatory")
    assert optional.status == "warning"
    assert not optional.launch_allowed
    assert mandatory.launch_allowed


def test_hard_time_limit_blocks_every_phase() -> None:
    for phase in ("compute", "aggregation"):
        snapshot = evaluate_cost(EFFECTIVE_STOP_UTC, phase=phase)
        assert snapshot.status == "hard_stop"
        assert not snapshot.launch_allowed


def test_projected_hard_cap_is_conservative() -> None:
    snapshot = evaluate_cost(
        LEDGER_EPOCH_UTC,
        eta_hours=(HARD_CAP_USD - STORAGE_RESERVE_USD) / Decimal("3.29"),
        phase="aggregation",
    )
    assert not snapshot.launch_allowed
    assert snapshot.projected_spend_usd >= HARD_CAP_USD


def test_parser_requires_timezone_and_accepts_z() -> None:
    assert parse_datetime("2026-08-18T09:13:18Z") == LEDGER_EPOCH_UTC
    with pytest.raises(ValueError, match="timezone"):
        parse_datetime("2026-08-18T09:13:18")
