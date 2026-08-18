#!/usr/bin/env python3
"""Local, immutable DCDN elapsed-time and estimated-cost launch guard.

This module deliberately has no provider API, credential, network, or Pod
lifecycle integration. It only evaluates the owner-approved frozen accounting
policy against a caller-supplied ETA and the local wall clock.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import ROUND_CEILING, ROUND_HALF_UP, Decimal
from typing import Literal

LEDGER_EPOCH_UTC = datetime(2026, 8, 18, 9, 13, 18, tzinfo=UTC)
POD_RATE_USD_PER_HOUR = Decimal("3.29")
STORAGE_RESERVE_USD = Decimal("25")
WARNING_USD = Decimal("370")
HARD_CAP_USD = Decimal("460")
MAX_ELAPSED = timedelta(hours=120)
ABSOLUTE_STOP_UTC = datetime(2026, 8, 24, 11, 0, 0, tzinfo=UTC)
RESERVE_DURATION = timedelta(hours=8)
ELAPSED_STOP_UTC = LEDGER_EPOCH_UTC + MAX_ELAPSED
EFFECTIVE_STOP_UTC = min(ELAPSED_STOP_UTC, ABSOLUTE_STOP_UTC)
RESERVE_START_UTC = EFFECTIVE_STOP_UTC - RESERVE_DURATION

Phase = Literal["compute", "aggregation"]
WorkClass = Literal["mandatory", "optional"]
_MONEY_QUANTUM = Decimal("0.01")
_MICROSECONDS_PER_HOUR = Decimal(3_600_000_000)


def parse_datetime(value: str) -> datetime:
    """Parse an ISO-8601 timestamp and normalize it to UTC."""

    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        raise ValueError("timestamp must include an explicit timezone")
    return parsed.astimezone(UTC)


def _decimal_hours(delta: timedelta) -> Decimal:
    microseconds = (
        delta.days * 86_400_000_000 + delta.seconds * 1_000_000 + delta.microseconds
    )
    return Decimal(microseconds) / _MICROSECONDS_PER_HOUR


def _add_decimal_hours(moment: datetime, hours: Decimal) -> datetime:
    microseconds = (hours * _MICROSECONDS_PER_HOUR).to_integral_value(
        rounding=ROUND_CEILING
    )
    return moment + timedelta(microseconds=int(microseconds))


def estimate_spend_usd(at: datetime) -> Decimal:
    """Estimate incremental compute plus the frozen storage reserve."""

    if at.tzinfo is None:
        raise ValueError("evaluation time must be timezone-aware")
    at_utc = at.astimezone(UTC)
    if at_utc < LEDGER_EPOCH_UTC:
        raise ValueError("evaluation time precedes immutable ledger_epoch_utc")
    elapsed_hours = _decimal_hours(at_utc - LEDGER_EPOCH_UTC)
    return STORAGE_RESERVE_USD + POD_RATE_USD_PER_HOUR * elapsed_hours


@dataclass(frozen=True)
class CostSnapshot:
    now_utc: datetime
    projected_end_utc: datetime
    phase: Phase
    work_class: WorkClass
    eta_hours: Decimal
    elapsed_hours: Decimal
    estimated_spend_usd: Decimal
    projected_spend_usd: Decimal
    status: str
    launch_allowed: bool
    reasons: tuple[str, ...]

    @staticmethod
    def _money(value: Decimal) -> str:
        return str(value.quantize(_MONEY_QUANTUM, rounding=ROUND_HALF_UP))

    def as_dict(self) -> dict[str, object]:
        return {
            "accounting_basis": "local_owner_approved_estimate",
            "control_plane_billing": "unavailable_and_owner_waived",
            "ledger_epoch_utc": LEDGER_EPOCH_UTC.isoformat().replace("+00:00", "Z"),
            "now_utc": self.now_utc.isoformat().replace("+00:00", "Z"),
            "projected_end_utc": self.projected_end_utc.isoformat().replace(
                "+00:00", "Z"
            ),
            "elapsed_stop_utc": ELAPSED_STOP_UTC.isoformat().replace("+00:00", "Z"),
            "absolute_stop_utc": ABSOLUTE_STOP_UTC.isoformat().replace("+00:00", "Z"),
            "effective_stop_utc": EFFECTIVE_STOP_UTC.isoformat().replace("+00:00", "Z"),
            "reserve_start_utc": RESERVE_START_UTC.isoformat().replace("+00:00", "Z"),
            "phase": self.phase,
            "work_class": self.work_class,
            "eta_hours": str(self.eta_hours),
            "elapsed_hours": str(self.elapsed_hours.quantize(Decimal("0.000001"))),
            "pod_rate_usd_per_hour": str(POD_RATE_USD_PER_HOUR),
            "storage_reserve_usd": str(STORAGE_RESERVE_USD),
            "warning_usd": str(WARNING_USD),
            "hard_cap_usd": str(HARD_CAP_USD),
            "estimated_spend_usd": self._money(self.estimated_spend_usd),
            "projected_spend_usd": self._money(self.projected_spend_usd),
            "status": self.status,
            "launch_allowed": self.launch_allowed,
            "reasons": list(self.reasons),
            "provider_actions_performed": False,
        }


def evaluate_cost(
    now: datetime,
    *,
    eta_hours: Decimal = Decimal(0),
    phase: Phase = "compute",
    work_class: WorkClass = "mandatory",
) -> CostSnapshot:
    """Evaluate whether a local task may start under the frozen policy."""

    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    if eta_hours < 0:
        raise ValueError("eta_hours must be non-negative")
    if phase not in ("compute", "aggregation"):
        raise ValueError(f"unknown phase: {phase}")
    if work_class not in ("mandatory", "optional"):
        raise ValueError(f"unknown work class: {work_class}")

    now_utc = now.astimezone(UTC)
    projected_end = _add_decimal_hours(now_utc, eta_hours)
    estimated = estimate_spend_usd(now_utc)
    projected = estimate_spend_usd(projected_end)
    reasons: list[str] = []

    if now_utc >= EFFECTIVE_STOP_UTC:
        reasons.append("effective hard time limit has been reached")
    if projected_end > EFFECTIVE_STOP_UTC:
        reasons.append("ETA crosses the effective hard time limit")
    if estimated >= HARD_CAP_USD:
        reasons.append("estimated spend has reached the hard cap")
    if projected >= HARD_CAP_USD:
        reasons.append("projected spend reaches the hard cap")

    if phase == "compute":
        if now_utc >= RESERVE_START_UTC:
            reasons.append("final eight-hour aggregation reserve has begun")
        elif projected_end > RESERVE_START_UTC:
            reasons.append("compute ETA crosses into the final aggregation reserve")

    warning_reached = max(estimated, projected) >= WARNING_USD
    if work_class == "optional" and warning_reached:
        reasons.append("optional work is disabled at the warning threshold")

    hard_now = now_utc >= EFFECTIVE_STOP_UTC or estimated >= HARD_CAP_USD
    if hard_now:
        status = "hard_stop"
    elif now_utc >= RESERVE_START_UTC:
        status = "aggregation_reserve"
    elif warning_reached:
        status = "warning"
    else:
        status = "ok"

    return CostSnapshot(
        now_utc=now_utc,
        projected_end_utc=projected_end,
        phase=phase,
        work_class=work_class,
        eta_hours=eta_hours,
        elapsed_hours=_decimal_hours(now_utc - LEDGER_EPOCH_UTC),
        estimated_spend_usd=estimated,
        projected_spend_usd=projected,
        status=status,
        launch_allowed=not reasons,
        reasons=tuple(reasons),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--now",
        type=parse_datetime,
        default=None,
        help="timezone-aware ISO-8601 time; defaults to the current UTC time",
    )
    parser.add_argument(
        "--eta-hours",
        type=Decimal,
        default=Decimal(0),
        help="conservative elapsed ETA for the proposed task",
    )
    parser.add_argument(
        "--phase", choices=("compute", "aggregation"), default="compute"
    )
    parser.add_argument(
        "--work-class", choices=("mandatory", "optional"), default="mandatory"
    )
    parser.add_argument(
        "--json", action="store_true", help="emit machine-readable JSON"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        snapshot = evaluate_cost(
            args.now or datetime.now(UTC),
            eta_hours=args.eta_hours,
            phase=args.phase,
            work_class=args.work_class,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error

    payload = snapshot.as_dict()
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        verdict = "ALLOW" if snapshot.launch_allowed else "BLOCK"
        print(
            f"{verdict}: status={snapshot.status}; "
            f"estimated=${payload['estimated_spend_usd']}; "
            f"projected=${payload['projected_spend_usd']}; "
            f"reserve_start={payload['reserve_start_utc']}"
        )
        for reason in snapshot.reasons:
            print(f"- {reason}")
    return 0 if snapshot.launch_allowed else 2


if __name__ == "__main__":
    raise SystemExit(main())
