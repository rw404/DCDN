"""Preflight validation for materialized public DCDN operands.

The optimized training loop may disable expensive all-value assertions only
after this validator has scanned the complete frozen operand view.  It checks
the exact ordered schema, finite fallback values, semantic denominator mask,
strict positivity of active denominators, and floor/coverage incidence while
retaining stable row-ID evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import torch

SCHEMA_CONTRACT = "dcdn-feature-schema-v1"


class OperandValidationError(ValueError):
    """Validation failure carrying bounded audit identifiers."""

    def __init__(
        self,
        message: str,
        *,
        row_ids: Sequence[str] = (),
        feature_names: Sequence[str] = (),
        kind: str,
    ) -> None:
        super().__init__(message)
        self.row_ids = tuple(row_ids)
        self.feature_names = tuple(feature_names)
        self.kind = kind


@dataclass(frozen=True)
class DenominatorFieldStats:
    name: str
    index: int
    total_rows: int
    valid_rows: int
    missing_rows: int
    coverage: float
    floored_rows: int
    floor_fraction_of_valid: float
    minimum_valid_value: float | None
    maximum_valid_value: float | None
    missing_row_ids: tuple[str, ...]
    floored_row_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OperandValidationReport:
    contract: str
    schema_sha256: str
    row_ids_sha256: str
    rows: int
    features: int
    denominator_features: int
    total_denominator_cells: int
    active_denominator_cells: int
    overall_coverage: float
    floored_active_cells: int
    overall_floor_fraction: float
    div_eps: float
    field_stats: tuple[DenominatorFieldStats, ...]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["field_stats"] = [field.to_dict() for field in self.field_stats]
        return result


def _canonical_feature_names(feature_names: Sequence[str]) -> tuple[str, ...]:
    if isinstance(feature_names, str | bytes):
        raise TypeError("feature_names must be an ordered sequence of names")
    names = tuple(feature_names)
    if not names:
        raise ValueError("feature_names must not be empty")
    if any(not isinstance(name, str) or not name for name in names):
        raise TypeError("every feature name must be a non-empty string")
    if len(set(names)) != len(names):
        raise ValueError("feature_names must be unique and ordered")
    return names


def _validate_sha256(name: str, value: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    if any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def canonical_schema_sha256(feature_names: Sequence[str]) -> str:
    """Hash exact feature identity *and order* under a versioned contract."""

    names = _canonical_feature_names(feature_names)
    payload = json.dumps(
        {"contract": SCHEMA_CONTRACT, "feature_names": list(names)},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_schema(
    feature_names: Sequence[str],
    *,
    expected_feature_names: Sequence[str] | None = None,
    expected_schema_sha256: str | None = None,
) -> str:
    names = _canonical_feature_names(feature_names)
    if expected_feature_names is not None:
        expected = _canonical_feature_names(expected_feature_names)
        if names != expected:
            first_mismatch = next(
                (
                    index
                    for index, (received, wanted) in enumerate(
                        zip(names, expected, strict=False)
                    )
                    if received != wanted
                ),
                min(len(names), len(expected)),
            )
            raise ValueError(
                "feature schema order mismatch at index "
                f"{first_mismatch}: received={names}, expected={expected}"
            )
    digest = canonical_schema_sha256(names)
    if expected_schema_sha256 is not None:
        expected_digest = _validate_sha256(
            "expected_schema_sha256", expected_schema_sha256
        )
        if digest != expected_digest:
            raise ValueError(
                f"feature schema SHA-256 mismatch: {digest} != {expected_digest}"
            )
    return digest


def canonical_denominator_mask(
    feature_names: Sequence[str],
    denominator_feature_names: Sequence[str],
) -> torch.Tensor:
    """Construct the one semantic mask from exact named raw/base operands."""

    names = _canonical_feature_names(feature_names)
    if isinstance(denominator_feature_names, str | bytes):
        raise TypeError("denominator_feature_names must be a sequence")
    denominators = tuple(denominator_feature_names)
    if any(not isinstance(name, str) or not name for name in denominators):
        raise TypeError("denominator feature names must be non-empty strings")
    if len(set(denominators)) != len(denominators):
        raise ValueError("denominator feature names must be unique")
    unknown = sorted(set(denominators) - set(names))
    if unknown:
        raise ValueError("unknown denominator features: " + ", ".join(unknown))
    selected = set(denominators)
    return torch.tensor([name in selected for name in names], dtype=torch.bool)


def _normalize_row_id(value: Any) -> str:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise TypeError("each row ID tensor must contain exactly one value")
        value = value.detach().cpu().item()
    if value is None:
        raise ValueError("row IDs must not be None")
    return f"{type(value).__name__}:{value}"


def _canonical_row_ids(row_ids: Sequence[Any], row_count: int) -> tuple[str, ...]:
    if isinstance(row_ids, str | bytes):
        raise TypeError("row_ids must be an ordered sequence")
    try:
        normalized = tuple(_normalize_row_id(value) for value in row_ids)
    except TypeError:
        raise
    if len(normalized) != row_count:
        raise ValueError(f"row_ids must contain exactly {row_count} entries")
    if len(set(normalized)) != len(normalized):
        raise ValueError("row_ids must be unique")
    return normalized


def _row_ids_sha256(row_ids: Sequence[str]) -> str:
    payload = json.dumps(
        list(row_ids), separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _canonical_mask(mask: Sequence[bool] | torch.Tensor, features: int) -> torch.Tensor:
    result = mask.detach() if isinstance(mask, torch.Tensor) else torch.as_tensor(mask)
    if result.dtype != torch.bool:
        raise TypeError("denominator_mask must contain booleans")
    if result.ndim != 1 or result.numel() != features:
        raise ValueError(
            f"denominator_mask must be one-dimensional with {features} entries"
        )
    return result.to(device="cpu", dtype=torch.bool).clone()


def _sample_bad_cells(
    bad: torch.Tensor,
    row_ids: tuple[str, ...],
    feature_names: tuple[str, ...],
    limit: int,
) -> tuple[tuple[str, ...], tuple[str, ...], str]:
    coordinates = torch.nonzero(bad, as_tuple=False)[:limit].cpu().tolist()
    sampled_rows: list[str] = []
    sampled_features: list[str] = []
    rendered: list[str] = []
    for row_index, feature_index in coordinates:
        row_id = row_ids[row_index]
        feature = feature_names[feature_index]
        if row_id not in sampled_rows:
            sampled_rows.append(row_id)
        if feature not in sampled_features:
            sampled_features.append(feature)
        rendered.append(f"({row_id}, {feature})")
    return tuple(sampled_rows), tuple(sampled_features), ", ".join(rendered)


def validate_materialized_operands(
    values: torch.Tensor,
    *,
    row_ids: Sequence[Any],
    feature_names: Sequence[str],
    denominator_mask: Sequence[bool] | torch.Tensor,
    denominator_validity: torch.Tensor | None = None,
    div_eps: float = 1e-2,
    expected_feature_names: Sequence[str] | None = None,
    expected_schema_sha256: str | None = None,
    max_row_id_samples: int = 20,
) -> OperandValidationReport:
    """Validate one complete materialized ``[rows, features]`` operand view."""

    if not isinstance(values, torch.Tensor):
        raise TypeError("values must be a tensor")
    if values.ndim != 2:
        raise ValueError("values must have shape [rows, features]")
    if values.shape[0] == 0:
        raise ValueError("the materialized operand view must contain rows")
    if values.dtype not in (torch.float32, torch.bfloat16):
        if values.dtype == torch.float16:
            raise TypeError("FP16 materialized operands are forbidden")
        raise TypeError("materialized operands must use FP32 or BF16")
    if (
        isinstance(max_row_id_samples, bool)
        or not isinstance(max_row_id_samples, int)
        or max_row_id_samples <= 0
    ):
        raise ValueError("max_row_id_samples must be a positive integer")
    div_eps = float(div_eps)
    if not math.isfinite(div_eps) or div_eps <= 0:
        raise ValueError("div_eps must be finite and positive")

    names = _canonical_feature_names(feature_names)
    if values.shape[1] != len(names):
        raise ValueError("values width must match the ordered feature schema")
    schema_digest = validate_schema(
        names,
        expected_feature_names=expected_feature_names,
        expected_schema_sha256=expected_schema_sha256,
    )
    normalized_row_ids = _canonical_row_ids(row_ids, values.shape[0])
    mask_cpu = _canonical_mask(denominator_mask, values.shape[1])
    mask = mask_cpu.to(device=values.device)

    if denominator_validity is None:
        validity = torch.ones_like(values, dtype=torch.bool)
    else:
        if not isinstance(denominator_validity, torch.Tensor):
            raise TypeError("denominator_validity must be a tensor or None")
        if denominator_validity.shape != values.shape:
            raise ValueError("denominator_validity must match values.shape")
        if denominator_validity.dtype != torch.bool:
            raise TypeError("denominator_validity must be boolean")
        if denominator_validity.device != values.device:
            raise ValueError("denominator_validity must share values.device")
        validity = denominator_validity

    finite = torch.isfinite(values)
    nonfinite = ~finite
    if torch.any(nonfinite):
        bad_rows, bad_features, rendered = _sample_bad_cells(
            nonfinite,
            normalized_row_ids,
            names,
            max_row_id_samples,
        )
        raise OperandValidationError(
            "all operands, including invalid-denominator fallbacks, must be "
            f"finite; sampled cells: {rendered}",
            row_ids=bad_rows,
            feature_names=bad_features,
            kind="nonfinite_fallback",
        )

    active = validity & mask.view(1, -1)
    nonpositive_active = active & (values <= 0)
    if torch.any(nonpositive_active):
        bad_rows, bad_features, rendered = _sample_bad_cells(
            nonpositive_active,
            normalized_row_ids,
            names,
            max_row_id_samples,
        )
        raise OperandValidationError(
            "active semantic denominators must be strictly positive; "
            f"sampled cells: {rendered}",
            row_ids=bad_rows,
            feature_names=bad_features,
            kind="nonpositive_active_denominator",
        )

    rows = int(values.shape[0])
    denominator_indices = torch.nonzero(mask_cpu, as_tuple=False).flatten().tolist()
    field_stats: list[DenominatorFieldStats] = []
    active_total = 0
    floored_total = 0
    for index in denominator_indices:
        valid_column_cpu = validity[:, index].detach().cpu()
        value_column_cpu = values[:, index].detach().float().cpu()
        valid_indices = torch.nonzero(valid_column_cpu, as_tuple=False).flatten()
        missing_indices = torch.nonzero(~valid_column_cpu, as_tuple=False).flatten()
        floored_indices = torch.nonzero(
            valid_column_cpu & (value_column_cpu < div_eps), as_tuple=False
        ).flatten()
        valid_count = int(valid_indices.numel())
        missing_count = int(missing_indices.numel())
        floored_count = int(floored_indices.numel())
        active_total += valid_count
        floored_total += floored_count
        if valid_count:
            valid_values = value_column_cpu[valid_column_cpu]
            minimum = float(valid_values.min().item())
            maximum = float(valid_values.max().item())
        else:
            minimum = None
            maximum = None
        missing_ids = tuple(
            normalized_row_ids[row]
            for row in missing_indices[:max_row_id_samples].tolist()
        )
        floored_ids = tuple(
            normalized_row_ids[row]
            for row in floored_indices[:max_row_id_samples].tolist()
        )
        field_stats.append(
            DenominatorFieldStats(
                name=names[index],
                index=index,
                total_rows=rows,
                valid_rows=valid_count,
                missing_rows=missing_count,
                coverage=valid_count / rows,
                floored_rows=floored_count,
                floor_fraction_of_valid=(
                    floored_count / valid_count if valid_count else 0.0
                ),
                minimum_valid_value=minimum,
                maximum_valid_value=maximum,
                missing_row_ids=missing_ids,
                floored_row_ids=floored_ids,
            )
        )

    total_cells = rows * len(denominator_indices)
    return OperandValidationReport(
        contract=SCHEMA_CONTRACT,
        schema_sha256=schema_digest,
        row_ids_sha256=_row_ids_sha256(normalized_row_ids),
        rows=rows,
        features=len(names),
        denominator_features=len(denominator_indices),
        total_denominator_cells=total_cells,
        active_denominator_cells=active_total,
        overall_coverage=(active_total / total_cells if total_cells else 0.0),
        floored_active_cells=floored_total,
        overall_floor_fraction=(floored_total / active_total if active_total else 0.0),
        div_eps=div_eps,
        field_stats=tuple(field_stats),
    )
