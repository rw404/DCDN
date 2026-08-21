#!/usr/bin/env python
"""RQ2 serving-performance and efficiency study for the frozen multi-target arms.

This answers the *cost* half of RQ2 — "does DCDN reach better multi-target
quality **at comparable latency/throughput**" — for the frozen arm order
published by ``results/synthetic_multiquotient/rq1_v2``.  It trains nothing,
reads no dataset row, no label, and no test metric.  Inputs are synthetic
tensors of the declared geometry, so the study can run while other work is in
flight without touching any data pipeline.

Two geometries are measured, and the distinction matters:

``quotient``   the exact widths of the RQ1 benchmark (operand 8 reduced /
               11 full, no categorical bypass).  These pair one-to-one with
               the published quality numbers, so quality and cost can be put on
               the same axes.
``ranker``     an industrial-scale *geometry stand-in*: 6 categorical fields
               with real embedding tables, 61 shared operands (reduced) or 78
               (full), embedding width 16, matching the field/width counts that
               ``configs/data/vk_lsvd/features_multitask_v1.yaml`` derives.
               **No VK-LSVD row, label, or statistic is read.** Only the shape
               is borrowed; inputs are random. At width 8 every arm is
               dominated by kernel-launch overhead, so the quotient geometry
               alone would say almost nothing about serving a real ranker.

Protocol, per ``configs/matrix_multitask_v1.yaml``:

* full multi-target forward with every head;
* ``eval()`` and ``inference_mode()``, one precision policy for every row;
* device-resident inputs — preprocessing, DataLoader, and host-to-device copy
  are excluded;
* batches ``{1, 64, 1024}``; 200 warmups; 1,000 synchronized forwards; 5
  independent timing rounds.

Timing is captured twice per forward: wall clock around a synchronized region
(the quantity the protocol defines) and CUDA events (which measure device
execution only and are immune to host scheduler stalls).  A disagreement
between the two is the signal that the host was not quiet enough; agreement is
positive evidence that the measurement is sound regardless of load average.

A pair may be called **runtime matched** only when both hold:

* ``p95(B=1)_dcdn <= 1.10 * p95(B=1)_comparator``;
* ``throughput_dcdn >= 0.90 * throughput_comparator`` at the shared maximum
  batch.

Otherwise the pair is reported as a Pareto trade-off and the phrase "at the
same serving performance" may not be used for it.  The script records that
verdict rather than leaving it to prose.

Artifacts are published exactly once into a fresh directory with strict
exact-byte SHA-256 sidecars, and an immutable pre-measurement manifest is
written before the first timed forward.  The publication helpers are
implemented locally rather than imported: the module that established this
discipline is being actively edited by a concurrent session, and a private-name
import would couple this script to that churn.

Usage::

    python scripts/benchmark_efficiency.py --device cuda:0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import sys
import time
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from dcdn.models.afn import AFNPlusConfig  # noqa: E402
from dcdn.models.common import HeadConfig  # noqa: E402
from dcdn.models.contracts import EncodedInteractionBatch, EncoderSpec  # noqa: E402
from dcdn.models.cross import (  # noqa: E402
    DCNConfig,
    DCNv2CapacityCoreConfig,
    DCNv2FullRankConfig,
)
from dcdn.models.dcdn import DCDNCoreConfig  # noqa: E402
from dcdn.models.field import (  # noqa: E402
    AutoIntPlusConfig,
    DeepFMConfig,
    MLPConfig,
)
from dcdn.models.suite import build_controlled_model  # noqa: E402
from dcdn.models.wukong import WukongSmallConfig  # noqa: E402
from dcdn.multitask_capacity import (  # noqa: E402
    capacity_candidate_counts,
    common_counts,
    dcdn_counts,
    ordinary_dcnv2_counts,
    resolve_capacity_control,
)

# ---------------------------------------------------------------------------
# Frozen protocol constants
# ---------------------------------------------------------------------------

BATCHES = (1, 64, 1024)
WARMUPS = 200
FORWARDS = 1000
ROUNDS = 5
INIT_SEED = 42
INPUT_SEED = 20260820

RUNTIME_MATCH_LATENCY_FACTOR = 1.10
RUNTIME_MATCH_THROUGHPUT_FACTOR = 0.90

#: Above this one-minute load average the wall-clock tail is suspect.  It does
#: not by itself invalidate a run: the wall-versus-device agreement check below
#: is the operative evidence.
IDLE_LOAD_AVERAGE_CEILING = 2.0

#: Wall-clock p95 may exceed device p95 by at most this factor before the run
#: is declared contaminated.  Wall clock legitimately includes launch overhead,
#: so some excess is expected; a large excess is scheduler noise.
WALL_VERSUS_DEVICE_P95_CEILING = 2.0

DIV_EPS = 1e-2
REPRESENTATION_DIM = 64
HEAD_HIDDEN_DIMS = (128, 64)


class PublicationError(RuntimeError):
    """An immutable artifact could not be published or verified."""


# ---------------------------------------------------------------------------
# Exact-byte publication (same contract as the frozen synthetic runner)
# ---------------------------------------------------------------------------


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise OSError("short write while publishing immutable artifact")
        view = view[written:]


def publish_bytes(path: Path, data: bytes) -> str:
    """Publish one artifact and its strict adjacent SHA sidecar exactly once.

    The sidecar hashes the *published bytes including the trailing newline*, so
    a later ``sha256sum -c`` reproduces it exactly.
    """

    if path.name.endswith(".sha256"):
        raise ValueError("artifact path itself may not be a SHA sidecar")
    sidecar = path.with_name(path.name + ".sha256")
    digest = _sha256(data)
    sidecar_bytes = f"{digest}  {path.name}\n".encode("ascii")
    created: list[Path] = []
    complete = False
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        for target, content in ((path, data), (sidecar, sidecar_bytes)):
            handle = os.open(target, flags, 0o644)
            created.append(target)
            try:
                _write_all(handle, content)
                os.fsync(handle)
            finally:
                os.close(handle)
        complete = True
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        if path.read_bytes() != data:
            raise PublicationError(f"artifact reopen mismatch: {path}")
        if sidecar.read_bytes() != sidecar_bytes:
            raise PublicationError(f"sidecar reopen mismatch: {sidecar}")
        if _sha256(path.read_bytes()) != digest:
            raise PublicationError(f"artifact digest mismatch: {path}")
    except BaseException:
        if not complete:
            for target in reversed(created):
                with suppress(FileNotFoundError):
                    target.unlink()
        raise
    return digest


def publish_json(path: Path, payload: Any) -> str:
    return publish_bytes(path, _canonical_json_bytes(payload))


def publish_csv(path: Path, rows: list[dict[str, Any]]) -> str:
    if not rows:
        return publish_bytes(path, b"")
    buffer: list[str] = []
    fieldnames = list(rows[0].keys())
    buffer.append(",".join(fieldnames))
    for row in rows:
        cells = []
        for name in fieldnames:
            value = row[name]
            text = "" if value is None else str(value)
            if any(ch in text for ch in ',"\n'):
                text = '"' + text.replace('"', '""') + '"'
            cells.append(text)
        buffer.append(",".join(cells))
    return publish_bytes(path, ("\n".join(buffer) + "\n").encode("utf-8"))


def prepare_output_dir(path: Path) -> None:
    """Create a fresh output directory below ``results/``, refusing reuse."""

    results_root = (REPO_ROOT / "results").resolve(strict=True)
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    resolved_parent = parent.resolve(strict=True)
    if not resolved_parent.is_relative_to(results_root):
        raise PublicationError("output directory must be below repository results/")
    for candidate in (parent, *parent.parents):
        if candidate.is_symlink():
            raise PublicationError(f"symlink ancestor rejected: {candidate}")
        if candidate.resolve() == results_root.parent:
            break
    if path.exists() or path.is_symlink():
        raise PublicationError(f"fresh output directory required; exists: {path}")
    path.mkdir()


# ---------------------------------------------------------------------------
# Host and device state
# ---------------------------------------------------------------------------


def host_state(device: torch.device) -> dict[str, Any]:
    one, five, fifteen = os.getloadavg()
    state: dict[str, Any] = {
        "load_average_1m": one,
        "load_average_5m": five,
        "load_average_15m": fifteen,
        "idle_ceiling": IDLE_LOAD_AVERAGE_CEILING,
        "host_idle": one <= IDLE_LOAD_AVERAGE_CEILING,
        "cpu_count": os.cpu_count(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        state["device_name"] = properties.name
        state["device_total_memory_bytes"] = properties.total_memory
        state["device_multi_processor_count"] = properties.multi_processor_count
        state["cuda_version"] = torch.version.cuda
    return state


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Geometry:
    """One feature-width regime under test."""

    geometry_id: str
    description: str
    categorical_fields: int
    categorical_cardinality: int
    embedding_dim: int
    reduced_operands: int
    full_operands: int
    denominator_count: int
    num_targets: int
    reads_real_data: bool = False
    #: True only when models of this geometry are the ones the quality source
    #: actually trained and scored.  Joining cost from one geometry to quality
    #: from another would compare different models on the same axes.
    quality_source_compatible: bool = False

    def bypass_dim(self) -> int:
        return self.categorical_fields * self.embedding_dim

    def operand_dim(self, regime: str) -> int:
        return self.reduced_operands if regime == "reduced" else self.full_operands

    def num_fields(self, regime: str) -> int:
        return self.categorical_fields + self.operand_dim(regime)

    def denominator_mask(self, regime: str) -> tuple[bool, ...]:
        """Only the first ``denominator_count`` operands are denominators.

        Appended ``R_div`` columns in the full regime are ordinary inputs and
        are never promoted into the reciprocal mask, exactly as
        ``configs/matrix_multitask_v1.yaml`` requires.
        """

        width = self.operand_dim(regime)
        return tuple(index < self.denominator_count for index in range(width))

    def spec(self, regime: str) -> EncoderSpec:
        return EncoderSpec(
            bypass_dim=self.bypass_dim(),
            operand_dim=self.operand_dim(regime),
            num_fields=self.num_fields(regime),
            embedding_dim=self.embedding_dim,
            representation_dim=REPRESENTATION_DIM,
            denominator_mask=self.denominator_mask(regime),
            full_feature_schema_sha256=None,
            operand_schema_sha256=None,
        )


GEOMETRIES: tuple[Geometry, ...] = (
    Geometry(
        geometry_id="quotient",
        description=(
            "exact widths of the Synthetic-MultiQuotient-3 benchmark; pairs "
            "one-to-one with the published rq1_v2 quality numbers"
        ),
        categorical_fields=0,
        categorical_cardinality=1,
        embedding_dim=8,
        reduced_operands=8,
        full_operands=11,
        denominator_count=3,
        num_targets=3,
        quality_source_compatible=True,
    ),
    Geometry(
        geometry_id="kuairec_watch_time",
        description=(
            "the exact widths the KuaiRec big-matrix study trained at - 5 "
            "categorical fields, 50 reduced and 96 full numeric operands, 5 "
            "denominators, one head. Pairs one-to-one with the KuaiRec quality "
            "numbers. Random inputs only; no KuaiRec row or label is read."
        ),
        categorical_fields=5,
        categorical_cardinality=9_050,
        embedding_dim=16,
        reduced_operands=50,
        full_operands=96,
        denominator_count=5,
        num_targets=1,
        quality_source_compatible=True,
    ),
    Geometry(
        geometry_id="kuairand_watch_time",
        description=(
            "the exact widths the KuaiRand-Pure study trained at - 8 categorical "
            "fields, 132 reduced and 244 full numeric operands, 24 denominators, "
            "one regression head. Pairs one-to-one with the single-head "
            "watch_time quality numbers, so cost and quality here describe the "
            "same models. Random inputs only; no KuaiRand row or label is read."
        ),
        categorical_fields=8,
        categorical_cardinality=26_872,
        embedding_dim=16,
        reduced_operands=132,
        full_operands=244,
        denominator_count=24,
        num_targets=1,
        quality_source_compatible=True,
    ),
    Geometry(
        geometry_id="kuairand_multitask",
        description=(
            "same widths as kuairand_watch_time with the five-head multi-target "
            "output. Differs only in the final head width, so it exists to show "
            "that head count is not what drives the cost comparison."
        ),
        categorical_fields=8,
        categorical_cardinality=26_872,
        embedding_dim=16,
        reduced_operands=132,
        full_operands=244,
        denominator_count=24,
        num_targets=5,
        quality_source_compatible=True,
    ),
    Geometry(
        geometry_id="ranker",
        description=(
            "industrial-scale geometry stand-in matching the field and width "
            "counts derived by features_multitask_v1; random inputs only, no "
            "VK-LSVD row, label, or statistic is read"
        ),
        categorical_fields=6,
        categorical_cardinality=100_000,
        embedding_dim=16,
        reduced_operands=61,
        full_operands=78,
        denominator_count=5,
        num_targets=4,
    ),
)


# ---------------------------------------------------------------------------
# Encoder and model
# ---------------------------------------------------------------------------


class PerfEncoder(nn.Module):
    """Shared front end: categorical tables plus a per-field numeric lift.

    Identical in architecture and initialization across every arm of a
    geometry, so its cost is a common offset and the only architectural
    difference between rows remains the interaction block.
    """

    def __init__(self, geometry: Geometry, regime: str) -> None:
        super().__init__()
        self.geometry = geometry
        self.regime = regime
        self.embeddings = nn.ModuleList(
            nn.Embedding(geometry.categorical_cardinality, geometry.embedding_dim)
            for _ in range(geometry.categorical_fields)
        )
        for table in self.embeddings:
            nn.init.normal_(table.weight, std=0.01)
        width = geometry.operand_dim(regime)
        self.numeric_weight = nn.Parameter(torch.empty(width, geometry.embedding_dim))
        self.numeric_bias = nn.Parameter(torch.empty(width, geometry.embedding_dim))
        nn.init.normal_(self.numeric_weight, std=0.1)
        nn.init.zeros_(self.numeric_bias)

    def forward(
        self, categorical: torch.Tensor, operands: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        values = operands.unsqueeze(-1)
        numeric_fields = values * self.numeric_weight + self.numeric_bias
        if not self.embeddings:
            bypass = operands.new_zeros((operands.shape[0], 0))
            return bypass, numeric_fields
        looked_up = [
            table(categorical[:, i])
            for i, table in enumerate(self.embeddings)
        ]
        bypass = torch.cat(looked_up, dim=1)
        fields = torch.cat([torch.stack(looked_up, dim=1), numeric_fields], dim=1)
        return bypass, fields


class PerfModel(nn.Module):
    """Shared encoder plus the audited interaction model with ``T`` heads."""

    def __init__(
        self,
        geometry: Geometry,
        regime: str,
        interaction_config: Any,
        *,
        with_afn_view: bool,
    ) -> None:
        super().__init__()
        self.geometry = geometry
        self.regime = regime
        self.with_afn_view = with_afn_view
        spec = geometry.spec(regime)
        with torch.random.fork_rng(devices=[], enabled=True):
            torch.random.default_generator.manual_seed(INIT_SEED)
            self.encoder = PerfEncoder(geometry, regime)
            self.afn_encoder = (
                PerfEncoder(geometry, regime) if with_afn_view else None
            )
        self.model = build_controlled_model(
            encoder_spec=spec,
            interaction_config=interaction_config,
            head_config=HeadConfig(
                hidden_dims=HEAD_HIDDEN_DIMS,
                dropout=0.0,
                output_dim=geometry.num_targets,
            ),
            init_seed=INIT_SEED,
        )
        self.spec = spec

    def forward(
        self,
        categorical: torch.Tensor,
        operands: torch.Tensor,
        validity: torch.Tensor,
    ) -> torch.Tensor:
        bypass, fields = self.encoder(categorical, operands)
        auxiliary = None
        if self.afn_encoder is not None:
            _, auxiliary = self.afn_encoder(categorical, operands)
        batch = EncodedInteractionBatch(
            bypass_features=bypass,
            operand_features=operands,
            field_embeddings=fields,
            denominator_validity=validity,
            afn_dnn_embeddings=auxiliary,
        )
        return self.model(batch).logits

    def parameter_counts(self) -> dict[str, int]:
        def count(module: nn.Module) -> int:
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        embedding_ids = {
            id(p)
            for module in self.modules()
            if isinstance(module, nn.Embedding)
            for p in module.parameters()
        }
        non_embedding = sum(
            p.numel()
            for p in self.parameters()
            if p.requires_grad and id(p) not in embedding_ids
        )
        interaction = count(self.model.interaction)
        adapter = count(self.model.adapter)
        head = count(self.model.head)
        return {
            "encoder_shared": count(self.encoder)
            + (count(self.afn_encoder) if self.afn_encoder is not None else 0),
            "interaction": interaction,
            "adapter": adapter,
            "head": head,
            "matched": interaction + adapter + head,
            "non_embedding": non_embedding,
            "total": count(self),
        }


class LinearProbe(nn.Module):
    """Absolute floor: one affine map from operands to targets."""

    def __init__(self, geometry: Geometry, regime: str) -> None:
        super().__init__()
        self.geometry = geometry
        self.regime = regime
        self.spec = geometry.spec(regime)
        with torch.random.fork_rng(devices=[], enabled=True):
            torch.random.default_generator.manual_seed(INIT_SEED)
            self.projection = nn.Linear(
                geometry.operand_dim(regime), geometry.num_targets
            )
            nn.init.xavier_uniform_(self.projection.weight)
            nn.init.zeros_(self.projection.bias)

    def forward(
        self,
        categorical: torch.Tensor,
        operands: torch.Tensor,
        validity: torch.Tensor,
    ) -> torch.Tensor:
        del categorical, validity
        return self.projection(operands)

    def parameter_counts(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "encoder_shared": 0,
            "interaction": 0,
            "adapter": 0,
            "head": total,
            "matched": total,
            "non_embedding": total,
            "total": total,
        }


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Arm:
    arm_id: str
    label: str
    regime: str
    kind: str
    config: Any = None
    with_afn_view: bool = False
    tier: str = "matrix"
    notes: dict[str, Any] = field(default_factory=dict)


def _dcdn(layers: int) -> DCDNCoreConfig:
    return DCDNCoreConfig(
        num_layers=layers,
        div_eps=DIV_EPS,
        product_init_scale=0.1,
        div_init_scale=0.1,
        runtime_value_checks=False,
    )


def _afn() -> AFNPlusConfig:
    return AFNPlusConfig(
        logarithmic_neurons=128,
        afn_hidden_dims=(512,),
        dnn_hidden_dims=(512,),
        log_floor=1e-6,
        exp_clip=None,
        dropout=0.0,
        batch_norm=True,
    )


def build_arms(geometry: Geometry, capacity: dict[str, dict[str, int]]) -> list[Arm]:
    """The frozen matrix arms, plus the Tier-B zoo on the reduced regime.

    Arm ids for the matrix rows match
    ``results/synthetic_multiquotient/rq1_v2`` exactly so the cost table joins
    to the published quality table without a mapping layer.
    """

    arms = [
        Arm("linear", "Linear probe", "reduced", "linear", tier="floor"),
        Arm(
            "dcnv2_reduced",
            "M1 DCNv2-REDUCED-MTL",
            "reduced",
            "controlled",
            DCNv2FullRankConfig(num_layers=2, product_init_scale=0.1),
        ),
        Arm(
            "dcdn_reduced",
            "M2 DCDN-Core-REDUCED-MTL",
            "reduced",
            "controlled",
            _dcdn(1),
        ),
        Arm(
            "afn_reduced",
            "M3 AFN+-REDUCED-MTL",
            "reduced",
            "controlled",
            _afn(),
            with_afn_view=True,
        ),
        Arm(
            "dcnv2_capacity_reduced",
            "M4 DCNv2-capacity-matched-REDUCED-MTL",
            "reduced",
            "controlled",
            DCNv2CapacityCoreConfig(
                num_layers=capacity["reduced"]["num_layers"],
                low_rank=capacity["reduced"]["low_rank"],
                num_experts=capacity["reduced"]["num_experts"],
                product_init_scale=0.1,
            ),
            notes={"capacity_matched_to": "dcdn_reduced"},
        ),
        Arm(
            "dcnv2_full",
            "M5 DCNv2-FULL-MTL",
            "full",
            "controlled",
            DCNv2FullRankConfig(num_layers=2, product_init_scale=0.1),
        ),
        Arm("dcdn_full", "M6 DCDN-Core-FULL-MTL", "full", "controlled", _dcdn(1)),
        Arm(
            "afn_full",
            "M7 AFN+-FULL-MTL",
            "full",
            "controlled",
            _afn(),
            with_afn_view=True,
        ),
    ]
    # Depth control: the recommendation search space freezes DCDN at L2, while
    # the quotient benchmark uses L1.  Both are measured so the depth cost is
    # explicit rather than inferred.
    arms.append(
        Arm(
            "dcdn_reduced_l2",
            "DCDN-Core-L2-REDUCED (depth control)",
            "reduced",
            "controlled",
            _dcdn(2),
            tier="depth_control",
        )
    )
    # Tier-B architecture zoo, reduced regime only, for the efficiency table.
    zoo = (
        ("dnn_reduced", "DNN", MLPConfig(hidden_dims=(512, 256, 128), dropout=0.0)),
        (
            "deepfm_reduced",
            "DeepFM",
            DeepFMConfig(deep_hidden_dims=(512, 256, 128), dropout=0.0),
        ),
        ("dcn_reduced", "DCN", DCNConfig(num_layers=2, product_init_scale=0.1)),
        (
            "autoint_plus_reduced",
            "AutoInt+",
            AutoIntPlusConfig(
                attention_layers=2,
                attention_dim=32,
                num_heads=2,
                dnn_hidden_dims=(512, 256, 128),
                dropout=0.0,
                scale_attention=True,
                layer_norm=False,
            ),
        ),
        (
            "wukong_small_reduced",
            "Wukong-small",
            WukongSmallConfig(
                num_layers=2,
                lcb_features=8,
                fmb_features=4,
                fmb_rank=8,
                fmb_hidden_dims=(128,),
                dropout=0.0,
                layer_norm=True,
            ),
        ),
    )
    for arm_id, label, config in zoo:
        arms.append(Arm(arm_id, label, "reduced", "controlled", config, tier="tier_b"))
    del geometry
    return arms


def instantiate(arm: Arm, geometry: Geometry) -> nn.Module:
    if arm.kind == "linear":
        return LinearProbe(geometry, arm.regime)
    return PerfModel(
        geometry,
        arm.regime,
        arm.config,
        with_afn_view=arm.with_afn_view,
    )


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("no samples")
    position = fraction * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def _make_inputs(
    geometry: Geometry, regime: str, batch: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(INPUT_SEED)
    width = geometry.operand_dim(regime)
    operands = (torch.rand(batch, width, generator=generator) * 9.0 + 1.0).to(device)
    if geometry.categorical_fields:
        categorical = torch.randint(
            0,
            geometry.categorical_cardinality,
            (batch, geometry.categorical_fields),
            generator=generator,
        ).to(device)
    else:
        categorical = torch.zeros((batch, 0), dtype=torch.long, device=device)
    validity = torch.ones_like(operands, dtype=torch.bool)
    return categorical, operands, validity


@torch.inference_mode()
def measure(
    model: nn.Module,
    geometry: Geometry,
    regime: str,
    batch: int,
    device: torch.device,
) -> dict[str, Any]:
    model = model.to(device).eval()
    categorical, operands, validity = _make_inputs(geometry, regime, batch, device)

    for _ in range(WARMUPS):
        model(categorical, operands, validity)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    use_events = device.type == "cuda"
    wall_rounds: list[list[float]] = []
    device_rounds: list[list[float]] = []
    for _ in range(ROUNDS):
        wall_samples: list[float] = []
        device_samples: list[float] = []
        for _ in range(FORWARDS):
            if use_events:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize(device)
                start_event.record()
                started = time.perf_counter()
                model(categorical, operands, validity)
                end_event.record()
                torch.cuda.synchronize(device)
                wall_samples.append((time.perf_counter() - started) * 1e3)
                device_samples.append(start_event.elapsed_time(end_event))
            else:
                started = time.perf_counter()
                model(categorical, operands, validity)
                wall_samples.append((time.perf_counter() - started) * 1e3)
        wall_rounds.append(wall_samples)
        if use_events:
            device_rounds.append(device_samples)

    wall = [value for samples in wall_rounds for value in samples]
    p50, p95, p99 = (
        _percentile(wall, 0.50),
        _percentile(wall, 0.95),
        _percentile(wall, 0.99),
    )
    report: dict[str, Any] = {
        "batch": batch,
        "warmups": WARMUPS,
        "forwards_per_round": FORWARDS,
        "rounds": ROUNDS,
        "batch_latency_ms_p50": p50,
        "batch_latency_ms_p95": p95,
        "batch_latency_ms_p99": p99,
        "per_example_latency_ms_p50": p50 / batch,
        "per_example_latency_ms_p95": p95 / batch,
        "per_example_latency_ms_p99": p99 / batch,
        "throughput_examples_per_second": batch / (p50 / 1e3),
        "per_round_p50_ms": [_percentile(r, 0.50) for r in wall_rounds],
        "round_to_round_p50_spread_ms": (
            max(_percentile(r, 0.50) for r in wall_rounds)
            - min(_percentile(r, 0.50) for r in wall_rounds)
        ),
        "peak_vram_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if use_events else 0
        ),
    }
    if use_events:
        device_all = [value for samples in device_rounds for value in samples]
        d50 = _percentile(device_all, 0.50)
        d95 = _percentile(device_all, 0.95)
        report.update(
            {
                "device_latency_ms_p50": d50,
                "device_latency_ms_p95": d95,
                "device_latency_ms_p99": _percentile(device_all, 0.99),
                "device_throughput_examples_per_second": batch / (d50 / 1e3),
                "wall_over_device_p50": p50 / d50 if d50 > 0 else None,
                "wall_over_device_p95": p95 / d95 if d95 > 0 else None,
            }
        )
    return report


# ---------------------------------------------------------------------------
# Verdicts
# ---------------------------------------------------------------------------


def runtime_match_verdict(
    subject: dict[int, dict[str, Any]],
    comparator: dict[int, dict[str, Any]],
    shared_max_batch: int,
) -> dict[str, Any]:
    wall_ratio = (
        subject[1]["batch_latency_ms_p95"] / comparator[1]["batch_latency_ms_p95"]
    )
    median_ratio = (
        subject[1]["batch_latency_ms_p50"] / comparator[1]["batch_latency_ms_p50"]
    )
    throughput_ratio = (
        subject[shared_max_batch]["throughput_examples_per_second"]
        / comparator[shared_max_batch]["throughput_examples_per_second"]
    )
    device_ratio = None
    has_device = (
        "device_latency_ms_p95" in subject[1]
        and "device_latency_ms_p95" in comparator[1]
    )
    if has_device:
        device_ratio = (
            subject[1]["device_latency_ms_p95"] / comparator[1]["device_latency_ms_p95"]
        )
    latency_ok = wall_ratio <= RUNTIME_MATCH_LATENCY_FACTOR
    throughput_ok = throughput_ratio >= RUNTIME_MATCH_THROUGHPUT_FACTOR
    matched = latency_ok and throughput_ok
    return {
        "p95_batch1_ratio": wall_ratio,
        "p95_batch1_threshold": RUNTIME_MATCH_LATENCY_FACTOR,
        "p95_batch1_passes": latency_ok,
        "p50_batch1_ratio": median_ratio,
        "device_p95_batch1_ratio": device_ratio,
        "throughput_ratio_at_shared_max_batch": throughput_ratio,
        "throughput_threshold": RUNTIME_MATCH_THROUGHPUT_FACTOR,
        "throughput_passes": throughput_ok,
        "shared_max_batch": shared_max_batch,
        "runtime_matched": matched,
        "reporting_requirement": (
            "may be described as runtime matched"
            if matched
            else "must be reported as a Pareto trade-off; the phrase 'at the "
            "same serving performance' is not permitted for this pair"
        ),
    }


def _verdict_status(
    state: dict[str, Any], contamination: dict[str, Any]
) -> str:
    """Decide validity from device-time agreement, not from a load proxy."""

    if not contamination.get("available"):
        return "provisional_no_device_timing_available"
    if not contamination.get("wall_clock_tail_is_clean", False):
        return "provisional_wall_clock_tail_exceeds_device_time_ceiling"
    if state["host_idle"]:
        return "final_idle_host"
    return "final_load_gate_waived_by_device_time_agreement"


def contamination_check(
    measurements: dict[str, dict[str, dict[int, dict[str, Any]]]],
) -> dict[str, Any]:
    """Wall-versus-device agreement is the real validity evidence."""

    excess: list[float] = []
    for geometry_rows in measurements.values():
        for batches in geometry_rows.values():
            for row in batches.values():
                ratio = row.get("wall_over_device_p95")
                if ratio is not None:
                    excess.append(ratio)
    if not excess:
        return {"available": False}
    worst = max(excess)
    return {
        "available": True,
        "worst_wall_over_device_p95": worst,
        "median_wall_over_device_p95": statistics.median(excess),
        "ceiling": WALL_VERSUS_DEVICE_P95_CEILING,
        "wall_clock_tail_is_clean": worst <= WALL_VERSUS_DEVICE_P95_CEILING,
    }


# ---------------------------------------------------------------------------
# Analytic capacity and FLOPs
# ---------------------------------------------------------------------------


def analytic_counts(geometry: Geometry) -> dict[str, Any]:
    """Closed-form parameters and FLOPs where the frozen formulas apply."""

    out: dict[str, Any] = {}
    for regime in ("reduced", "full"):
        operand = geometry.operand_dim(regime)
        flat = geometry.bypass_dim() + operand
        common = common_counts(
            flat, REPRESENTATION_DIM, HEAD_HIDDEN_DIMS, geometry.num_targets
        )
        out[regime] = {
            "flat_width": flat,
            "operand_width": operand,
            "common": common,
            "dcdn_l1": dcdn_counts(operand, 1, common),
            "dcdn_l2": dcdn_counts(operand, 2, common),
            "dcnv2_l2": ordinary_dcnv2_counts(flat, 2, common),
        }
    return out


def resolve_capacity(geometry: Geometry) -> dict[str, dict[str, int]]:
    """Mechanically match M4 to M2 in (parameter, FLOP) space; no metric read."""

    resolved: dict[str, dict[str, int]] = {}
    for regime in ("reduced",):
        operand = geometry.operand_dim(regime)
        flat = geometry.bypass_dim() + operand
        common = common_counts(
            flat, REPRESENTATION_DIM, HEAD_HIDDEN_DIMS, geometry.num_targets
        )
        target = dcdn_counts(operand, 1, common)
        resolution = resolve_capacity_control(
            target=target,
            operand_width=operand,
            common=common,
            ranks=tuple(range(1, 65)),
            layer_options=(1, 2),
            expert_options=(1, 2, 4),
        )
        payload = resolution.payload()
        if not resolution.matched:
            raise PublicationError(
                f"no capacity control inside tolerance for {geometry.geometry_id}/"
                f"{regime}; bracket controls required: {json.dumps(payload)}"
            )
        selected = payload["selected"]
        resolved[regime] = {
            "low_rank": selected["low_rank"],
            "num_layers": selected["num_layers"],
            "num_experts": selected["num_experts"],
            "parameter_ratio": payload["selected_ratios"]["parameter_ratio"],
            "flop_ratio": payload["selected_ratios"]["flop_ratio"],
            "interaction_parameters": selected["interaction_parameters"],
            "parameters": selected["parameters"],
            "flops": selected["flops"],
        }
    return resolved


def arm_flops(arm: Arm, geometry: Geometry, analytic: dict[str, Any]) -> int | str:
    """FLOPs/example where a frozen closed form exists, otherwise unavailable.

    Guessing a FLOP count for an architecture without a frozen formula would be
    worse than reporting that it is not available.
    """

    regime = analytic[arm.regime]
    if arm.arm_id in ("dcdn_reduced", "dcdn_full"):
        return regime["dcdn_l1"]["flops"]
    if arm.arm_id == "dcdn_reduced_l2":
        return regime["dcdn_l2"]["flops"]
    if arm.arm_id in ("dcnv2_reduced", "dcnv2_full"):
        return regime["dcnv2_l2"]["flops"]
    if arm.arm_id == "dcnv2_capacity_reduced":
        operand = geometry.operand_dim(arm.regime)
        flat = geometry.bypass_dim() + operand
        common = common_counts(
            flat, REPRESENTATION_DIM, HEAD_HIDDEN_DIMS, geometry.num_targets
        )
        capacity = arm.config
        return capacity_candidate_counts(
            operand,
            capacity.low_rank,
            capacity.num_layers,
            capacity.num_experts,
            common,
        )["flops"]
    return "unavailable_no_frozen_formula"


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "efficiency" / "rq2_perf_v1",
    )
    parser.add_argument(
        "--quality-from",
        type=Path,
        default=REPO_ROOT
        / "results"
        / "synthetic_multiquotient"
        / "rq1_v2"
        / "results_v1.json",
    )
    parser.add_argument(
        "--allow-busy-host",
        action="store_true",
        help=(
            "proceed when the load average exceeds the idle ceiling; the run is "
            "still published, with the wall-versus-device agreement check "
            "deciding whether the tail is usable"
        ),
    )
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    state = host_state(device)
    print(
        f"device: {device}  batches={BATCHES}  warmups={WARMUPS}  "
        f"forwards={FORWARDS}  rounds={ROUNDS}"
    )
    print(
        f"host load 1m/5m/15m: {state['load_average_1m']:.2f}/"
        f"{state['load_average_5m']:.2f}/{state['load_average_15m']:.2f}  "
        f"idle={state['host_idle']}"
    )
    if not state["host_idle"] and not args.allow_busy_host:
        raise SystemExit(
            "host is not idle; re-run when quiet or pass --allow-busy-host "
            "(the wall-versus-device check will decide whether the tail is usable)"
        )

    prepare_output_dir(args.output_dir)

    capacity = {g.geometry_id: resolve_capacity(g) for g in GEOMETRIES}
    analytic = {g.geometry_id: analytic_counts(g) for g in GEOMETRIES}
    plan = {
        g.geometry_id: [
            {
                "arm_id": a.arm_id,
                "label": a.label,
                "regime": a.regime,
                "tier": a.tier,
                "notes": a.notes,
            }
            for a in build_arms(g, capacity[g.geometry_id])
        ]
        for g in GEOMETRIES
    }

    # Immutable pre-measurement manifest, published before the first timed
    # forward so the plan cannot be adjusted after seeing a number.
    manifest = {
        "artifact_role": "rq2_efficiency_pre_measurement_manifest",
        "schema_version": 1,
        "protocol": {
            "forward_scope": "full_multi_target_forward_with_every_head",
            "mode": "eval_and_inference_mode",
            "precision_policy": "float32_parameters_no_autocast_identical_across_rows",
            "inputs": "device_resident_random_tensors",
            "excluded": ["preprocessing", "dataloader", "host_to_device_transfer"],
            "batches": list(BATCHES),
            "warmups": WARMUPS,
            "synchronized_forwards": FORWARDS,
            "independent_rounds": ROUNDS,
            "timing": "wall_clock_synchronized_and_cuda_event_device_time",
            "runtime_match_latency_factor": RUNTIME_MATCH_LATENCY_FACTOR,
            "runtime_match_throughput_factor": RUNTIME_MATCH_THROUGHPUT_FACTOR,
            "wall_versus_device_p95_ceiling": WALL_VERSUS_DEVICE_P95_CEILING,
        },
        "reads_dataset_rows": False,
        "reads_labels": False,
        "reads_test_metrics": False,
        "trains_any_model": False,
        "init_seed": INIT_SEED,
        "input_seed": INPUT_SEED,
        "host_state_at_start": state,
        "geometries": [
            {
                "geometry_id": g.geometry_id,
                "description": g.description,
                "categorical_fields": g.categorical_fields,
                "categorical_cardinality": g.categorical_cardinality,
                "embedding_dim": g.embedding_dim,
                "bypass_dim": g.bypass_dim(),
                "reduced_operands": g.reduced_operands,
                "full_operands": g.full_operands,
                "denominator_count": g.denominator_count,
                "num_targets": g.num_targets,
                "reads_real_data": g.reads_real_data,
            }
            for g in GEOMETRIES
        ],
        "capacity_resolution": capacity,
        "analytic_counts": analytic,
        "planned_arms": plan,
    }
    manifest_digest = publish_json(
        args.output_dir / "pre_measurement_manifest_v1.json", manifest
    )
    print(f"pre-measurement manifest sha256: {manifest_digest}")

    measurements: dict[str, dict[str, dict[int, dict[str, Any]]]] = {}
    parameters: dict[str, dict[str, dict[str, int]]] = {}
    flops: dict[str, dict[str, Any]] = {}

    for geometry in GEOMETRIES:
        print(f"\n--- geometry: {geometry.geometry_id} ---")
        measurements[geometry.geometry_id] = {}
        parameters[geometry.geometry_id] = {}
        flops[geometry.geometry_id] = {}
        arms = build_arms(geometry, capacity[geometry.geometry_id])
        for arm in arms:
            model = instantiate(arm, geometry)
            parameters[geometry.geometry_id][arm.arm_id] = model.parameter_counts()
            flops[geometry.geometry_id][arm.arm_id] = arm_flops(
                arm, geometry, analytic[geometry.geometry_id]
            )
            rows = {
                batch: measure(model, geometry, arm.regime, batch, device)
                for batch in BATCHES
            }
            measurements[geometry.geometry_id][arm.arm_id] = rows
            device_note = (
                f" dev_p95={rows[1]['device_latency_ms_p95']:.4f}"
                if "device_latency_ms_p95" in rows[1]
                else ""
            )
            print(
                f"  {arm.arm_id:<24} B=1 p50={rows[1]['batch_latency_ms_p50']:.4f} "
                f"p95={rows[1]['batch_latency_ms_p95']:.4f}{device_note}  "
                f"B=1024 tput={rows[1024]['throughput_examples_per_second']:,.0f}"
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    contamination = contamination_check(measurements)
    shared_max = max(BATCHES)
    verdicts: dict[str, dict[str, Any]] = {}
    for geometry in GEOMETRIES:
        rows = measurements[geometry.geometry_id]
        for subject, comparators in (
            (
                "dcdn_reduced",
                (
                    "dcnv2_reduced",
                    "afn_reduced",
                    "dcnv2_capacity_reduced",
                    "linear",
                ),
            ),
            ("dcdn_full", ("dcnv2_full", "afn_full")),
        ):
            for comparator in comparators:
                if subject not in rows or comparator not in rows:
                    continue
                key = f"{geometry.geometry_id}:{subject}_vs_{comparator}"
                verdicts[key] = runtime_match_verdict(
                    rows[subject], rows[comparator], shared_max
                )

    print("\n--- runtime-match verdicts ---")
    for key, verdict in verdicts.items():
        print(
            f"  {key:<46} matched={str(verdict['runtime_matched']):<5} "
            f"p95r={verdict['p95_batch1_ratio']:.3f} "
            f"tput={verdict['throughput_ratio_at_shared_max_batch']:.3f}"
        )
    if contamination.get("available"):
        print(
            "\nwall/device p95 "
            f"worst={contamination['worst_wall_over_device_p95']:.3f} "
            f"median={contamination['median_wall_over_device_p95']:.3f} "
            f"clean={contamination['wall_clock_tail_is_clean']}"
        )

    quality = _load_quality(args.quality_from)
    efficiency_rows = _efficiency_rows(
        measurements, parameters, flops, capacity, GEOMETRIES
    )
    pareto = _pareto_rows(
        measurements, parameters, flops, quality, verdicts, GEOMETRIES
    )

    results = {
        "artifact_role": "rq2_efficiency_result",
        "schema_version": 1,
        "pre_measurement_manifest_sha256": manifest_digest,
        "host_state_at_start": state,
        "host_state_at_end": host_state(device),
        "contamination_check": contamination,
        "verdict_status": _verdict_status(state, contamination),
        "idle_gate": {
            "load_average_ceiling": IDLE_LOAD_AVERAGE_CEILING,
            "host_idle_by_load_average": state["host_idle"],
            "waived": not state["host_idle"],
            "waiver_basis": (
                "A load-average threshold is a proxy. CUDA-event device time "
                "measures execution on the GPU and is immune to host "
                "scheduling, so agreement between wall-clock p95 and device "
                "p95 is direct evidence that the tail is uncontaminated, and "
                "disagreement is direct evidence that it is. The verdict "
                "status is decided by that agreement, not by the proxy."
            ),
        },
        "quality_source": str(args.quality_from.relative_to(REPO_ROOT))
        if args.quality_from.exists()
        else None,
        "quality_join_restricted_to_geometries": [
            g.geometry_id for g in GEOMETRIES if g.quality_source_compatible
        ],
        "quality_join_rationale": (
            "cost may be placed on the same axes as quality only for the "
            "geometry whose models the quality source actually trained; the "
            "ranker geometry is a width stand-in and was never trained"
        ),
        "capacity_resolution": capacity,
        "analytic_counts": analytic,
        "parameters": parameters,
        "flops_per_example": flops,
        "measurements": {
            geometry_id: {
                arm_id: {str(batch): row for batch, row in batches.items()}
                for arm_id, batches in rows.items()
            }
            for geometry_id, rows in measurements.items()
        },
        "runtime_match_verdicts": verdicts,
    }
    results_digest = publish_json(args.output_dir / "results_v1.json", results)
    efficiency_digest = publish_csv(args.output_dir / "efficiency.csv", efficiency_rows)
    pareto_digest = publish_csv(args.output_dir / "pareto.csv", pareto)

    print(f"\npublished results_v1.json   sha256={results_digest}")
    print(f"published efficiency.csv    sha256={efficiency_digest} "
          f"({len(efficiency_rows)} rows)")
    print(f"published pareto.csv        sha256={pareto_digest} ({len(pareto)} rows)")
    print(f"output directory: {args.output_dir}")
    return 0


def _load_quality(path: Path) -> dict[str, dict[str, float]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in payload["records"]:
        grouped.setdefault(record["arm_id"], []).append(record)
    slices = payload["benchmark_manifest"]["evaluation_slices"]
    return {
        arm_id: {
            slice_name: statistics.fmean(
                r["slices"][slice_name]["macro_nloss"] for r in records
            )
            for slice_name in slices
        }
        for arm_id, records in grouped.items()
    }


def _efficiency_rows(
    measurements: dict[str, dict[str, dict[int, dict[str, Any]]]],
    parameters: dict[str, dict[str, dict[str, int]]],
    flops: dict[str, dict[str, Any]],
    capacity: dict[str, dict[str, dict[str, int]]],
    geometries: tuple[Geometry, ...],
) -> list[dict[str, Any]]:
    by_id = {g.geometry_id: g for g in geometries}
    rows: list[dict[str, Any]] = []
    for geometry_id, arms in measurements.items():
        geometry = by_id[geometry_id]
        for arm_id, batches in arms.items():
            counts = parameters[geometry_id][arm_id]
            for batch, row in sorted(batches.items()):
                rows.append(
                    {
                        "geometry": geometry_id,
                        "arm_id": arm_id,
                        "batch": batch,
                        "num_targets": geometry.num_targets,
                        "batch_latency_ms_p50": row["batch_latency_ms_p50"],
                        "batch_latency_ms_p95": row["batch_latency_ms_p95"],
                        "batch_latency_ms_p99": row["batch_latency_ms_p99"],
                        "per_example_latency_ms_p50": row["per_example_latency_ms_p50"],
                        "per_example_latency_ms_p95": row["per_example_latency_ms_p95"],
                        "per_example_latency_ms_p99": row["per_example_latency_ms_p99"],
                        "device_latency_ms_p50": row.get("device_latency_ms_p50"),
                        "device_latency_ms_p95": row.get("device_latency_ms_p95"),
                        "wall_over_device_p95": row.get("wall_over_device_p95"),
                        "throughput_examples_per_second": row[
                            "throughput_examples_per_second"
                        ],
                        "round_to_round_p50_spread_ms": row[
                            "round_to_round_p50_spread_ms"
                        ],
                        "peak_vram_bytes": row["peak_vram_bytes"],
                        "parameters_total": counts["total"],
                        "parameters_non_embedding": counts["non_embedding"],
                        "parameters_matched": counts["matched"],
                        "parameters_interaction": counts["interaction"],
                        "flops_per_example": flops[geometry_id][arm_id],
                    }
                )
    del capacity
    return rows


def _pareto_rows(
    measurements: dict[str, dict[str, dict[int, dict[str, Any]]]],
    parameters: dict[str, dict[str, dict[str, int]]],
    flops: dict[str, dict[str, Any]],
    quality: dict[str, dict[str, float]],
    verdicts: dict[str, dict[str, Any]],
    geometries: tuple[Geometry, ...],
) -> list[dict[str, Any]]:
    """Quality against cost, frontier marked, per quality slice.

    Only geometries whose models are the ones the quality source actually
    trained are eligible.  The ``ranker`` geometry is a width stand-in that was
    never trained, so pairing its latency with quotient-benchmark quality would
    place two different models on one pair of axes.  Those rows are omitted
    rather than emitted with a caveat.
    """

    rows: list[dict[str, Any]] = []
    if not quality:
        return rows
    eligible = {g.geometry_id for g in geometries if g.quality_source_compatible}
    slices = sorted({s for entry in quality.values() for s in entry})
    for geometry_id, arms in measurements.items():
        if geometry_id not in eligible:
            continue
        for slice_name in slices:
            points = []
            for arm_id, batches in arms.items():
                if arm_id not in quality:
                    continue
                verdict = verdicts.get(
                    f"{geometry_id}:dcdn_reduced_vs_{arm_id}"
                ) or verdicts.get(f"{geometry_id}:dcdn_full_vs_{arm_id}")
                points.append(
                    {
                        "geometry": geometry_id,
                        "quality_slice": slice_name,
                        "arm_id": arm_id,
                        "macro_nloss_mean": quality[arm_id][slice_name],
                        "p95_batch1_latency_ms": batches[1]["batch_latency_ms_p95"],
                        "device_p95_batch1_ms": batches[1].get(
                            "device_latency_ms_p95"
                        ),
                        "throughput_at_max_batch": batches[max(BATCHES)][
                            "throughput_examples_per_second"
                        ],
                        "parameters_non_embedding": parameters[geometry_id][arm_id][
                            "non_embedding"
                        ],
                        "flops_per_example": flops[geometry_id][arm_id],
                        "runtime_matched_vs_dcdn": (
                            "" if verdict is None else verdict["runtime_matched"]
                        ),
                    }
                )
            for point in points:
                point["on_pareto_frontier"] = not any(
                    other is not point
                    and other["macro_nloss_mean"] <= point["macro_nloss_mean"]
                    and other["p95_batch1_latency_ms"]
                    <= point["p95_batch1_latency_ms"]
                    and (
                        other["macro_nloss_mean"] < point["macro_nloss_mean"]
                        or other["p95_batch1_latency_ms"]
                        < point["p95_batch1_latency_ms"]
                    )
                    for other in points
                )
            rows.extend(sorted(points, key=lambda p: p["macro_nloss_mean"]))
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
