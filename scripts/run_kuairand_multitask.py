#!/usr/bin/env python
"""RQ2 (quality) and RQ3 on KuaiRand-Pure: the seven-row multi-target matrix.

This is the first mandatory *recommendation* cell of the override. It answers
the two questions the synthetic benchmark cannot:

**RQ2 quality** — does DCDN reach better multi-target quality than DCNv2 and
AFN+ on the reduced feature view, where runtime parity has been established
separately (`docs/rq2_fused_inference.md`)? And what is honestly lost against
the full feature view?

**RQ3** — is any gain explained by the reciprocal architecture or merely by
parameters and FLOPs? Answered by `M4`, a DCNv2 low-rank mixture matched to
`M2` within +/-5% on trainable non-embedding parameters *and* FLOPs/example,
selected mechanically with no validation metric read.

Five heads, per `configs/data/kuairand_pure/targets_rq23_v1.yaml`: `long_view`,
`like`, `forward`, `hate` (binary) and `watch_time` (normalized clipped
regression, constant fitted on train only).

**The locked test window is never opened.** `scripts/data/kuairand_pure_materialize.py`
drops 2022-05-06..08 by date before reading any label, and this script only
ever sees the `train` and `validation` arrays it wrote. Every number here is a
**validation** number; the headline test remains sealed for the freeze
ceremony.

Usage::

    python scripts/run_kuairand_multitask.py --device cuda:0
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from dcdn.metrics import (  # noqa: E402
    group_concordance,
    masked_binary_target_metrics,
    watch_time_metrics,
)
from dcdn.models.afn import AFNPlusConfig  # noqa: E402
from dcdn.models.common import HeadConfig  # noqa: E402
from dcdn.models.contracts import EncodedInteractionBatch, EncoderSpec  # noqa: E402
from dcdn.models.cross import (  # noqa: E402
    DCNConfig,
    DCNv2CapacityCoreConfig,
    DCNv2FullRankConfig,
)
from dcdn.models.euler import EulerNetConfig  # noqa: E402
from dcdn.models.dcdn import DCDNCoreConfig  # noqa: E402
from dcdn.models.field import (  # noqa: E402
    AutoIntPlusConfig,
    DeepFMConfig,
    MLPConfig,
)
from dcdn.models.suite import build_controlled_model  # noqa: E402
from dcdn.models.wukong import WukongSmallConfig  # noqa: E402
from dcdn.multitask import (  # noqa: E402
    MultiTargetSpec,
    TargetSpec,
    compute_task_constants,
    macro_nloss,
)
from dcdn.multitask_capacity import (  # noqa: E402
    common_counts,
    dcdn_counts,
    resolve_capacity_control,
)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


beff = _load("benchmark_efficiency", REPO_ROOT / "scripts" / "benchmark_efficiency.py")

EMBEDDING_DIM = 16
REPRESENTATION_DIM = 64
HEAD_HIDDEN_DIMS = (128, 64)
DIV_EPS = 1e-2

#: `c_user_id` is column 0 of the materialized categorical block, and is the
#: grouping key for every user-grouped metric **on the KuaiRand and KuaiRec
#: views**. It is not universal: the VK-LSVD views carry no user column in the
#: categorical block at all (its six fields have cardinalities 55, 4, 82, 24,
#: 13, 22) and ship the grouping key as a separate `user_ids` array instead.
#: Both readers below therefore prefer an explicit `user_ids` entry when the
#: view supplies one and fall back to this column when it does not, so the
#: KuaiRand and KuaiRec numbers are bit-identical to before.
USER_COLUMN = 0


def grouping_key(split_data: dict) -> "np.ndarray":
    """The user id for user-grouped metrics, however this view stores it."""
    stored = split_data.get("user_ids")
    if stored is not None:
        return stored.cpu().numpy()
    return split_data["categorical"][:, USER_COLUMN].cpu().numpy()

TUNING_SEED = 42
#: The frozen matrix fixes `final_seeds: [42, 43, 44]`. Two facts make three
#: seeds too few HERE, and both are properties of this study rather than of the
#: frozen protocol:
#:
#: * the matrix also records `seed_42_permitted_for: validation_tuning_only`,
#:   and this study evaluates on validation because the test split is locked,
#:   so seed 42's number is the one the learning rate was selected on and is
#:   not an independent measurement - leaving two clean seeds;
#: * the observed per-seed spread is large (capacity-matched DCNv2 ranged
#:   0.937 / 1.790 / 1.697 in the pilot), so two clean seeds separate nothing.
#:
#: The frozen three are therefore run and reported as the protocol table, and
#: the extra seeds are run as a clearly-labelled stability annex. The annex is
#: not the frozen cell and never substitutes for it; it exists so a contrast is
#: not called established, or dismissed, on two noisy points.
FINAL_SEEDS = (42, 43, 44)
ANNEX_SEEDS = (45, 46, 47, 48, 49)
# Four points, not two. The first pass used {3e-4, 1e-3} and the two-layer
# cross arms swung from 12.46 to 2.10 across that single step, which means a
# two-point grid can easily report the penalty of an unlucky learning rate as
# though it were an architectural difference. The grid is widened identically
# for every arm, so no arm gets a tuning budget another one lacks.
LEARNING_RATES = (1e-4, 3e-4, 1e-3, 3e-3)

BATCH_SIZE = 4096
MAX_STEPS = 3000
VALIDATE_EVERY = 150
PATIENCE = 5
WEIGHT_DECAY = 1e-6
GRAD_CLIP = 1.0

TARGETS = ("long_view", "like", "forward", "hate", "watch_time")
KINDS = ("binary", "binary", "binary", "binary", "regression")
SPEC = MultiTargetSpec(
    dataset_id="kuairand_pure",
    targets=tuple(
        TargetSpec(name=n, kind=k) for n, k in zip(TARGETS, KINDS, strict=True)
    ),
    primary_confirmatory_target="long_view",
)


@dataclass(frozen=True)
class Arm:
    arm_id: str
    label: str
    regime: str
    config: Any
    with_afn_view: bool = False


class KuaiRandEncoder(nn.Module):
    """Categorical embeddings plus a per-field numeric lift, shared by all arms."""

    def __init__(self, cardinalities: list[int], numeric_width: int) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList(
            nn.Embedding(card, EMBEDDING_DIM) for card in cardinalities
        )
        for table in self.embeddings:
            nn.init.normal_(table.weight, std=0.01)
        self.numeric_weight = nn.Parameter(torch.empty(numeric_width, EMBEDDING_DIM))
        self.numeric_bias = nn.Parameter(torch.empty(numeric_width, EMBEDDING_DIM))
        nn.init.normal_(self.numeric_weight, std=0.1)
        nn.init.zeros_(self.numeric_bias)

    def forward(
        self, categorical: torch.Tensor, numeric: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        looked_up = [
            table(categorical[:, i])
            for i, table in enumerate(self.embeddings)
        ]
        bypass = torch.cat(looked_up, dim=1)
        numeric_fields = numeric.unsqueeze(-1) * self.numeric_weight + self.numeric_bias
        fields = torch.cat([torch.stack(looked_up, dim=1), numeric_fields], dim=1)
        return bypass, fields


class KuaiRandModel(nn.Module):
    """Shared encoder, audited interaction block, per-target heads."""

    def __init__(
        self,
        cardinalities: list[int],
        numeric_width: int,
        denominator_mask: tuple[bool, ...],
        config: Any,
        init_seed: int,
        *,
        with_afn_view: bool,
        output_dim: int | None = None,
    ) -> None:
        super().__init__()
        spec = EncoderSpec(
            bypass_dim=len(cardinalities) * EMBEDDING_DIM,
            operand_dim=numeric_width,
            num_fields=len(cardinalities) + numeric_width,
            embedding_dim=EMBEDDING_DIM,
            representation_dim=REPRESENTATION_DIM,
            denominator_mask=denominator_mask,
            full_feature_schema_sha256=None,
            operand_schema_sha256=None,
        )
        with torch.random.fork_rng(devices=[], enabled=True):
            torch.random.default_generator.manual_seed(init_seed)
            self.encoder = KuaiRandEncoder(cardinalities, numeric_width)
            self.afn_encoder = (
                KuaiRandEncoder(cardinalities, numeric_width) if with_afn_view else None
            )
        self.model = build_controlled_model(
            encoder_spec=spec,
            interaction_config=config,
            head_config=HeadConfig(
                hidden_dims=HEAD_HIDDEN_DIMS,
                dropout=0.0,
                # Defaults to the five multi-target heads; the single-target
                # follow-up in `run_kuairand_single_target.py` passes 1 so that
                # everything upstream of the head stays bit-for-bit identical
                # between the two studies.
                output_dim=len(TARGETS) if output_dim is None else output_dim,
            ),
            init_seed=init_seed,
        )
        self.spec = spec

    def forward(
        self,
        categorical: torch.Tensor,
        numeric: torch.Tensor,
        validity: torch.Tensor,
    ) -> torch.Tensor:
        bypass, fields = self.encoder(categorical, numeric)
        auxiliary = None
        if self.afn_encoder is not None:
            _, auxiliary = self.afn_encoder(categorical, numeric)
        batch = EncodedInteractionBatch(
            bypass_features=bypass,
            operand_features=numeric,
            field_embeddings=fields,
            denominator_validity=validity,
            afn_dnn_embeddings=auxiliary,
        )
        return self.model(batch).logits

    def counts(self) -> dict[str, int]:
        def count(module: nn.Module) -> int:
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        embedding_ids = {
            id(p)
            for module in self.modules()
            if isinstance(module, nn.Embedding)
            for p in module.parameters()
        }
        interaction = count(self.model.interaction)
        return {
            "interaction": interaction,
            "adapter": count(self.model.adapter),
            "head": count(self.model.head),
            "matched": interaction + count(self.model.adapter) + count(self.model.head),
            "non_embedding": sum(
                p.numel()
                for p in self.parameters()
                if p.requires_grad and id(p) not in embedding_ids
            ),
            "total": count(self),
        }


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


def _dcdn() -> DCDNCoreConfig:
    return DCDNCoreConfig(
        num_layers=1,
        div_eps=DIV_EPS,
        product_init_scale=0.1,
        div_init_scale=0.1,
        runtime_value_checks=False,
    )


def build_arms(capacity: dict[str, int]) -> tuple[Arm, ...]:
    return (
        Arm("dcnv2_reduced", "M1 DCNv2-REDUCED-MTL", "reduced",
            DCNv2FullRankConfig(num_layers=2, product_init_scale=0.1)),
        Arm("dcdn_reduced", "M2 DCDN-Core-REDUCED-MTL", "reduced", _dcdn()),
        Arm("afn_reduced", "M3 AFN+-REDUCED-MTL", "reduced", _afn(),
            with_afn_view=True),
        Arm("dcnv2_capacity_reduced", "M4 DCNv2-capacity-matched-REDUCED-MTL",
            "reduced",
            DCNv2CapacityCoreConfig(
                num_layers=capacity["num_layers"],
                low_rank=capacity["low_rank"],
                num_experts=capacity["num_experts"],
                product_init_scale=0.1,
            )),
        Arm("dcnv2_full", "M5 DCNv2-FULL-MTL", "full",
            DCNv2FullRankConfig(num_layers=2, product_init_scale=0.1)),
        Arm("dcdn_full", "M6 DCDN-Core-FULL-MTL", "full", _dcdn()),
        Arm("afn_full", "M7 AFN+-FULL-MTL", "full", _afn(), with_afn_view=True),
    )


#: The five architectures outside the frozen M1-M7 matrix. They exist to place
#: DCDN against the field rather than against three chosen comparators, and they
#: are reported SEPARATELY from M1-M7 so the frozen matrix is not silently
#: widened.
#:
#: Every one runs on the FULL view, deliberately. The question is whether DCDN
#: on the REDUCED inputs can stand against the field given EVERYTHING - all the
#: engineered divisions included. Putting the field on the reduced view too
#: would answer a narrower question, and would flatter DCDN by denying its
#: competitors the very features that were built for them.
def build_field_arms() -> tuple[Arm, ...]:
    return (
        Arm("mlp_full", "F1 DNN/MLP-FULL", "full",
            MLPConfig(hidden_dims=(400, 400, 400), dropout=0.0)),
        Arm("deepfm_full", "F2 DeepFM-FULL", "full",
            DeepFMConfig(deep_hidden_dims=(400, 400, 400), dropout=0.0)),
        Arm("dcn_full", "F3 DCN-original-FULL", "full",
            DCNConfig(num_layers=2, product_init_scale=0.1)),
        Arm("autoint_full", "F4 AutoInt+-FULL", "full",
            AutoIntPlusConfig(
                attention_layers=3, attention_dim=64, num_heads=2,
                dnn_hidden_dims=(400, 400), dropout=0.0,
                scale_attention=True, layer_norm=True,
            )),
        Arm("wukong_full", "F5 Wukong-small-FULL", "full",
            WukongSmallConfig(
                num_layers=2, lcb_features=16, fmb_features=16, fmb_rank=8,
                fmb_hidden_dims=(256, 256), dropout=0.0, layer_norm=True,
            )),
        # EulerNet is the second algebraic prior in the field, after AFN+: in
        # polar form a monomial with arbitrary real exponents is linear, so a
        # learned coefficient of -1 is reachable and division is inside its
        # hypothesis class. Three layers of sixteen order-fields is the source
        # paper's mid-size setting; it is a published configuration like the
        # other four field arms, not a capacity-matched comparator.
        Arm("eulernet_full", "F6 EulerNet-FULL", "full",
            EulerNetConfig(
                num_layers=3, fields_per_layer=16, dropout=0.0, exp_clip=10.0,
            )),
        Arm("eulernet_reduced", "F7 EulerNet-REDUCED", "reduced",
            EulerNetConfig(
                num_layers=3, fields_per_layer=16, dropout=0.0, exp_clip=10.0,
            )),
    )


#: The two preconditioners. Both are run and both are published; see the
#: comment in `Views.__init__` for why the choice is not pre-registered.
#: `none` is the correct choice for a dataset whose own contract already
#: mandates a fitted normalization at materialization time - KuaiRec divides
#: every P and E column by max(train q99, 1.0) - because rescaling again here
#: would normalize twice.
PRECONDITIONERS = ("geometric", "rms", "none")


def _column_scale(block: np.ndarray, kind: str) -> np.ndarray:
    """Per-column train scale, guarded to stay positive and finite.

    `rms` leaves each column with unit second moment. `geometric` divides by
    the train geometric mean of the strictly positive entries, which sets the
    column's median near one.
    """

    values = block.astype(np.float64)
    if kind == "none":
        return np.ones(values.shape[1], dtype=np.float64)
    if kind == "rms":
        scale = np.sqrt(np.mean(np.square(values), axis=0))
    elif kind == "geometric":
        scale = np.ones(values.shape[1], dtype=np.float64)
        for index in range(values.shape[1]):
            positive = values[:, index]
            positive = positive[positive > 0.0]
            if positive.size:
                scale[index] = np.exp(np.mean(np.log(positive)))
    else:
        raise ValueError(f"unknown preconditioner: {kind}")
    scale[~np.isfinite(scale) | (scale <= 0.0)] = 1.0
    return scale


class Views:
    """The materialized train/validation arrays, on device."""

    def __init__(
        self, root: Path, device: torch.device, preconditioner: str = "geometric"
    ) -> None:
        manifest = json.loads((root / "manifest_v1.json").read_text(encoding="utf-8"))
        self.manifest = manifest
        self.cardinalities = manifest["cardinalities"]
        # The engineered division block is called R on KuaiRand and E on
        # KuaiRec. Both mean the same thing - the manually materialized
        # divisions of historical counters that the REDUCED view removes - so
        # the name is read from the manifest rather than hard-coded.
        self.engineered_block = "R" if "R_names" in manifest else "E"
        self.blocks = ("P", "V", self.engineered_block)
        p_count = len(manifest["P_names"])
        v_count = len(manifest["V_names"])
        r_count = len(manifest[f"{self.engineered_block}_names"])
        # The frozen mask is aligned to P on KuaiRand and to P+V on KuaiRec, so
        # accept either and pad to the reduced width. V and the engineered
        # block are never denominators.
        mask = list(manifest["denominator_mask"])
        if len(mask) == p_count:
            mask = mask + [False] * v_count
        if len(mask) != p_count + v_count:
            raise ValueError(
                f"denominator mask has {len(mask)} entries; expected "
                f"{p_count} or {p_count + v_count}"
            )
        self.mask_reduced = tuple(mask)
        self.mask_full = tuple(mask + [False] * r_count)
        self.reduced_width = p_count + v_count
        self.full_width = p_count + v_count + r_count

        # ------------------------------------------------------------------
        # Input preconditioning
        # ------------------------------------------------------------------
        # The frozen feature contract fixes the natural-unit scales and forbids
        # fitting them ("Count and millisecond scales are fixed, not fitted"),
        # so the materialized P columns carry their true magnitudes - the
        # all-time tab duration sum reaches 9.8e7. Two multiplicative
        # interaction layers over operands that large overflow float32, and
        # every arm diverged on the first tuning trial.
        #
        # The rescale applied here is therefore a MODEL INPUT preconditioner,
        # not a feature change, and it is deliberately PURELY MULTIPLICATIVE:
        # each column is divided by its train-only root-mean-square. No shift,
        # no log. That choice is load-bearing for fairness. A log or z-score
        # would turn a quotient of two counters into a difference, which is
        # exactly the structure DCDN's reciprocal branch is supposed to have
        # to discover - it would hand the additive baselines the answer. Under
        # x -> x/s the ratio n/d is unchanged up to the constant s_d/s_n, which
        # any of the compared architectures can absorb into its weights, so the
        # transform is neutral between arms. Positivity, and hence the
        # denominator validity mask, is preserved because every s is positive.
        # Which of the two scales is used is NOT pre-registered, because the
        # first attempt (`rms`) was chosen before training and then observed to
        # make DCDN diverge: under it the median denominator lands at 0.43 and
        # the first percentile at 2e-4, so the model's own declared `div_eps`
        # floor of 1e-2 saturates the reciprocal carrier for a large share of
        # rows. Switching to `geometric` after seeing that is a post-hoc choice
        # and cannot be laundered into a principled one, so the matrix is run
        # under BOTH and both are published. Agreement between them is the
        # evidence that the conclusion does not rest on this knob.
        self.preconditioner = preconditioner
        train_payload = np.load(root / "train.npz")
        self.input_scale = {
            block: _column_scale(train_payload[block], preconditioner)
            for block in self.blocks
        }
        # V is a set of 0/1 validity indicators with fixed semantics; leave it
        # exactly as materialized so a flag never becomes a non-unit number.
        self.input_scale["V"] = np.ones_like(self.input_scale["V"])

        self.data: dict[str, dict[str, torch.Tensor]] = {}
        for split in ("train", "validation"):
            payload = np.load(root / f"{split}.npz")
            blocks = {
                block: (payload[block] / self.input_scale[block]).astype(np.float32)
                for block in self.blocks
            }
            numeric_reduced = np.concatenate([blocks["P"], blocks["V"]], axis=1)
            numeric_full = np.concatenate(
                [blocks[name] for name in self.blocks], axis=1
            )
            labels = payload["labels"]
            self.data[split] = {
                "categorical": torch.from_numpy(payload["categorical"]).to(device),
                "reduced": torch.from_numpy(numeric_reduced).to(device),
                "full": torch.from_numpy(numeric_full).to(device),
                "labels": torch.from_numpy(labels).to(device),
                "masks": torch.ones(labels.shape, dtype=torch.bool, device=device),
            }

    def numeric(self, split: str, regime: str) -> torch.Tensor:
        return self.data[split][regime]

    def validity(self, split: str, regime: str) -> torch.Tensor:
        numeric = self.numeric(split, regime)
        mask = self.mask_reduced if regime == "reduced" else self.mask_full
        flags = torch.tensor(mask, device=numeric.device)
        # A denominator coordinate is valid when it is strictly positive; the
        # materializer already floors eligible fields, so this is a guard.
        return (~flags) | (numeric > 0)


EVAL_BATCH = 32768


@torch.no_grad()
def forward_in_batches(
    model: nn.Module,
    categorical: torch.Tensor,
    numeric: torch.Tensor,
    validity: torch.Tensor,
) -> torch.Tensor:
    """Score a whole split in slices.

    A single forward over the full validation split would allocate a field
    tensor of several gigabytes for the wide FULL regime, and twice that for
    AFN+, which owns a second embedding view. Slicing keeps peak memory flat
    and changes no result: the model is in eval mode with no cross-row state.
    """

    outputs = []
    for start in range(0, categorical.shape[0], EVAL_BATCH):
        stop = start + EVAL_BATCH
        outputs.append(
            model(categorical[start:stop], numeric[start:stop], validity[start:stop])
        )
    return torch.cat(outputs, dim=0)


def train_one(
    model: nn.Module,
    views: Views,
    arm: Arm,
    constants,
    device: torch.device,
    learning_rate: float,
    seed: int,
) -> dict[str, Any]:
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8,
        weight_decay=WEIGHT_DECAY,
    )
    train = views.data["train"]
    valid = views.data["validation"]
    train_numeric = views.numeric("train", arm.regime)
    valid_numeric = views.numeric("validation", arm.regime)
    train_validity = views.validity("train", arm.regime)
    valid_validity = views.validity("validation", arm.regime)

    generator = torch.Generator(device="cpu").manual_seed(seed)
    best = {"macro": float("inf"), "state": None, "step": -1}
    stale = 0
    nonfinite = 0
    started = time.perf_counter()
    rows = train_numeric.shape[0]

    for step in range(1, MAX_STEPS + 1):
        model.train()
        index = torch.randint(0, rows, (BATCH_SIZE,), generator=generator).to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(
            train["categorical"][index], train_numeric[index], train_validity[index]
        )
        if not bool(torch.isfinite(logits).all()):
            nonfinite += 1
            if nonfinite > 1:
                raise FloatingPointError(f"{arm.arm_id}: repeated non-finite logits")
            continue
        loss = macro_nloss(
            SPEC, constants, logits, train["labels"][index], train["masks"][index]
        ).aggregate
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        if step % VALIDATE_EVERY == 0:
            model.eval()
            value = float(
                macro_nloss(
                    SPEC, constants,
                    torch.nan_to_num(
                        forward_in_batches(
                            model, valid["categorical"], valid_numeric, valid_validity
                        )
                    ),
                    valid["labels"], valid["masks"],
                ).aggregate
            )
            if value < best["macro"]:
                best = {
                    "macro": value,
                    "state": {
                        k: v.detach().clone() for k, v in model.state_dict().items()
                    },
                    "step": step,
                }
                stale = 0
            else:
                stale += 1
                if stale >= PATIENCE:
                    break

    if best["state"] is None:
        raise RuntimeError(f"{arm.arm_id}: no validation checkpoint recorded")
    model.load_state_dict(best["state"])
    return {
        "validation_macro_nloss": best["macro"],
        "best_step": best["step"],
        "nonfinite_training_batches": nonfinite,
        "seconds": time.perf_counter() - started,
    }




class PredictionStore:
    """Per-row validation predictions, so a new metric never needs a retrain.

    Adding user-GAUC and X-AUC to this study cost a full retrain of every arm
    and seed, purely because predictions were computed inline and discarded
    with the model. Keeping them makes any future metric a seconds-long
    recomputation instead.

    They go to `$DCDN_DATA_CACHE`, never to the repository: at float32 and
    311,490 validation rows a single run is roughly 350 MB, which has no place
    in git. float32 rather than float16 is deliberate - X-AUC counts tied
    predictions explicitly, and half precision would manufacture ties that the
    model did not produce.

    Opt-in via `--save-predictions`. With the flag absent this class is never
    constructed and the run behaves exactly as it did before.
    """

    def __init__(self, root: Path, study: str) -> None:
        self.path = root / "predictions" / f"{study}.npz"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.arrays: dict[str, np.ndarray] = {}

    def add_context(self, views: Views, spec) -> None:
        valid = views.data["validation"]
        self.arrays["labels"] = valid["labels"].cpu().numpy()
        self.arrays["masks"] = valid["masks"].cpu().numpy()
        self.arrays["user_ids"] = grouping_key(valid)
        self.arrays["target_names"] = np.array(list(spec.names))

    def add(self, arm_id: str, seed: int, predictions: np.ndarray) -> None:
        self.arrays[f"pred__{arm_id}__{seed}"] = predictions.astype(np.float32)

    def write(self) -> str:
        with tempfile.NamedTemporaryFile(
            dir=self.path.parent, suffix=".npz", delete=False
        ) as handle:
            temporary = Path(handle.name)
        np.savez(temporary, **self.arrays)
        os.replace(temporary, self.path)
        digest = hashlib.sha256(self.path.read_bytes()).hexdigest()
        return digest


def frozen_metric_set(
    spec,
    predictions: np.ndarray,
    labels: np.ndarray,
    masks: np.ndarray,
    user_ids: np.ndarray,
    c_seconds: float,
) -> dict[str, Any]:
    """Every metric `configs/metrics_multitask_v1.yaml` freezes, per head.

    Binary heads get log-loss, ROC-AUC, AUPRC and uniform user-GAUC with its
    coverage. The regression head gets normalized MSE, MAE in seconds, and the
    continuous pairwise ordering metrics: global X-AUC and its user-grouped
    counterpart. All of it comes from `src/dcdn/metrics.py`, which is the frozen
    implementation - nothing is recomputed locally.

    KuaiRand-Pure ships no session identifier, so `session_gauc` is reported as
    null with an explicit reason. The frozen `session_policy` forbids
    synthesizing sessions to fill the gap.
    """

    report: dict[str, Any] = {}
    for index, target in enumerate(spec.targets):
        column = predictions[:, index]
        truth = labels[:, index]
        observed = masks[:, index].astype(bool)
        if target.kind == "binary":
            probability = 1.0 / (1.0 + np.exp(-np.clip(column, -60.0, 60.0)))
            binary = masked_binary_target_metrics(
                truth, probability, user_ids, observed
            )
            report[target.name] = {
                "kind": "binary",
                "log_loss": binary.log_loss,
                "roc_auc": binary.roc_auc,
                "auprc": binary.auprc,
                "user_gauc": binary.user_gauc.value,
                "user_gauc_group_coverage": binary.user_gauc.group_coverage,
                "user_gauc_row_coverage": binary.user_gauc.row_coverage,
                "user_gauc_retained_groups": binary.user_gauc.retained_groups,
                "user_gauc_total_groups": binary.user_gauc.total_groups,
                "user_gauc_weighting": binary.user_gauc.weighting,
                "session_gauc": None,
                "session_gauc_availability": (
                    "not_available_kuairand_pure_has_no_session_identifier"
                ),
                "observed_rows": binary.observed_rows,
            }
            continue

        selected_truth = np.clip(truth[observed], 0.0, 1.0)
        selected_prediction = column[observed]
        watch = watch_time_metrics(
            selected_truth, selected_prediction,
            duration_constant_seconds=c_seconds,
        )
        grouped = group_concordance(
            c_seconds * selected_truth,
            c_seconds * np.clip(selected_prediction, 0.0, 1.0),
            user_ids[observed],
            weighting="uniform",
        )
        report[target.name] = {
            "kind": "regression",
            "normalized_mse": watch.normalized_mse,
            "mae_seconds": watch.mae_seconds,
            "mse_seconds2": watch.mse_seconds2,
            "global_xauc": watch.global_xauc.value,
            "global_xauc_comparable_pair_coverage": (
                watch.global_xauc.comparable_pair_coverage
            ),
            "global_xauc_comparable_pairs": watch.global_xauc.comparable_pairs,
            "user_xgauc": grouped.value,
            "user_xgauc_group_coverage": grouped.group_coverage,
            "user_xgauc_row_coverage": grouped.row_coverage,
            "user_xgauc_retained_groups": grouped.retained_groups,
            "user_xgauc_total_groups": grouped.total_groups,
            "user_xgauc_weighting": grouped.weighting,
            "session_gauc": None,
            "session_gauc_availability": (
                "not_available_kuairand_pure_has_no_session_identifier"
            ),
            "observed_rows": int(observed.sum()),
        }
    return report


@torch.no_grad()
def evaluate(model: nn.Module, views: Views, arm: Arm, constants) -> dict[str, Any]:
    model.eval()
    valid = views.data["validation"]
    numeric = views.numeric("validation", arm.regime)
    validity = views.validity("validation", arm.regime)
    logits = forward_in_batches(model, valid["categorical"], numeric, validity)
    nonfinite = int((~torch.isfinite(logits)).sum())
    safe = torch.nan_to_num(logits)
    report = macro_nloss(
        SPEC, constants, safe, valid["labels"], valid["masks"]
    ).as_report(SPEC)
    report["nonfinite_predictions"] = nonfinite
    # ROC-AUC per binary head, by ranks, without an external dependency.
    labels = valid["labels"].cpu().numpy()
    scores = safe.cpu().numpy()
    auc: dict[str, float] = {}
    for index, (name, kind) in enumerate(zip(TARGETS, KINDS, strict=True)):
        if kind != "binary":
            continue
        auc[name] = _roc_auc(scores[:, index], labels[:, index])
    report["roc_auc"] = auc
    report["_predictions"] = scores
    report["frozen_metrics"] = frozen_metric_set(
        SPEC, scores, labels,
        valid["masks"].cpu().numpy(),
        grouping_key(valid),
        float(views.manifest["watch_time_c_seconds"]),
    )
    watch_index = TARGETS.index("watch_time")
    normalized_mae = float(
        np.abs(scores[:, watch_index] - labels[:, watch_index]).mean()
    )
    report["watch_time_mae"] = normalized_mae
    # The contract asks for regression error in natural units. The head predicts
    # `clip(play_seconds, 0, c) / c`, so seconds are recovered by the same frozen
    # train-only constant the materializer used to build the label.
    report["watch_time_mae_seconds"] = normalized_mae * float(
        views.manifest["watch_time_c_seconds"]
    )
    return report


def _roc_auc(score: np.ndarray, label: np.ndarray) -> float:
    order = np.argsort(score, kind="stable")
    ranks = np.empty(len(score), dtype=np.float64)
    sorted_score = score[order]
    start = 0
    for position in range(1, len(sorted_score) + 1):
        at_end = position == len(sorted_score)
        if at_end or sorted_score[position] != sorted_score[start]:
            ranks[order[start:position]] = (start + position + 1) / 2.0
            start = position
    positives = float(label.sum())
    negatives = float(len(label) - positives)
    if positives == 0 or negatives == 0:
        return float("nan")
    return float(
        (ranks[label == 1].sum() - positives * (positives + 1) / 2)
        / (positives * negatives)
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--views",
        type=Path,
        default=Path(__import__("os").environ.get("DCDN_DATA_CACHE", "/tmp"))
        / "runtime"
        / "kuairand_pure_multitask_v1",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "kuairand_multitask" / "rq23_v1",
    )
    parser.add_argument(
        "--preconditioner", choices=PRECONDITIONERS, default="rms"
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="persist per-row validation predictions to the data cache so a "
             "future metric can be computed without retraining",
    )
    parser.add_argument(
        "--annex-seeds",
        action="store_true",
        help="also run the stability-annex seeds beyond the frozen three",
    )
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    beff.prepare_output_dir(args.output_dir)
    views = Views(args.views, device, preconditioner=args.preconditioner)
    print(
        f"views: train={views.data['train']['labels'].shape[0]:,} "
        f"validation={views.data['validation']['labels'].shape[0]:,} "
        f"reduced={views.reduced_width} full={views.full_width} "
        f"denominators={sum(views.mask_reduced)}"
    )

    constants = compute_task_constants(
        SPEC, views.data["train"]["labels"], views.data["train"]["masks"]
    )
    print("frozen task constants: " + "  ".join(
        f"{n}={v:.6g}" for n, v in zip(SPEC.names, constants.values, strict=True)
    ))

    # Mechanical capacity control: no validation metric is read here.
    common = common_counts(
        len(views.cardinalities) * EMBEDDING_DIM + views.reduced_width,
        REPRESENTATION_DIM, HEAD_HIDDEN_DIMS, len(TARGETS),
    )
    target_counts = dcdn_counts(views.reduced_width, 1, common)
    resolution = resolve_capacity_control(
        target=target_counts, operand_width=views.reduced_width, common=common,
        ranks=tuple(range(1, 129)), layer_options=(1, 2), expert_options=(1, 2, 4),
    )
    payload = resolution.payload()
    if not resolution.matched:
        raise SystemExit(f"no capacity control inside tolerance: {json.dumps(payload)}")
    capacity = payload["selected"]
    print(
        f"capacity control: rank={capacity['low_rank']} "
        f"layers={capacity['num_layers']} experts={capacity['num_experts']} "
        f"param_ratio={payload['selected_ratios']['parameter_ratio']:.4f} "
        f"flop_ratio={payload['selected_ratios']['flop_ratio']:.4f}"
    )

    arms = build_arms(capacity)

    def make(arm: Arm, seed: int) -> KuaiRandModel:
        width = views.reduced_width if arm.regime == "reduced" else views.full_width
        mask = views.mask_reduced if arm.regime == "reduced" else views.mask_full
        return KuaiRandModel(
            views.cardinalities, width, mask, arm.config, seed,
            with_afn_view=arm.with_afn_view,
        )

    manifest = {
        "artifact_role": "kuairand_pure_rq23_pre_training_manifest",
        "schema_version": 1,
        "dataset_id": "kuairand_pure",
        "targets": list(TARGETS),
        "target_kinds": list(KINDS),
        "objective_spec_sha256": SPEC.contract_sha256,
        "task_constants_sha256": constants.contract_sha256,
        "view_manifest_sha256": hashlib.sha256(
            (args.views / "manifest_v1.json").read_bytes()
        ).hexdigest(),
        "capacity_resolution": payload,
        # The input preconditioner is part of what produced these numbers, so
        # it is hashed here rather than left implicit in the code.
        "input_preconditioner": {
            "kind": f"per_column_train_{views.preconditioner}_divide",
            "alternative_also_published": [
                k for k in PRECONDITIONERS if k != views.preconditioner
            ],
            "pre_registered": False,
            "shift": "none",
            "fitted_on": "train_only",
            "applied_to_blocks": ["P", "R"],
            "identity_on_blocks": ["V"],
            "ratio_preserving": True,
            "reason": (
                "frozen feature contract forbids fitted scales, but raw "
                "counters reach 9.8e7 and overflow float32 through two "
                "multiplicative interaction layers; a purely multiplicative "
                "rescale bounds magnitude without linearizing quotients. The "
                "specific scale was selected after observing divergence and is "
                "therefore reported under both variants, not pre-registered."
            ),
            "scale_sha256": {
                block: hashlib.sha256(
                    np.ascontiguousarray(scale, dtype=np.float64).tobytes()
                ).hexdigest()
                for block, scale in sorted(views.input_scale.items())
            },
        },
        "arm_order": [a.arm_id for a in arms],
        "final_seeds": list(FINAL_SEEDS),
        "annex_seeds": list(ANNEX_SEEDS) if args.annex_seeds else [],
        "seed_42_is_tuning_contaminated_on_validation": True,
        "tuning_seed": TUNING_SEED,
        "learning_rates": list(LEARNING_RATES),
        "test_unlocked": False,
        "evaluated_split": "validation",
        "locked_test_window_read": False,
    }
    digest = beff.publish_json(
        args.output_dir / "pre_training_manifest_v1.json", manifest
    )
    print(f"pre-training manifest sha256: {digest}\n")

    store = None
    if args.save_predictions:
        store = PredictionStore(
            Path(os.environ.get("DCDN_DATA_CACHE", "/tmp")), args.output_dir.name
        )
        store.add_context(views, SPEC)

    print("--- tuning (seed 42, validation-selected) ---")
    selected: dict[str, float] = {}
    for arm in arms:
        trials = []
        for learning_rate in LEARNING_RATES:
            outcome = train_one(
                make(arm, TUNING_SEED), views, arm, constants, device,
                learning_rate, TUNING_SEED,
            )
            trials.append({"learning_rate": learning_rate, **outcome})
            print(
                f"  {arm.arm_id:<24} lr={learning_rate:<7} "
                f"val Macro-NLoss={outcome['validation_macro_nloss']:.6f} "
                f"({outcome['seconds']:.0f}s)"
            )
        best = min(trials, key=lambda t: t["validation_macro_nloss"])
        selected[arm.arm_id] = best["learning_rate"]

    frozen = {
        "selected_learning_rates": selected,
        "capacity_resolution": payload,
        "objective_spec_sha256": SPEC.contract_sha256,
        "task_constants_sha256": constants.contract_sha256,
    }
    frozen_digest = beff.publish_json(
        args.output_dir / "selected_configs_v1.json", frozen
    )
    print(f"\nfrozen selected configs sha256: {frozen_digest}\n")

    seeds = tuple(FINAL_SEEDS) + (tuple(ANNEX_SEEDS) if args.annex_seeds else ())
    listed = ", ".join(str(s) for s in seeds)
    print(f"--- final evaluation (seeds {listed}; validation) ---")
    records: list[dict[str, Any]] = []
    for arm in arms:
        for seed in seeds:
            model = make(arm, seed)
            counts = model.counts()
            training = train_one(
                model, views, arm, constants, device, selected[arm.arm_id], seed
            )
            report = evaluate(model, views, arm, constants)
            records.append({
                "arm_id": arm.arm_id, "label": arm.label, "regime": arm.regime,
                "seed_class": (
                    "frozen_protocol" if seed in FINAL_SEEDS else "stability_annex"
                ),
                "tuning_contaminated": seed == TUNING_SEED,
                "seed": seed, "learning_rate": selected[arm.arm_id],
                "parameters": counts, "training": training, "validation": report,
            })
            stored = report.pop("_predictions")
            if store is not None:
                store.add(arm.arm_id, seed, stored)
            print(
                f"  {arm.arm_id:<24} seed={seed} "
                f"Macro-NLoss={report['macro_nloss']:.6f} "
                f"long_view_AUC={report['roc_auc']['long_view']:.4f} "
                f"watch_MAE={report['watch_time_mae']:.4f}"
            )
            del model
            torch.cuda.empty_cache()

    predictions_digest = store.write() if store is not None else None
    if predictions_digest:
        print(f"\npredictions cached: {store.path} sha256={predictions_digest}")

    results = {
        "artifact_role": "kuairand_pure_rq23_result",
        "schema_version": 1,
        "pre_training_manifest_sha256": digest,
        "selected_configs_sha256": frozen_digest,
        "evaluated_split": "validation",
        "test_unlocked": False,
        "targets": list(TARGETS),
        # Where the raw per-row validation predictions live, when they were
        # kept. Outside the repository by design; the digest pins the file.
        "cached_predictions_sha256": predictions_digest,
        "cached_predictions_path": (
            str(store.path) if store is not None else None
        ),
        "records": records,
    }
    results_digest = beff.publish_json(args.output_dir / "results_v1.json", results)
    print(f"\npublished results_v1.json sha256={results_digest}")
    print(f"output directory: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
