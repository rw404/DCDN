from __future__ import annotations

import pytest
import torch
from torch import nn

from dcdn.core import DCDNCore
from dcdn.fused_inference import (
    TRITON_AVAILABLE,
    FusedCrossNet,
    FusedDCDNCore,
    FusedInferenceError,
    max_abs_difference,
)

CUDA = torch.cuda.is_available()
requires_gpu = pytest.mark.skipif(
    not (CUDA and TRITON_AVAILABLE), reason="fused path requires CUDA and Triton"
)

#: Float32 GEMM reassociation only; anything larger is a real disagreement.
TOLERANCE = 1e-3

#: (features, layers, denominators) covering both study geometries and depths.
GEOMETRIES = ((8, 1, 3), (11, 1, 3), (61, 1, 5), (61, 2, 5), (78, 1, 5), (157, 2, 5))


class ReferenceCross(nn.Module):
    """The full-rank DCNv2 recurrence exactly as ``cross.py`` implements it."""

    def __init__(self, dim: int, layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(dim, dim) for _ in range(layers))

    def forward(self, anchor: torch.Tensor) -> torch.Tensor:
        state = anchor
        for layer in self.layers:
            state = state + anchor * layer(state)
        return state


def _core(features: int, layers: int, denominators: int, device: str = "cpu"):
    mask = [index < denominators for index in range(features)]
    core = DCDNCore(
        features, layers, mask, div_eps=1e-2, runtime_value_checks=False
    )
    return core.to(device).eval()


def _positive(batch: int, features: int, device: str, seed: int = 0):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return (torch.rand(batch, features, generator=generator) * 9.0 + 1.0).to(device)


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------


def test_module_reports_triton_availability_without_importing_it_at_use() -> None:
    assert isinstance(TRITON_AVAILABLE, bool)


def test_fused_construction_fails_cleanly_without_triton(monkeypatch) -> None:
    import dcdn.fused_inference as module

    monkeypatch.setattr(module, "TRITON_AVAILABLE", False)
    with pytest.raises(FusedInferenceError, match="Triton is required"):
        module._require_triton()


# ---------------------------------------------------------------------------
# Numerical equivalence — the property the whole study rests on
# ---------------------------------------------------------------------------


@requires_gpu
@pytest.mark.parametrize(("features", "layers", "denominators"), GEOMETRIES)
@pytest.mark.parametrize("batch", (1, 64, 1024))
def test_fused_dcdn_matches_the_audited_core(
    features: int, layers: int, denominators: int, batch: int
) -> None:
    core = _core(features, layers, denominators, "cuda")
    fused = FusedDCDNCore.from_core(core).to("cuda").eval()
    x0 = _positive(batch, features, "cuda")
    with torch.inference_mode():
        assert max_abs_difference(core(x0), fused(x0)) < TOLERANCE


@requires_gpu
@pytest.mark.parametrize(("features", "layers", "denominators"), GEOMETRIES)
def test_fused_carrier_matches_the_audited_carrier(
    features: int, layers: int, denominators: int
) -> None:
    core = _core(features, layers, denominators, "cuda")
    fused = FusedDCDNCore.from_core(core).to("cuda").eval()
    x0 = _positive(256, features, "cuda")
    validity = torch.ones_like(x0, dtype=torch.bool)
    with torch.inference_mode():
        expected = core.reciprocal_carrier(x0, validity)
        assert max_abs_difference(expected, fused.reciprocal_carrier(x0)) < 1e-6


@requires_gpu
def test_fused_carrier_is_zero_outside_the_mask() -> None:
    core = _core(8, 1, 3, "cuda")
    fused = FusedDCDNCore.from_core(core).to("cuda").eval()
    with torch.inference_mode():
        carrier = fused.reciprocal_carrier(_positive(16, 8, "cuda"))
    assert float(carrier[:, :3].min()) > 0.0
    assert float(carrier[:, 3:].abs().max()) == 0.0


@requires_gpu
def test_fused_carrier_applies_the_reciprocal_floor() -> None:
    core = _core(4, 1, 2, "cuda")
    fused = FusedDCDNCore.from_core(core).to("cuda").eval()
    tiny = torch.full((2, 4), 1e-6, device="cuda")
    with torch.inference_mode():
        carrier = fused.reciprocal_carrier(tiny)
    # 1 / div_eps = 100, not 1e6.
    assert float(carrier[:, :2].max()) == pytest.approx(100.0, rel=1e-5)


@requires_gpu
@pytest.mark.parametrize("layers", (1, 2))
@pytest.mark.parametrize("features", (8, 61, 157, 174))
def test_fused_cross_matches_the_reference_recurrence(
    features: int, layers: int
) -> None:
    torch.manual_seed(0)
    reference = ReferenceCross(features, layers).to("cuda").eval()
    fused = FusedCrossNet.from_interaction(reference).to("cuda").eval()
    anchor = _positive(64, features, "cuda")
    with torch.inference_mode():
        assert max_abs_difference(reference(anchor), fused(anchor)) < TOLERANCE


# ---------------------------------------------------------------------------
# Preconditions
# ---------------------------------------------------------------------------


@requires_gpu
def test_gating_validity_is_rejected_not_ignored() -> None:
    """The fused carrier implements the all-valid case only."""

    core = _core(8, 1, 3, "cuda")
    fused = FusedDCDNCore.from_core(core).to("cuda").eval()
    x0 = _positive(4, 8, "cuda")
    validity = torch.ones_like(x0, dtype=torch.bool)
    validity[0, 0] = False
    with torch.inference_mode(), pytest.raises(FusedInferenceError, match="all-valid"):
        fused(x0, validity)


@requires_gpu
def test_all_true_validity_is_accepted() -> None:
    core = _core(8, 1, 3, "cuda")
    fused = FusedDCDNCore.from_core(core).to("cuda").eval()
    x0 = _positive(4, 8, "cuda")
    with torch.inference_mode():
        both = fused(x0, torch.ones_like(x0, dtype=torch.bool))
        plain = fused(x0)
    assert torch.equal(both, plain)


@requires_gpu
def test_wrong_width_is_rejected() -> None:
    core = _core(8, 1, 3, "cuda")
    fused = FusedDCDNCore.from_core(core).to("cuda").eval()
    with torch.inference_mode(), pytest.raises(FusedInferenceError, match="shape"):
        fused(_positive(4, 7, "cuda"))


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------


@requires_gpu
def test_fused_weights_carry_no_gradient_and_no_extra_capacity() -> None:
    core = _core(61, 2, 5, "cuda")
    fused = FusedDCDNCore.from_core(core).to("cuda")
    assert not any(p.requires_grad for p in fused.parameters())
    # Concatenation regroups the same weights; it must not add or drop any.
    assert sum(p.numel() for p in fused.parameters()) == sum(
        p.numel() for p in core.parameters()
    )


@requires_gpu
def test_fused_weight_layout_is_product_then_division() -> None:
    core = _core(8, 1, 3, "cuda")
    fused = FusedDCDNCore.from_core(core).to("cuda")
    weight = fused._fused_weight[0]
    assert weight.shape == (16, 8)
    assert torch.equal(weight[:8], core.layers[0].product_block.weight)
    assert torch.equal(weight[8:], core.layers[0].division_block.weight)


@requires_gpu
def test_batch_of_one_is_preserved() -> None:
    core = _core(61, 1, 5, "cuda")
    fused = FusedDCDNCore.from_core(core).to("cuda").eval()
    with torch.inference_mode():
        assert fused(_positive(1, 61, "cuda")).shape == (1, 61)


@requires_gpu
def test_fused_forward_is_deterministic() -> None:
    core = _core(61, 2, 5, "cuda")
    fused = FusedDCDNCore.from_core(core).to("cuda").eval()
    x0 = _positive(128, 61, "cuda")
    with torch.inference_mode():
        assert torch.equal(fused(x0), fused(x0))


# ---------------------------------------------------------------------------
# The validity guard must be movable off the hot path without being weakened
# ---------------------------------------------------------------------------


def test_validity_guard_defaults_to_checking_every_call() -> None:
    """Correctness is the default; the fast path must be opted into."""

    core = FusedDCDNCore.from_core(_core(6, 2, 2))
    assert core.validate_validity_per_call is True


def test_verify_all_valid_accepts_an_all_valid_tensor() -> None:
    core = FusedDCDNCore.from_core(_core(6, 2, 2))
    core.verify_all_valid(torch.ones(4, dtype=torch.bool))
    core.verify_all_valid(None)


def test_verify_all_valid_rejects_a_gating_tensor() -> None:
    """The same rejection the per-call guard performed, off the hot path."""

    core = FusedDCDNCore.from_core(_core(6, 2, 2))
    with pytest.raises(FusedInferenceError, match="all-valid"):
        core.verify_all_valid(torch.zeros(4, dtype=torch.bool))
    with pytest.raises(FusedInferenceError, match="all-valid"):
        core.verify_all_valid(torch.tensor([True, True, False, True]))


def test_disabling_the_per_call_check_does_not_change_any_output() -> None:
    """Turning the guard off is a scheduling change, never a numeric one."""

    if not TRITON_AVAILABLE or not torch.cuda.is_available():
        pytest.skip("needs Triton and CUDA")
    core = FusedDCDNCore.from_core(_core(6, 2, 2)).cuda()
    inputs = torch.rand(8, core.features_count, device="cuda") + 0.5
    validity = torch.ones(8, core.features_count, dtype=torch.bool, device="cuda")

    core.validate_validity_per_call = True
    checked = core(inputs, validity)
    core.validate_validity_per_call = False
    unchecked = core(inputs, validity)
    assert torch.equal(checked, unchecked)


def test_disabled_guard_stops_rejecting_so_the_caller_must_verify() -> None:
    """States the contract the flag creates, so it cannot be set casually."""

    if not TRITON_AVAILABLE or not torch.cuda.is_available():
        pytest.skip("needs Triton and CUDA")
    core = FusedDCDNCore.from_core(_core(6, 2, 2)).cuda()
    inputs = torch.rand(4, core.features_count, device="cuda") + 0.5
    gating = torch.zeros(4, core.features_count, dtype=torch.bool, device="cuda")

    core.validate_validity_per_call = True
    with pytest.raises(FusedInferenceError):
        core(inputs, gating)

    # With the check hoisted the module trusts the caller; that is exactly why
    # `verify_all_valid` must be run once first.
    core.validate_validity_per_call = False
    core(inputs, gating)
