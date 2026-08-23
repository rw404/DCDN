"""EulerNet's defining property: a learned exponent of -1 computes a quotient.

This is the reason the arm exists. `DCDN_HANDOFF.md` names EulerNet-like
approaches as prior art the novelty claim must respect, because in polar form a
monomial with arbitrary real exponents is a linear map on log-amplitude and
phase - so division is inside the hypothesis class rather than approximated.

The tests below check the algebra directly by setting the order matrix, not by
training. A model that cannot represent the quotient exactly when handed the
right coefficients would not be EulerNet, whatever it learns.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from dcdn.models.euler import (  # noqa: E402
    EulerInteractionLayer,
    EulerNetConfig,
    EulerNetInteraction,
)
from dcdn.models.contracts import EncoderSpec  # noqa: E402


def _quotient_layer() -> EulerInteractionLayer:
    """One layer wired to compute field0 / field1 and nothing else."""
    layer = EulerInteractionLayer(
        in_fields=2, out_fields=1, embedding_dim=1, dropout=0.0, exp_clip=None)
    with torch.no_grad():
        layer.orders.copy_(torch.tensor([[1.0], [-1.0]]))
        layer.bias_amplitude.zero_()
        layer.bias_phase.zero_()
        layer.implicit.weight.zero_()
        layer.implicit.bias.zero_()
        # LayerNorm over a width-1 embedding would divide the answer by its own
        # standard deviation, which is zero here; identity keeps the algebra
        # visible. The trained model keeps its norms.
        layer.norm_real = torch.nn.Identity()
        layer.norm_imaginary = torch.nn.Identity()
    return layer


def test_order_minus_one_computes_the_exact_quotient() -> None:
    layer = _quotient_layer()
    numerator = torch.tensor([[[3.0], [0.0]]])
    denominator = torch.tensor([[[4.0], [0.0]]])
    # A real positive z has phase 0, so the imaginary part carries no signal
    # and the real output must be exactly n/d.
    real = torch.tensor([[[3.0], [4.0]]])
    imaginary = torch.zeros_like(real)
    out_real, _ = layer(real, imaginary)
    assert out_real.shape == (1, 1, 1)
    assert out_real.item() == pytest.approx(3.0 / 4.0, rel=1e-5)
    del numerator, denominator


@pytest.mark.parametrize("n,d", [(1.0, 8.0), (7.0, 2.0), (0.5, 0.25), (9.0, 9.0)])
def test_quotient_holds_across_magnitudes(n: float, d: float) -> None:
    layer = _quotient_layer()
    real = torch.tensor([[[n], [d]]])
    out_real, _ = layer(real, torch.zeros_like(real))
    assert out_real.item() == pytest.approx(n / d, rel=1e-5)


def test_the_floor_is_the_only_thing_between_it_and_a_zero_denominator() -> None:
    """A zero denominator is bounded by MODULUS_FLOOR, not by a clamp on the ratio."""
    layer = _quotient_layer()
    real = torch.tensor([[[1.0], [0.0]]])
    out_real, _ = layer(real, torch.zeros_like(real))
    assert torch.isfinite(out_real).all()
    assert out_real.item() > 1e3


def test_negative_exponents_survive_the_declared_clip() -> None:
    config = EulerNetConfig(num_layers=1, fields_per_layer=4, dropout=0.0, exp_clip=10.0)
    spec = EncoderSpec(
        bypass_dim=32, operand_dim=6, num_fields=8, embedding_dim=4,
        representation_dim=16, denominator_mask=tuple([False] * 6),
        full_feature_schema_sha256=None, operand_schema_sha256=None)
    block = EulerNetInteraction(spec, config)
    with torch.no_grad():
        block.layers[0].orders.fill_(-1.0)
    from dcdn.models.contracts import EncodedInteractionBatch
    batch = EncodedInteractionBatch(
        bypass_features=torch.rand(4, 32),
        operand_features=torch.rand(4, 6) + 0.5,
        field_embeddings=torch.rand(4, 8, 4) + 0.5,
    )
    out = block(batch)
    assert out.shape == (4, 2 * 4 * 4)
    assert torch.isfinite(out).all()


def test_zero_field_does_not_poison_the_backward_pass() -> None:
    """`atan2(0, 0)` returns 0 with a NaN gradient; the phase must be guarded.

    A numeric operand of exactly zero produces an exactly zero field embedding,
    which is 1.02% of cells on the VK-LSVD ur view. Before the guard, one such
    cell made every upstream gradient NaN on the first backward pass and the
    training harness rejected the arm as divergent.
    """
    layer = EulerInteractionLayer(
        in_fields=3, out_fields=2, embedding_dim=4, dropout=0.0, exp_clip=10.0)
    real = torch.rand(6, 3, 4, requires_grad=True)
    imaginary = torch.rand(6, 3, 4, requires_grad=True)
    with torch.no_grad():
        real[0, 0] = 0.0
        imaginary[0, 0] = 0.0
    out_real, out_imaginary = layer(real, imaginary)
    (out_real.square().sum() + out_imaginary.square().sum()).backward()
    assert torch.isfinite(real.grad).all()
    assert torch.isfinite(imaginary.grad).all()
    for name, parameter in layer.named_parameters():
        assert torch.isfinite(parameter.grad).all(), name
