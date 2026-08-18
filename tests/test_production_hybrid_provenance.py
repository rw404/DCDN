from __future__ import annotations

import hashlib
import sys
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from torch import nn

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPOSITORY_ROOT / "src"
for import_root in (REPOSITORY_ROOT, SRC_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from dcdn.hybrid import (  # noqa: E402
    PROVIDED_BLOCK_FILENAME,
    PROVIDED_BLOCK_SHA256,
    ExperimentalProductionHybridAdapter,
    ExperimentalProductionHybridOneBlock,
    production_hybrid_parameter_formula,
    validate_provided_block_structure,
)
from DCDNBlock_provided import DCDNBlock  # noqa: E402

REFERENCE_CORE_SHA256 = (
    "6b2cdb5db0c248fcd960c42d4ce14873820cc65d382741721214166d5e83d9ac"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_user_supplied_provenance_hashes_are_unchanged():
    assert PROVIDED_BLOCK_FILENAME == "DCDNBlock_provided.py"
    assert _sha256(REPOSITORY_ROOT / PROVIDED_BLOCK_FILENAME) == PROVIDED_BLOCK_SHA256
    assert _sha256(REPOSITORY_ROOT / "DCDNCore_reference.py") == REFERENCE_CORE_SHA256


@pytest.mark.parametrize("features_count", [1, 3, 17])
def test_production_hybrid_topology_and_parameter_formula(features_count):
    block = DCDNBlock(features_count=features_count)
    validate_provided_block_structure(block, features_count)
    actual = sum(parameter.numel() for parameter in block.parameters())
    expected = production_hybrid_parameter_formula(features_count)
    assert actual == expected == 3 * features_count**2 + 67 * features_count + 48
    assert isinstance(block.block, nn.Linear) and block.block.bias is not None
    assert isinstance(block.div_block, nn.Linear) and block.div_block.bias is not None
    assert isinstance(block.fc_block[0], nn.Linear)
    assert block.fc_block[0].out_features == 32
    assert isinstance(block.fc_block[1], nn.LeakyReLU)
    assert isinstance(block.fc_block[2], nn.BatchNorm1d)
    assert block.fc_block[2].num_features == 8
    assert isinstance(block.fc_block[3], nn.Dropout)
    assert block.fc_block[3].p == pytest.approx(0.1)
    assert block.projector.in_features == features_count + 32
    assert block.projector.out_features == features_count


def test_production_batchnorm_buffers_are_persistent_and_track_batches():
    block = DCDNBlock(features_count=4)
    batch_norm = block.fc_block[2]
    state = block.state_dict()
    assert "fc_block.2.running_mean" in state
    assert "fc_block.2.running_var" in state
    assert "fc_block.2.num_batches_tracked" in state
    assert batch_norm.running_mean.shape == (8,)
    assert batch_norm.running_var.shape == (8,)
    assert batch_norm.num_batches_tracked.item() == 0

    block.train()
    x_0 = torch.randn(6, 8, 4)
    x_0[..., 0] += 0.5
    block(x_0, x_0)
    assert batch_norm.num_batches_tracked.item() == 1
    assert not torch.equal(
        batch_norm.running_mean, torch.zeros_like(batch_norm.running_mean)
    )


def test_training_forward_reproduces_after_bn_state_and_dropout_rng_restore():
    torch.manual_seed(77)
    block = DCDNBlock(features_count=3).train()
    x_0 = torch.randn(5, 8, 3)
    initial_state = deepcopy(block.state_dict())
    initial_rng = torch.random.get_rng_state().clone()
    first = block(x_0, x_0)
    first_state = deepcopy(block.state_dict())

    block.load_state_dict(initial_state)
    torch.random.set_rng_state(initial_rng)
    second = block(x_0, x_0)
    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
    for key, value in first_state.items():
        torch.testing.assert_close(block.state_dict()[key], value)


def test_experimental_adapter_matches_immutable_source_in_fp32_eval():
    torch.manual_seed(11)
    block = DCDNBlock(features_count=4, div_eps=1e-2).eval()
    adapter = ExperimentalProductionHybridAdapter(block, features_count=4).eval()
    x_0 = torch.tensor(
        [
            [
                [0.001, -0.004, 2.0, -3.0],
                [0.02, 0.5, -1.0, 4.0],
                [1.0, 2.0, 3.0, 4.0],
                [-1.0, -2.0, -3.0, -4.0],
                [0.0, 0.01, -0.01, 10.0],
                [3.0, 2.0, 1.0, 0.5],
                [0.3, 0.4, 0.5, 0.6],
                [-0.3, -0.4, -0.5, -0.6],
            ]
        ],
        dtype=torch.float32,
    )
    x_i = 0.7 * x_0 + 0.2
    with torch.no_grad():
        source_output = block(x_0, x_i)
        adapter_output = adapter(x_0, x_i)
    torch.testing.assert_close(adapter_output, source_output, rtol=0.0, atol=0.0)
    assert adapter_output.shape == x_i.shape
    assert adapter_output.dtype == x_i.dtype
    assert adapter_output.device == x_i.device
    assert torch.isfinite(adapter_output).all()


def test_minimal_wrapper_uses_exactly_one_block_and_x_i_equals_x_0():
    wrapper = ExperimentalProductionHybridOneBlock.from_provided_source(
        features_count=3
    ).eval()
    provided_blocks = [
        module for module in wrapper.modules() if isinstance(module, DCDNBlock)
    ]
    assert len(provided_blocks) == 1

    observed = []

    def capture_inputs(_module, inputs):
        observed.append(inputs)

    hook = wrapper.adapter.register_forward_pre_hook(capture_inputs)
    x_0 = torch.rand(2, 8, 3) + 0.1
    with torch.no_grad():
        output = wrapper(x_0)
    hook.remove()
    assert len(observed) == 1
    assert observed[0][0] is x_0
    assert observed[0][1] is x_0
    assert output.shape == x_0.shape
    metadata = wrapper.provenance_metadata()
    assert metadata["depth"] == 1
    assert metadata["initial_state_contract"] == "x_i=x_0"
    assert (
        "end-to-end production wrapper remains incomplete" in metadata["wrapper_scope"]
    )


def test_adapter_rejects_broadcastable_shapes_dtype_nonfinite_and_bad_eps():
    block = DCDNBlock(features_count=3)
    adapter = ExperimentalProductionHybridAdapter(block, features_count=3).eval()
    x_0 = torch.ones(2, 8, 3)
    with pytest.raises(ValueError, match=r"\[B, 8, 3\]"):
        adapter(torch.ones(2, 1, 3), torch.ones(2, 1, 3))
    with pytest.raises(ValueError, match="identical"):
        adapter(x_0, torch.ones(1, 8, 3))
    with pytest.raises(TypeError, match="identical dtypes"):
        adapter(x_0, x_0.to(torch.bfloat16))
    with pytest.raises(TypeError, match="FP16"):
        adapter(x_0.half(), x_0.half())
    with pytest.raises(TypeError, match="FP32 or BF16"):
        adapter(x_0.double(), x_0.double())
    nonfinite = x_0.clone()
    nonfinite[1, 3, 2] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        adapter(nonfinite, nonfinite)

    block.div_eps = 0.0
    with pytest.raises(ValueError, match="positive"):
        adapter(x_0, x_0)
    with pytest.raises(ValueError, match="positive"):
        ExperimentalProductionHybridAdapter(
            DCDNBlock(features_count=3, div_eps=-1.0), features_count=3
        )


def test_adapter_rejects_mutated_production_topology():
    wrong_bn = DCDNBlock(features_count=3)
    wrong_bn.fc_block[2] = nn.BatchNorm1d(7)
    with pytest.raises(ValueError, match=r"BatchNorm1d\(8\)"):
        ExperimentalProductionHybridAdapter(wrong_bn, features_count=3)

    wrong_dropout = DCDNBlock(features_count=3)
    wrong_dropout.fc_block[3] = nn.Dropout(0.2)
    with pytest.raises(ValueError, match=r"Dropout\(0.1\)"):
        ExperimentalProductionHybridAdapter(wrong_dropout, features_count=3)


def test_adapter_bfloat16_path_is_finite_when_platform_supports_it():
    adapter = (
        ExperimentalProductionHybridAdapter.from_provided_source(
            features_count=3, div_eps=1e-2
        )
        .eval()
        .to(dtype=torch.bfloat16)
    )
    x_0 = torch.tensor(
        [
            [[0.001, -0.001, 2.0]] * 8,
            [[3.0, 4.0, 5.0]] * 8,
        ],
        dtype=torch.bfloat16,
    )
    try:
        with torch.no_grad():
            output = adapter(x_0, x_0)
    except RuntimeError as exc:  # pragma: no cover - platform capability gate.
        pytest.skip(f"BF16 hybrid execution unavailable: {exc}")
    assert output.dtype == torch.bfloat16
    assert output.shape == x_0.shape
    assert torch.isfinite(output).all()


def test_adapter_provenance_explicitly_separates_source_and_guard_delta():
    adapter = ExperimentalProductionHybridAdapter.from_provided_source(features_count=2)
    metadata = adapter.provenance_metadata()
    assert metadata["architecture"] == "DCDN-Hybrid-Prod-L1-PLE8"
    assert metadata["source_sha256"] == PROVIDED_BLOCK_SHA256
    assert metadata["source_unchanged"] is True
    assert metadata["semantic_denominator_mask"] is False
    assert metadata["adapter_reciprocal_compute_dtype"] == "float32"
    assert "adapter adds validation" in metadata["source_guard_delta"]
    assert not hasattr(adapter, "denominator_mask")
