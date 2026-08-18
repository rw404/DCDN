"""Pinned read-only source metadata for the clean-room model equations."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

_FUXICTR = {
    "repository": "https://github.com/reczoo/FuxiCTR.git",
    "commit": "0245987fdc1d43c36164ea496a02324885a8557a",
    "license": "Apache-2.0",
    "license_sha256": (
        "ed24b2447e7415e4baeaf4366d12db81e8a21224f3f1a69c08b8e30807b2ebaa"
    ),
}


_SOURCE_PROVENANCE: dict[str, dict[str, Any]] = {
    "dnn": {
        "implementation": "project clean-room baseline",
        "upstream_source": None,
        "note": "generic ReLU MLP block; architecture must be frozen in config",
    },
    "deepfm": {
        **_FUXICTR,
        "reference_path": "model_zoo/DeepFM/DeepFM_torch/src/DeepFM.py",
        "reference_sha256": (
            "326fc8158d68e6e8bd922050c81ee44dd165f96e073a6e64ade8b3f703725e32"
        ),
        "clean_room_basis": "linear + FM second order + deep branch equations",
    },
    "dcn": {
        **_FUXICTR,
        "reference_path": "model_zoo/DCN/DCN_torch/src/DCN.py",
        "reference_sha256": (
            "f6d4e6a937c5e4cac6055c7b988cd2ed1c7720facff52d4bc9ecaf9e87037371"
        ),
        "cross_layer_path": "fuxictr/pytorch/layers/interactions/cross_net.py",
        "cross_layer_sha256": (
            "bd1627cec81db448e7147b06e64e2457a58ee12e4dbec2a7c0c7310a1f7c07d3"
        ),
    },
    "autoint_plus": {
        **_FUXICTR,
        "reference_path": "model_zoo/AutoInt/src/AutoInt.py",
        "reference_sha256": (
            "fdde3cc7f0d0b482bdc5d17967cf48f28f5d8b85babe9d71e6d45960c998cd9e"
        ),
        "required_branches": ["multi_head_self_attention", "parallel_dnn"],
    },
    "dcnv2_full_rank": {
        **_FUXICTR,
        "reference_path": "model_zoo/DCNv2/src/DCNv2.py",
        "reference_sha256": (
            "78fc7d1c4f129dcbb750e410e53e075a646cacf5462ebd186ce0cf687418f4a9"
        ),
        "cross_layer_path": "fuxictr/pytorch/layers/interactions/cross_net.py",
        "cross_layer_sha256": (
            "bd1627cec81db448e7147b06e64e2457a58ee12e4dbec2a7c0c7310a1f7c07d3"
        ),
        "parameterization": "full_rank_dense_flat_C_plus_P",
    },
    "dcnv2_capacity_matched_core": {
        **_FUXICTR,
        "reference_path": "fuxictr/pytorch/layers/interactions/cross_net.py",
        "reference_sha256": (
            "bd1627cec81db448e7147b06e64e2457a58ee12e4dbec2a7c0c7310a1f7c07d3"
        ),
        "parameterization": "low_rank_mixture_of_experts_operand_core",
        "clean_room_shape_fix": "never use unconstrained squeeze; B=1 preserved",
    },
    "afn_plus": {
        **_FUXICTR,
        "reference_path": "model_zoo/AFN/src/AFN.py",
        "reference_sha256": (
            "cceba7bba9bd60f11f84a43f00f473fdf615dfdc1de46223d46c11d6252e28d2"
        ),
        "official_repository": "https://github.com/WeiyuCheng/AFN-AAAI-20.git",
        "official_commit": "25d7bd995bd0a3c71f562b3d332a0628c3b30d28",
        "official_license": "MIT",
        "official_license_sha256": (
            "b6a2bc1dc211e2286b0af8106deb43ff704d09c9e1131eb62851e76b10e7d160"
        ),
        "official_reference_path": "src/AFN.py",
        "official_reference_sha256": (
            "e6728d48a41ae683dd3b16b6a1c8cdfa75e15ee1be9ae1abb7fe97708d77979f"
        ),
        "controlled_loss_adaptation": (
            "one common task loss; official auxiliary branch losses omitted"
        ),
    },
    "wukong_small": {
        **_FUXICTR,
        "reference_path": "model_zoo/WuKong/src/WuKong.py",
        "reference_sha256": (
            "f0be1a2024532acc5ab03129abff5c8b289fb2554d21c58e2db9020845d268d2"
        ),
        "second_reference_repository": (
            "https://github.com/clabrugere/wukong-recommendation.git"
        ),
        "second_reference_commit": "8a0c14ccee50e02d40201d4dc665ee9fcebeb6b9",
        "second_reference_license": "MIT",
        "second_reference_license_sha256": (
            "fcb8346bda486d130ed8e1eed4d145d07277e0d40a9c7747a0e69878dc619d7d"
        ),
        "second_reference_path": "model/pytorch/wukong.py",
        "second_reference_sha256": (
            "56a05170af970f1bd238e264757abd8986c6a89c2a435665fc164231be17ff4e"
        ),
        "paper_record": "https://proceedings.mlr.press/v235/zhang24ao.html",
        "paper_pdf_origin": (
            "https://raw.githubusercontent.com/mlresearch/v235/main/assets/"
            "zhang24ao/zhang24ao.pdf"
        ),
        "paper_arxiv_identifier": "arXiv:2403.02545",
        "paper_local_pdf_sha256": (
            "8f4f539a174d3ff12653694d7e3d16f2c3b0c0651770a97445ac4826dcb5fabd"
        ),
        "configuration_policy": "no constructor scale defaults",
    },
    "dcdn_core": {
        "implementation": "src/dcdn/core.py",
        "implementation_sha256_at_audit": (
            "faeb8b3b8d7cb60eec7416e7b881b34963ef1f3664a9be454e9562a4603468c6"
        ),
        "reference": "DCDNCore_reference.py",
        "reference_sha256": (
            "6b2cdb5db0c248fcd960c42d4ce14873820cc65d382741721214166d5e83d9ac"
        ),
        "production_hybrid_provenance": "DCDNBlock_provided.py",
        "production_hybrid_sha256": (
            "d58c31c03096d112a47718a643fc91bc5dd9b5a53cd7941948145b34335ec1bb"
        ),
        "architecture_boundary": "public core only; production hybrid excluded",
    },
}


def source_provenance(model_key: str) -> dict[str, Any]:
    if model_key not in _SOURCE_PROVENANCE:
        raise KeyError(f"unknown model provenance key: {model_key}")
    return deepcopy(_SOURCE_PROVENANCE[model_key])
