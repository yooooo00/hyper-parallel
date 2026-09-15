# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""MXFP8 setup shared by high-performance MLP and expert modules."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from hyper_parallel.components.quantization.functional.npu_mxfp8 import (
    validate_npu_runtime,
    validate_npu_swiglu_runtime,
)
from hyper_parallel.components.quantization.quantizers.mxfp8 import MXFP8Quantizer


def make_mxfp8_quantizer(
    use_mxfp8: bool,
    fused_swiglu_quant: bool,
    context: Mapping[str, Any] | None,
) -> MXFP8Quantizer | None:
    """Resolve the optional local-compute path without another enable switch.

    Args:
        use_mxfp8: Enable MXFP8 matrix multiplication.
        fused_swiglu_quant: Fuse SwiGLU and MXFP8 quantization in both passes.
        context: Trainer mesh context; local compute currently excludes TP/CP/PP.

    Returns:
        A validated MXFP8 quantizer, or None for the original compute path.
    """
    if fused_swiglu_quant and not use_mxfp8:
        raise ValueError("fused_swiglu_quant=True requires use_mxfp8=True.")
    if not use_mxfp8:
        return None
    if any((context or {}).get(axis) for axis in ("tp", "cp", "pp")):
        raise NotImplementedError("MXFP8 high-performance modules require TP=CP=PP=1.")
    validate_npu_runtime()
    if fused_swiglu_quant:
        validate_npu_swiglu_runtime()
    return MXFP8Quantizer()
