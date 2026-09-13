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
"""Local MXFP8 expert computation, independent of routing and EP communication."""

from collections.abc import Callable

import torch  # pylint: disable=forbidden-backend-import
import torch.nn.functional as F  # pylint: disable=forbidden-backend-import

from hyper_parallel.components.quantization.functional.mxfp8_gmm_func import npu_quant_grouped_linear
from hyper_parallel.components.quantization.functional.mxfp8_swiglu_quant_func import mxfp8_swiglu_quant
from hyper_parallel.components.quantization.quantizers.mxfp8 import MXFP8Quantizer


def mxfp8_grouped_swiglu(
    inputs: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    counts: torch.Tensor,
    quantizer: MXFP8Quantizer,
    *,
    fused_swiglu_quant: bool = False,
    activation_func: Callable[[torch.Tensor], torch.Tensor] = F.silu,
) -> torch.Tensor:
    """Compute two local GMMs with standard [expert, out, in] weights.

    Args:
        inputs: Expert-major BF16/FP16 tokens, with no internal communication.
        gate_up_weight: Local packed Gate/Up weights.
        down_weight: Local Down weights.
        counts: Number of tokens for each local expert, including empty experts.
        quantizer: MXFP8 recipe for both projections.
        fused_swiglu_quant: Fuse activation/derivative with dual-axis quantization.
        activation_func: Gate activation used by the unfused reference path.

    Returns:
        High-precision expert-major outputs, without routing probability weights.
    """
    gate_up = npu_quant_grouped_linear(inputs, gate_up_weight, counts, quantizer, group_list_type=1)
    if fused_swiglu_quant and inputs.shape[0] != 0:
        intermediate = mxfp8_swiglu_quant(gate_up, quantizer, group_index=counts.cumsum(0))
    else:
        gate, up = gate_up.chunk(2, dim=-1)
        intermediate = activation_func(gate) * up
    return npu_quant_grouped_linear(intermediate, down_weight, counts, quantizer, group_list_type=1)
