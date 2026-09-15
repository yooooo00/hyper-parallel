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
"""Dense and grouped SwiGLU with fused MXFP8 forward and backward quantization."""

from typing import Optional

import torch  # pylint: disable=forbidden-backend-import
from torch.autograd.function import once_differentiable  # pylint: disable=forbidden-backend-import

from hyper_parallel.components.quantization.quantizers.mxfp8 import MXFP8Quantizer
from hyper_parallel.components.quantization.tensor import MXFP8Tensor


class _MXFP8SwiGLUQuantFunction(torch.autograd.Function):
    """Pass prequantized activations and gradients between adjacent MXFP8 Linears."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        inputs: torch.Tensor,
        quantizer: MXFP8Quantizer,
        group_index: Optional[torch.Tensor],
    ) -> MXFP8Tensor:
        """Keep the logical leading dimensions while quantizing a 2D matrix.

        Args:
            ctx: Autograd context retaining the original activation.
            inputs: Packed high-precision Gate/Up activation.
            quantizer: MXFP8 recipe and operator adapter.
            group_index: Cumulative expert token boundaries, or None for Dense.
        """
        matrix = inputs.reshape(-1, inputs.shape[-1])
        row_data, row_scale, col_data, col_scale = quantizer.npu_ops.swiglu_mx_quant_dual_axis(
            matrix, quant_dtype=quantizer.quant_dtype, group_index=group_index,
        )
        if ctx.needs_input_grad[0]:
            ctx.save_for_backward(matrix, group_index)
            ctx.input_shape = inputs.shape
            ctx.quantizer = quantizer
        return MXFP8Tensor(
            shape=(*inputs.shape[:-1], inputs.shape[-1] // 2), dtype=inputs.dtype,
            quantizer=quantizer, row_data=row_data, row_scale=row_scale,
            col_data=col_data, col_scale=col_scale,
        )

    @staticmethod
    @once_differentiable
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[MXFP8Tensor, None, None]:
        """Return an MXFP8 gradient directly to the preceding bias-free Linear.

        Args:
            ctx: Context saved by forward.
            grad_output: High-precision gradient produced by FC2 dgrad.
        """
        matrix, group_index = ctx.saved_tensors
        quantizer = ctx.quantizer
        row_data, row_scale, col_data, col_scale = quantizer.npu_ops.swiglu_backward_mx_quant_dual_axis(
            matrix, grad_output.reshape(-1, grad_output.shape[-1]).contiguous(),
            quant_dtype=quantizer.quant_dtype, group_index=group_index,
        )
        return MXFP8Tensor(
            shape=ctx.input_shape, dtype=matrix.dtype,
            quantizer=quantizer, row_data=row_data, row_scale=row_scale,
            col_data=col_data, col_scale=col_scale,
        ), None, None


def mxfp8_swiglu_quant(
    inputs: torch.Tensor,
    quantizer: MXFP8Quantizer,
    *,
    group_index: Optional[torch.Tensor] = None,
) -> MXFP8Tensor:
    """Fuse last-axis SwiGLU and MX quantization between adjacent projections.

    Args:
        inputs: BF16/FP16 packed [gate | up] projection, with arbitrary leading
            dimensions. Its sole gradient consumer must be an MXFP8 Linear/GMM;
            ordinary bias addition or branching cannot consume the MX gradient.
        quantizer: MXFP8 quantizer used by the adjacent Linear operations.
        group_index: Cumulative expert token boundaries for expert-major 2D
            inputs. Column quantization restarts at each expert boundary.
            Dense callers leave this as None. The consumer GMM must use the
            same grouping; a prequantized carrier does not encode group metadata.

    Returns:
        A logical BF16/FP16 carrier with 2D E4M3 payloads and packed E8M0 scales.
        Its backward also returns a prequantized carrier to the first Linear.
    """
    return _MXFP8SwiGLUQuantFunction.apply(inputs, quantizer, group_index)
