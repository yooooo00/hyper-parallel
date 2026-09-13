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
"""HF-compatible packed Dense MLP with opt-in SwiGLU + MXFP8 quantization."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch  # pylint: disable=forbidden-backend-import
from torch import nn  # pylint: disable=forbidden-backend-import

from hyper_parallel.models.replacement import module_replacement
from hyper_parallel.components.quantization.functional.mxfp8_swiglu_quant_func import (
    mxfp8_swiglu_quant,
)
from hyper_parallel.components.quantization.modules.mxfp8_linear import (
    replace_mxfp8_linear,
)
from hyper_parallel.components.modules.swiglu_mlp import SwiGLUMLP


@module_replacement
class MXFP8SwiGLUMLP(SwiGLUMLP):
    """Pack HF Gate/Up weights once and connect two MXFP8 Dense projections.

    Select this MLP-level replacement instead of selecting its child Linears.
    The inherited weight transforms preserve HF checkpoint import/export.
    """

    def __init__(
        self,
        *,
        module: nn.Module,
        module_fqn: str = "",
        context: Mapping[str, Any] | None = None,
        fused_swiglu_quant: bool = True,
    ) -> None:
        """Build a bias-free Gate/Up MLP with an optional fused activation.

        Args:
            module: HF MLP with gate_proj, up_proj, and down_proj.
            module_fqn: Source MLP name supplied by replacement.
            context: Trainer replacement context.
            fused_swiglu_quant: Fuse SwiGLU and quantization in both passes.
                False uses the same packed weights with separate activation
                and quantization calls.
        """
        if module.gate_proj.bias is not None or module.up_proj.bias is not None:
            raise ValueError("SwiGLU MX quant fusion requires bias-free Gate/Up projections.")
        super().__init__(module=module, module_fqn=module_fqn, context=context)
        self.fused_swiglu_quant = fused_swiglu_quant
        for name in ("linear_fc1", "linear_fc2"):
            setattr(self, name, replace_mxfp8_linear(
                module=getattr(self, name), module_fqn=f"{module_fqn}.{name}", context=context or {},
            ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run FC1, fused SwiGLU + quant, and FC2.

        Args:
            x: BF16/FP16 hidden states.

        Returns:
            MLP output with the input's logical dtype and leading dimensions.
        """
        if not self.fused_swiglu_quant:
            return super().forward(x)
        intermediate = self.linear_fc1(x)
        intermediate = mxfp8_swiglu_quant(intermediate, self.linear_fc2.quantizer)
        return self.linear_fc2(intermediate)
