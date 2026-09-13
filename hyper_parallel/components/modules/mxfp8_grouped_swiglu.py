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
"""High-performance experts with MXFP8 local compute and the shared EP entry."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch  # pylint: disable=forbidden-backend-import
from torch import nn  # pylint: disable=forbidden-backend-import
from transformers.activations import SiLUActivation

from hyper_parallel.components.modules.grouped_experts import GroupedExperts
from hyper_parallel.components.quantization.functional.mxfp8_grouped_swiglu_func import mxfp8_grouped_swiglu
from hyper_parallel.components.quantization.functional.npu_mxfp8 import validate_npu_gmm_runtime
from hyper_parallel.components.quantization.quantizers.mxfp8 import MXFP8Quantizer
from hyper_parallel.models.replacement import module_replacement


@module_replacement
class MXFP8GroupedSwiGLU(GroupedExperts):
    """Reuse GroupedExperts routing and checkpoint conversion; replace local math.

    Select this factory instead of a second GroupedExperts replacement. EP must
    use its grouped compute entry; TP/CP/PP and biased/ungated experts are not supported.
    """

    def __init__(
        self,
        *,
        module: nn.Module,
        module_fqn: str = "",
        context: Mapping[str, Any] | None = None,
        fused_swiglu_quant: bool = False,
    ) -> None:
        """Build MXFP8 local experts, optionally fusing SwiGLU and quantization.

        Args:
            module: Source HF packed expert container.
            module_fqn: Source name supplied by replacement.
            context: Trainer mesh context; EP is supported through grouped compute.
            fused_swiglu_quant: Enable forward and backward MXFP8 fusion.
        """
        context = context or {}
        if any(context.get(axis) for axis in ("tp", "cp", "pp")):
            raise NotImplementedError("MXFP8GroupedSwiGLU requires TP=CP=PP=1.")
        super().__init__(module=module, module_fqn=module_fqn, context=context)
        hidden_act = getattr(self.config, "hidden_act", None) or getattr(self.config, "hidden_activation", None)
        hidden_act = hidden_act or getattr(self.config, "mlp_hidden_act", "silu")
        if not self.has_gate or self.add_bias or hidden_act not in ("silu", "swiglu"):
            raise ValueError("MXFP8GroupedSwiGLU requires bias-free SwiGLU experts.")
        source_activation = getattr(module, "act_fn", None)
        if source_activation is not None and not isinstance(source_activation, (nn.SiLU, SiLUActivation)):
            raise ValueError("MXFP8GroupedSwiGLU requires the standard SiLU gate activation.")
        if self.use_2d_experts or not self.router_gating_in_fp32:
            raise ValueError("MXFP8GroupedSwiGLU requires 3D weights and routing weights after GMM2.")
        if self.hidden_size % 32 or self.intermediate_size % 32:
            raise ValueError("MXFP8GroupedSwiGLU requires hidden/intermediate sizes divisible by 32.")
        validate_npu_gmm_runtime()
        self.quantizer = MXFP8Quantizer()
        self.fused_swiglu_quant = fused_swiglu_quant
        self.requires_grouped_expert_compute = True

    def _grouped_gemm_expert_forward(
        self,
        gate_up_proj: torch.Tensor,
        down_proj: torch.Tensor,
        permuted: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        permuted_probs: torch.Tensor | None,
    ) -> torch.Tensor:
        """Consume local tokens; inherited callers own permutation and probability weighting."""
        del permuted_probs
        return mxfp8_grouped_swiglu(
            permuted, gate_up_proj.transpose(-2, -1), down_proj.transpose(-2, -1),
            tokens_per_expert.to(device=permuted.device), self.quantizer,
            fused_swiglu_quant=self.fused_swiglu_quant,
        )
