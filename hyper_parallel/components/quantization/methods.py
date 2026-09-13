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
"""Instance-local MXFP8 compute injection without replacing modules or parameters."""

from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatchcase
from typing import Any

import torch  # pylint: disable=forbidden-backend-import
from torch import nn  # pylint: disable=forbidden-backend-import
from transformers.activations import SiLUActivation

from hyper_parallel.components.quantization.config import LowPrecisionConfig
from hyper_parallel.components.quantization.functional.mxfp8_grouped_swiglu_func import mxfp8_grouped_swiglu
from hyper_parallel.components.quantization.functional.mxfp8_linear_func import mxfp8_linear
from hyper_parallel.components.quantization.functional.mxfp8_swiglu_quant_func import mxfp8_swiglu_quant
from hyper_parallel.components.quantization.functional.npu_mxfp8 import validate_npu_gmm_runtime, validate_npu_runtime
from hyper_parallel.components.quantization.quantizers.mxfp8 import MXFP8Quantizer
from hyper_parallel.distributed._builder.forward_rewriter import _install_bound_forward


@dataclass
class _MethodState:
    quantizer: MXFP8Quantizer
    fused: bool
    transposed: bool = False
    packed_mlp: bool = False


def _linear_forward(module: nn.Module, input: torch.Tensor) -> torch.Tensor:  # pylint: disable=redefined-builtin
    output = mxfp8_linear(input, module.weight, module._hp_low_precision_methods.quantizer)
    return output if module.bias is None else output + module.bias


def _mlp_forward(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    state = module._hp_low_precision_methods
    if state.packed_mlp:
        weight = module.linear_fc1.weight
        down = module.linear_fc2
    else:
        # Keep HF Parameter identities and checkpoint keys: this is a temporary
        # differentiable concatenation, not a persistent packed weight/cache.
        weight = torch.cat((module.gate_proj.weight, module.up_proj.weight), dim=0)
        down = module.down_proj
    gate_up = mxfp8_linear(x, weight, state.quantizer)
    if state.fused:
        intermediate = mxfp8_swiglu_quant(gate_up, state.quantizer)
    else:
        gate, up = gate_up.chunk(2, dim=-1)
        intermediate = torch.nn.functional.silu(gate) * up
    output = mxfp8_linear(intermediate, down.weight, state.quantizer)
    return output if down.bias is None else output + down.bias


def _expert_major(
    module: nn.Module,
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
    scores: torch.Tensor | None = None,
) -> torch.Tensor:
    del scores  # Both supported callers apply routing probabilities after GMM2.
    state = module._hp_low_precision_methods
    gate_up, down = module.gate_up_proj, module.down_proj
    if state.transposed:
        gate_up, down = gate_up.transpose(-2, -1), down.transpose(-2, -1)
    return mxfp8_grouped_swiglu(
        x, gate_up, down, num_tokens_per_expert.to(x.device), state.quantizer,
        fused_swiglu_quant=state.fused,
    )


def _experts_forward(
    module: nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    hidden_shape = hidden_states.shape
    flat = hidden_states.reshape(-1, hidden_shape[-1])
    indices = top_k_index.reshape(-1)
    order = indices.argsort()
    top_k = top_k_index.shape[-1]
    token_indices = torch.arange(flat.shape[0], device=flat.device).repeat_interleave(top_k)
    counts = torch.bincount(indices, minlength=module.num_experts)
    output = module.forward_expert_major(flat[token_indices[order]], counts)
    # HF casts each routed contribution back before accumulating into hidden states.
    output = (output * top_k_weights.reshape(-1)[order].unsqueeze(-1)).to(hidden_states.dtype)
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(order.numel(), device=order.device)
    return output[inverse].reshape(flat.shape[0], top_k, hidden_shape[-1]).sum(1).reshape(hidden_shape)


def _check_silu(module: nn.Module) -> None:
    activation = getattr(module, "act_fn", None)
    config = getattr(module, "config", None)
    hidden_act = getattr(config, "hidden_act", None) or getattr(config, "hidden_activation", "silu")
    if hidden_act not in ("silu", "swiglu"):
        raise ValueError("MXFP8 method injection requires SwiGLU.")
    if activation is not None and not isinstance(activation, (nn.SiLU, SiLUActivation)):
        raise ValueError("MXFP8 method injection requires the standard SiLU activation.")


def _validate_target(module: nn.Module, kind: str) -> None:
    if hasattr(module, "_hp_low_precision_methods") or hasattr(module, "quantizer"):
        raise ValueError("Choose either a quantized replacement or method injection for a module, not both.")
    if kind == "linear":
        if type(module) is not nn.Linear:  # pylint: disable=unidiomatic-typecheck
            raise TypeError("linear targets must be ordinary nn.Linear modules.")
        weights = [module.weight]
    elif kind == "mlp":
        _check_silu(module)
        names = ("linear_fc1", "linear_fc2") if hasattr(module, "linear_fc1") else (
            "gate_proj", "up_proj", "down_proj")
        projections = [getattr(module, name, None) for name in names]
        if not all(type(projection) is nn.Linear for projection in projections):  # pylint: disable=unidiomatic-typecheck
            raise TypeError("mlp targets require ordinary packed or separate Gate/Up/Down Linear children.")
        if any(projection.bias is not None for projection in projections[:-1]):
            raise ValueError("Fused MXFP8 MLP requires bias-free Gate/Up projections.")
        if any(projection._forward_hooks or projection._forward_pre_hooks for projection in projections):
            raise ValueError("MLP method injection cannot bypass child Linear forward hooks.")
        weights = [projection.weight for projection in projections]
    else:
        _check_silu(module)
        weights = [getattr(module, name, None) for name in ("gate_up_proj", "down_proj")]
        if not all(isinstance(weight, nn.Parameter) and weight.ndim == 3 for weight in weights):
            raise TypeError("experts targets require packed 3D Gate/Up and Down parameters.")
        if any(getattr(module, flag, False) for flag in ("has_bias", "add_bias")):
            raise ValueError("MXFP8 expert methods require bias-free weights.")
        if not getattr(module, "router_gating_in_fp32", True):
            raise ValueError("MXFP8 expert methods require routing probabilities after GMM2.")
    if any(dimension % 32 for weight in weights for dimension in weight.shape[-2:]):
        raise ValueError("MXFP8 method targets require matrix dimensions divisible by 32.")


def apply_low_precision_methods(
    model: nn.Module,
    config: LowPrecisionConfig | None,
    *,
    context: dict[str, Any] | None = None,
) -> nn.Module:
    """Bind low-precision compute before sharding/FSDP, preserving model state.

    Args:
        model: Model after optional structural replacements, before sharding.
        config: Explicit linear/mlp/experts FQN-glob targets and MXFP8 recipe.
        context: Trainer topology flags; this path requires TP=CP=PP=1.

    Returns:
        The same model, with identical modules, parameters and checkpoint keys.
        No process-global operator or class methods are modified.
    """
    if config is None or not config.enabled or not config.targets:
        return model
    if any((context or {}).get(axis) for axis in ("tp", "cp", "pp")):
        raise NotImplementedError("Low-precision method injection requires TP=CP=PP=1.")
    selected = {}
    aliases = list(model.named_modules(remove_duplicate=False))
    for kind, patterns in config.targets.items():
        for pattern in patterns:
            matches = [(name, module) for name, module in aliases if fnmatchcase(name, pattern)]
            if not matches:
                raise ValueError(f"low_precision.targets pattern matched no module: {pattern!r}")
            for name, module in matches:
                previous = selected.get(id(module))
                if previous is not None and previous[2] != kind:
                    raise ValueError(f"Conflicting low-precision target kinds for {name!r}")
                selected[id(module)] = (name, module, kind)
    for _, module, kind in selected.values():
        if any(id(child) in selected for child in module.modules() if child is not module):
            raise ValueError("Choose an MLP target or its Linear children, not both.")
        _validate_target(module, kind)
    validate_npu_runtime()
    if any(kind == "experts" for _, _, kind in selected.values()):
        validate_npu_gmm_runtime()
    for _, module, kind in selected.values():
        module._hp_low_precision_methods = _MethodState(
            MXFP8Quantizer(), config.fused_swiglu_quant,
            transposed=bool(getattr(module, "is_transposed", False)),
            packed_mlp=hasattr(module, "linear_fc1"),
        )
        if kind == "experts":
            has_expert_entry = callable(getattr(module, "forward_expert_major", None))
            _install_bound_forward(module, _expert_major, method_name="forward_expert_major")
            if not has_expert_entry:
                module.num_experts = module.gate_up_proj.shape[0]
                _install_bound_forward(module, _experts_forward)
            module.requires_grouped_expert_compute = True
        else:
            _install_bound_forward(module, _linear_forward if kind == "linear" else _mlp_forward)
    return model
