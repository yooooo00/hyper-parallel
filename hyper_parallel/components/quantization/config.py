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
"""Typed configuration for NPU low-precision model conversion."""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class LowPrecisionConfig:
    """Configure build-time NPU low-precision conversion.

    Only format/scaling pairs backed by a complete Dense implementation are
    accepted.
    """

    enabled: bool = False
    format: Literal["mxfp8_e4m3", "hif8"] = "mxfp8_e4m3"
    scaling: Literal["mx_block", "current"] = "mx_block"
    # Method targets preserve module and Parameter identities; an empty mapping
    # leaves the existing explicit replacement workflow unchanged.
    targets: dict[str, list[str]] = field(default_factory=dict)
    fused_swiglu_quant: bool = False

    def __post_init__(self) -> None:
        """Validate the supported format/scaling combinations."""

        if self.targets and self.format != "mxfp8_e4m3":
            raise NotImplementedError("Low-precision method targets currently support MXFP8 only.")
        unknown = set(self.targets) - {"linear", "mlp", "experts"}
        if unknown:
            raise ValueError(f"Unknown low_precision.targets kinds: {sorted(unknown)}")
        if self.fused_swiglu_quant and self.targets and not any(self.targets.get(k) for k in ("mlp", "experts")):
            raise ValueError("SwiGLU fusion needs an mlp or experts target, not standalone Linear targets.")

        if not isinstance(self.enabled, bool):
            raise ValueError(
                "LowPrecisionConfig.enabled must be a bool, "
                f"but got {type(self.enabled).__name__}."
            )
        supported = {
            ("mxfp8_e4m3", "mx_block"),
            ("hif8", "current"),
        }
        if (self.format, self.scaling) not in supported:
            raise ValueError(
                "Unsupported low-precision format/scaling combination "
                f"{self.format!r}/{self.scaling!r}; expected one of "
                "mxfp8_e4m3/mx_block or hif8/current."
            )
