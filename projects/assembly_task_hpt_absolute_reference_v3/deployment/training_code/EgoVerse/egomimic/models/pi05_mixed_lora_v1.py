"""Dependency-free LoRA injection for EgoVerse pi0.5 mixed training.

The base OpenPI parameters keep their original names.  Low-rank parameters
are attached directly to selected ``nn.Linear`` modules and applied through a
forward hook, so an ordinary Lightning checkpoint remains loadable without a
PEFT dependency.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class Pi05MixedLoraConfig:
    rank: int = 16
    alpha: float = 32.0
    dropout: float = 0.05
    vision_last_n_blocks: int = 2
    train_multimodal_projector: bool = True
    train_action_projections: bool = True
    train_camera_slot_embeddings: bool = True

    def validate(self) -> None:
        if self.rank <= 0:
            raise ValueError("LoRA rank must be positive")
        if self.alpha <= 0:
            raise ValueError("LoRA alpha must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("LoRA dropout must be in [0, 1)")
        if self.vision_last_n_blocks < 0:
            raise ValueError("vision_last_n_blocks must be non-negative")


@dataclass(frozen=True)
class Pi05MixedLoraSummary:
    total_parameters: int
    trainable_parameters: int
    adapter_parameters: int
    fully_trainable_parameters: int
    injected_linear_modules: int
    language_modules: int
    action_expert_modules: int
    vision_modules: int
    trainable_module_names: tuple[str, ...]

    @property
    def trainable_fraction(self) -> float:
        return self.trainable_parameters / max(self.total_parameters, 1)

    def to_dict(self) -> dict:
        result = asdict(self)
        result["trainable_fraction"] = self.trainable_fraction
        return result


_TRANSFORMER_TARGETS = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
}
_VISION_TARGETS = {
    "q_proj",
    "k_proj",
    "v_proj",
    "out_proj",
    "fc1",
    "fc2",
}
_VISION_LAYER_RE = re.compile(
    r"\.paligemma(?:\.model)?\.vision_tower\.vision_model\.encoder\.layers\.(\d+)\."
)
_LANGUAGE_LAYER_RE = re.compile(
    r"\.paligemma(?:\.model)?\.language_model\.layers\."
)
_MULTIMODAL_PROJECTOR_RE = re.compile(
    r"\.paligemma(?:\.model)?\.multi_modal_projector\."
)


def _vision_depth(policy: nn.Module) -> int:
    indices: list[int] = []
    for name, _ in policy.named_modules():
        match = _VISION_LAYER_RE.search(f".{name}.")
        if match is not None:
            indices.append(int(match.group(1)))
    return max(indices) + 1 if indices else 0


def _category(name: str, *, vision_first_block: int) -> str | None:
    leaf = name.rsplit(".", 1)[-1]
    padded = f".{name}."
    if ".gemma_expert.model.layers." in padded and leaf in _TRANSFORMER_TARGETS:
        return "action_expert"
    if _LANGUAGE_LAYER_RE.search(padded) is not None and leaf in _TRANSFORMER_TARGETS:
        return "language"
    match = _VISION_LAYER_RE.search(padded)
    if (
        match is not None
        and int(match.group(1)) >= vision_first_block
        and leaf in _VISION_TARGETS
    ):
        return "vision"
    return None


def _attach_lora(module: nn.Linear, config: Pi05MixedLoraConfig) -> int:
    if hasattr(module, "egoverse_lora_a") or hasattr(module, "egoverse_lora_b"):
        raise RuntimeError("LoRA was already injected into this module")
    dtype = module.weight.dtype
    device = module.weight.device
    a = nn.Parameter(
        torch.empty(config.rank, module.in_features, dtype=dtype, device=device)
    )
    b = nn.Parameter(
        torch.zeros(module.out_features, config.rank, dtype=dtype, device=device)
    )
    nn.init.kaiming_uniform_(a, a=math.sqrt(5))
    module.register_parameter("egoverse_lora_a", a)
    module.register_parameter("egoverse_lora_b", b)
    scale = config.alpha / config.rank

    def add_low_rank_update(
        linear: nn.Linear,
        inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> torch.Tensor:
        value = F.dropout(inputs[0], p=config.dropout, training=linear.training)
        update = F.linear(
            F.linear(value, linear.egoverse_lora_a), linear.egoverse_lora_b
        )
        return output + update * scale

    module.register_forward_hook(add_low_rank_update)
    return a.numel() + b.numel()


def configure_pi05_mixed_lora(
    policy: nn.Module, config: Pi05MixedLoraConfig
) -> Pi05MixedLoraSummary:
    """Freeze the base and enable audited LoRA plus small interface modules."""

    config.validate()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)

    vision_depth = _vision_depth(policy)
    if config.vision_last_n_blocks and vision_depth == 0:
        raise RuntimeError("Could not find SigLIP transformer blocks")
    vision_first_block = max(vision_depth - config.vision_last_n_blocks, 0)

    counts = {"language": 0, "action_expert": 0, "vision": 0}
    adapter_parameters = 0
    injected_names: list[str] = []
    for name, module in policy.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        category = _category(name, vision_first_block=vision_first_block)
        if category is None:
            continue
        adapter_parameters += _attach_lora(module, config)
        counts[category] += 1
        injected_names.append(name)

    if counts["language"] == 0 or counts["action_expert"] == 0:
        raise RuntimeError(
            "LoRA target audit failed: language and action-expert targets are required"
        )
    if config.vision_last_n_blocks and counts["vision"] == 0:
        raise RuntimeError("LoRA target audit failed: no SigLIP target was injected")

    fully_trainable_names: list[str] = []
    fully_trainable_parameters = 0
    for name, parameter in policy.named_parameters():
        if "egoverse_lora_" in name:
            continue
        train_fully = False
        if config.train_action_projections and any(
            token in name
            for token in (
                "action_in_proj",
                "action_out_proj",
                "time_mlp_in",
                "time_mlp_out",
            )
        ):
            train_fully = True
        if (
            config.train_multimodal_projector
            and _MULTIMODAL_PROJECTOR_RE.search(f".{name}.") is not None
        ):
            train_fully = True
        if config.train_camera_slot_embeddings and name.endswith(
            "egoverse_camera_slot_embeddings"
        ):
            train_fully = True
        if train_fully:
            parameter.requires_grad_(True)
            fully_trainable_names.append(name)
            fully_trainable_parameters += parameter.numel()

    total = sum(parameter.numel() for parameter in policy.parameters())
    trainable = sum(
        parameter.numel() for parameter in policy.parameters() if parameter.requires_grad
    )
    if trainable != adapter_parameters + fully_trainable_parameters:
        raise RuntimeError(
            "Trainable parameter accounting mismatch: "
            f"{trainable=} {adapter_parameters=} {fully_trainable_parameters=}"
        )
    return Pi05MixedLoraSummary(
        total_parameters=total,
        trainable_parameters=trainable,
        adapter_parameters=adapter_parameters,
        fully_trainable_parameters=fully_trainable_parameters,
        injected_linear_modules=len(injected_names),
        language_modules=counts["language"],
        action_expert_modules=counts["action_expert"],
        vision_modules=counts["vision"],
        trainable_module_names=tuple(injected_names + fully_trainable_names),
    )
