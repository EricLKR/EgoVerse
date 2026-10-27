"""LoRA adaptation of pi0.5 Expert-Action V4 Bounded for PIPER."""

from __future__ import annotations

import logging
from collections.abc import Mapping

from egomimic.algo.pi_egoverse_expert_action_v4_bounded import (
    PIEgoVerseExpertActionV4Bounded,
)
from egomimic.models.pi05_mixed_lora_v1 import (
    Pi05MixedLoraConfig,
    configure_pi05_mixed_lora,
)

logger = logging.getLogger(__name__)


class PIEgoVerseExpertActionV4BoundedLoRA(PIEgoVerseExpertActionV4Bounded):
    def __init__(self, *args, lora: Mapping | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        config = Pi05MixedLoraConfig(**dict(lora or {}))
        summary = configure_pi05_mixed_lora(self.nets["policy"], config)
        auxiliary_trainable = sum(
            parameter.numel()
            for parameter in self.nets["auxiliary"].parameters()
            if parameter.requires_grad
        )
        total = sum(parameter.numel() for parameter in self.nets.parameters())
        trainable = sum(
            parameter.numel()
            for parameter in self.nets.parameters()
            if parameter.requires_grad
        )
        self.lora_config = config
        self.lora_summary = summary.to_dict()
        self.finetune_parameter_stats = {
            "total": total,
            "trainable": trainable,
            "frozen": total - trainable,
            "trainable_percent": 100.0 * trainable / total,
            "policy_lora_and_interfaces": summary.trainable_parameters,
            "auxiliary_heads": auxiliary_trainable,
        }
        logger.info(
            "pi0.5 V4 bounded LoRA: trainable=%d/%d (%.4f%%), "
            "policy=%d auxiliary=%d targets(language=%d expert=%d vision=%d)",
            trainable,
            total,
            100.0 * trainable / total,
            summary.trainable_parameters,
            auxiliary_trainable,
            summary.language_modules,
            summary.action_expert_modules,
            summary.vision_modules,
        )
