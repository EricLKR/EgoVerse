"""LoRA variant of the mixed EgoVerse pi0.5 Expert-Action V3 policy."""

from __future__ import annotations

import logging
from collections.abc import Mapping

from egomimic.algo.pi_egoverse_expert_action_v3 import PIEgoVerseExpertActionV3
from egomimic.models.pi05_mixed_lora_v1 import (
    Pi05MixedLoraConfig,
    configure_pi05_mixed_lora,
)

logger = logging.getLogger(__name__)


class PIEgoVerseExpertActionV3LoRA(PIEgoVerseExpertActionV3):
    """Keep the V3 domain masks/heads while adapting the shared pi0.5 backbone."""

    def __init__(self, *args, lora: Mapping | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        config = Pi05MixedLoraConfig(**dict(lora or {}))
        summary = configure_pi05_mixed_lora(self.nets["policy"], config)

        # Auxiliary joint/gripper heads are deliberately small and are trained
        # in full. Human samples cannot update their robot-only outputs because
        # Expert-Action V3 applies exact per-domain masks.
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
            "Mixed pi0.5 LoRA configured: trainable=%d/%d (%.4f%%), "
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
