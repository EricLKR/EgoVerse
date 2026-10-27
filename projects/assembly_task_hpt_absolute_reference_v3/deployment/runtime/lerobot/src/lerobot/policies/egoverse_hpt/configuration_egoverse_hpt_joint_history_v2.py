"""Inference configuration for four-frame EgoVerse joint History V2."""

from __future__ import annotations

from dataclasses import dataclass

from lerobot.policies.egoverse_hpt.configuration_egoverse_hpt_joint import (
    EgoVerseHPTJointConfig,
)
from lerobot.configs.policies import PreTrainedConfig


@PreTrainedConfig.register_subclass("egoverse_hpt_joint_history_v2")
@dataclass
class EgoVerseHPTJointHistoryV2Config(EgoVerseHPTJointConfig):
    observation_history: int = 4
    observation_history_step_ms: float = 100.0
    observation_history_max_skew_ms: float = 50.0
    n_obs_steps: int = 4

    def __post_init__(self):
        super().__post_init__()
        if self.observation_history < 2:
            raise ValueError("History V2 requires observation_history >= 2")
        if self.n_obs_steps != self.observation_history:
            raise ValueError("n_obs_steps must equal observation_history")
        if self.observation_history_step_ms <= 0:
            raise ValueError("observation_history_step_ms must be positive")
        if self.observation_history_max_skew_ms <= 0:
            raise ValueError("observation_history_max_skew_ms must be positive")

