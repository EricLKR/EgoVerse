#!/usr/bin/env python
"""Pre/post-processor pipelines for EgoVerseHPTPolicy.

Deliberately minimal: `prepare_observation_for_inference` (called before the
preprocessor, in lerobot.utils.control_utils.predict_action) already converts
the observation to tensors, adds the batch dimension, normalizes images to
[0, 1] CHW, and moves everything to the target device -- so there's nothing
left for a preprocessor to do here. And EgoVerseHPTPolicy.predict_action_chunk
does its own normalization internally (egomimic's norm_stats + the FK-based
unit conversions), matching `EgoVerseHPTConfig.normalization_mapping` being
IDENTITY everywhere -- so no NormalizerProcessorStep/UnnormalizerProcessorStep
is used here; those would be no-ops for IDENTITY anyway, and skipping them
avoids depending on `dataset_stats` (which this policy never uses) being
present at all.
"""

from typing import Any

from lerobot.policies.egoverse_hpt.configuration_egoverse_hpt import EgoVerseHPTConfig
from lerobot.processor import (
    DeviceProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME


def make_egoverse_hpt_pre_post_processors(
    config: EgoVerseHPTConfig,
    dataset_stats: dict[str, dict[str, Any]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=[],  # observation already tensor/batched/on-device by the time this runs
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=[DeviceProcessorStep(device="cpu")],  # select_action's output may be on cuda
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
