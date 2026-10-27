#!/usr/bin/env python
"""Inference-only configuration for a joint-space EgoVerse HPT checkpoint.

The policy consumes raw PIPER feedback plus the two RGB cameras, applies the
normalization statistics stored in the Lightning checkpoint, and emits direct
absolute joint targets.  Cartesian pose conversion and IK are deliberately not
part of this policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode, PolicyFeature


@PreTrainedConfig.register_subclass("egoverse_hpt_joint")
@dataclass
class EgoVerseHPTJointConfig(PreTrainedConfig):
    ckpt_path: str = ""
    joint_limits_deg: list | None = None
    gripper_stroke_mm: float = 100.0

    front_camera_key: str = "435if"
    wrist_camera_key: str = "405"
    # Assembly New1 was trained with a third fixed side camera.  This key is
    # intentionally explicit so a two-camera runtime cannot silently execute
    # a three-camera checkpoint with the missing slot zero-filled.
    side_camera_key: str = "side"
    expected_image_height: int = 480
    expected_image_width: int = 640
    resize_mismatched_images: bool = False

    # The recorded policy horizon covers 30 control frames and is resampled to
    # 100 model points.  query_frequency counts real 30-Hz control ticks.
    action_horizon_frames: int = 30
    query_frequency: int = 10
    num_inference_steps: int = 10
    # Reuse the same diffusion noise at every replan.  Observations may change,
    # but identical observations no longer produce unrelated chunks merely
    # because the global RNG advanced.
    inference_seed: int = 42

    # This deployment entrypoint is intended for the deterministic residual
    # joint head.  Keeping the contract strict prevents an older Cartesian or
    # joint Flow-Matching checkpoint from being executed accidentally.  The
    # legacy joint head can still be inspected by opting out explicitly.
    require_residual_joint_head: bool = True

    # Physical output validation.  Predictions outside the calibrated range
    # are clipped and counted in the audit report before conversion to the
    # normalized command coordinates used by the PIPER LeRobot driver.
    joint_limit_margin_rad: float = 0.0
    gripper_threshold: float = 0.5
    # Driver-coordinate trajectory limits applied to the complete selected
    # chunk before it enters the command queue.  PIPER's LeRobot driver uses
    # [-100, 100] for each calibrated arm joint.
    max_joint_command_step: float = 0.50
    max_joint_command_accel: float = 0.10

    # Optional ACT-style temporal ensemble in direct joint space.  Each
    # prediction is placed on an absolute 30-Hz policy timeline.  Replanning
    # therefore adds another prediction for the same future ticks instead of
    # restarting a FIFO chunk at offset zero.
    temporal_ensemble: bool = False
    temporal_replan_interval: int = 5
    temporal_ensemble_decay: float = 0.10

    # Zero-phase smoothing of the six physical joint-radian trajectories
    # before resampling them onto control ticks.  The binary gripper channel
    # bypasses this filter entirely.
    chunk_smoothing: bool = False
    chunk_smoothing_window: int = 11
    chunk_smoothing_sigma: float = 2.0

    n_obs_steps: int = 1
    chunk_size: int = 100
    input_features: dict[str, PolicyFeature] = field(default_factory=dict)
    output_features: dict[str, PolicyFeature] = field(default_factory=dict)

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,
        }
    )

    def __post_init__(self):
        super().__post_init__()
        if not self.ckpt_path:
            raise ValueError("EgoVerseHPTJointConfig.ckpt_path is required")
        if self.joint_limits_deg is None or len(self.joint_limits_deg) != 6:
            raise ValueError("joint_limits_deg must contain six calibrated (min,max) pairs")
        for index, pair in enumerate(self.joint_limits_deg):
            if len(pair) != 2 or float(pair[1]) <= float(pair[0]):
                raise ValueError(f"invalid joint_limits_deg[{index}]={pair}")
        if self.gripper_stroke_mm <= 0:
            raise ValueError("gripper_stroke_mm must be positive")
        if self.action_horizon_frames <= 0:
            raise ValueError("action_horizon_frames must be positive")
        if not 0 < self.query_frequency <= self.action_horizon_frames:
            raise ValueError("query_frequency must be in (0, action_horizon_frames]")
        if self.num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be positive")
        if self.inference_seed < 0:
            raise ValueError("inference_seed must be non-negative")
        if self.expected_image_height <= 0 or self.expected_image_width <= 0:
            raise ValueError("expected image dimensions must be positive")
        if self.joint_limit_margin_rad < 0:
            raise ValueError("joint_limit_margin_rad must be non-negative")
        if not 0.0 <= self.gripper_threshold <= 1.0:
            raise ValueError("gripper_threshold must be in [0,1]")
        if self.max_joint_command_step <= 0:
            raise ValueError("max_joint_command_step must be positive")
        if not 0 < self.max_joint_command_accel <= self.max_joint_command_step:
            raise ValueError(
                "max_joint_command_accel must be in (0,max_joint_command_step]"
            )
        if not 0 < self.temporal_replan_interval <= self.action_horizon_frames:
            raise ValueError(
                "temporal_replan_interval must be in (0,action_horizon_frames]"
            )
        if self.temporal_ensemble_decay < 0:
            raise ValueError("temporal_ensemble_decay must be non-negative")
        if (
            self.chunk_smoothing_window <= 0
            or self.chunk_smoothing_window > self.chunk_size
            or self.chunk_smoothing_window % 2 == 0
        ):
            raise ValueError(
                "chunk_smoothing_window must be a positive odd value no larger than chunk_size"
            )
        if self.chunk_smoothing_sigma <= 0:
            raise ValueError("chunk_smoothing_sigma must be positive")

    def get_optimizer_preset(self):
        raise NotImplementedError("Joint HPT wrapper is inference-only")

    def get_scheduler_preset(self):
        raise NotImplementedError("Joint HPT wrapper is inference-only")

    def validate_features(self) -> None:
        pass

    @property
    def observation_delta_indices(self) -> list | None:
        return None

    @property
    def action_delta_indices(self) -> list | None:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> list | None:
        return None
