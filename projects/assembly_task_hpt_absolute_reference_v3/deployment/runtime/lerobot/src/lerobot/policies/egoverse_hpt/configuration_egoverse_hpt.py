#!/usr/bin/env python
"""Config for wrapping an EgoVerse HPT checkpoint (PyTorch Lightning .ckpt,
trained via egomimic/trainHydra.py) as a lerobot PreTrainedPolicy, so
`lerobot-record --policy.path=...` can drive the real Piper arm with it --
reusing lerobot's camera capture, Piper CAN I/O, control loop, and safety
limits instead of writing a bespoke rollout script.

This only works in an environment with BOTH `lerobot` and `egomimic`
importable (see EgoVerse's `emimic` venv, which has both installed).
"""

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode, PolicyFeature


@PreTrainedConfig.register_subclass("egoverse_hpt")
@dataclass
class EgoVerseHPTConfig(PreTrainedConfig):
    # Path to the EgoVerse Lightning checkpoint (e.g. .../checkpoints/last.ckpt).
    # NOT a lerobot-style pretrained_model directory -- loaded via
    # egomimic's own ModelWrapper.load_from_checkpoint, not lerobot's
    # safetensors mechanism (see modeling_egoverse_hpt.py's from_pretrained
    # override).
    ckpt_path: str = ""

    # REQUIRED, no safe default: this specific Piper unit's calibrated
    # per-joint (min, max) limits in real degrees, 6 pairs, joint_0..joint_5.
    # Needed to convert the policy's predicted (real, physical) joint angles
    # into the normalized [-100, 100] command space lerobot's Piper robot
    # class expects -- see PiperConfig.joint_limits_deg / find_real_joint_limits.py
    # for how to obtain this. Get this WRONG and the arm receives a
    # correctly-shaped but wrongly-scaled command -- there is no way for this
    # policy to detect that on its own, so it is intentionally not defaulted.
    joint_limits_deg: list | None = None

    # PIPER-L gripper stroke (mm) -- see piper_kinematics.py's GRIPPER_STROKE_MM.
    gripper_stroke_mm: float = 100.0

    # Must match how THIS ckpt was trained (see egomimic/rldb/embodiment/piper.py):
    #   "wristframe" -- ckpt trained with Piper.get_transform_list(mode=
    #       "cartesian_wristframe"): the model predicts each step as a pose
    #       *relative to the current observed EEF pose*. predict_action_chunk
    #       composes each predicted delta onto the live FK'd anchor pose
    #       (obs_mat @ delta_mat) before IK.
    #   "camframe" -- ckpt trained with mode="cartesian_camframe" (the
    #       EgoVerse paper's actual canonical cross-embodiment representation):
    #       the model predicts the EEF pose *in the external camera's frame*
    #       (T_cam<-ee). predict_action_chunk composes each predicted pose
    #       with the fixed camera extrinsics (Piper.EXTRINSICS[front_camera_key],
    #       T_base<-cam) to recover the base_link-frame IK target -- a plain
    #       fixed-matrix multiply, no live anchor needed (unlike wristframe).
    #   "absolute" -- ckpt trained with mode="cartesian_base": the model
    #       predicts an absolute base_link-frame target pose directly. No
    #       composition -- the predicted pose goes straight to IK.
    # Pointing this at the wrong value for a given ckpt silently miscomputes
    # every IK target -- there's no way for this policy to detect that itself.
    # (plain str, not Literal: this venv's draccus has no decoder registered
    # for typing.Literal -- see __post_init__ for the actual validation.)
    action_frame: str = "absolute"

    # Which lerobot camera keys map to Piper's two training cameras. Default
    # matches piper.py's (unverified against the physical rig) assumption:
    # 435if (D435i) = front, 405 (D405) = wrist. Swap if that's backwards.
    front_camera_key: str = "435if"
    wrist_camera_key: str = "405"

    # The Piper training action covers 30 raw 30-Hz control frames, then gets
    # resampled to the shared HPT head's 100 points. `query_frequency` is the
    # number of RAW control frames to execute before re-querying, not Hz and
    # not a direct slice of the 100 model points. The policy maps raw offset k
    # to round(k * model_horizon / action_horizon_frames), exactly like the
    # offline receding-horizon evaluator. qf5 was selected on validation for
    # the best ResNet checkpoint; use qf10 if measured inference+IK latency
    # cannot stay below qf5's 167 ms budget at 30 Hz.
    action_horizon_frames: int = 30
    query_frequency: int = 5

    # Backwards-compatible alias for old launch commands. Leave None for the
    # validated `query_frequency` behavior. If supplied, it replaces
    # query_frequency but is still interpreted in RAW control frames and is
    # resampled correctly -- it no longer means "take the first N model rows".
    n_action_steps: int | None = None

    # Flow-matching Euler integration steps used during deployment. Offline
    # validation found 10 better than the 20-step no-retraining control.
    num_inference_steps: int = 10

    # Both ResNet image stems were trained on RGB CHW 480x640 frames. A shape
    # mismatch is rejected by default because silently reshaping or changing
    # crop/aspect ratio creates a large deployment distribution shift.
    expected_image_height: int = 480
    expected_image_width: int = 640
    resize_mismatched_images: bool = False

    # Training actions use a binary 0/1 gripper command. The observed gripper
    # state remains continuous; only the predicted command is thresholded.
    gripper_threshold: float = 0.5

    # The training/validation MultiDataset normalizes proprioception with the
    # checkpoint's own norm_stats before HPT sees it. Deployment must do the
    # same. Keep the opt-out only for controlled offline A/B diagnostics; live
    # deployment should always leave this enabled.
    normalize_proprio_input: bool = True

    # Optional base-frame safety envelope. For execution these should be set
    # from the demonstrated workspace with a conservative physical margin.
    # They remain optional so checkpoint-only/offline tests do not invent a
    # workspace. The dedicated deployment script refuses execute mode without
    # explicit bounds.
    workspace_xyz_min: list[float] | None = None
    workspace_xyz_max: list[float] | None = None

    # IK (see PiperKinematicsSolver.ik in egomimic/robot/piper/piper_kinematics.py).
    ik_num_retries: int = 8
    ik_pos_tolerance: float = 2e-3
    ik_rot_tolerance: float = 3e-2
    ik_seed_noise_std: float = 0.15

    # Every predicted step whose IK doesn't converge holds the previous joint
    # target instead of executing an unvalidated solution. This is the
    # up-to-N consecutive holds allowed before select_action raises instead
    # of continuing to hold silently (0 = never raise, just keep holding).
    max_consecutive_ik_failures: int = 5

    n_obs_steps: int = 1
    chunk_size: int = 100  # matches Piper.ACTION_HORIZON / the trained model's action_horizon
    input_features: dict[str, PolicyFeature] = field(default_factory=dict)
    output_features: dict[str, PolicyFeature] = field(default_factory=dict)

    # Bypass lerobot's own mean/std normalization pipeline entirely -- this
    # policy does its own normalization internally (egomimic's norm_stats +
    # the Piper embodiment's FK-based unit conversions), so lerobot's
    # pre/post-processors should be pure passthrough.
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
            raise ValueError("EgoVerseHPTConfig.ckpt_path is required.")
        if self.joint_limits_deg is None:
            raise ValueError(
                "EgoVerseHPTConfig.joint_limits_deg is required -- this arm's calibrated "
                "per-joint limits, 6x(min,max) in real degrees. There is no safe default; "
                "using the wrong unit's calibration silently miscalibrates every command. "
                "See find_real_joint_limits.py / PiperConfig.joint_limits_deg."
            )
        if len(self.joint_limits_deg) != 6:
            raise ValueError(f"joint_limits_deg needs 6 (min, max) pairs, got {len(self.joint_limits_deg)}")
        if self.action_frame not in ("wristframe", "camframe", "absolute"):
            raise ValueError(
                f"action_frame must be 'wristframe', 'camframe', or 'absolute', got {self.action_frame!r}."
            )
        if self.action_horizon_frames <= 0:
            raise ValueError("action_horizon_frames must be positive.")
        effective_qf = self.n_action_steps if self.n_action_steps is not None else self.query_frequency
        if not 0 < effective_qf <= self.action_horizon_frames:
            raise ValueError(
                "query_frequency/n_action_steps must be in (0, action_horizon_frames], "
                f"got {effective_qf} for horizon {self.action_horizon_frames}."
            )
        if self.num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be positive.")
        if self.expected_image_height <= 0 or self.expected_image_width <= 0:
            raise ValueError("expected image height and width must be positive.")
        if not 0.0 <= self.gripper_threshold <= 1.0:
            raise ValueError("gripper_threshold must be in [0, 1].")
        if (self.workspace_xyz_min is None) != (self.workspace_xyz_max is None):
            raise ValueError("workspace_xyz_min and workspace_xyz_max must be supplied together.")
        if self.workspace_xyz_min is not None:
            if len(self.workspace_xyz_min) != 3 or len(self.workspace_xyz_max) != 3:
                raise ValueError("workspace XYZ bounds must each contain exactly 3 values.")
            if any(
                lo >= hi
                for lo, hi in zip(self.workspace_xyz_min, self.workspace_xyz_max, strict=True)
            ):
                raise ValueError("each workspace_xyz_min value must be smaller than workspace_xyz_max.")

    def get_optimizer_preset(self):
        raise NotImplementedError("EgoVerseHPTConfig is inference-only -- train via egomimic/trainHydra.py.")

    def get_scheduler_preset(self):
        raise NotImplementedError("EgoVerseHPTConfig is inference-only -- train via egomimic/trainHydra.py.")

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
