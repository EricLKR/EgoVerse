#!/usr/bin/env python
"""LeRobot inference wrapper for EgoVerse's direct joint-space HPT policy.

Input
-----
* PIPER feedback: six joints in the SDK's raw 0.001-degree unit and gripper
  aperture in raw 0.001-mm.
* Front, wrist and side RGB frames for Assembly New1 checkpoints.

Output
------
Seven absolute LeRobot/PIPER command coordinates: six calibrated joints in
[-100, 100] and a binary gripper in {0, 100}.  The model itself predicts six
absolute joint angles in radians plus a binary gripper in {0, 1}.  There is no
FK, Cartesian target, workspace gate, or IK anywhere in this module.
"""

from __future__ import annotations

import logging
import time
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

from lerobot.policies.egoverse_hpt.configuration_egoverse_hpt_joint import (
    EgoVerseHPTJointConfig,
)
from lerobot.policies.pretrained import PreTrainedPolicy

logger = logging.getLogger(__name__)


def model_indices_for_control_ticks(
    num_model_steps: int, action_horizon_frames: int, query_frequency: int
) -> list[int]:
    """Map real control ticks onto a model trajectory of arbitrary length."""
    if num_model_steps <= 0 or action_horizon_frames <= 0:
        raise ValueError("model and action horizons must be positive")
    if not 0 < query_frequency <= action_horizon_frames:
        raise ValueError("query_frequency must be in (0, action_horizon_frames]")
    if action_horizon_frames == 1:
        return [0]
    # Both trajectories include their endpoints.  Therefore the exact affine
    # map is [0, action_horizon_frames-1] -> [0, num_model_steps-1], not a
    # division by the number of samples.  In particular, 30 ticks cover model
    # indices 0..99 for Clean V2 and 0..29 for native-30 V3.
    return [
        int(
            round(
                offset
                * (num_model_steps - 1)
                / (action_horizon_frames - 1)
            )
        )
        for offset in range(query_frequency)
    ]


def absolute_time_joint_ensemble(
    plans: list[tuple[int, int, np.ndarray]],
    policy_tick: int,
    decay: float,
) -> tuple[np.ndarray, dict]:
    """Blend predictions that refer to the same absolute control tick.

    ``plans`` entries are ``(plan_id, start_tick, actions)``.  Arm values are
    affine calibrated joint coordinates, so their weighted average is exactly
    equivalent to averaging the corresponding joint radians.  The gripper is
    kept binary through a weighted vote.
    """
    if policy_tick < 0 or decay < 0:
        raise ValueError("policy_tick and temporal decay must be non-negative")
    candidates: list[np.ndarray] = []
    plan_ids: list[int] = []
    plan_starts: list[int] = []
    source_offsets: list[int] = []
    ages: list[int] = []
    for plan_id, start_tick, actions in plans:
        array = np.asarray(actions, dtype=np.float64)
        offset = policy_tick - int(start_tick)
        if array.ndim != 2 or array.shape[1] != 7:
            raise ValueError(f"temporal plan must have shape (T,7), got {array.shape}")
        if 0 <= offset < len(array):
            candidates.append(array[offset])
            plan_ids.append(int(plan_id))
            plan_starts.append(int(start_tick))
            source_offsets.append(int(offset))
            ages.append(int(offset))
    if not candidates:
        raise RuntimeError(f"no temporal prediction covers policy tick {policy_tick}")
    values = np.stack(candidates, axis=0)
    weights = np.exp(-float(decay) * np.asarray(ages, dtype=np.float64))
    weights /= weights.sum()
    output = np.empty(7, dtype=np.float64)
    output[:6] = np.sum(values[:, :6] * weights[:, None], axis=0)
    open_vote = float(np.sum((values[:, 6] >= 50.0) * weights))
    output[6] = 100.0 if open_vote >= 0.5 else 0.0
    joint_spread_rms = float(
        np.sqrt(np.mean(np.square(values[:, :6] - output[None, :6])))
    )
    report = {
        "policy_tick": int(policy_tick),
        "vote_count": len(candidates),
        "plan_ids": plan_ids,
        "plan_start_ticks": plan_starts,
        "source_offsets": source_offsets,
        "weights": weights.tolist(),
        "open_vote": open_vote,
        "blended_raw_driver_action": output.tolist(),
        "joint_spread_rms_driver": joint_spread_rms,
        "ensemble_space": "absolute_time_aligned_joint_driver_coordinates",
    }
    return output.astype(np.float32), report


def smooth_joint_action_chunk(
    actions: np.ndarray,
    window: int,
    sigma: float,
) -> np.ndarray:
    """Zero-phase Gaussian smoothing for arm joints, preserving the gripper.

    Edge padding avoids introducing a phase delay or pulling the first target
    toward zero.  Since each joint is smoothed independently in physical
    radians, the operation cannot mix joint semantics or create fractional
    gripper commands.
    """
    array = np.asarray(actions)
    if array.ndim != 2 or array.shape[1] != 7:
        raise ValueError(f"joint action chunk must have shape (T,7), got {array.shape}")
    if window <= 0 or window > len(array) or window % 2 == 0:
        raise ValueError("smoothing window must be positive, odd, and no larger than T")
    if sigma <= 0:
        raise ValueError("smoothing sigma must be positive")
    if not np.isfinite(array).all():
        raise ValueError("cannot smooth a non-finite action chunk")
    radius = window // 2
    coordinates = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * np.square(coordinates / float(sigma)))
    kernel /= kernel.sum()
    output = array.astype(np.float64, copy=True)
    for joint_index in range(6):
        padded = np.pad(array[:, joint_index], (radius, radius), mode="edge")
        output[:, joint_index] = np.convolve(padded, kernel, mode="valid")
    # Preserve every discrete prediction exactly; thresholding remains later.
    output[:, 6] = array[:, 6]
    return output.astype(array.dtype, copy=False)


def adjacent_joint_rms(actions: np.ndarray) -> tuple[float, float]:
    """Return mean/max adjacent joint-radian RMS for audit logging."""
    joints = np.asarray(actions, dtype=np.float64)[:, :6]
    if len(joints) < 2:
        return 0.0, 0.0
    rms = np.sqrt(np.mean(np.square(np.diff(joints, axis=0)), axis=1))
    return float(rms.mean()), float(rms.max())


def prepare_rgb_chw_image(
    image: np.ndarray | torch.Tensor,
    *,
    expected_height: int,
    expected_width: int,
    resize_mismatch: bool,
    name: str,
) -> torch.Tensor:
    """Return one finite RGB float32 CHW image in [0,1]."""
    tensor = image.detach().cpu() if isinstance(image, torch.Tensor) else torch.as_tensor(image)
    if tensor.ndim == 4:
        if tensor.shape[0] != 1:
            raise ValueError(f"{name} must contain one live frame, got {tuple(tensor.shape)}")
        tensor = tensor[0]
    if tensor.ndim != 3:
        raise ValueError(f"{name} must be CHW or HWC, got {tuple(tensor.shape)}")
    if tensor.shape[0] in (3, 4):
        tensor = tensor[:3]
    elif tensor.shape[-1] in (3, 4):
        tensor = tensor[..., :3].permute(2, 0, 1)
    else:
        raise ValueError(f"{name} has no recognizable RGB channel: {tuple(tensor.shape)}")
    tensor = tensor.contiguous().float()
    if tensor.numel() and float(tensor.max()) > 1.5:
        tensor = tensor / 255.0
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains NaN or Inf")
    expected = (expected_height, expected_width)
    if tuple(tensor.shape[-2:]) != expected:
        if not resize_mismatch:
            raise ValueError(
                f"{name} has HW={tuple(tensor.shape[-2:])}; checkpoint expects {expected}"
            )
        tensor = F.interpolate(
            tensor.unsqueeze(0), size=expected, mode="bilinear", align_corners=False
        )[0]
    return tensor.clamp_(0.0, 1.0)


class EgoVerseHPTJointPolicy(PreTrainedPolicy):
    config_class = EgoVerseHPTJointConfig
    name = "egoverse_hpt_joint"

    def __init__(self, config: EgoVerseHPTJointConfig):
        super().__init__(config)
        from egomimic.pl_utils.pl_model import ModelWrapper
        from egomimic.rldb.embodiment.embodiment import get_embodiment_id

        self._embodiment_id = get_embodiment_id("piper_left_arm")
        self._joint_low_rad = np.radians(
            np.asarray([pair[0] for pair in config.joint_limits_deg], dtype=np.float64)
        )
        self._joint_high_rad = np.radians(
            np.asarray([pair[1] for pair in config.joint_limits_deg], dtype=np.float64)
        )
        self._action_queue: deque[torch.Tensor] = deque()
        self._chunk_sequence_id = 0
        self._active_chunk_size = 0
        self._active_chunk_consumed = 0
        self._last_planned_velocity_driver = np.zeros(6, dtype=np.float64)
        self._pending_planned_velocity_driver: np.ndarray | None = None
        self._temporal_plans: list[tuple[int, int, np.ndarray]] = []
        self._policy_tick = 0
        self._last_temporal_replan_tick: int | None = None
        self.last_ensemble_report: dict | None = None
        self.last_prediction_report: dict | None = None

        logger.info("[egoverse_hpt_joint] loading %s", config.ckpt_path)
        model = ModelWrapper.load_from_checkpoint(
            config.ckpt_path, weights_only=False, map_location=config.device
        )
        model.to(config.device)
        model.eval()
        model.model.device = config.device
        self._egoverse_model = model
        self._algo = model.model

        self._validate_checkpoint_contract()

        if getattr(self._algo, "diffusion", False):
            from egomimic.models.denoising_policy import DenoisingPolicy

            for head_name, head in self._algo.nets["policy"].heads.items():
                if isinstance(head, DenoisingPolicy):
                    head.num_inference_steps = config.num_inference_steps
                    logger.info(
                        "[egoverse_hpt_joint] heads[%r].num_inference_steps=%d",
                        head_name,
                        config.num_inference_steps,
                    )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_name_or_path,
        *,
        config: EgoVerseHPTJointConfig | None = None,
        **kwargs,
    ):
        if config is None:
            config = EgoVerseHPTJointConfig.from_pretrained(pretrained_name_or_path, **kwargs)
        return cls(config)

    def _resolve_joint_head(self):
        """Return the network head that produces PIPER ``actions_joints``."""
        algo = self._algo
        heads = algo.nets["policy"].heads
        # Shared-Action V3 deliberately has one temporal head.  The same head
        # emits the common pose plus the PIPER-only joints and gripper through
        # predict_all(), so no independent auxiliary network exists.
        shared_head = heads["shared"] if "shared" in heads else None
        if (
            shared_head is not None
            and getattr(shared_head, "contract_name", None)
            in {"hpt_mixed_shared_action_v3", "hpt_expert_action_v3"}
            and (
                "actions_joints"
                in algo.auxiliary_ac_keys.get("piper_left_arm", [])
                or getattr(shared_head, "contract_name", None)
                == "hpt_expert_action_v3"
            )
        ):
            return "shared:actions_joints", shared_head
        if algo.shared_ac_key == "actions_joints" and "shared" in heads:
            return "shared", heads["shared"]
        auxiliary_name = f"piper_left_arm_actions_joints"
        if (
            "actions_joints" in algo.auxiliary_ac_keys.get("piper_left_arm", [])
            and auxiliary_name in heads
        ):
            return auxiliary_name, heads[auxiliary_name]
        if (
            algo.ac_keys.get(self._embodiment_id) == "actions_joints"
            and "piper_left_arm" in heads
        ):
            return "piper_left_arm", heads["piper_left_arm"]
        raise RuntimeError("checkpoint has no network head for PIPER actions_joints")

    def _validate_checkpoint_contract(self) -> None:
        algo = self._algo
        if self._embodiment_id not in algo.ac_keys:
            raise RuntimeError("checkpoint has no piper_left_arm action key")
        action_keys = {algo.ac_keys[self._embodiment_id]}
        if algo.shared_ac_key:
            action_keys.add(algo.shared_ac_key)
        action_keys.update(algo.auxiliary_ac_keys.get("piper_left_arm", []))
        heads = algo.nets["policy"].heads
        shared_head = heads["shared"] if "shared" in heads else None
        if getattr(shared_head, "contract_name", None) == "hpt_expert_action_v3":
            # Expert-Action V3 owns the PIPER joint projection inside its
            # shared 15-D Flow head, so it deliberately has no auxiliary key.
            action_keys.add("actions_joints")
        if "actions_joints" not in action_keys:
            raise RuntimeError(
                "checkpoint is not joint-space: no actions_joints primary/shared/auxiliary head"
            )
        proprio_key = "observations.state.joints"
        proprio_keys = set(algo.proprio_keys.get(self._embodiment_id, []))
        if proprio_key not in proprio_keys:
            raise RuntimeError(
                f"checkpoint lacks {proprio_key}; found proprio keys {sorted(proprio_keys)}"
            )
        stats = algo.norm_stats.norm_stats.get(self._embodiment_id, {})
        for key in (proprio_key, "actions_joints"):
            if key not in stats:
                raise RuntimeError(f"checkpoint normalization statistics lack {key}")

        head_name, head = self._resolve_joint_head()
        head_class = type(head).__name__
        contract_name = getattr(head, "contract_name", None)
        contract_version = getattr(head, "contract_version", None)
        residual_head_classes = {
            "ResidualJointPolicy",
            "ResidualJointPolicyCleanV2",
            "ResidualJointPolicyMixedCleanV2",
            "AbsoluteReferenceJointPolicyV3",
        }
        clean_v2_contracts = {
            "piper_residual_joint_clean_v2",
            "piper_absolute_reference_joint_v3",
        }
        shared_v3_contracts = {
            "hpt_mixed_shared_action_v3",
            "hpt_expert_action_v3",
        }
        is_residual = (
            head_class in residual_head_classes
            or contract_name in clean_v2_contracts
            or contract_name in shared_v3_contracts
        )
        if self.config.require_residual_joint_head and not is_residual:
            raise RuntimeError(
                "checkpoint is not the deterministic residual-joint architecture: "
                f"actions_joints head {head_name!r} is {head_class}. "
                "Use the residual checkpoint, or explicitly opt into the legacy "
                "joint head only for offline compatibility checks."
            )
        if is_residual:
            if not getattr(head, "requires_data_for_inference", False):
                raise RuntimeError(
                    "ResidualJointPolicy must require current proprio during inference"
                )
            expected = {"action_horizon": int(self.config.chunk_size)}
            # Expert-Action V3 stores its six-joint layout as module-level
            # slices rather than a ``joint_dims`` instance attribute.
            if contract_name != "hpt_expert_action_v3":
                expected["joint_dims"] = 6
            # Clean V2 resamples 30 native frames to 100 model points and
            # therefore stores a separate control_horizon. V3 is native-30.
            if contract_name not in shared_v3_contracts:
                expected["control_horizon"] = int(
                    self.config.action_horizon_frames
                )
            actual = {key: int(getattr(head, key, -1)) for key in expected}
            if actual != expected:
                raise RuntimeError(
                    "residual checkpoint trajectory contract mismatch: "
                    f"expected={expected}, actual={actual}"
                )
        self.checkpoint_report = {
            "action_head_name": head_name,
            "action_head_class": head_class,
            "policy_contract_name": contract_name,
            "policy_contract_version": contract_version,
            "is_residual_joint_head": is_residual,
            "requires_current_proprio": bool(
                getattr(head, "requires_data_for_inference", False)
            ),
            "model_action_horizon": int(getattr(head, "action_horizon", -1)),
            "native_control_horizon": int(
                getattr(head, "control_horizon", self.config.action_horizon_frames)
            ),
            "joint_dims": int(getattr(head, "joint_dims", -1)),
            "command_generation": "direct_absolute_joint_radians_no_ik",
        }

    def reset(self):
        self._action_queue.clear()
        self._chunk_sequence_id = 0
        self._active_chunk_size = 0
        self._active_chunk_consumed = 0
        self._last_planned_velocity_driver.fill(0.0)
        self._pending_planned_velocity_driver = None
        self._temporal_plans.clear()
        self._policy_tick = 0
        self._last_temporal_replan_tick = None
        self.last_ensemble_report = None
        self.last_prediction_report = None

    def get_optim_params(self) -> dict:
        raise NotImplementedError("EgoVerseHPTJointPolicy is inference-only")

    def forward(self, batch: dict[str, torch.Tensor]):
        raise NotImplementedError("Train joint HPT through egomimic/trainHydra.py")

    def _joint_rad_to_driver(self, joint_rad: np.ndarray) -> np.ndarray:
        span = self._joint_high_rad - self._joint_low_rad
        command = (joint_rad - self._joint_low_rad) / span * 200.0 - 100.0
        return np.clip(command, -100.0, 100.0)

    def _filter_driver_action(
        self,
        raw_driver_action: np.ndarray,
        current_driver: np.ndarray,
        previous_velocity: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Rate-limit one raw model target against fresh measured feedback."""
        action = np.asarray(raw_driver_action, dtype=np.float64).reshape(7).copy()
        current = np.asarray(current_driver, dtype=np.float64).reshape(6)
        velocity = np.asarray(previous_velocity, dtype=np.float64).reshape(6)
        if not (
            np.isfinite(action).all()
            and np.isfinite(current).all()
            and np.isfinite(velocity).all()
        ):
            raise ValueError("non-finite input to joint command trajectory filter")
        max_step = float(self.config.max_joint_command_step)
        max_accel = float(self.config.max_joint_command_accel)
        desired_velocity = np.clip(action[:6] - current, -max_step, max_step)
        next_velocity = np.clip(
            desired_velocity,
            velocity - max_accel,
            velocity + max_accel,
        )
        next_command = np.clip(current + next_velocity, -100.0, 100.0)
        next_velocity = next_command - current
        action[:6] = next_command
        return action.astype(np.float32), next_velocity

    def _preview_filtered_chunk(
        self, raw_driver_actions: np.ndarray, current_driver: np.ndarray
    ) -> np.ndarray:
        """Preview filtering assuming perfect tracking; does not mutate state."""
        actions = np.asarray(raw_driver_actions, dtype=np.float64)
        current = np.asarray(current_driver, dtype=np.float64).reshape(6).copy()
        velocity = self._last_planned_velocity_driver.copy()
        output = np.empty_like(actions, dtype=np.float32)
        for index, raw_action in enumerate(actions):
            safe_action, velocity = self._filter_driver_action(
                raw_action, current, velocity
            )
            output[index] = safe_action
            current = safe_action[:6].astype(np.float64)
        return output

    def _select_joint_output(self, action_out: dict) -> tuple[torch.Tensor, str]:
        algo = self._algo
        # Robot-only checkpoint: actions_joints is the shared head.
        if algo.shared_ac_key == "actions_joints" and "shared" in action_out:
            return action_out["shared"], "shared"
        # Future mixed checkpoint: actions_joints is a robot auxiliary head.
        if "actions_joints" in action_out:
            return action_out["actions_joints"], "actions_joints"
        # Separate-domain checkpoint: robot primary head is actions_joints.
        if (
            algo.ac_keys.get(self._embodiment_id) == "actions_joints"
            and "piper_left_arm" in action_out
        ):
            return action_out["piper_left_arm"], "piper_left_arm"
        raise RuntimeError(
            f"model forward returned {sorted(action_out)} but none is a robot actions_joints head"
        )

    def _predict_joint_actions(
        self, features, data: dict
    ) -> tuple[torch.Tensor, torch.Tensor | None, str]:
        """Execute the checkpoint's native joint projection.

        V3 must call the exact shared head serialized by Lightning. Older
        checkpoints retain their standalone joint-head inference path.  The
        Expert-Action V3 head natively decodes a 30-step physical joint tensor
        from 100-step normalization statistics, so return that raw tensor
        directly instead of asking the generic normalizer to broadcast 100
        statistics over 30 predictions.
        """
        output_head, joint_head = self._resolve_joint_head()
        if getattr(joint_head, "contract_name", None) == "hpt_expert_action_v3":
            outputs = joint_head.sample_outputs(features, data)
            if "joint_raw" not in outputs:
                raise RuntimeError("Expert-Action V3 omitted joint_raw")
            raw = outputs["joint_raw"]
            normalized = joint_head._normalize(raw, data, "actions_joints")
            return normalized, raw, output_head
        if getattr(joint_head, "contract_name", None) == "hpt_mixed_shared_action_v3":
            outputs = joint_head.predict_all(features, data)
            if "actions_joints" not in outputs:
                raise RuntimeError(
                    "HPT Shared-Action V3 predict_all() omitted actions_joints"
                )
            return outputs["actions_joints"], None, output_head
        prediction = (
            joint_head(features, data=data)
            if getattr(joint_head, "requires_data_for_inference", False)
            else joint_head(features)
        )
        return prediction, None, output_head

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        device = self.config.device

        def to_numpy(value) -> np.ndarray:
            if isinstance(value, torch.Tensor):
                return value.detach().cpu().numpy()
            return np.asarray(value)

        raw_state = to_numpy(batch["observation.state"]).reshape(-1)[:7].astype(np.float64)
        if raw_state.shape != (7,) or not np.isfinite(raw_state).all():
            raise RuntimeError(f"invalid raw PIPER state: shape={raw_state.shape} {raw_state}")
        # SDK feedback units: joints are 0.001 degree and gripper is
        # 0.001 mm.  Keep the conversion local so importing this direct-joint
        # wrapper never imports or constructs a kinematics solver.
        obs_joint = (raw_state[:6] * 1e-3 * np.pi / 180.0).astype(np.float32)
        obs_gripper = np.clip(
            raw_state[6:7] * 1e-3 / self.config.gripper_stroke_mm, 0.0, 1.0
        ).astype(np.float32)
        proprio_physical = np.concatenate([obs_joint, obs_gripper]).astype(np.float32)

        front_key = f"observation.images.{self.config.front_camera_key}"
        wrist_key = f"observation.images.{self.config.wrist_camera_key}"
        side_key = f"observation.images.{self.config.side_camera_key}"
        front = prepare_rgb_chw_image(
            batch[front_key],
            expected_height=self.config.expected_image_height,
            expected_width=self.config.expected_image_width,
            resize_mismatch=self.config.resize_mismatched_images,
            name=front_key,
        )
        wrist = prepare_rgb_chw_image(
            batch[wrist_key],
            expected_height=self.config.expected_image_height,
            expected_width=self.config.expected_image_width,
            resize_mismatch=self.config.resize_mismatched_images,
            name=wrist_key,
        )
        side = prepare_rgb_chw_image(
            batch[side_key],
            expected_height=self.config.expected_image_height,
            expected_width=self.config.expected_image_width,
            resize_mismatch=self.config.resize_mismatched_images,
            name=side_key,
        )

        horizon = int(self.config.chunk_size)
        # Dataset labels and normalization tensors remain 100 steps for both
        # HPT variants. Expert-Action V3 internally decodes a native 30-step
        # joint trajectory, but its training data contract still receives the
        # original 100-step placeholders and pad mask.
        joint_action_stats = self._algo.norm_stats.norm_stats.get(
            self._embodiment_id, {}
        ).get("actions_joints")
        if joint_action_stats is None:
            raise RuntimeError("checkpoint lacks actions_joints normalization statistics")
        norm_mode = self._algo.norm_stats.norm_mode
        horizon_stat_name = {
            "quantile": "quantile_1",
            "minmax": "min",
            "zscore": "mean",
        }.get(norm_mode)
        if horizon_stat_name is None or horizon_stat_name not in joint_action_stats:
            raise RuntimeError(
                f"cannot resolve actions_joints horizon for normalization mode {norm_mode!r}"
            )
        joint_stat_shape = np.asarray(joint_action_stats[horizon_stat_name]).shape
        training_action_horizon = (
            int(joint_stat_shape[0]) if len(joint_stat_shape) == 2 else horizon
        )
        dummy_action = np.repeat(
            proprio_physical[None, :], training_action_horizon, axis=0
        )
        # This placeholder is only used to satisfy HPT's inference data
        # contract. Clean V2 correctly rejects non-binary gripper labels, so
        # never copy a partially open measured aperture into the label slot.
        dummy_action[:, 6] = float(obs_gripper[0] >= self.config.gripper_threshold)
        hpt_batch = {
            "observations.images.front_img_1": front.unsqueeze(0).to(device),
            "observations.images.wrist_img": wrist.unsqueeze(0).to(device),
            "observations.images.side_img": side.unsqueeze(0).to(device),
            "observations.state.joints": torch.from_numpy(proprio_physical)
            .unsqueeze(0)
            .to(device),
            "actions_joints": torch.from_numpy(dummy_action).unsqueeze(0).to(device),
            "pad_mask": torch.ones(
                (1, training_action_horizon, 1), dtype=torch.bool, device=device
            ),
            "embodiment": torch.tensor([self._embodiment_id], device=device),
        }

        # Mixed Clean V2 uses a shared 9D Cartesian action for human/robot
        # cotraining and an auxiliary 7D joint action for PIPER deployment.
        # HPT's data adapter still requires the shared action key even though
        # deployment evaluates only the joint head. Supply a neutral midpoint
        # placeholder with the exact dimensionality recorded in this
        # checkpoint's normalization statistics.
        shared_action_key = self._algo.shared_ac_key
        if shared_action_key and shared_action_key not in hpt_batch:
            shared_stats = self._algo.norm_stats.norm_stats.get(
                self._embodiment_id, {}
            ).get(shared_action_key)
            if shared_stats is None:
                raise RuntimeError(
                    f"checkpoint lacks normalization statistics for shared action "
                    f"{shared_action_key!r}"
                )
            if norm_mode == "quantile":
                low_name, high_name = "quantile_1", "quantile_99"
            elif norm_mode == "minmax":
                low_name, high_name = "min", "max"
            elif norm_mode == "zscore":
                low_name = high_name = "mean"
            else:
                raise RuntimeError(f"unsupported checkpoint normalization mode {norm_mode!r}")
            low = np.asarray(shared_stats[low_name], dtype=np.float32)
            high = np.asarray(shared_stats[high_name], dtype=np.float32)
            neutral = 0.5 * (low + high)
            if neutral.ndim == 1:
                neutral_horizon = np.repeat(neutral[None, :], horizon, axis=0)
            elif neutral.ndim == 2:
                # Expert-Action V3 keeps the shared pose target at its original
                # 100-step training horizon while its PIPER joint branch is a
                # native 30-step output.  This placeholder is normalized by
                # the shared-action statistics and is not an executed target,
                # so preserve the statistics' own temporal dimension.
                neutral_horizon = neutral
            else:
                raise RuntimeError(
                    f"shared action statistics for {shared_action_key!r} have "
                    f"unexpected shape {neutral.shape}"
                )
            hpt_batch[shared_action_key] = torch.from_numpy(
                neutral_horizon
            ).unsqueeze(0).to(device)

        hpt_batch = self._algo.norm_stats.normalize(hpt_batch, self._embodiment_id)
        proprio_model_input = (
            hpt_batch["observations.state.joints"][0].detach().cpu().numpy()
        )
        if not np.isfinite(proprio_model_input).all():
            raise RuntimeError("normalized joint proprio contains NaN or Inf")

        ac_key = self._algo.ac_keys[self._embodiment_id]
        data = self._algo._robomimic_to_hpt_data(
            hpt_batch,
            self._algo.camera_keys[self._embodiment_id],
            self._algo.proprio_keys[self._embodiment_id],
            self._algo.lang_keys[self._embodiment_id],
            ac_key,
            self._algo.auxiliary_ac_keys.get("piper_left_arm", []),
        )
        # Clean V2 and Expert-Action V3 decode residuals around the *physical*
        # current joints.  Their heads therefore need the proprio normalizer in
        # the HPT data dictionary, not only the already-normalized state tensor.
        # The training dataset supplied this metadata through its batch
        # contract; live deployment constructs the batch directly and must
        # attach the same immutable checkpoint statistics explicitly.
        proprio_key = "observations.state.joints"
        proprio_stats = self._algo.norm_stats.norm_stats.get(
            self._embodiment_id, {}
        ).get(proprio_key)
        if proprio_stats is None:
            raise RuntimeError(
                f"checkpoint lacks normalization statistics for {proprio_key!r}"
            )
        norm_mode = self._algo.norm_stats.norm_mode
        norm_fields = {
            "quantile": ("quantile_1", "quantile_99"),
            "minmax": ("min", "max"),
            "zscore": ("mean", "std"),
        }.get(norm_mode)
        if norm_fields is None:
            raise RuntimeError(
                f"unsupported checkpoint normalization mode {norm_mode!r}"
            )
        data["_proprio_norm_mode"] = norm_mode
        reference = data["state_joints"]
        for stat_name in norm_fields:
            if stat_name not in proprio_stats:
                raise RuntimeError(
                    f"checkpoint {proprio_key!r} stats lack {stat_name!r}"
                )
            data[f"_proprio_norm_{stat_name}"] = torch.as_tensor(
                proprio_stats[stat_name],
                device=reference.device,
                dtype=reference.dtype,
            )
        inference_started = time.perf_counter()
        inference_device = torch.device(device)
        cuda_devices = []
        if inference_device.type == "cuda":
            cuda_devices = [
                inference_device.index
                if inference_device.index is not None
                else torch.cuda.current_device()
            ]
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(int(self.config.inference_seed))
            if inference_device.type == "cuda":
                torch.cuda.manual_seed_all(int(self.config.inference_seed))
            policy_net = self._algo.nets["policy"]
            features, _ = policy_net.forward_features("piper_left_arm", data)
            if self._algo.diffusion:
                features = (features, "piper_left_arm")
            pred_norm, pred_raw, output_head = self._predict_joint_actions(
                features, data
            )
        pred_norm_np = pred_norm[0].detach().cpu().numpy()
        pred = (
            pred_raw[0].detach().cpu().numpy()
            if pred_raw is not None
            else self._algo.norm_stats.unnormalize(
                {"actions_joints": pred_norm}, self._embodiment_id
            )["actions_joints"][0].detach().cpu().numpy()
        )
        inference_ms = (time.perf_counter() - inference_started) * 1e3

        if pred.ndim != 2 or pred.shape[1] != 7:
            raise RuntimeError(f"expected joint action (T,7), got {pred.shape}")
        if pred_norm_np.shape != pred.shape or not np.isfinite(pred_norm_np).all():
            raise RuntimeError("normalized joint action is malformed or non-finite")
        if not np.isfinite(pred).all():
            raise RuntimeError("physical joint action contains NaN or Inf")

        pred_for_control = (
            smooth_joint_action_chunk(
                pred,
                self.config.chunk_smoothing_window,
                self.config.chunk_smoothing_sigma,
            )
            if self.config.chunk_smoothing
            else pred.copy()
        )
        raw_adjacent_rms = adjacent_joint_rms(pred)
        smoothed_adjacent_rms = adjacent_joint_rms(pred_for_control)

        selected_control_ticks = (
            self.config.action_horizon_frames
            if self.config.temporal_ensemble
            else self.config.query_frequency
        )
        query_indices = model_indices_for_control_ticks(
            pred.shape[0], self.config.action_horizon_frames, selected_control_ticks
        )
        full_control_indices = model_indices_for_control_ticks(
            pred.shape[0],
            self.config.action_horizon_frames,
            self.config.action_horizon_frames,
        )
        full_control_gripper_raw = pred[full_control_indices, 6].copy()
        full_control_gripper_commands_pct = np.where(
            full_control_gripper_raw >= self.config.gripper_threshold,
            100.0,
            0.0,
        ).astype(np.float32)
        selected_unsmoothed = pred[query_indices].copy()
        selected = pred_for_control[query_indices].copy()
        margin = float(self.config.joint_limit_margin_rad)
        low = self._joint_low_rad + margin
        high = self._joint_high_rad - margin
        if np.any(high <= low):
            raise RuntimeError("joint_limit_margin_rad collapses a calibrated joint range")
        before_clip = selected[:, :6].copy()
        selected[:, :6] = np.clip(selected[:, :6], low, high)
        clipped_mask = np.abs(selected[:, :6] - before_clip) > 1e-9
        selected[:, 6] = (selected[:, 6] >= self.config.gripper_threshold).astype(np.float32)

        raw_driver_actions = np.empty_like(selected, dtype=np.float32)
        raw_driver_actions[:, :6] = np.stack(
            [self._joint_rad_to_driver(row) for row in selected[:, :6]], axis=0
        ).astype(np.float32)
        raw_driver_actions[:, 6] = selected[:, 6] * 100.0
        current_driver = self._joint_rad_to_driver(obs_joint)
        preview_driver_actions = self._preview_filtered_chunk(
            raw_driver_actions, current_driver
        )

        self._chunk_sequence_id += 1
        self.last_prediction_report = {
            "policy_space": "absolute_joint_radians_plus_binary_gripper",
            "checkpoint_contract": self.checkpoint_report,
            "output_head": output_head,
            "proprio_physical": proprio_physical.tolist(),
            "proprio_model_input": proprio_model_input.tolist(),
            "model_chunk_shape": list(pred.shape),
            "query_frequency": int(self.config.query_frequency),
            "selected_control_ticks": int(selected_control_ticks),
            "temporal_ensemble": bool(self.config.temporal_ensemble),
            "temporal_replan_interval": int(self.config.temporal_replan_interval),
            "temporal_ensemble_decay": float(self.config.temporal_ensemble_decay),
            "chunk_smoothing": bool(self.config.chunk_smoothing),
            "chunk_smoothing_window": int(self.config.chunk_smoothing_window),
            "chunk_smoothing_sigma": float(self.config.chunk_smoothing_sigma),
            "chunk_raw_adjacent_joint_rms_rad_mean": raw_adjacent_rms[0],
            "chunk_raw_adjacent_joint_rms_rad_max": raw_adjacent_rms[1],
            "chunk_smoothed_adjacent_joint_rms_rad_mean": smoothed_adjacent_rms[0],
            "chunk_smoothed_adjacent_joint_rms_rad_max": smoothed_adjacent_rms[1],
            "query_indices": query_indices,
            "full_control_indices": full_control_indices,
            "full_control_gripper_raw": full_control_gripper_raw.tolist(),
            "full_control_gripper_commands_pct": (
                full_control_gripper_commands_pct.tolist()
            ),
            "num_inference_steps": int(self.config.num_inference_steps),
            "inference_seed": int(self.config.inference_seed),
            "model_inference_ms": inference_ms,
            "joint_limit_clipped_values": int(clipped_mask.sum()),
            "joint_limit_clipped_steps": int(clipped_mask.any(axis=1).sum()),
            "full_model_chunk_columns": [
                "joint_1_rad",
                "joint_2_rad",
                "joint_3_rad",
                "joint_4_rad",
                "joint_5_rad",
                "joint_6_rad",
                "gripper_open",
            ],
            "full_model_chunk_normalized": pred_norm_np.tolist(),
            "full_model_chunk_unnormalized": pred.tolist(),
            "full_model_chunk_smoothed": pred_for_control.tolist(),
            "selected_physical_actions_unsmoothed": selected_unsmoothed.tolist(),
            "selected_physical_actions": selected.tolist(),
            "selected_raw_driver_actions": raw_driver_actions.tolist(),
            "selected_driver_actions": preview_driver_actions.tolist(),
            "trajectory_filter": {
                "max_joint_command_step": float(self.config.max_joint_command_step),
                "max_joint_command_accel": float(self.config.max_joint_command_accel),
                "current_driver_anchor": current_driver.tolist(),
            },
        }
        return torch.from_numpy(raw_driver_actions)

    @torch.no_grad()
    def peek_action(self, batch: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        if self.config.temporal_ensemble:
            should_replan = (
                self._last_temporal_replan_tick is None
                or self._policy_tick - self._last_temporal_replan_tick
                >= self.config.temporal_replan_interval
            )
            if should_replan:
                chunk = self.predict_action_chunk(batch, **kwargs).detach().cpu().numpy()
                self._temporal_plans.append(
                    (self._chunk_sequence_id, self._policy_tick, chunk.copy())
                )
                self._last_temporal_replan_tick = self._policy_tick
                self._active_chunk_size = len(chunk)
                self._active_chunk_consumed = 0
            raw_action, self.last_ensemble_report = absolute_time_joint_ensemble(
                self._temporal_plans,
                self._policy_tick,
                self.config.temporal_ensemble_decay,
            )
        elif not self._action_queue:
            chunk = self.predict_action_chunk(batch, **kwargs)
            self._action_queue.extend(row.detach().cpu() for row in chunk)
            self._active_chunk_size = len(self._action_queue)
            self._active_chunk_consumed = 0
            raw_action = self._action_queue[0].numpy()
            self.last_ensemble_report = None
        else:
            raw_action = self._action_queue[0].numpy()
            self.last_ensemble_report = None
        raw_state_value = batch["observation.state"]
        if isinstance(raw_state_value, torch.Tensor):
            raw_state = raw_state_value.detach().cpu().numpy()
        else:
            raw_state = np.asarray(raw_state_value)
        raw_state = raw_state.reshape(-1)[:7].astype(np.float64)
        if raw_state.shape != (7,) or not np.isfinite(raw_state).all():
            raise RuntimeError(f"invalid raw PIPER state while filtering: {raw_state}")
        current_joint_rad = raw_state[:6] * 1e-3 * np.pi / 180.0
        current_driver = self._joint_rad_to_driver(current_joint_rad)
        safe_action, pending_velocity = self._filter_driver_action(
            raw_action,
            current_driver,
            self._last_planned_velocity_driver,
        )
        self._pending_planned_velocity_driver = pending_velocity
        return torch.from_numpy(safe_action)

    @torch.no_grad()
    def advance_action(self) -> None:
        if not self.config.temporal_ensemble and not self._action_queue:
            raise RuntimeError("cannot advance an empty action queue")
        if self._pending_planned_velocity_driver is None:
            raise RuntimeError("cannot advance before peeking a filtered action")
        if self.config.temporal_ensemble:
            self._policy_tick += 1
            self._active_chunk_consumed = (
                self._policy_tick - int(self._last_temporal_replan_tick or 0)
            )
            self._temporal_plans = [
                plan
                for plan in self._temporal_plans
                if plan[1] + len(plan[2]) > self._policy_tick
            ]
        else:
            self._action_queue.popleft()
            self._active_chunk_consumed += 1
        self._last_planned_velocity_driver = self._pending_planned_velocity_driver
        self._pending_planned_velocity_driver = None

    @torch.no_grad()
    def cancel_pending_action(self) -> None:
        """Discard an uncommitted filter preview without consuming its action.

        The deployment runtime calls this only when a command was materially
        changed after filtering.  The same action is then recomputed from the
        next fresh feedback sample instead of committing a velocity state that
        the robot never received.
        """
        self._pending_planned_velocity_driver = None

    @torch.no_grad()
    def select_action(self, batch: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        action = self.peek_action(batch, **kwargs)
        self.advance_action()
        return action

    def queue_status(self) -> dict[str, int | bool]:
        if self.config.temporal_ensemble:
            vote_count = (
                int(self.last_ensemble_report["vote_count"])
                if self.last_ensemble_report is not None
                else 0
            )
            return {
                "chunk_id": int(self._chunk_sequence_id),
                "chunk_size": int(self.config.action_horizon_frames),
                "action_offset": int(self._active_chunk_consumed),
                "pending_actions": int(
                    max(0, self.config.action_horizon_frames - self._active_chunk_consumed)
                ),
                "temporal_ensemble": True,
                "policy_tick": int(self._policy_tick),
                "active_plan_count": int(len(self._temporal_plans)),
                "ensemble_vote_count": vote_count,
            }
        return {
            "chunk_id": int(self._chunk_sequence_id),
            "chunk_size": int(self._active_chunk_size),
            "action_offset": int(self._active_chunk_consumed),
            "pending_actions": int(len(self._action_queue)),
            "temporal_ensemble": False,
        }
