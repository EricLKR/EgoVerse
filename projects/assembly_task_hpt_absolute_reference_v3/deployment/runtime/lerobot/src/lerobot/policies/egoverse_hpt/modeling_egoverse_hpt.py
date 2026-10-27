#!/usr/bin/env python
"""lerobot PreTrainedPolicy wrapper around an EgoVerse HPT checkpoint.

Reuses, rather than reimplements, the pieces already validated elsewhere:
  - egomimic.rldb.embodiment.piper.Piper.get_transform_list -- same transform
    pipeline used at training time, applied here to one live frame instead of
    a zarr-loaded batch.
  - HPT._robomimic_to_hpt_data / HPTModel.forward -- the model's own
    data-plumbing and inference call, exactly as egomimic's rollout.py and
    eval_hpt.py already use it.
  - PiperKinematicsSolver.fk / .ik (egomimic/robot/piper/piper_kinematics.py)
    -- FK for the observation, IK for turning the predicted cartesian action
    back into a joint command.

Only tested offline (cross-checked against egomimic's own eval_hpt.py on
recorded validation data) as of writing -- NOT yet run against the real arm.
See the module-level safety notes in select_action.
"""

from __future__ import annotations

import logging
import time
from collections import deque

import numpy as np
import torch
import torch.nn.functional
from scipy.spatial.transform import Rotation as ScipyRotation

from lerobot.policies.egoverse_hpt.configuration_egoverse_hpt import EgoVerseHPTConfig
from lerobot.policies.pretrained import PreTrainedPolicy

logger = logging.getLogger(__name__)


def model_indices_for_control_ticks(
    num_model_steps: int, action_horizon_frames: int, query_frequency: int
) -> list[int]:
    """Map raw 30-Hz control offsets to the model's resampled chunk.

    This is intentionally identical to PiperRollout and
    piper_plot_checkpoint_eval.py: ``round(offset * T / raw_horizon)``.
    Directly taking ``pred[:query_frequency]`` is wrong when T != raw_horizon.
    """
    if num_model_steps <= 0 or action_horizon_frames <= 0:
        raise ValueError("model and raw action horizons must be positive")
    if not 0 < query_frequency <= action_horizon_frames:
        raise ValueError("query_frequency must be in (0, action_horizon_frames]")
    return [
        min(num_model_steps - 1, int(round(offset * num_model_steps / action_horizon_frames)))
        for offset in range(query_frequency)
    ]


def prepare_rgb_chw_image(
    image: np.ndarray | torch.Tensor,
    *,
    expected_height: int,
    expected_width: int,
    resize_mismatch: bool,
    name: str,
) -> torch.Tensor:
    """Return one RGB float32 CHW image in [0, 1] with strict shape checks."""
    tensor = image.detach().cpu() if isinstance(image, torch.Tensor) else torch.as_tensor(image)
    if tensor.ndim == 4:
        if tensor.shape[0] != 1:
            raise ValueError(f"{name} must contain exactly one live frame, got {tuple(tensor.shape)}")
        tensor = tensor[0]
    if tensor.ndim != 3:
        raise ValueError(f"{name} must be CHW or HWC, got {tuple(tensor.shape)}")
    if tensor.shape[0] in (3, 4):
        tensor = tensor[:3]
    elif tensor.shape[-1] in (3, 4):
        tensor = tensor[..., :3].permute(2, 0, 1)
    else:
        raise ValueError(f"{name} has no recognizable RGB channel dimension: {tuple(tensor.shape)}")
    tensor = tensor.contiguous().float()
    if tensor.numel() and tensor.max() > 1.5:
        tensor = tensor / 255.0
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains NaN or Inf")
    actual_hw = tuple(tensor.shape[-2:])
    expected_hw = (expected_height, expected_width)
    if actual_hw != expected_hw:
        if not resize_mismatch:
            raise ValueError(
                f"{name} is {actual_hw}, but the checkpoint was trained on {expected_hw}. "
                "Configure the camera to the training resolution or explicitly enable resizing."
            )
        tensor = torch.nn.functional.interpolate(
            tensor.unsqueeze(0), size=expected_hw, mode="bilinear", align_corners=False
        )[0]
    return tensor.clamp_(0.0, 1.0)


class EgoVerseHPTPolicy(PreTrainedPolicy):
    config_class = EgoVerseHPTConfig
    name = "egoverse_hpt"

    def __init__(self, config: EgoVerseHPTConfig):
        super().__init__(config)
        # Heavy imports deferred to construction time -- this module is only
        # ever loaded in an env with egomimic installed (EgoVerse's `emimic`
        # venv), and importing at module level would break `--help` / policy
        # discovery in envs that don't have it.
        from egomimic.rldb.embodiment.embodiment import get_embodiment_id
        from egomimic.rldb.embodiment.piper import Piper as PiperEmbodiment
        from egomimic.robot.piper.piper_kinematics import PiperKinematicsSolver

        self._PiperEmbodiment = PiperEmbodiment
        self._embodiment_id = get_embodiment_id("piper_left_arm")

        low = np.array([p[0] for p in config.joint_limits_deg], dtype=np.float64)
        high = np.array([p[1] for p in config.joint_limits_deg], dtype=np.float64)
        if np.any(high <= low):
            raise ValueError(f"joint_limits_deg has a non-increasing (min, max) pair: {config.joint_limits_deg}")

        # Real bug found via hardware testing: the generic agilex reference
        # URDF's <limit> tags are idealized/nominal (e.g. exactly [0, 180] deg
        # for joint2) and can be a couple degrees narrower than what a real
        # unit's mechanical stops actually allow. IK's retry-clamping
        # (torch.clamp against self.chain.low/high) then silently snaps any
        # retry landing in that gap -- INCLUDING the seed itself -- to the
        # boundary, which can make even a trivial "hold current pose" IK
        # solve fail to converge if the arm happens to be resting right at
        # that boundary (as it was: joint2=-2.06 deg, joint3=1.98 deg, both
        # valid per this unit's real calibration but outside the URDF's
        # nominal limits). Pass THIS unit's real calibrated limits (the same
        # ones already required for the [-100,100] command conversion below)
        # so IK's joint-limit enforcement matches reality instead of the
        # idealized URDF.
        self._kinematics = PiperKinematicsSolver(
            device="cpu", joint_limits_rad=(np.radians(low), np.radians(high))
        )
        # NOTE: obs-only. Piper's "cartesian_base" and "cartesian_wristframe"
        # modes build IDENTICAL observations.state.ee_pose (both leave obs in
        # base_link frame -- only the *action* differs between the two); this
        # is fixed at "cartesian_base" regardless of config.action_frame for
        # that reason. The dummy "actions_cartesian" it also produces here is
        # never consumed (see below).
        self._transform_list = PiperEmbodiment.get_transform_list(mode="cartesian_base")

        self._action_queue: deque = deque()
        self._chunk_sequence_id = 0
        self._active_chunk_size = 0
        self._active_chunk_consumed = 0
        self._consecutive_ik_failures = 0
        self.last_prediction_report: dict | None = None

        self._joint_low_deg = low
        self._joint_high_deg = high

        # "camframe" ckpts predict T_cam<-ee directly; reverting to base_link
        # for IK is a fixed-matrix multiply (T_base<-ee = T_base<-cam @
        # T_cam<-ee), no live anchor needed (unlike wristframe) -- see
        # predict_action_chunk. Load once here rather than per-call.
        self._cam_to_base_mat: np.ndarray | None = None
        if config.action_frame == "camframe":
            extrinsics = PiperEmbodiment.EXTRINSICS
            if extrinsics is None or config.front_camera_key not in extrinsics:
                raise ValueError(
                    f"action_frame='camframe' needs Piper.EXTRINSICS[{config.front_camera_key!r}] "
                    "(the calibrated camera<-base extrinsics this ckpt was trained with), but it's "
                    f"missing. Available keys: {list(extrinsics.keys()) if extrinsics else None}. "
                    "See egomimic/scripts/calibrate_camera/calibrate_piper_extrinsics.py."
                )
            self._cam_to_base_mat = extrinsics[config.front_camera_key].astype(np.float64)

        # Loaded here (not only in from_pretrained): `make_policy` calls
        # `cls(config)` directly for the `--policy.type=egoverse_hpt` fresh-
        # construction path (as opposed to `--policy.path=<dir>`), which
        # would otherwise skip weight loading entirely and silently run on
        # an untrained model. `ckpt_path` is a required config field either
        # way, so loading eagerly here is always correct and makes both
        # invocation styles work.
        from egomimic.pl_utils.pl_model import ModelWrapper

        logger.info(f"[egoverse_hpt] Loading EgoVerse checkpoint from {config.ckpt_path}")
        model = ModelWrapper.load_from_checkpoint(
            config.ckpt_path, weights_only=False, map_location=config.device
        )
        model.to(config.device)
        model.eval()
        model.model.device = config.device
        self._egoverse_model = model

        # Match the validated deployment-time override rather than silently
        # using the training config's integration count.
        if getattr(model.model, "diffusion", False):
            from egomimic.models.denoising_policy import DenoisingPolicy

            for head_name, head in model.model.nets["policy"].heads.items():
                if isinstance(head, DenoisingPolicy):
                    head.num_inference_steps = config.num_inference_steps
                    logger.info(
                        f"[egoverse_hpt] diffusion=true checkpoint -- set "
                        f"heads[{head_name!r}].num_inference_steps = "
                        f"{config.num_inference_steps}"
                    )

    # ------------------------------------------------------------------
    # Overridden to skip lerobot's HF-hub/safetensors loading mechanism --
    # the EgoVerse checkpoint is a PyTorch Lightning .ckpt loaded above in
    # __init__ (which from_pretrained's default implementation already calls
    # via `cls(config, **kwargs)`), not a bare state_dict this class would
    # otherwise try to fetch/load a second time.
    # ------------------------------------------------------------------
    @classmethod
    def from_pretrained(cls, pretrained_name_or_path, *, config: EgoVerseHPTConfig | None = None, **kwargs):
        if config is None:
            config = EgoVerseHPTConfig.from_pretrained(pretrained_name_or_path, **kwargs)
        return cls(config)

    # ------------------------------------------------------------------
    def reset(self):
        self._action_queue.clear()
        self._chunk_sequence_id = 0
        self._active_chunk_size = 0
        self._active_chunk_consumed = 0
        self._consecutive_ik_failures = 0
        self.last_prediction_report = None

    def get_optim_params(self) -> dict:
        raise NotImplementedError("EgoVerseHPTPolicy is inference-only -- train via egomimic/trainHydra.py.")

    def forward(self, batch: dict[str, torch.Tensor]):
        raise NotImplementedError(
            "EgoVerseHPTPolicy is inference-only (wraps an already-trained EgoVerse checkpoint). "
            "Train/fine-tune via egomimic/trainHydra.py, not lerobot-train."
        )

    # ------------------------------------------------------------------
    def _rad_to_normalized_deg(self, joint_rad: np.ndarray) -> np.ndarray:
        """Real joint angles (radians) -> lerobot/Piper's [-100, 100] command
        space, using THIS unit's calibrated limits (config.joint_limits_deg).
        Inverse of piper_sdk_interface.py's raw_deg001_to_norm."""
        deg = np.degrees(joint_rad)
        span = self._joint_high_deg - self._joint_low_deg
        norm = (deg - self._joint_low_deg) / span * 200.0 - 100.0
        return np.clip(norm, -100.0, 100.0)

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        """Runs one live observation through FK -> HPT -> IK. Returns a
        (n_action_steps, 7) tensor: 6 joints + gripper, in the SAME
        normalized [-100, 100]/[0, 100] space lerobot's Piper.send_action
        expects (matching the recorded dataset's `action` convention)."""
        from egomimic.robot.piper.piper_kinematics import raw_gripper_to_frac, raw_joint_to_rad
        from egomimic.utils.pose_utils import _xyz6d_to_matrix, _xyzwxyz_to_matrix

        device = self.config.device
        algo = self._egoverse_model.model  # HPT instance

        def _to_numpy_cpu(x) -> np.ndarray:
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
            return np.asarray(x)

        # --- 1. Live observation -> FK -> the same pre-transform batch shape
        #     the Piper zarr conversion / key_map produce. ---
        joint_raw = _to_numpy_cpu(batch["observation.state"]).reshape(-1)[:7].astype(np.float64)
        joint_rad = raw_joint_to_rad(joint_raw[:6])
        gripper_frac = raw_gripper_to_frac(joint_raw[6:7])
        obs_ee_pose = self._kinematics.fk(joint_rad.astype(np.float32))  # (7,) XYZWXYZ

        horizon = self._PiperEmbodiment.ACTION_HORIZON
        front_key = f"observation.images.{self.config.front_camera_key}"
        wrist_key = f"observation.images.{self.config.wrist_camera_key}"
        front_img = prepare_rgb_chw_image(
            batch[front_key],
            expected_height=self.config.expected_image_height,
            expected_width=self.config.expected_image_width,
            resize_mismatch=self.config.resize_mismatched_images,
            name=front_key,
        )
        wrist_img = prepare_rgb_chw_image(
            batch[wrist_key],
            expected_height=self.config.expected_image_height,
            expected_width=self.config.expected_image_width,
            resize_mismatch=self.config.resize_mismatched_images,
            name=wrist_key,
        )

        pre_transform_batch = {
            "obs_ee_pose": torch.from_numpy(obs_ee_pose).float(),
            "obs_gripper": torch.from_numpy(gripper_frac).float(),
            # Dummy action-chunk placeholders: required by transform_list's key
            # names, never actually consumed by HPTModel.forward (only
            # HPT.compute_loss reads "action", which this policy never calls).
            "action_ee_pose": torch.from_numpy(obs_ee_pose).float().unsqueeze(0).repeat(horizon, 1),
            "action_gripper": torch.from_numpy(gripper_frac).float().unsqueeze(0).repeat(horizon, 1),
        }
        for transform in self._transform_list:
            pre_transform_batch = transform.transform(pre_transform_batch)
        # pre_transform_batch now has "observations.state.ee_pose" (real) and
        # "actions_cartesian" (dummy, unused) from Piper's own transform list.

        hpt_batch = {
            "observations.images.front_img_1": front_img.unsqueeze(0).to(device),
            "observations.images.wrist_img": wrist_img.unsqueeze(0).to(device),
            "observations.state.ee_pose": pre_transform_batch["observations.state.ee_pose"]
            .unsqueeze(0)
            .to(device),
            "actions_cartesian": pre_transform_batch["actions_cartesian"].unsqueeze(0).to(device),
            "pad_mask": torch.ones((1, horizon, 1), dtype=torch.bool, device=device),
            "embodiment": torch.tensor([self._embodiment_id], device=device),
        }

        # MultiDataset normalizes proprioception with the checkpoint's own
        # quantile statistics during training and validation.  Live deployment
        # constructs this batch directly, so it must apply that same
        # normalization explicitly before HPT sees it.  Without this step the
        # model receives physical xyz/rotation/gripper values in a feature
        # space where it was trained on normalized values.
        proprio_key = "observations.state.ee_pose"
        proprio_physical = hpt_batch[proprio_key].detach().cpu().numpy()
        if self.config.normalize_proprio_input:
            proprio_stats = algo.norm_stats.norm_stats.get(self._embodiment_id, {}).get(
                proprio_key
            )
            if proprio_stats is None:
                raise RuntimeError(
                    f"checkpoint has no normalization statistics for {proprio_key}"
                )
            hpt_batch = algo.norm_stats.normalize(hpt_batch, self._embodiment_id)
        proprio_model_input = hpt_batch[proprio_key].detach().cpu().numpy()
        if not np.isfinite(proprio_model_input).all():
            raise RuntimeError("normalized HPT proprio input contains NaN or Inf")

        # --- 2. HPT forward pass (same call rollout.py / eval_hpt.py use). ---
        ac_key = algo.ac_keys[self._embodiment_id]
        data = algo._robomimic_to_hpt_data(
            hpt_batch,
            algo.camera_keys[self._embodiment_id],
            algo.proprio_keys[self._embodiment_id],
            algo.lang_keys[self._embodiment_id],
            ac_key,
            algo.auxiliary_ac_keys.get("piper_left_arm", []),
        )
        inference_started = time.perf_counter()
        action_out = algo.nets["policy"].forward("piper_left_arm", data)
        pred_key = "shared" if algo.shared_ac_key else "piper_left_arm"
        pred_norm = action_out[pred_key]  # (1, T, 10), normalized (model's own training-time norm)

        unnorm = algo.norm_stats.unnormalize({algo.shared_ac_key: pred_norm}, self._embodiment_id)
        pred_norm_np = pred_norm[0].detach().cpu().numpy()
        pred = (
            unnorm[algo.shared_ac_key][0].detach().cpu().numpy()
        )  # (T, 10): xyz(3) + 6D rot(6) + gripper(1)
        inference_ms = (time.perf_counter() - inference_started) * 1e3
        if pred.ndim != 2 or pred.shape[1] != 10:
            raise RuntimeError(f"expected HPT action shape (T, 10), got {pred.shape}")
        if pred_norm_np.shape != pred.shape:
            raise RuntimeError(
                f"normalized/unnormalized HPT action shapes differ: {pred_norm_np.shape} vs {pred.shape}"
            )
        if not np.isfinite(pred_norm_np).all():
            raise RuntimeError("normalized HPT action chunk contains NaN or Inf; refusing to log/use it")
        if not np.isfinite(pred).all():
            raise RuntimeError("HPT action chunk contains NaN or Inf; refusing to run IK")

        # --- 3. Predicted cartesian action -> joint command via IK, one step
        #     at a time, each seeded from the previous solution within this
        #     chunk. ---
        query_frequency = (
            self.config.n_action_steps
            if self.config.n_action_steps is not None
            else self.config.query_frequency
        )
        query_indices = model_indices_for_control_ticks(
            pred.shape[0], self.config.action_horizon_frames, query_frequency
        )
        # Seed from the FRESH, actually-sensed joint position (this call's
        # own `joint_rad`, from `batch["observation.state"]`), never from
        # `self._last_joint_rad` (the previous chunk's own predicted/IK'd
        # target). Real bug found via hardware testing: the robot layer's
        # `max_relative_target` safety clamp means the arm frequently does
        # NOT reach wherever the previous chunk asked for -- seeding the next
        # chunk's IK from that unexecuted assumption (rather than reality)
        # let the policy's internal notion of "current pose" drift further
        # and further from the real arm state every time a clamp fired,
        # producing ever-larger commanded jumps that got clamped even harder
        # -- a self-reinforcing loop that looked like the arm just stalling
        # partway through a reach instead of completing it.
        seed = joint_rad.astype(np.float32)
        # Anchor for "wristframe" ckpts: the SAME single observed pose used as
        # `target_world` when the model was trained (see
        # Piper.get_transform_list(mode="cartesian_wristframe")) -- i.e. the
        # live FK pose from step 1 above, fixed for this whole chunk (not
        # re-anchored per step; matches how the training-time transform only
        # ever takes one anchor per chunk).
        obs_mat = _xyzwxyz_to_matrix(obs_ee_pose[None, :].astype(np.float64))[0]
        joint_actions = []
        step_reports = []
        ik_started = time.perf_counter()
        for raw_offset, model_index in enumerate(query_indices):
            xyz = pred[model_index, :3]
            c1_c2 = pred[model_index, 3:9]
            raw_gripper = float(np.clip(pred[model_index, 9], 0.0, 1.0))
            gripper_t = float(raw_gripper >= self.config.gripper_threshold)

            invalid_reason = None
            try:
                pred_mat = _xyz6d_to_matrix(
                    np.concatenate([xyz, c1_c2])[None, :].astype(np.float64)
                )[0]
                if self.config.action_frame == "wristframe":
                    # Revert: absolute base_link pose = anchor @ predicted_delta.
                    abs_mat = obs_mat @ pred_mat
                elif self.config.action_frame == "camframe":
                    # T_base<-ee = T_base<-cam @ T_cam<-ee.
                    abs_mat = self._cam_to_base_mat @ pred_mat
                else:
                    abs_mat = pred_mat
                if not np.isfinite(abs_mat).all():
                    raise ValueError("pose matrix contains NaN or Inf")
                quat_xyzw = ScipyRotation.from_matrix(abs_mat[:3, :3]).as_quat()
                target_pose = np.concatenate(
                    [abs_mat[:3, 3], quat_xyzw[[3, 0, 1, 2]]]
                )  # XYZWXYZ
                if self.config.workspace_xyz_min is not None:
                    xyz_min = np.asarray(self.config.workspace_xyz_min, dtype=np.float64)
                    xyz_max = np.asarray(self.config.workspace_xyz_max, dtype=np.float64)
                    if np.any(target_pose[:3] < xyz_min) or np.any(target_pose[:3] > xyz_max):
                        invalid_reason = (
                            f"base-frame xyz {target_pose[:3].tolist()} is outside "
                            f"[{xyz_min.tolist()}, {xyz_max.tolist()}]"
                        )
            except (ValueError, np.linalg.LinAlgError) as exc:
                invalid_reason = f"invalid Cartesian pose: {exc}"
                target_pose = None

            if invalid_reason is None:
                q_sol, converged, pos_err = self._kinematics.ik(
                    target_pose,
                    seed_joint_angles_rad=seed,
                    num_retries=self.config.ik_num_retries,
                    pos_tolerance=self.config.ik_pos_tolerance,
                    rot_tolerance=self.config.ik_rot_tolerance,
                    seed_noise_std=self.config.ik_seed_noise_std,
                )
            else:
                q_sol, converged, pos_err = seed, False, float("nan")

            if not converged:
                self._consecutive_ik_failures += 1
                logger.warning(
                    f"[egoverse_hpt] unsafe target or IK failure "
                    f"(reason={invalid_reason or 'IK non-convergence'}, pos_err={pos_err:.4f}m) "
                    f"at raw offset {raw_offset}/model index {model_index}; "
                    f"holding previous joint target instead of executing this step "
                    f"({self._consecutive_ik_failures}/{self.config.max_consecutive_ik_failures} consecutive)."
                )
                q_sol = seed  # SAFETY: never execute an unvalidated IK solution -- hold instead.
                if (
                    self.config.max_consecutive_ik_failures > 0
                    and self._consecutive_ik_failures >= self.config.max_consecutive_ik_failures
                ):
                    raise RuntimeError(
                        f"[egoverse_hpt] {self._consecutive_ik_failures} consecutive IK failures -- "
                        "stopping rather than continuing to hold indefinitely. Likely causes: the "
                        "policy is predicting poses outside the arm's reachable workspace, or "
                        "joint_limits_deg doesn't match this unit's actual calibration."
                    )
            else:
                self._consecutive_ik_failures = 0

            seed = q_sol
            norm_joints = self._rad_to_normalized_deg(q_sol)
            norm_gripper = gripper_t * 100.0
            joint_actions.append(np.concatenate([norm_joints, [norm_gripper]]))
            step_reports.append(
                {
                    "raw_offset": raw_offset,
                    "model_index": model_index,
                    "predicted_xyz_base_m": None if target_pose is None else target_pose[:3].tolist(),
                    "raw_gripper_prediction": raw_gripper,
                    "binary_gripper_command": int(gripper_t),
                    "ik_converged": bool(converged),
                    "ik_position_error_m": None if not np.isfinite(pos_err) else float(pos_err),
                    "rejection_reason": invalid_reason,
                    "normalized_joint_command": norm_joints.tolist(),
                }
            )

        self.last_prediction_report = {
            "action_frame": self.config.action_frame,
            "proprio_normalization_applied": bool(self.config.normalize_proprio_input),
            "proprio_physical": proprio_physical.tolist(),
            "proprio_model_input": proprio_model_input.tolist(),
            "model_chunk_shape": list(pred.shape),
            "action_horizon_frames": self.config.action_horizon_frames,
            "query_frequency": query_frequency,
            "query_indices": query_indices,
            "num_inference_steps": self.config.num_inference_steps,
            "model_inference_ms": inference_ms,
            "ik_total_ms": (time.perf_counter() - ik_started) * 1e3,
            "full_model_chunk_columns": [
                "x",
                "y",
                "z",
                "rotation_6d_0",
                "rotation_6d_1",
                "rotation_6d_2",
                "rotation_6d_3",
                "rotation_6d_4",
                "rotation_6d_5",
                "gripper",
            ],
            # Exact 100x10 network output before normalization statistics are
            # inverted. Kept for diagnosing saturation or distribution shift.
            "full_model_chunk_normalized": pred_norm_np.tolist(),
            # Exact 100x10 physical-space prediction consumed below by the
            # Cartesian-to-joint conversion. For action_frame="absolute", xyz
            # is in base_link metres and gripper is the raw continuous value
            # before clipping, thresholding, or binary conversion.
            "full_model_chunk_unnormalized": pred.tolist(),
            "steps": step_reports,
        }
        return torch.from_numpy(np.stack(joint_actions).astype(np.float32))

    @torch.no_grad()
    def queue_status(self) -> dict[str, int]:
        """Return progress through the currently buffered control actions.

        ``action_offset`` is the zero-based action currently at the head of
        the queue. When the queue is empty after consuming a chunk it equals
        ``chunk_size`` until the next call to :meth:`peek_action` generates a
        new chunk.
        """
        return {
            "chunk_id": int(self._chunk_sequence_id),
            "chunk_size": int(self._active_chunk_size),
            "action_offset": int(self._active_chunk_consumed),
            "pending_actions": int(len(self._action_queue)),
        }

    @torch.no_grad()
    def peek_action(self, batch: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        """Return the next queued action without consuming it.

        Deployment can use this together with :meth:`advance_action` to make
        queue progress depend on measured robot feedback. The existing
        :meth:`select_action` method below retains its consume-every-call
        behavior for lerobot compatibility and offline evaluation.
        """
        # `last_prediction_report` is an event payload, not persistent state:
        # deployment logs should contain the large 100-step report only on a
        # frame that actually ran inference, not duplicate it while a queued
        # target is held for multiple feedback cycles.
        self.last_prediction_report = None
        if len(self._action_queue) == 0:
            chunk = self.predict_action_chunk(batch, **kwargs)
            self._action_queue.extend(chunk)
            self._chunk_sequence_id += 1
            self._active_chunk_size = len(chunk)
            self._active_chunk_consumed = 0
        return self._action_queue[0]

    def advance_action(self) -> torch.Tensor:
        """Consume one action after the caller confirms feedback progress."""
        if len(self._action_queue) == 0:
            raise RuntimeError("cannot advance an empty EgoVerse action queue")
        action = self._action_queue.popleft()
        self._active_chunk_consumed += 1
        return action

    @torch.no_grad()
    def select_action(self, batch: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        """Returns ONE action (7,), normalized, ready for
        lerobot Piper.send_action. Refills the internal queue by calling
        predict_action_chunk when empty -- same action-queue pattern as
        lerobot's own ACTPolicy.

        Safety note: this method can raise (see predict_action_chunk) after
        repeated IK non-convergence rather than silently execute a bad
        command -- the caller (lerobot-record's control loop) will stop on
        that exception. This is intentional: for a freshly-built, not-yet
        hardware-tested policy, stopping loudly is the correct failure mode.
        """
        action = self.peek_action(batch, **kwargs)
        self.advance_action()
        return action
