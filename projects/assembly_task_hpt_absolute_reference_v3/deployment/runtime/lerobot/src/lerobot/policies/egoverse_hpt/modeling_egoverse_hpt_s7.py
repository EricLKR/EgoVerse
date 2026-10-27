#!/usr/bin/env python
"""EgoVerseHPTS7Policy: EgoVerseHPTPolicy + live S7 short-horizon effect token, for P2/P3 checkpoints
(trained with an `s7_token` proprio stem -- see egoverse_jepa's phase_c_s7_precompute.py /
phase_c_copy_inject.py / hpt_cotrain_piper_human_s7.yaml). The base EgoVerseHPTPolicy class is untouched;
this is a new, additive subclass so plain P0-style checkpoints keep using the simpler, already-verified
base adapter unchanged.

S7 itself runs in a SEPARATE persistent subprocess (see s7_worker_bridge.py) using egoverse_jepa's own
`egomimic` fork -- confirmed directly that fork and this deployment env's `egomimic` (needed for real
Piper robot control) cannot coexist under one import name in one process (hpt_nets.py alone has 165 lines
of real divergence between them).

IMPORTANT, disclosed limitation: this has been verified only via offline dry-run (a real recorded frame,
not a live camera feed) -- the raw camera array's exact shape/dtype convention for a camera key
(s7_source_camera_key) that isn't part of the trained policy's own image pipeline has NOT been confirmed
against the real lerobot camera capture path. `_extract_raw_frame` is written defensively (handles both
CHW and HWC, uint8 and float) but the FIRST real-hardware run should sanity-check this before trusting any
downstream prediction -- e.g. by logging the raw array's .shape/.dtype on the first call (already done,
see the one-time log line below) and confirming it matches expectations before proceeding to a real
rollout.
"""
from __future__ import annotations

import logging
import time
from collections import deque

import numpy as np
import torch
from scipy.spatial.transform import Rotation as ScipyRotation

from lerobot.policies.egoverse_hpt.configuration_egoverse_hpt_s7 import EgoVerseHPTS7Config
from lerobot.policies.egoverse_hpt.modeling_egoverse_hpt import EgoVerseHPTPolicy

logger = logging.getLogger(__name__)

# Native Aria-format resolution S7's frozen resnet was trained on -- MUST match
# egoverse_jepa's mfp.NATIVE_W/NATIVE_H (640x480) exactly.
_S7_TARGET_W, _S7_TARGET_H = 640, 480
# Robot camera's own native capture resolution -- MUST match phase_c_s7_precompute.py's read_robot_frame
# assumption (720x1280, confirmed against the actual piper_zarr_teddy_bear recordings).
_ROBOT_NATIVE_W, _ROBOT_NATIVE_H = 1280, 720


class EgoVerseHPTS7Policy(EgoVerseHPTPolicy):
    config_class = EgoVerseHPTS7Config
    name = "egoverse_hpt_s7"

    def __init__(self, config: EgoVerseHPTS7Config):
        super().__init__(config)
        from lerobot.policies.egoverse_hpt.s7_worker_bridge import get_s7_worker

        self._s7_worker = get_s7_worker(device=str(config.device))
        self._frame_history: deque[tuple[float, np.ndarray]] = deque()
        self._history_seconds = config.s7_history_buffer_seconds
        self._delta_seconds = config.s7_delta_seconds
        self._tol_frac = config.s7_timestamp_tolerance_frac
        self._logged_raw_shape_once = False

        algo = self._egoverse_model.model
        proprio_keys = algo.proprio_keys.get(self._embodiment_id, [])
        if "s7_token" not in proprio_keys:
            logger.warning(
                "[egoverse_hpt_s7] Loaded checkpoint's proprio_keys=%s does NOT include 's7_token' -- "
                "this looks like a plain P0-style checkpoint, not a P2/P3 one. The live S7 token will "
                "still be computed but silently ignored by _robomimic_to_hpt_data (matches the same "
                "if-key-in-batch behavior the base adapter always had). Use EgoVerseHPTPolicy (no _s7 "
                "suffix) for P0 checkpoints instead -- it's simpler and doesn't need the S7 worker at all.",
                proprio_keys,
            )

    def reset(self):
        super().reset()
        self._frame_history.clear()

    @staticmethod
    def _extract_raw_frame(raw) -> np.ndarray:
        """raw: whatever batch[s7_source_camera_key] is -- CHW or HWC, uint8 [0,255] or float [0,1].
        Returns HWC uint8, ready for PIL-style crop/resize."""
        arr = raw.detach().cpu().numpy() if isinstance(raw, torch.Tensor) else np.asarray(raw)
        if arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[0] != arr.shape[-1]:
            arr = arr.transpose(1, 2, 0)  # CHW -> HWC
        if arr.dtype != np.uint8:
            arr = (arr * 255.0).clip(0, 255).astype(np.uint8) if arr.max() <= 1.5 else arr.astype(np.uint8)
        return arr

    def _crop_resize_for_s7(self, raw_hwc_uint8: np.ndarray) -> np.ndarray:
        """720x1280 (or whatever the raw capture is) -> center-crop to 4:3 -> resize to 640x480 -> CHW
        float32 [0,1]. Same transform as phase_c_s7_precompute.py's read_robot_frame, so the live token
        distribution matches what training saw."""
        from PIL import Image

        h, w = raw_hwc_uint8.shape[:2]
        target_w = int(h * 4 / 3)
        if target_w <= w:
            left = (w - target_w) // 2
            cropped = raw_hwc_uint8[:, left:left + target_w]
        else:
            # capture is already narrower than 4:3 (unexpected but handled) -- crop height instead
            target_h = int(w * 3 / 4)
            top = (h - target_h) // 2
            cropped = raw_hwc_uint8[top:top + target_h, :]
        img = Image.fromarray(cropped).resize((_S7_TARGET_W, _S7_TARGET_H), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return arr.transpose(2, 0, 1)  # HWC -> CHW

    def _push_frame_and_get_past(self, batch: dict, now: float) -> np.ndarray | None:
        raw = batch.get(f"observation.images.{self.config.s7_source_camera_key}")
        if raw is None:
            logger.warning(
                "[egoverse_hpt_s7] No 'observation.images.%s' in batch -- s7_source_camera_key may be "
                "wrong, or this camera isn't being captured. S7 token will be all-zero this step.",
                self.config.s7_source_camera_key,
            )
            return None, None

        raw_hwc = self._extract_raw_frame(raw)
        if not self._logged_raw_shape_once:
            logger.info(
                "[egoverse_hpt_s7] First raw S7-source frame: shape=%s dtype=%s -- verify this matches "
                "the real camera's native resolution before trusting downstream predictions.",
                raw_hwc.shape, raw_hwc.dtype,
            )
            self._logged_raw_shape_once = True

        cur_frame = self._crop_resize_for_s7(raw_hwc)
        self._frame_history.append((now, cur_frame))
        while self._frame_history and (now - self._frame_history[0][0]) > self._history_seconds:
            self._frame_history.popleft()

        target_t = now - self._delta_seconds
        best_t, best_frame, best_err = None, None, float("inf")
        for t, frame in self._frame_history:
            err = abs(t - target_t)
            if err < best_err:
                best_err, best_t, best_frame = err, t, frame

        if best_frame is None or best_err > self._tol_frac * self._delta_seconds:
            return cur_frame, None  # no valid past frame yet (e.g. just after reset()) -- caller zeroes the token
        return cur_frame, best_frame

    def _compute_s7_token(self, batch: dict) -> np.ndarray:
        now = time.monotonic()
        cur_frame, past_frame = self._push_frame_and_get_past(batch, now)
        if cur_frame is None or past_frame is None:
            return np.zeros(24, dtype=np.float32)  # matches the training-time zero-fill for early frames
        try:
            return self._s7_worker.compute_token(past_frame, cur_frame)
        except Exception as e:
            logger.error("[egoverse_hpt_s7] S7 worker call failed (%s) -- using zero token this step.", e)
            return np.zeros(24, dtype=np.float32)

    # ------------------------------------------------------------------
    # predict_action_chunk: same as the base class, with ONE addition (s7_token computed + injected into
    # hpt_batch before the forward pass). Duplicated rather than wrapped -- the base method has no
    # extension seam for an extra proprio key, and monkeypatching internals felt more fragile than a
    # clearly-marked, side-by-side override for a safety-critical real-robot control path.
    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        from egomimic.robot.piper.piper_kinematics import raw_gripper_to_frac, raw_joint_to_rad
        from egomimic.utils.pose_utils import _xyz6d_to_matrix, _xyzwxyz_to_matrix

        device = self.config.device
        algo = self._egoverse_model.model

        def _to_numpy_cpu(x) -> np.ndarray:
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
            return np.asarray(x)

        s7_token = self._compute_s7_token(batch)  # <-- the one new step, computed before everything else
        # so a slow/failed S7 call is visible as latency/logging here, not hidden inside the forward pass.

        joint_raw = _to_numpy_cpu(batch["observation.state"]).reshape(-1)[:7].astype(np.float64)
        joint_rad = raw_joint_to_rad(joint_raw[:6])
        gripper_frac = raw_gripper_to_frac(joint_raw[6:7])
        obs_ee_pose = self._kinematics.fk(joint_rad.astype(np.float32))

        horizon = self._PiperEmbodiment.ACTION_HORIZON
        front_key = f"observation.images.{self.config.front_camera_key}"
        wrist_key = f"observation.images.{self.config.wrist_camera_key}"
        front_np = _to_numpy_cpu(batch[front_key])
        wrist_np = _to_numpy_cpu(batch[wrist_key])
        front_img = torch.as_tensor(front_np).reshape(3, *front_np.shape[-2:]).float()
        wrist_img = torch.as_tensor(wrist_np).reshape(3, *wrist_np.shape[-2:]).float()
        if front_img.max() > 1.5:
            front_img = front_img / 255.0
        if wrist_img.max() > 1.5:
            wrist_img = wrist_img / 255.0

        pre_transform_batch = {
            "obs_ee_pose": torch.from_numpy(obs_ee_pose).float(),
            "obs_gripper": torch.from_numpy(gripper_frac).float(),
            "action_ee_pose": torch.from_numpy(obs_ee_pose).float().unsqueeze(0).repeat(horizon, 1),
            "action_gripper": torch.from_numpy(gripper_frac).float().unsqueeze(0).repeat(horizon, 1),
        }
        for transform in self._transform_list:
            pre_transform_batch = transform.transform(pre_transform_batch)

        hpt_batch = {
            "observations.images.front_img_1": front_img.unsqueeze(0).to(device),
            "observations.images.wrist_img": wrist_img.unsqueeze(0).to(device),
            "observations.state.ee_pose": pre_transform_batch["observations.state.ee_pose"].unsqueeze(0).to(device),
            "actions_cartesian": pre_transform_batch["actions_cartesian"].unsqueeze(0).to(device),
            "s7_token": torch.from_numpy(s7_token).float().unsqueeze(0).to(device),  # <-- the injection
            "pad_mask": torch.ones((1, horizon, 1), dtype=torch.bool, device=device),
            "embodiment": torch.tensor([self._embodiment_id], device=device),
        }

        ac_key = algo.ac_keys[self._embodiment_id]
        data = algo._robomimic_to_hpt_data(
            hpt_batch,
            algo.camera_keys[self._embodiment_id],
            algo.proprio_keys[self._embodiment_id],
            algo.lang_keys[self._embodiment_id],
            ac_key,
            algo.auxiliary_ac_keys.get("piper_left_arm", []),
        )
        action_out = algo.nets["policy"].forward("piper_left_arm", data)
        pred_key = "shared" if algo.shared_ac_key else "piper_left_arm"
        pred_norm = action_out[pred_key]

        unnorm = algo.norm_stats.unnormalize({algo.shared_ac_key: pred_norm}, self._embodiment_id)
        pred = unnorm[algo.shared_ac_key][0].cpu().numpy()

        n_steps = min(self.config.n_action_steps, pred.shape[0])
        # Seed from the FRESH sensed joint position, never a persisted cross-chunk value -- matches the
        # base class's own hardware-tested fix (see its predict_action_chunk comment): seeding from an
        # assumed-reached previous target compounds drift under max_relative_target clamping.
        seed = joint_rad.astype(np.float32)
        obs_mat = _xyzwxyz_to_matrix(obs_ee_pose[None, :].astype(np.float64))[0]
        joint_actions = []
        for t in range(n_steps):
            xyz = pred[t, :3]
            c1_c2 = pred[t, 3:9]
            gripper_t = float(np.clip(pred[t, 9], 0.0, 1.0))
            pred_mat = _xyz6d_to_matrix(np.concatenate([xyz, c1_c2])[None, :].astype(np.float64))[0]
            if self.config.action_frame == "wristframe":
                abs_mat = obs_mat @ pred_mat
            else:
                abs_mat = pred_mat
            quat_xyzw = ScipyRotation.from_matrix(abs_mat[:3, :3]).as_quat()
            target_pose = np.concatenate([abs_mat[:3, 3], quat_xyzw[[3, 0, 1, 2]]])

            q_sol, converged, pos_err = self._kinematics.ik(
                target_pose,
                seed_joint_angles_rad=seed,
                num_retries=self.config.ik_num_retries,
                pos_tolerance=self.config.ik_pos_tolerance,
                rot_tolerance=self.config.ik_rot_tolerance,
                seed_noise_std=self.config.ik_seed_noise_std,
            )
            if not converged:
                self._consecutive_ik_failures += 1
                logger.warning(
                    "[egoverse_hpt_s7] IK did not converge (pos_err=%.4fm) at chunk step %d; holding "
                    "previous joint target (%d/%d consecutive).",
                    pos_err, t, self._consecutive_ik_failures, self.config.max_consecutive_ik_failures,
                )
                q_sol = seed
                if (self.config.max_consecutive_ik_failures > 0
                        and self._consecutive_ik_failures >= self.config.max_consecutive_ik_failures):
                    raise RuntimeError(
                        f"[egoverse_hpt_s7] {self._consecutive_ik_failures} consecutive IK failures -- stopping."
                    )
            else:
                self._consecutive_ik_failures = 0

            seed = q_sol
            norm_joints = self._rad_to_normalized_deg(q_sol)
            norm_gripper = gripper_t * 100.0
            joint_actions.append(np.concatenate([norm_joints, [norm_gripper]]))

        return torch.from_numpy(np.stack(joint_actions).astype(np.float32))
