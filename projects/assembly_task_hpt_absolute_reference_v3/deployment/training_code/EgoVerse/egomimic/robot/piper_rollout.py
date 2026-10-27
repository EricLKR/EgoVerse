"""Real-hardware rollout for a trained Piper HPT/FMPolicy checkpoint.

This is the missing last leg of the Piper coordinate pipeline: training already builds
`cartesian_camframe` (Piper.get_transform_list(mode="cartesian_camframe"), see
egomimic/rldb/embodiment/piper.py) -- the EgoVerse/EgoMimic papers' canonical
camera-centered cross-embodiment representation (a^H_{t:t+k} = [(T_t^device)^-1
T_{t+i}^device p_{t+i}] in both papers' notation) -- but nothing previously converted a
trained policy's camera-frame prediction back into a real joint command on the arm. This
script closes that loop:

    predicted action (camframe, 9D xyz+6D-rotation + 1D gripper, from get_transform_list's
    XYZWXYZ_to_XYZ6D convention)
        -> _xyz6d_to_matrix                                  # 6D rotation -> SE3
        -> Piper.EXTRINSICS[cam] @ (camframe pose)            # cam frame -> base_link frame
           (exact inverse of ActionChunkCoordinateFrameTransform's `target_se3.inverse() @
           chunk_se3` used at training time -- same extrinsics matrix, opposite direction)
        -> PiperKinematicsSolver.ik(..., seed_joint_angles_rad=<live joint state>)
        -> normalized [-100, 100] / [0, 100] joint command    -> LerobotPiper.send_action

`--mode cartesian_base` is also supported (skips the extrinsics step -- action is already
base_link-frame) for checkpoints trained solo, no cotraining. `cartesian_wristframe` is
NOT supported here: unlike a single fixed camera-frame anchor, wristframe's target is the
current *observed* pose, which is exactly the frame this script continuously re-observes,
so composing it into an absolute base-frame target needs a different (untested) derivation
than either mode above. Point a wristframe checkpoint at this script and it raises rather
than silently doing the wrong thing.

WHAT THIS DOES NOT KNOW AND WILL NOT GUESS (verify before running):
  - Wrist camera (405/D405) serial number. No default -- required via --wrist-camera-serial.
    Front camera (435if/D435i) serial defaults to the unit calibrate_piper_extrinsics.py
    was run against (327122075831); override with --front-camera-serial if this is a
    different physical unit.
  - Which physical camera is "front" vs "wrist". Same unverified assumption
    egomimic/rldb/embodiment/piper.py's _get_keymap and piper_lerobot_to_zarr.py both
    carry (435if=front, 405=wrist) -- confirm against the rig before trusting rollout
    viz/behavior, not just before trusting training.
  - Extrinsics confidence: Piper.EXTRINSICS was fit from only 11/20 valid hand-eye samples
    (see piper.py's class docstring). Treat cartesian_camframe rollouts as
    working-but-not-fully-validated until re-calibrated with more samples.
  - Run it from the venv at /mnt/wd10/tongxi/egoverse_piper/EgoVerse/emimic (its python
    has lightning + lerobot + pytorch_kinematics + zarr 3 together; this module imports
    cleanly there). NOT the conda env also named `emimic` (no lightning/lerobot) and NOT
    /mnt/wd10/tongxi/EgoVerse_v1/emimic (points at a different egomimic fork).
  - The raw action horizon the checkpoint was trained with. Piper.ACTION_HORIZON is now 30
    frames (1.0 s); checkpoints trained before that change used 100 (3.3 s). The predicted
    CHUNK_LENGTH (100)-step chunk is spread over `--action-horizon-frames` control ticks, so
    passing the wrong value plays the trajectory back 3.3x too fast or too slow.

SAFETY: this executes model-predicted, IK-solved joint commands on the real arm in a
closed loop. `--max-relative-target` bounds per-step motion (same clamp
calibrate_piper_extrinsics.py relies on) but this is still autonomous motion driven by a
policy that has not been validated at all yet in cartesian_camframe mode. Start with
--speed-percent low (e.g. 10-15), keep a hand on the E-stop, and watch the printed
IK-convergence / position-error log every tick.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from egomimic.models.denoising_policy import DenoisingPolicy
from egomimic.pl_utils.pl_model import ModelWrapper
from egomimic.rldb.embodiment.piper import Piper
from egomimic.robot.piper.gripper_utils import update_binary_gripper_state
from egomimic.robot.piper.piper_kinematics import (
    GRIPPER_STROKE_MM,
    PiperKinematicsSolver,
    raw_gripper_to_frac,
    raw_joint_to_rad,
)
from egomimic.robot.robot_utils import RateLoop
from egomimic.utils.pose_utils import _matrix_to_xyzwxyz, _xyz6d_to_matrix

_SUPPORTED_MODES = ("cartesian_camframe", "cartesian_base")

# Same unit this script's extrinsics/joint-limits provenance traces back to
# (calibrate_piper_extrinsics.py / piper_recover_cmd_ee_pose.py). Override if this is a
# different physical arm.
DEFAULT_FRONT_CAMERA_SERIAL = "327122075831"
DEFAULT_CALIBRATION_PATH = (
    "~/.cache/huggingface/lerobot/calibration/robots/piper/my_piper_follower.json"
)


def load_joint_limits_deg(calibration_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Loads this unit's REAL calibrated per-joint limits from the same lerobot
    calibration cache file piper_recover_cmd_ee_pose.py already validated (created
    2026-08-19, loaded automatically by lerobot's Piper.__init__ on every connect).
    Duplicated here (not imported) because scripts/ isn't a package and
    piper_recover_cmd_ee_pose.py isn't meant to be imported -- same reasoning
    calibrate_piper_extrinsics.py's own hardcoded JOINT_LIMITS_DEG copy already used.
    """
    path = Path(calibration_path).expanduser()
    with open(path) as f:
        cal = json.load(f)
    joint_keys = [f"joint_{i}" for i in range(6)]
    min_deg = np.array([cal[k]["range_min"] / 1000.0 for k in joint_keys])
    max_deg = np.array([cal[k]["range_max"] / 1000.0 for k in joint_keys])
    return min_deg, max_deg


def _deg_to_normalized(deg: np.ndarray, low_deg: np.ndarray, high_deg: np.ndarray) -> np.ndarray:
    """Real degrees -> [-100, 100], exact inverse of
    piper_recover_cmd_ee_pose.py's action_pct_to_real_deg / calibrate_piper_extrinsics.py's
    normalized_from_deg."""
    return np.clip((deg - low_deg) / (high_deg - low_deg) * 200.0 - 100.0, -100.0, 100.0)


class PiperRollout:
    def __init__(
        self,
        policy_path: str,
        mode: str = "cartesian_camframe",
        port: str = "can_follower",
        robot_id: str = "my_piper_follower",
        front_camera_serial: str = DEFAULT_FRONT_CAMERA_SERIAL,
        wrist_camera_serial: str | None = None,
        calibration_path: str = DEFAULT_CALIBRATION_PATH,
        max_relative_target: float = 5.0,
        speed_percent: int = 15,
        query_frequency: int = 20,
        num_ik_retries: int = 16,
        gripper_stroke_mm: float = GRIPPER_STROKE_MM,
        action_horizon_frames: int | None = None,
        gripper_close_threshold: float = 0.3,
        gripper_open_threshold: float = 0.7,
    ):
        self.action_horizon = int(action_horizon_frames or Piper.ACTION_HORIZON)
        if not (0 < query_frequency <= self.action_horizon):
            raise ValueError(
                f"query_frequency ({query_frequency}) must be in (0, action_horizon_frames="
                f"{self.action_horizon}] -- a chunk only covers action_horizon_frames control ticks."
            )
        if mode not in _SUPPORTED_MODES:
            raise ValueError(
                f"Unsupported mode {mode!r}; expected one of {_SUPPORTED_MODES}. "
                "cartesian_wristframe rollout needs a different (unimplemented) derivation -- "
                "see this module's docstring."
            )
        if wrist_camera_serial is None:
            raise ValueError(
                "--wrist-camera-serial is required -- no default exists in this codebase "
                "(never recorded anywhere). Find it (e.g. `rs-enumerate-devices`) rather than guess."
            )

        self.mode = mode
        self.query_frequency = query_frequency
        self.num_ik_retries = num_ik_retries
        self.gripper_stroke_mm = gripper_stroke_mm
        self.gripper_close_threshold = float(gripper_close_threshold)
        self.gripper_open_threshold = float(gripper_open_threshold)
        # Validate threshold ordering before connecting to hardware.
        update_binary_gripper_state(
            0.5,
            None,
            close_threshold=self.gripper_close_threshold,
            open_threshold=self.gripper_open_threshold,
        )
        self._gripper_binary_state: float | None = None

        low_deg, high_deg = load_joint_limits_deg(calibration_path)
        self.low_deg, self.high_deg = low_deg, high_deg
        self.kin = PiperKinematicsSolver(
            device="cpu",
            joint_limits_rad=(np.deg2rad(low_deg), np.deg2rad(high_deg)),
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[piper_rollout] loading policy from {policy_path}")
        self.policy = self._load_policy(policy_path)

        self.extrinsics = Piper.EXTRINSICS
        if self.mode == "cartesian_camframe":
            if self.extrinsics is None:
                raise ValueError("Piper.EXTRINSICS is None -- cannot run cartesian_camframe rollout.")
            self.cam_key = next(iter(self.extrinsics))
            self.base_T_cam = self.extrinsics[self.cam_key]  # (4,4), world<-cam == base<-cam

        self.robot, self.cameras_wh = self._connect_hardware(
            port, robot_id, front_camera_serial, wrist_camera_serial,
            max_relative_target, speed_percent,
        )

        self._cached_chunk = None  # (T, 10) most recent model prediction, base_link-frame xyzwxyz+gripper
        self._chunk_tick = 0

    # ------------------------------------------------------------------ policy

    def _load_policy(self, policy_path: str):
        policy = ModelWrapper.load_from_checkpoint(policy_path, weights_only=False, map_location="cpu")
        policy = policy.to(self.device)
        policy.eval()
        policy.model.device = self.device
        if getattr(policy.model, "diffusion", False):
            for head in policy.model.nets.policy.heads:
                if isinstance(policy.model.nets.policy.heads[head], DenoisingPolicy):
                    policy.model.nets.policy.heads[head].num_inference_steps = 10
        return policy

    # ---------------------------------------------------------------- hardware

    def _connect_hardware(
        self, port, robot_id, front_serial, wrist_serial, max_relative_target, speed_percent
    ):
        # Imported lazily -- only the env actually driving the arm needs lerobot/pyrealsense2
        # importable; this lets the module still be imported (e.g. for tests) without them.
        from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
        from lerobot.robots.piper.config_piper import PiperConfig
        from lerobot.robots.piper.piper import Piper as LerobotPiper

        # front_1 <-> 435if (D435i, 1280x720), wrist_1 <-> 405 (D405, 848x480): matches
        # CAMERA_ZARR_KEYS in piper_lerobot_to_zarr.py and Piper._get_keymap -- unverified
        # against the physical rig, see module docstring.
        cameras = {
            "435if": RealSenseCameraConfig(serial_number_or_name=front_serial, fps=30, width=1280, height=720),
            "405": RealSenseCameraConfig(serial_number_or_name=wrist_serial, fps=30, width=848, height=480),
        }
        robot = LerobotPiper(
            PiperConfig(
                port=port,
                id=robot_id,
                max_relative_target=max_relative_target,
                speed_percent=speed_percent,
                joint_limits_deg=list(zip(self.low_deg.tolist(), self.high_deg.tolist())),
                cameras=cameras,
            )
        )
        # calibrate=False: this unit's calibration is already persisted (the same file
        # load_joint_limits_deg just read) and loaded automatically on connect -- see
        # piper_recover_cmd_ee_pose.py's docstring.
        robot.connect(calibrate=False)
        wh = {k: (v.width, v.height) for k, v in cameras.items()}
        return robot, wh

    def get_obs(self) -> dict:
        return self.robot.get_observation()

    # --------------------------------------------------------------- obs -> model batch

    def _build_obs_batch(self, obs: dict) -> dict:
        joint_raw = np.array([obs[f"joint_{i}.pos"] for i in range(6)], dtype=np.float64)  # 0.001deg counts
        gripper_raw = np.array([obs["joint_6.pos"]], dtype=np.float64)  # 0.001mm counts

        joint_rad = raw_joint_to_rad(joint_raw)
        obs_ee_pose = self.kin.fk(joint_rad.astype(np.float32)).astype(np.float64)  # (7,) xyzwxyz, base_link
        obs_gripper = raw_gripper_to_frac(gripper_raw, stroke_mm=self.gripper_stroke_mm)  # (1,) in [0,1]
        dummy_gripper_command = (
            self._gripper_binary_state
            if self._gripper_binary_state is not None
            else float(obs_gripper[0] >= 0.5)
        )

        # Real transform list needs an action-side chunk to run structurally (ConcatKeys,
        # InterpolatePose, ...) even though we only want its obs_key output here -- same
        # trick egomimic/robot/rollout.py's process_obs_for_transform_list uses for Eva
        # (repeat the current obs as a dummy "action"). The dummy chunk is discarded below;
        # only observations.state.ee_pose is kept.
        horizon = self.action_horizon
        batch = {
            "action_ee_pose": np.tile(obs_ee_pose, (horizon, 1)),
            # forward_eval does not consume the action target, but the shared
            # preprocessing path expects it to exist. Keep this dummy target
            # binary so inference follows the same schema as training.
            "action_gripper": np.full((horizon, 1), dummy_gripper_command, dtype=np.float64),
            "obs_ee_pose": obs_ee_pose,
            "obs_gripper": obs_gripper,
        }
        for t in Piper.get_transform_list(mode=self.mode, stride=1):
            batch = t.transform(batch)

        state = batch["observations.state.ee_pose"]  # (10,) tensor, post XYZWXYZ_to_XYZ6D + gripper concat
        if not isinstance(state, torch.Tensor):
            state = torch.as_tensor(state)

        front = self._image_to_tensor(obs["435if"])
        wrist = self._image_to_tensor(obs["405"])

        model_batch = {
            # keep rollout-local, PI-schema, and canonical dataset-key aliases together so
            # checkpoints trained under any of Piper._get_keymap's keymap_mode variants can
            # all resolve the same tensors -- same defensive pattern
            # egomimic/robot/rollout.py's process_obs_for_transform_list uses for Eva.
            "front_img_1": front,
            "base_0_rgb": front,
            "observations.images.front_img_1": front,
            "wrist_img": wrist,
            "left_wrist_0_rgb": wrist,
            "observations.images.wrist_img": wrist,
            "observations.state.ee_pose": state,
            "pad_mask": torch.ones((1, Piper.CHUNK_LENGTH, 1), dtype=torch.bool),
        }
        return model_batch

    @staticmethod
    def _image_to_tensor(img: np.ndarray) -> torch.Tensor:
        # lerobot's RealSenseCameraConfig defaults to color_mode=RGB (unlike Eva's cv2/ROS
        # BGR source in egomimic/robot/rollout.py) -- NO channel swap here. Verify against
        # the actual camera output before trusting this; a silent RGB/BGR mismatch would
        # degrade the frozen ResNet's features without erroring.
        t = torch.from_numpy(np.ascontiguousarray(img))  # (H, W, 3) uint8
        t = t.permute(2, 0, 1).to(dtype=torch.float32) / 255.0  # (3, H, W)
        return t

    # ------------------------------------------------------------- model -> action chunk

    def predict_action_chunk(self, obs: dict) -> np.ndarray:
        batch = self._build_obs_batch(obs)
        wrapped = {"piper_left_arm": batch}
        processed = self.policy.model.process_batch_for_training(wrapped)
        with torch.no_grad():
            preds = self.policy.model.forward_eval(processed)["piper_left_arm_actions_cartesian"]
        return preds.detach().cpu().numpy().reshape(-1, 10)  # (T, 10): xyz(3)+c1(3)+c2(3)+gripper(1)

    def _chunk_to_base_xyzwxyz(self, chunk: np.ndarray) -> np.ndarray:
        """(T, 10) predicted [xyz, c1, c2, gripper] -> (T, 8) [xyzwxyz base_link pose, gripper]."""
        pose6d, gripper = chunk[:, :9], chunk[:, 9:10]
        cam_mat = _xyz6d_to_matrix(pose6d)  # (T, 4, 4)
        if self.mode == "cartesian_camframe":
            # Exact inverse of the training-time transform (target_se3.inverse() @
            # chunk_se3, mode="xyzwxyz", target_world=Piper.EXTRINSICS[cam]): re-apply the
            # SAME extrinsics matrix in the forward direction, base<-cam @ cam<-ee = base<-ee.
            base_mat = self.base_T_cam[None, :, :] @ cam_mat
        else:  # cartesian_base: chunk is already base_link-frame
            base_mat = cam_mat
        base_xyzwxyz = _matrix_to_xyzwxyz(base_mat)  # (T, 7)
        return np.concatenate([base_xyzwxyz, np.clip(gripper, 0.0, 1.0)], axis=-1)  # (T, 8)

    # ------------------------------------------------------------------------- IK -> cmd

    def _pose_to_action_dict(self, target_xyzwxyz: np.ndarray, gripper_frac: float, seed_joint_rad: np.ndarray):
        joint_rad, converged, pos_err = self.kin.ik(
            target_xyzwxyz, seed_joint_angles_rad=seed_joint_rad, num_retries=self.num_ik_retries
        )
        if not converged:
            print(
                f"[piper_rollout] WARNING: IK did not converge (pos_err={pos_err:.4f}m) -- "
                "holding current position for this tick rather than executing an untrusted solve."
            )
            return None, pos_err, converged
        joint_deg = np.rad2deg(joint_rad)
        joint_norm = _deg_to_normalized(joint_deg, self.low_deg, self.high_deg)
        action = {f"joint_{i}.pos": float(joint_norm[i]) for i in range(6)}
        self._gripper_binary_state = update_binary_gripper_state(
            float(np.clip(gripper_frac, 0.0, 1.0)),
            self._gripper_binary_state,
            close_threshold=self.gripper_close_threshold,
            open_threshold=self.gripper_open_threshold,
        )
        action["joint_6.pos"] = self._gripper_binary_state * 100.0
        return action, pos_err, converged

    # ------------------------------------------------------------------------------- loop

    def rollout_step(self, obs: dict, seed_joint_rad: np.ndarray):
        """Returns (action_dict_or_None, pos_err, converged) for this tick. Re-queries the
        policy every `query_frequency` ticks (receding horizon); reuses the cached chunk
        otherwise. The chunk's CHUNK_LENGTH (100) resampled steps span `action_horizon`
        raw frames of real time (the same InterpolatePose stretch training applied), so
        control tick k maps to chunk step round(k * CHUNK_LENGTH / action_horizon) -- the
        inverse of Eva's `_downsample_chunk` (100 -> resampled_action_len).
        `seed_joint_rad` should be the just-read live joint state, not a previous IK
        solution, so IK error never compounds across ticks."""
        if self._cached_chunk is None or self._chunk_tick >= self.query_frequency:
            raw_chunk = self.predict_action_chunk(obs)
            self._cached_chunk = self._chunk_to_base_xyzwxyz(raw_chunk)
            self._chunk_tick = 0
        n_steps = self._cached_chunk.shape[0]
        step = min(n_steps - 1, int(round(self._chunk_tick * n_steps / self.action_horizon)))
        pose8 = self._cached_chunk[step]
        self._chunk_tick += 1
        return self._pose_to_action_dict(pose8[:7], float(pose8[7]), seed_joint_rad)

    def run(self, frequency: float = 20.0, num_steps: int | None = None):
        try:
            with RateLoop(frequency=frequency, max_iterations=num_steps) as loop:
                for i in loop:
                    obs = self.get_obs()
                    joint_rad = raw_joint_to_rad(
                        np.array([obs[f"joint_{j}.pos"] for j in range(6)], dtype=np.float64)
                    )
                    action, pos_err, converged = self.rollout_step(obs, joint_rad.astype(np.float32))
                    if action is not None:
                        sent = self.robot.send_action(action)
                        print(f"[piper_rollout] step {i}: pos_err={pos_err:.4f}m converged={converged} sent={sent}")
        except KeyboardInterrupt:
            print("[piper_rollout] KeyboardInterrupt -- stopping.")
        finally:
            self.robot.disconnect()


def build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy-path", required=True)
    p.add_argument("--mode", default="cartesian_camframe", choices=_SUPPORTED_MODES)
    p.add_argument("--port", default="can_follower")
    p.add_argument("--robot-id", default="my_piper_follower")
    p.add_argument("--front-camera-serial", default=DEFAULT_FRONT_CAMERA_SERIAL)
    p.add_argument("--wrist-camera-serial", required=True)
    p.add_argument("--calibration-path", default=DEFAULT_CALIBRATION_PATH)
    p.add_argument("--max-relative-target", type=float, default=5.0)
    p.add_argument("--speed-percent", type=int, default=15)
    p.add_argument("--query-frequency", type=int, default=20)
    p.add_argument(
        "--gripper-close-threshold",
        type=float,
        default=0.3,
        help="Predictions at or below this value switch the command to fully closed.",
    )
    p.add_argument(
        "--gripper-open-threshold",
        type=float,
        default=0.7,
        help="Predictions at or above this value switch the command to fully open.",
    )
    p.add_argument(
        "--action-horizon-frames", type=int, default=None,
        help=f"Raw-frame horizon the checkpoint was trained with (default Piper.ACTION_HORIZON="
             f"{Piper.ACTION_HORIZON}; pass 100 for checkpoints trained before that change).",
    )
    p.add_argument("--frequency", type=float, default=30.0, help="Control loop Hz; data was recorded at 30.")
    p.add_argument("--num-steps", type=int, default=None)
    return p


def main():
    args = build_arg_parser().parse_args()
    rollout = PiperRollout(
        policy_path=args.policy_path,
        mode=args.mode,
        port=args.port,
        robot_id=args.robot_id,
        front_camera_serial=args.front_camera_serial,
        wrist_camera_serial=args.wrist_camera_serial,
        calibration_path=args.calibration_path,
        max_relative_target=args.max_relative_target,
        speed_percent=args.speed_percent,
        query_frequency=args.query_frequency,
        action_horizon_frames=args.action_horizon_frames,
        gripper_close_threshold=args.gripper_close_threshold,
        gripper_open_threshold=args.gripper_open_threshold,
    )
    rollout.run(frequency=args.frequency, num_steps=args.num_steps)


if __name__ == "__main__":
    main()
