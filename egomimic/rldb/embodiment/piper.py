from __future__ import annotations

from typing import Literal

import numpy as np

from egomimic.rldb.embodiment.embodiment import Embodiment
from egomimic.rldb.zarr.action_chunk_transforms import (
    ActionChunkCoordinateFrameTransform,
    ConcatKeys,
    DeleteKeys,
    InterpolateStep,
    InterpolatePose,
    NumpyToTensor,
    PoseCoordinateFrameTransform,
    Transform,
    XYZWXYZ_to_XYZ6D,
)
from egomimic.utils.pose_utils import _matrix_to_xyzwxyz


class Piper(Embodiment):
    """Single-arm AgileX PIPER-L follower embodiment.

    Structurally this is Eva minus the second arm, minus the egocentric camera,
    plus one real difference: PIPER has no onboard EE-pose sensor, only 6 joint
    angles + a gripper aperture. `obs_ee_pose` here is derived *offline*, via FK
    over the recorded joint stream (see `egomimic/robot/piper/piper_kinematics.py`),
    at Zarr-conversion time -- not read live off the arm the way Eva's is.

    Cartesian pose chunks are built the same way `Human`'s canonical pipeline
    does: by stacking *future* `obs_ee_pose` frames rather than a commanded pose.
    Gripper targets are different: they use binary open/close intent recovered
    from the recorded command as `cmd_gripper`; continuous `obs_gripper` remains
    in proprioception so an object-blocked grasp is not mistaken for a partial
    open command. Eva uses a true `cmd_ee_pose` instead;
    a Piper equivalent has since been recovered (FK over the calibrated raw
    `action` stream, ~1cm median lead over obs -- see
    scripts/piper_process/piper_recover_cmd_ee_pose.py, written to the
    `piper_zarr_teddy_bear_cmdee` copies) but is not wired in here yet.

    ACTION_HORIZON is in *raw recorded frames* (30 fps), and is the time span
    the action chunk covers before InterpolatePose resamples it to CHUNK_LENGTH
    steps for the shared head. It must match the cotraining partner in
    *seconds*, not in steps -- both embodiments' chunks are resampled to the same
    CHUNK_LENGTH, so a mismatched raw horizon silently teaches the shared action
    head two different timescales. Human.ACTION_HORIZON is 30 (1.0 s @ 30 fps);
    Eva's is 45 (1.5 s). This was 100 (3.3 s, chosen to match the ACT baseline's
    chunk_size) through the first cotraining runs -- a 3.3x timescale mismatch
    against the human side; keep 100 only for solo-Piper/ACT-comparison runs.

    EXTRINSICS (camera_key -> world<-cam 4x4, world = robot base_link here) are
    required for `cartesian_camframe` mode, which re-expresses ee_pose in the
    external camera's frame. Left as `None` until this rig is calibrated (do
    not fill this in with a guessed/typical transform -- an uncalibrated
    extrinsics matrix is worse than none, since it would silently corrupt
    every downstream cartesian-frame use).

    For cotraining against `Human`, `cartesian_wristframe` mode (no extrinsics
    needed, see `_build_piper_wristframe_transform_list`'s docstring) is a
    valid embodiment-agnostic representation, but NOT the one the EgoVerse
    paper's own method actually uses -- the paper's canonical cotraining
    representation is camera-centered (mirrors `Eva.get_transform_list(mode=
    "cartesian")`, i.e. this rig's `cartesian_camframe` mode), which needs a
    real extrinsics calibration. See `calibrate_camera/
    calibrate_piper_extrinsics.py` for how EXTRINSICS below was obtained
    (eye-to-hand hand-eye calibration, `cv2.calibrateHandEye`, ArUco 5x5 id=0
    marker rigidly mounted on the gripper).

    CAUTION: that calibration ran with only 11 valid samples (of 20
    attempted) -- above the ~8 minimum for `calibrateHandEye` to produce a
    mathematically valid rotation, but below the ~15-20 with a good
    position+orientation spread you'd want for full confidence. Treat this
    as a working-but-not-fully-validated calibration; if `cartesian_camframe`
    results look geometrically off, re-run that script with more samples
    (and a more securely mounted tag -- tape came loose mid-run last time)
    before assuming the bug is elsewhere.

    Use plain `cartesian_base` mode for solo Piper training/validation (no
    cotraining) -- keeps ee_pose in the arm's own base_link frame throughout.
    """

    # world<-cam 4x4 per external camera, robot base_link frame. See this
    # class's docstring for calibration provenance/confidence caveats. Re-run
    # 2026-09-18 (calibrate_piper_extrinsics.py) after the FK fix -- the
    # earlier calibration used the buggy agx_arm_urdf-derived FK as its
    # kin.fk() ground truth (see piper_kinematics.py's module docstring), so
    # this replaces rather than supplements it; the old matrix is not kept
    # here (see git history / piper_extrinsics_result.txt for provenance).
    EXTRINSICS: dict[str, np.ndarray] | None = {
        "435if": np.array(
            [
                [-4.27838100e-02, -9.45660090e-01, 3.22329860e-01, 2.85056900e-02],
                [-9.98482910e-01, 2.92787200e-02, -4.66329200e-02, -2.51857600e-01],
                [3.46614800e-02, -3.23835990e-01, -9.45478100e-01, 6.51945020e-01],
                [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
            ]
        ),
    }

    # Raw frames (30 fps) per action chunk == Human.ACTION_HORIZON, so cotrained
    # chunks span the same 1.0 s. See the class docstring before changing.
    ACTION_HORIZON = 30
    # Steps every chunk is resampled to for the shared action head (FMPolicy
    # action_horizon / CrossTransformer act_seq in the hpt_*_piper*.yaml configs).
    CHUNK_LENGTH = 100

    @classmethod
    def _get_keymap(cls, keymap_mode: str):
        is_pi = keymap_mode.endswith("_pi")
        base_mode = keymap_mode[: -len("_pi")] if is_pi else keymap_mode
        front_key = "base_0_rgb" if is_pi else cls.VIZ_IMAGE_KEY
        wrist_key = "left_wrist_0_rgb" if is_pi else "observations.images.wrist_img"

        if base_mode not in ("cartesian_base", "cartesian_camframe"):
            raise ValueError(
                f"Unsupported keymap_mode '{keymap_mode}' for {cls.__name__}. "
                "Expected 'cartesian_base' or 'cartesian_camframe' (optionally with a '_pi' suffix)."
            )

        horizon = cls.ACTION_HORIZON
        return {
            # NOTE: which of the two recorded camera streams is "front" vs "wrist"
            # hasn't been confirmed against the physical rig -- 435if (D435i, wide
            # FOV, 720p) is mapped to front_1 and 405 (D405, short-range, 480p) to
            # wrist_1 on the (unverified) assumption that the D405 is gripper/wrist
            # -mounted, which is AgileX's common reference layout for this arm.
            # Swap the two zarr_key values below if that's backwards for this rig.
            front_key: {"key_type": "camera_keys", "zarr_key": "images.front_1"},
            wrist_key: {"key_type": "camera_keys", "zarr_key": "images.wrist_1"},
            "action_ee_pose": {
                "key_type": "action_keys",
                "zarr_key": "obs_ee_pose",
                "horizon": horizon,
            },
            "action_gripper": {
                "key_type": "action_keys",
                "zarr_key": "cmd_gripper",
                "horizon": horizon,
            },
            "obs_ee_pose": {"key_type": "proprio_keys", "zarr_key": "obs_ee_pose"},
            "obs_gripper": {"key_type": "proprio_keys", "zarr_key": "obs_gripper"},
        }

    @classmethod
    def get_transform_list(
        cls,
        mode: Literal[
            "cartesian_base", "cartesian_camframe", "cartesian_wristframe"
        ] = "cartesian_base",
        stride: int = 1,
    ) -> list[Transform]:
        """`stride=1`: PIPER data is already recorded at the training fps (no
        subsampling needed, unlike ~30fps-native Aria human data with stride=3).
        """
        if mode == "cartesian_base":
            return _build_piper_base_frame_transform_list(stride=stride)
        if mode == "cartesian_camframe":
            if cls.EXTRINSICS is None:
                raise ValueError(
                    "Piper.EXTRINSICS is None -- 'cartesian_camframe' mode needs a real "
                    "camera<-base extrinsics calibration for this rig before it can re-express "
                    "ee_pose in the camera frame. Use 'cartesian_base' until this rig is "
                    "calibrated (see calibrate_camera/), or set Piper.EXTRINSICS yourself if "
                    "you already have it."
                )
            return _build_piper_camframe_transform_list(stride=stride)
        if mode == "cartesian_wristframe":
            return _build_piper_wristframe_transform_list(stride=stride)
        raise ValueError(f"Unsupported transform_list mode '{mode}' for {cls.__name__}")


def _interpolate_chunk_transforms(
    pose_key: str, gripper_key: str, *, chunk_length: int, stride: int
) -> list[Transform]:
    """Resample a raw `ACTION_HORIZON`-frame chunk to `chunk_length` steps --
    slerp for the xyzwxyz pose and zero-order hold for the binary gripper
    command. Every mode needs this now that ACTION_HORIZON (30) != CHUNK_LENGTH
    (100); step resampling preserves exact 0/1 gripper targets."""
    return [
        InterpolatePose(
            new_chunk_length=chunk_length,
            action_key=pose_key,
            output_action_key=pose_key,
            stride=stride,
            mode="xyzwxyz",
        ),
        InterpolateStep(
            new_chunk_length=chunk_length,
            action_key=gripper_key,
            output_action_key=gripper_key,
            stride=stride,
        ),
    ]


def _build_piper_base_frame_transform_list(
    *,
    action_ee_pose: str = "action_ee_pose",
    action_gripper: str = "action_gripper",
    obs_ee_pose: str = "obs_ee_pose",
    obs_gripper: str = "obs_gripper",
    actions_key: str = "actions_cartesian",
    obs_key: str = "observations.state.ee_pose",
    chunk_length: int = 100,
    stride: int = 1,
) -> list[Transform]:
    """No frame conversion: ee_pose stays in the arm's own base_link frame (where
    FK put it). Action chunks are stacked future obs poses from the zarr
    key_map's `horizon`, resampled to `chunk_length`. Converts quat->6D rotation
    (7D->9D, `XYZWXYZ_to_XYZ6D`) so the shared action/state keys have no
    yaw-wraparound discontinuity at +-pi -- raw ypr was triggering constant
    "Bounds violation" rejection-sampling right at that boundary. See
    `_matrix_to_xyz6d`'s docstring.
    """
    return [
        *_interpolate_chunk_transforms(
            action_ee_pose, action_gripper, chunk_length=chunk_length, stride=stride
        ),
        XYZWXYZ_to_XYZ6D(keys=[action_ee_pose, obs_ee_pose]),
        ConcatKeys(
            key_list=[action_ee_pose, action_gripper],
            new_key_name=actions_key,
            delete_old_keys=True,
        ),
        ConcatKeys(
            key_list=[obs_ee_pose, obs_gripper],
            new_key_name=obs_key,
            delete_old_keys=True,
        ),
        NumpyToTensor(keys=[actions_key, obs_key]),
    ]


def _build_piper_wristframe_transform_list(
    *,
    action_ee_pose: str = "action_ee_pose",
    action_gripper: str = "action_gripper",
    obs_ee_pose: str = "obs_ee_pose",
    obs_gripper: str = "obs_gripper",
    action_ee_pose_wristframe: str = "action_ee_pose_wristframe",
    actions_key: str = "actions_cartesian",
    obs_key: str = "observations.state.ee_pose",
    chunk_length: int = 100,
    stride: int = 1,
) -> list[Transform]:
    """Actions expressed relative to Piper's own current observed EEF pose
    ("wrist frame") instead of the absolute base_link frame -- the same
    embodiment-agnostic action representation Eva and Human already use for
    real cross-embodiment cotraining (see eva.py's
    `_build_eva_bimanual_eef_frame_transform_list` and human.py's
    `_build_human_cartesian_eef_frame_transform_list` /
    human_single_arm.py's `get_transform_list`).

    No extrinsics/calibration needed here, unlike Eva/Human's
    world->camframe->wristframe pipeline: `target_world^-1 @ chunk_world`
    cancels out whatever *fixed, consistent* frame the inputs share, and
    Piper's own base_link is already exactly that (it's static -- the FK
    root). Eva/Human only need the camframe detour because their "base"
    frame (Eva's own base_link / the human's world-frame SLAM pose) isn't
    otherwise meaningful across arms/episodes for the *observation* they
    want to keep; the wristframe *action* trick itself never required it.

    Proprio (`obs_key`) stays in base_link frame (informative, matches FK,
    and is consumed by a per-domain stem so it never needs to be
    cross-embodiment-comparable). Only the action -- which flows through
    HPT's *shared* action head -- becomes relative-pose.
    """
    return [
        ActionChunkCoordinateFrameTransform(
            target_world=obs_ee_pose,
            chunk_world=action_ee_pose,
            transformed_key_name=action_ee_pose_wristframe,
            mode="xyzwxyz",
        ),
        *_interpolate_chunk_transforms(
            action_ee_pose_wristframe, action_gripper, chunk_length=chunk_length, stride=stride
        ),
        XYZWXYZ_to_XYZ6D(keys=[action_ee_pose_wristframe, obs_ee_pose]),
        ConcatKeys(
            key_list=[action_ee_pose_wristframe, action_gripper],
            new_key_name=actions_key,
            delete_old_keys=True,
        ),
        ConcatKeys(
            key_list=[obs_ee_pose, obs_gripper],
            new_key_name=obs_key,
            delete_old_keys=True,
        ),
        DeleteKeys(keys_to_delete=[action_ee_pose]),
        NumpyToTensor(keys=[actions_key, obs_key]),
    ]


def _build_piper_camframe_transform_list(
    *,
    target_world: str = "extrinsics_pose",
    action_ee_pose: str = "action_ee_pose",
    action_gripper: str = "action_gripper",
    obs_ee_pose: str = "obs_ee_pose",
    obs_gripper: str = "obs_gripper",
    obs_ee_pose_camframe: str = "obs_ee_pose_camframe",
    action_ee_pose_camframe: str = "action_ee_pose_camframe",
    actions_key: str = "actions_cartesian",
    obs_key: str = "observations.state.ee_pose",
    chunk_length: int = 100,
    stride: int = 1,
) -> list[Transform]:
    """Re-expresses base_link-frame ee_pose in the (single) external camera's
    frame, mirroring Eva's per-arm camframe conversion. Requires `Piper.EXTRINSICS`
    to carry a real calibration -- see the class docstring.
    """
    from egomimic.rldb.embodiment.piper import Piper  # avoid a module-level cycle

    extrinsics = Piper.EXTRINSICS
    cam_key = next(iter(extrinsics))
    extrinsics_pose = _matrix_to_xyzwxyz(extrinsics[cam_key][None, :])[0]
    extra_batch_key = {target_world: extrinsics_pose}

    return [
        ActionChunkCoordinateFrameTransform(
            target_world=target_world,
            chunk_world=action_ee_pose,
            transformed_key_name=action_ee_pose_camframe,
            extra_batch_key=extra_batch_key,
            mode="xyzwxyz",
        ),
        *_interpolate_chunk_transforms(
            action_ee_pose_camframe, action_gripper, chunk_length=chunk_length, stride=stride
        ),
        # NOTE: no extra_batch_key here (PoseCoordinateFrameTransform doesn't
        # accept one) -- target_world is already in the batch by this point,
        # injected by the action transform above via its own extra_batch_key
        # (transforms run sequentially over the same mutated batch dict).
        PoseCoordinateFrameTransform(
            target_world=target_world,
            pose_world=obs_ee_pose,
            transformed_key_name=obs_ee_pose_camframe,
            mode="xyzwxyz",
        ),
        XYZWXYZ_to_XYZ6D(keys=[action_ee_pose_camframe, obs_ee_pose_camframe]),
        ConcatKeys(
            key_list=[action_ee_pose_camframe, action_gripper],
            new_key_name=actions_key,
            delete_old_keys=True,
        ),
        ConcatKeys(
            key_list=[obs_ee_pose_camframe, obs_gripper],
            new_key_name=obs_key,
            delete_old_keys=True,
        ),
        # Real bug found via audit: leaving the raw base_link `action_ee_pose`/
        # `obs_ee_pose` keys undeleted (unlike cartesian_base/cartesian_wristframe,
        # which consume them directly) let them leak into the norm-stats
        # inference pass -- which auto-discovers quantile bounds for whatever
        # keys remain in the transformed batch -- causing spurious "Bounds
        # violation" rejection-sampling retries against keys nobody meant to
        # constrain (harmless for the model itself, which only reads named
        # keys via _robomimic_to_hpt_data, but real added dataloader noise).
        DeleteKeys(keys_to_delete=[target_world, action_ee_pose, obs_ee_pose]),
        NumpyToTensor(keys=[actions_key, obs_key]),
    ]
