"""Single-arm keymap/transform helpers for `Human` embodiment data that only
carries one side's `{arm}.obs_ee_pose` — no `left.*`/`right.*` counterpart at
all in the zarr store (e.g. `human_left_arm` episodes like
egoverse_jepa/aria_gen2_data/zarr). `Human`'s own cartesian keymap/transform
pipeline (egomimic/rldb/embodiment/human.py) unconditionally expects both
sides, so a plain `Human.get_keymap(...)` KeyErrors on this data.

Kept as standalone functions rather than a change to `human.py` — same
Transform primitives (from action_chunk_transforms), just single-sided.

The action is expressed in **wrist frame** (relative to the observer's own
currently-observed hand pose), mirroring
`human._build_human_cartesian_eef_frame_transform_list` and
`eva._build_eva_bimanual_eef_frame_transform_list` -- the actual
embodiment-agnostic action representation the paper's real cotraining uses.
An earlier version of this module stopped one step short, leaving the action
in **head frame** (world -> head, but never head -> wrist): that's still
anchored to wherever the demonstrator's head/camera happened to be pointing,
which is a *different* reference every episode and every frame, and isn't
comparable to Piper's action at all. Piper's matching mode is
`Piper.get_transform_list(mode="cartesian_wristframe")` (see piper.py), which
gets the same relative-pose property directly from base_link -- no
extrinsics needed on either side for the *action*; only the head-frame
projection of the human's *observation* needs `obs_head_pose`, since (unlike
Piper's static base_link) the human's raw ee_pose is expressed in a
per-episode SLAM/world frame that means nothing on its own. Output stays
XYZWXYZ quaternion internally then converts to 6D rotation (`XYZWXYZ_to_XYZ6D`),
matching Piper's convention, so the two cotrain cleanly via `PadGripperZeros`
in `cartesian_padded` mode (Human's ee_pose has no gripper; Piper's does).

Hydra usage (mirrors Human.get_keymap/get_transform_list's call signature):
    key_map:
      _target_: egomimic.rldb.embodiment.human_single_arm.get_keymap
      keymap_mode: cartesian_padded
      arm: left
    transform_list:
      _target_: egomimic.rldb.embodiment.human_single_arm.get_transform_list
      mode: cartesian_padded
      arm: left
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch

from egomimic.rldb.embodiment.human import Human
from egomimic.rldb.zarr.action_chunk_transforms import (
    ActionChunkCoordinateFrameTransform,
    ConcatKeys,
    DeleteKeys,
    InterpolatePose,
    NumpyToTensor,
    PoseCoordinateFrameTransform,
    Transform,
    XYZWXYZ_to_XYZ6D,
)

_SUPPORTED_MODES = ("cartesian", "cartesian_padded", "cartesian_camframe", "cartesian_camframe_padded")


class _PadTrailingZero(Transform):
    """Appends one zero channel to a key's last dim (..., D) -> (..., D+1).

    Unlike `PadGripperZeros` (bimanual-only: hardcoded 12D ypr -> 14D, zero
    inserted mid-array after each arm's 6 dims), this is a plain trailing pad
    for any shape/dtype -- used to give a gripper-less single-arm human pose
    (9D: xyz + 6D rotation, see XYZWXYZ_to_XYZ6D) the same trailing dim as
    Piper's pose+gripper (10D), so the two can share one action key/head for
    cotraining.
    """

    def __init__(self, key: str):
        self.key = key

    def transform(self, batch: dict) -> dict:
        arr = batch[self.key]
        if isinstance(arr, torch.Tensor):
            pad = torch.zeros((*arr.shape[:-1], 1), dtype=arr.dtype)
            batch[self.key] = torch.cat([arr, pad], dim=-1)
        else:
            arr = np.asarray(arr)
            pad = np.zeros((*arr.shape[:-1], 1), dtype=arr.dtype)
            batch[self.key] = np.concatenate([arr, pad], axis=-1)
        return batch


def get_keymap(
    keymap_mode: str = "cartesian",
    has_head_pose: bool = True,
    arm: Literal["left", "right"] = "left",
    norm_mode: bool = False,
):
    """``norm_mode``: mirrors Human/base Embodiment's flag, used by
    trainHydra's norm-stats pass -- strips camera_keys so that pass doesn't
    decode images it doesn't need.
    """
    if keymap_mode not in _SUPPORTED_MODES:
        raise ValueError(
            f"Unsupported keymap_mode {keymap_mode!r}; expected one of {_SUPPORTED_MODES}."
        )
    horizon = Human.ACTION_HORIZON
    key_map = {
        Human.VIZ_IMAGE_KEY: {"key_type": "camera_keys", "zarr_key": "images.front_1"},
        f"{arm}.action_ee_pose": {
            "key_type": "action_keys",
            "zarr_key": f"{arm}.obs_ee_pose",
            "horizon": horizon,
        },
        f"{arm}.obs_ee_pose": {"key_type": "proprio_keys", "zarr_key": f"{arm}.obs_ee_pose"},
    }
    if has_head_pose:
        key_map["obs_head_pose"] = {"key_type": "proprio_keys", "zarr_key": "obs_head_pose"}
    if norm_mode:
        to_delete = [k for k, v in key_map.items() if v.get("key_type") == "camera_keys"]
        for k in to_delete:
            del key_map[k]
    return key_map


def get_transform_list(
    mode: str = "cartesian",
    stride: int = 3,
    arm: Literal["left", "right"] = "left",
    chunk_length: int = 100,
    actions_key: str = "actions_cartesian",
) -> list[Transform]:
    """
    "cartesian"/"cartesian_padded": wrist-frame action (world->head->wrist,
        i.e. relative to the observer's own current hand pose). A VALID
        embodiment-agnostic representation, but NOT what the EgoVerse paper's
        own cotraining actually uses -- see the module docstring's history.
    "cartesian_camframe"/"cartesian_camframe_padded": head-frame ONLY
        (world->head, stopping there) -- the paper's actual canonical
        representation, matching Eva's real cotraining function
        (`_build_eva_bimanual_transform_list`, mode="cartesian") and Piper's
        `cartesian_camframe` mode exactly (both do a single frame projection
        for obs AND action, no further wrist-relative step). For egocentric
        human data, "camera frame" IS head frame -- the head-mounted device
        is the camera, so no separate extrinsics are needed the way Piper's
        external, non-moving camera needs `Piper.EXTRINSICS`.
    """
    if mode not in _SUPPORTED_MODES:
        raise ValueError(f"Unsupported mode {mode!r}; expected one of {_SUPPORTED_MODES}.")

    if mode in ("cartesian_camframe", "cartesian_camframe_padded"):
        return _build_human_camframe_transform_list(
            stride=stride,
            arm=arm,
            chunk_length=chunk_length,
            padded=(mode == "cartesian_camframe_padded"),
            actions_key=actions_key,
        )

    target_world = "obs_head_pose"
    action_world = f"{arm}.action_ee_pose"
    obs_pose = f"{arm}.obs_ee_pose"
    action_headframe = f"{arm}.action_ee_pose_headframe"
    obs_headframe = f"{arm}.obs_ee_pose_headframe"
    action_wristframe = f"{arm}.action_ee_pose_wristframe"
    obs_key = "observations.state.ee_pose"

    transform_list: list[Transform] = [
        # Stage 1: world (per-episode SLAM frame) -> head frame. Needed only
        # because the human's raw ee_pose isn't otherwise meaningful on its
        # own (unlike Piper's static base_link); this alone is NOT the fix
        # (see module docstring) -- it's just a well-defined common frame to
        # take the Stage 2 delta in.
        ActionChunkCoordinateFrameTransform(
            target_world=target_world,
            chunk_world=action_world,
            transformed_key_name=action_headframe,
            mode="xyzwxyz",
        ),
        PoseCoordinateFrameTransform(
            target_world=target_world,
            pose_world=obs_pose,
            transformed_key_name=obs_headframe,
            mode="xyzwxyz",
        ),
        InterpolatePose(
            new_chunk_length=chunk_length,
            action_key=action_headframe,
            output_action_key=action_headframe,
            stride=stride,
            mode="xyzwxyz",
        ),
        # Stage 2 (the actual fix): head frame -> wrist frame, i.e. the
        # action chunk expressed relative to the single currently-observed
        # hand pose (`obs_headframe`) rather than the moving head/camera.
        # This is the embodiment-agnostic representation -- matches Piper's
        # `cartesian_wristframe` mode and Eva/Human's real cotraining
        # pipelines (eva.py/human.py).
        ActionChunkCoordinateFrameTransform(
            target_world=obs_headframe,
            chunk_world=action_headframe,
            transformed_key_name=action_wristframe,
            mode="xyzwxyz",
        ),
        # quat->6D rotation (7D->9D): matches Piper's convention, and unlike
        # ypr has no yaw-wraparound discontinuity at +-pi -- that was
        # triggering constant "Bounds violation" rejection-sampling in the
        # shared action key. See egomimic/utils/pose_utils.py's
        # _matrix_to_xyz6d docstring.
        XYZWXYZ_to_XYZ6D(keys=[action_wristframe, obs_headframe]),
        ConcatKeys(key_list=[action_wristframe], new_key_name=actions_key, delete_old_keys=True),
        ConcatKeys(key_list=[obs_headframe], new_key_name=obs_key, delete_old_keys=True),
        DeleteKeys(keys_to_delete=[action_world, obs_pose, target_world, action_headframe]),
    ]
    if mode == "cartesian_padded":
        transform_list.append(_PadTrailingZero(key=actions_key))
    transform_list.append(NumpyToTensor(keys=[actions_key, obs_key]))
    return transform_list


def _build_human_camframe_transform_list(
    *,
    stride: int = 3,
    arm: Literal["left", "right"] = "left",
    chunk_length: int = 100,
    padded: bool = False,
    actions_key: str = "actions_cartesian",
) -> list[Transform]:
    """Stage 1 ONLY: world (per-episode SLAM frame) -> head frame, for BOTH
    obs and action -- matches Eva's canonical `_build_eva_bimanual_transform_list`
    (mode="cartesian") and Piper's `cartesian_camframe` mode exactly. This is
    the paper's actual cotraining representation (see piper.py's class
    docstring / the EgoVerse paper's a^H_(t:t+k) formula), NOT the wrist-frame
    variant `get_transform_list(mode="cartesian"/"cartesian_padded")` builds.

    No extrinsics needed here (unlike Piper): the human's "camera" is the
    head-mounted device itself, so head frame already IS camera frame for
    this egocentric data -- `obs_head_pose` is exactly the anchor Piper's
    fixed extrinsics matrix would otherwise be standing in for.
    """
    target_world = "obs_head_pose"
    action_world = f"{arm}.action_ee_pose"
    obs_pose = f"{arm}.obs_ee_pose"
    action_headframe = f"{arm}.action_ee_pose_headframe"
    obs_headframe = f"{arm}.obs_ee_pose_headframe"
    obs_key = "observations.state.ee_pose"

    transform_list: list[Transform] = [
        ActionChunkCoordinateFrameTransform(
            target_world=target_world,
            chunk_world=action_world,
            transformed_key_name=action_headframe,
            mode="xyzwxyz",
        ),
        PoseCoordinateFrameTransform(
            target_world=target_world,
            pose_world=obs_pose,
            transformed_key_name=obs_headframe,
            mode="xyzwxyz",
        ),
        InterpolatePose(
            new_chunk_length=chunk_length,
            action_key=action_headframe,
            output_action_key=action_headframe,
            stride=stride,
            mode="xyzwxyz",
        ),
        XYZWXYZ_to_XYZ6D(keys=[action_headframe, obs_headframe]),
        ConcatKeys(key_list=[action_headframe], new_key_name=actions_key, delete_old_keys=True),
        ConcatKeys(key_list=[obs_headframe], new_key_name=obs_key, delete_old_keys=True),
        DeleteKeys(keys_to_delete=[action_world, obs_pose, target_world]),
    ]
    if padded:
        transform_list.append(_PadTrailingZero(key=actions_key))
    transform_list.append(NumpyToTensor(keys=[actions_key, obs_key]))
    return transform_list
