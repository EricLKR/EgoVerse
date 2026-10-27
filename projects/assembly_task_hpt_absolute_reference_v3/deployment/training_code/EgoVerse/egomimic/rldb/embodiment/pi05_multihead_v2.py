"""Four-frame key maps/transforms for EgoVerse pi0.5 MultiHead V2."""

from __future__ import annotations

import copy
from typing import Literal

import numpy as np

from egomimic.rldb.embodiment.human import Human
from egomimic.rldb.embodiment.piper_human_joint import PiperHumanJoint
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


def _add_history(
    key_map: dict, observation_history: int, observation_stride: int, norm_mode: bool
) -> dict:
    if observation_history < 2 or observation_stride < 1:
        raise ValueError("MultiHead V2 requires history>=2 and stride>=1")
    result = copy.deepcopy(key_map)
    for key, info in list(result.items()):
        if info.get("key_type") in ("camera_keys", "proprio_keys"):
            info["history"] = int(observation_history)
            info["history_stride"] = int(observation_stride)
        if norm_mode and info.get("key_type") in ("camera_keys", "annotation_keys"):
            del result[key]
    return result


def get_piper_keymap(
    keymap_mode: str = "joint_cartesian_camframe",
    norm_mode: bool = False,
    annotation_key=None,
    observation_history: int = 4,
    observation_stride: int = 3,
) -> dict:
    key_map = PiperHumanJoint.get_keymap(
        keymap_mode=keymap_mode,
        norm_mode=False,
        annotation_key=annotation_key,
    )
    return _add_history(
        key_map, observation_history, observation_stride, norm_mode
    )


class _ActionFromCurrentHead(Transform):
    def __init__(self, target_history: str, chunk_world: str, output_key: str):
        self.target_history = target_history
        self.chunk_world = chunk_world
        self.output_key = output_key

    def transform(self, batch: dict) -> dict:
        history = np.asarray(batch[self.target_history])
        if history.ndim != 2 or history.shape[-1] != 7:
            raise ValueError(f"head history must be [T,7], got {history.shape}")
        helper = ActionChunkCoordinateFrameTransform(
            target_world="_pi05_v2_current_head",
            chunk_world=self.chunk_world,
            transformed_key_name=self.output_key,
            mode="xyzwxyz",
        )
        temporary = dict(batch)
        temporary["_pi05_v2_current_head"] = history[-1]
        temporary = helper.transform(temporary)
        batch[self.output_key] = temporary[self.output_key]
        return batch


class _PoseHistoryInHeadFrame(Transform):
    def __init__(self, head_history: str, pose_history: str, output_key: str):
        self.head_history = head_history
        self.pose_history = pose_history
        self.output_key = output_key

    def transform(self, batch: dict) -> dict:
        heads = np.asarray(batch[self.head_history])
        poses = np.asarray(batch[self.pose_history])
        if heads.shape != poses.shape or heads.ndim != 2 or heads.shape[-1] != 7:
            raise ValueError("head and hand histories must be synchronized [T,7]")
        output = []
        for head, pose in zip(heads, poses, strict=True):
            helper = PoseCoordinateFrameTransform(
                target_world="_pi05_v2_head",
                pose_world="_pi05_v2_hand",
                transformed_key_name="_pi05_v2_pose",
                mode="xyzwxyz",
            )
            transformed = helper.transform(
                {"_pi05_v2_head": head, "_pi05_v2_hand": pose}
            )
            output.append(transformed["_pi05_v2_pose"])
        batch[self.output_key] = np.stack(output, axis=0)
        return batch


def get_human_keymap(
    keymap_mode: str = "cartesian_camframe",
    has_head_pose: bool = True,
    arm: Literal["left", "right"] = "left",
    norm_mode: bool = False,
    observation_history: int = 4,
    observation_stride: int = 3,
) -> dict:
    if keymap_mode != "cartesian_camframe" or not has_head_pose:
        raise ValueError("MultiHead V2 human data requires camera-frame head pose")
    history = {
        "history": int(observation_history),
        "history_stride": int(observation_stride),
    }
    key_map = {
        Human.VIZ_IMAGE_KEY: {
            "key_type": "camera_keys",
            "zarr_key": "images.front_1",
            **history,
        },
        f"{arm}.action_ee_pose": {
            "key_type": "action_keys",
            "zarr_key": f"{arm}.obs_ee_pose",
            "horizon": Human.ACTION_HORIZON,
        },
        f"{arm}.obs_ee_pose": {
            "key_type": "proprio_keys",
            "zarr_key": f"{arm}.obs_ee_pose",
            **history,
        },
        "obs_head_pose": {
            "key_type": "proprio_keys",
            "zarr_key": "obs_head_pose",
            **history,
        },
    }
    if norm_mode:
        del key_map[Human.VIZ_IMAGE_KEY]
    return key_map


def get_human_transform_list(
    mode: str = "cartesian_camframe",
    stride: int = 3,
    arm: Literal["left", "right"] = "left",
    chunk_length: int = 100,
    actions_key: str = "actions_cartesian_pose",
) -> list[Transform]:
    if mode != "cartesian_camframe":
        raise ValueError("MultiHead V2 human transform requires cartesian_camframe")
    action_world = f"{arm}.action_ee_pose"
    pose_world = f"{arm}.obs_ee_pose"
    action_camera = f"{arm}.action_ee_pose_camframe"
    pose_camera = f"{arm}.obs_ee_pose_camframe"
    state_key = "observations.state.ee_pose"
    return [
        _ActionFromCurrentHead("obs_head_pose", action_world, action_camera),
        _PoseHistoryInHeadFrame("obs_head_pose", pose_world, pose_camera),
        InterpolatePose(
            new_chunk_length=chunk_length,
            action_key=action_camera,
            output_action_key=action_camera,
            stride=stride,
            mode="xyzwxyz",
        ),
        XYZWXYZ_to_XYZ6D(keys=[action_camera, pose_camera]),
        ConcatKeys(
            key_list=[action_camera],
            new_key_name=actions_key,
            delete_old_keys=True,
        ),
        ConcatKeys(
            key_list=[pose_camera], new_key_name=state_key, delete_old_keys=True
        ),
        DeleteKeys(keys_to_delete=[action_world, pose_world, "obs_head_pose"]),
        NumpyToTensor(keys=[actions_key, state_key]),
    ]
