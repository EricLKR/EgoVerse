"""PIPER representation for joint-space deployment with human cotraining.

Each robot sample exposes two synchronized action targets:

* ``actions_cartesian_pose``: a 9D camera-frame pose trajectory shared with
  the human embodiment (xyz + continuous 6D rotation).  It is an auxiliary
  representation-learning target and deliberately contains no gripper bit.
* ``actions_joints``: six absolute commanded joint angles in radians followed
  by the binary gripper command.  This is the only head used for deployment.

The Cartesian target is derived from the stored FK audit signal ``cmd_ee_pose``
and the calibrated fixed front-camera extrinsics.  No IK is used in this data
path or in deployment.
"""

from __future__ import annotations

from typing import Literal

from egomimic.rldb.embodiment.piper import Piper
from egomimic.rldb.embodiment.piper_joint import PiperJoint
from egomimic.rldb.zarr.action_chunk_transforms import (
    ActionChunkCoordinateFrameTransform,
    ConcatKeys,
    DeleteKeys,
    InterpolateLinear,
    InterpolatePose,
    InterpolateStep,
    NumpyToTensor,
    Transform,
    XYZWXYZ_to_XYZ6D,
)
from egomimic.utils.pose_utils import _matrix_to_xyzwxyz


class PiperHumanJoint(PiperJoint):
    SHARED_ACTION_KEY = "actions_cartesian_pose"

    @classmethod
    def _get_keymap(cls, keymap_mode: str):
        if keymap_mode != "joint_cartesian_camframe":
            raise ValueError(
                "PiperHumanJoint requires keymap_mode='joint_cartesian_camframe', "
                f"got {keymap_mode!r}"
            )
        return {
            cls.VIZ_IMAGE_KEY: {
                "key_type": "camera_keys",
                "zarr_key": "images.front_1",
            },
            "observations.images.wrist_img": {
                "key_type": "camera_keys",
                "zarr_key": "images.wrist_1",
            },
            "action_joint": {
                "key_type": "action_keys",
                "zarr_key": "cmd_joint",
                "horizon": cls.ACTION_HORIZON,
            },
            "action_gripper": {
                "key_type": "action_keys",
                "zarr_key": "cmd_gripper",
                "horizon": cls.ACTION_HORIZON,
            },
            "action_ee_pose": {
                "key_type": "action_keys",
                "zarr_key": "cmd_ee_pose",
                "horizon": cls.ACTION_HORIZON,
            },
            "obs_joint": {"key_type": "proprio_keys", "zarr_key": "obs_joint"},
            "obs_gripper": {
                "key_type": "proprio_keys",
                "zarr_key": "obs_gripper",
            },
        }

    @classmethod
    def get_transform_list(
        cls,
        mode: Literal["joint_cartesian_camframe"] = "joint_cartesian_camframe",
        stride: int = 1,
    ) -> list[Transform]:
        if mode != "joint_cartesian_camframe":
            raise ValueError(
                "PiperHumanJoint requires mode='joint_cartesian_camframe', "
                f"got {mode!r}"
            )
        if Piper.EXTRINSICS is None or not Piper.EXTRINSICS:
            raise ValueError(
                "PiperHumanJoint requires calibrated Piper.EXTRINSICS for the "
                "shared camera-frame pose target"
            )

        target_world = "extrinsics_pose"
        action_pose_cam = "action_ee_pose_camframe"
        cam_key = next(iter(Piper.EXTRINSICS))
        extrinsics_pose = _matrix_to_xyzwxyz(Piper.EXTRINSICS[cam_key][None, :])[0]

        return [
            ActionChunkCoordinateFrameTransform(
                target_world=target_world,
                chunk_world="action_ee_pose",
                transformed_key_name=action_pose_cam,
                extra_batch_key={target_world: extrinsics_pose},
                mode="xyzwxyz",
            ),
            InterpolatePose(
                new_chunk_length=cls.CHUNK_LENGTH,
                action_key=action_pose_cam,
                output_action_key=action_pose_cam,
                stride=stride,
                mode="xyzwxyz",
            ),
            XYZWXYZ_to_XYZ6D(keys=[action_pose_cam]),
            ConcatKeys(
                key_list=[action_pose_cam],
                new_key_name=cls.SHARED_ACTION_KEY,
                delete_old_keys=True,
            ),
            InterpolateLinear(
                new_chunk_length=cls.CHUNK_LENGTH,
                action_key="action_joint",
                output_action_key="action_joint",
                stride=stride,
            ),
            InterpolateStep(
                new_chunk_length=cls.CHUNK_LENGTH,
                action_key="action_gripper",
                output_action_key="action_gripper",
                stride=stride,
            ),
            ConcatKeys(
                key_list=["action_joint", "action_gripper"],
                new_key_name=cls.ACTION_KEY,
                delete_old_keys=True,
            ),
            ConcatKeys(
                key_list=["obs_joint", "obs_gripper"],
                new_key_name=cls.PROPRIO_KEY,
                delete_old_keys=True,
            ),
            DeleteKeys(keys_to_delete=[target_world, "action_ee_pose"]),
            NumpyToTensor(
                keys=[cls.SHARED_ACTION_KEY, cls.ACTION_KEY, cls.PROPRIO_KEY]
            ),
        ]

