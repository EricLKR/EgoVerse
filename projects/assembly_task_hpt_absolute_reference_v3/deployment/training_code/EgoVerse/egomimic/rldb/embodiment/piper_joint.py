"""PIPER joint-space training representation.

Actions are six calibrated joint commands in radians plus a binary gripper
command.  Proprioception is six measured joints in radians plus the continuous
measured gripper aperture.  ``cmd_ee_pose`` remains stored in each Zarr episode
as an FK-derived audit/evaluation signal, but is deliberately not a model input
or target.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from egomimic.rldb.embodiment.embodiment import get_embodiment
from egomimic.rldb.embodiment.piper import Piper
from egomimic.rldb.zarr.action_chunk_transforms import (
    ConcatKeys,
    InterpolateLinear,
    InterpolateStep,
    NumpyToTensor,
    Transform,
)
from egomimic.utils.type_utils import _to_numpy


class PiperJoint(Piper):
    ACTION_KEY = "actions_joints"
    PROPRIO_KEY = "observations.state.joints"

    @classmethod
    def _get_keymap(cls, keymap_mode: str):
        if keymap_mode != "joint":
            raise ValueError(
                f"Unsupported keymap_mode {keymap_mode!r}; PiperJoint requires 'joint'"
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
            "obs_joint": {"key_type": "proprio_keys", "zarr_key": "obs_joint"},
            "obs_gripper": {
                "key_type": "proprio_keys",
                "zarr_key": "obs_gripper",
            },
        }

    @classmethod
    def get_transform_list(
        cls, mode: Literal["joint"] = "joint", stride: int = 1
    ) -> list[Transform]:
        if mode != "joint":
            raise ValueError(f"PiperJoint requires mode='joint', got {mode!r}")
        return [
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
            NumpyToTensor(keys=[cls.ACTION_KEY, cls.PROPRIO_KEY]),
        ]

    @classmethod
    def viz_gt_preds(
        cls,
        predictions,
        batch,
        image_key,
        action_key=ACTION_KEY,
        **kwargs,
    ):
        """Render six joint curves and binary gripper GT/prediction for validation."""
        import matplotlib.pyplot as plt

        embodiment_id = int(batch["embodiment"].reshape(-1)[0].item())
        embodiment_name = get_embodiment(embodiment_id).lower()
        gt = _to_numpy(batch[action_key])
        pred = _to_numpy(predictions[f"{embodiment_name}_{action_key}"])
        images = []
        for gt_i, pred_i in zip(gt, pred):
            fig, axes = plt.subplots(4, 2, figsize=(10, 8), sharex=True)
            axes = axes.reshape(-1)
            frames = np.arange(gt_i.shape[0])
            for joint_index in range(6):
                axes[joint_index].plot(
                    frames, np.rad2deg(gt_i[:, joint_index]), color="black", label="GT"
                )
                axes[joint_index].plot(
                    frames,
                    np.rad2deg(pred_i[:, joint_index]),
                    color="#d62728",
                    label="model",
                )
                axes[joint_index].set_ylabel(f"J{joint_index + 1} (deg)")
                axes[joint_index].grid(alpha=0.25)
            axes[6].step(frames, gt_i[:, 6] >= 0.5, where="post", color="black", label="GT")
            axes[6].step(
                frames, pred_i[:, 6] >= 0.5, where="post", color="#1b9e77", label="model"
            )
            axes[6].set_yticks([0, 1], labels=["close", "open"])
            axes[6].set_ylabel("gripper")
            axes[6].grid(alpha=0.25)
            axes[6].set_xlabel("action tick")
            axes[7].axis("off")
            axes[0].legend(loc="best", fontsize=8)
            axes[6].legend(loc="best", fontsize=8)
            fig.tight_layout()
            fig.canvas.draw()
            rgba = np.asarray(fig.canvas.buffer_rgba())
            images.append(rgba[..., :3].copy())
            plt.close(fig)
        return np.stack(images)
