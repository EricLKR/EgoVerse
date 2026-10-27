"""DINOv3-aware keymaps for the Piper/human HPT vision ablation.

The existing embodiment keymaps remain the source of truth for actions,
proprioception, and raw RGB naming.  These helpers only add precomputed DINOv3
patch tokens and reclassify raw RGB as metadata for DINO runs.  Keeping raw RGB
in each sample lets the offline evaluator render plots, while marking it as
metadata prevents :class:`egomimic.algo.hpt.HPT` from treating it as another
trainable camera modality.

The embedding pipeline writes ViT-L/16 patch tokens as ``(196, 1024)`` arrays:

* ``images.front_1`` -> ``dino.front_1`` -> ``dino_front_img``
* ``images.wrist_1`` -> ``dino.wrist_1`` -> ``dino_wrist_img``

Do not use these keymaps before the embeddings have been appended to every
episode in the requested train/validation split.
"""

from __future__ import annotations

from typing import Literal

from egomimic.rldb.embodiment.human_single_arm import get_keymap as get_human_keymap
from egomimic.rldb.embodiment.piper import Piper


PiperDINOv3Variant = Literal["front", "both"]


def _raw_cameras_as_metadata(key_map: dict) -> None:
    """Keep decoded RGB available to visualization without exposing it to HPT."""
    for spec in key_map.values():
        if spec.get("key_type") == "camera_keys" and str(
            spec.get("zarr_key", "")
        ).startswith("images."):
            spec["key_type"] = "metadata_keys"


def get_piper_dinov3_keymap(
    keymap_mode: str = "cartesian_base",
    variant: PiperDINOv3Variant = "front",
    norm_mode: bool = False,
    annotation_key: str | None = None,
) -> dict:
    """Return Piper keys for front-only or front+wrist DINOv3 tokens.

    ``norm_mode`` deliberately delegates to the unchanged Piper keymap and
    adds no DINO arrays.  The norm-stat pass therefore reads only action and
    proprioception, exactly like the ResNet baseline.
    """
    if variant not in ("front", "both"):
        raise ValueError(f"variant must be 'front' or 'both', got {variant!r}")

    key_map = Piper.get_keymap(
        keymap_mode=keymap_mode,
        norm_mode=norm_mode,
        annotation_key=annotation_key,
    )
    if norm_mode:
        return key_map

    _raw_cameras_as_metadata(key_map)
    key_map["dino_front_img"] = {
        "key_type": "camera_keys",
        "zarr_key": "dino.front_1",
    }
    if variant == "both":
        key_map["dino_wrist_img"] = {
            "key_type": "camera_keys",
            "zarr_key": "dino.wrist_1",
        }
    return key_map


def get_human_dinov3_keymap(
    keymap_mode: str = "cartesian_camframe_padded",
    has_head_pose: bool = True,
    arm: Literal["left", "right"] = "left",
    norm_mode: bool = False,
) -> dict:
    """Return the single-arm human keymap with front-camera DINOv3 tokens."""
    key_map = get_human_keymap(
        keymap_mode=keymap_mode,
        has_head_pose=has_head_pose,
        arm=arm,
        norm_mode=norm_mode,
    )
    if norm_mode:
        return key_map

    _raw_cameras_as_metadata(key_map)
    key_map["dino_front_img"] = {
        "key_type": "camera_keys",
        "zarr_key": "dino.front_1",
    }
    return key_map
