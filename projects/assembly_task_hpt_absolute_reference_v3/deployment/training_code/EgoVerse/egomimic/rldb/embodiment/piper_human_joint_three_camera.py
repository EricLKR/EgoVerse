"""Three-camera PIPER mixed-training data contract for Assembly New1.

This is a new embodiment adapter rather than a modification of the proven
two-camera Assembly v1 adapter.  The numeric/action contract is inherited
unchanged; only the additional fixed side-camera image is exposed.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from egomimic.rldb.embodiment.piper import Piper
from egomimic.rldb.embodiment.piper_human_joint import PiperHumanJoint


def _install_recording_extrinsics() -> None:
    """Install the dataset-embedded mount calibration for smoke/preflight.

    The existing Assembly-v1 loader accepts only independently accepted files
    with ``status=verified``.  This new task uses a different camera mount and
    deliberately keeps the weaker provenance label instead of falsely marking
    it verified.  Formal launchers add a separate user-acceptance gate.
    """
    raw_path = os.environ.get("PIPER_RECORDING_EXTRINSICS_JSON")
    if not raw_path:
        return
    path = Path(raw_path).expanduser().resolve()
    payload = json.loads(path.read_text())
    if payload.get("status") != "recording_calibration":
        raise ValueError(f"unexpected recording extrinsics status: {path}")
    if payload.get("matrix_convention") != "base_world_from_camera_4x4":
        raise ValueError(f"unexpected recording extrinsics convention: {path}")
    matrix = np.asarray(payload.get("matrix"), dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"recording extrinsics must be finite 4x4: {path}")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise ValueError(f"recording extrinsics rotation is not orthonormal: {path}")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
        raise ValueError(f"recording extrinsics rotation determinant is not +1: {path}")
    Piper.EXTRINSICS = {str(payload.get("camera_key", "head")): matrix}


_install_recording_extrinsics()


class PiperHumanJointThreeCamera(PiperHumanJoint):
    @classmethod
    def _get_keymap(cls, keymap_mode: str):
        mapping = super()._get_keymap(keymap_mode)
        mapping["observations.images.side_img"] = {
            "key_type": "camera_keys",
            "zarr_key": "images.side_1",
        }
        return mapping
