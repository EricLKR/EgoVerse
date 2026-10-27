"""Native 30-step data contracts for pi0.5 Expert-Action V3."""

from __future__ import annotations

from egomimic.rldb.embodiment.piper_human_joint import PiperHumanJoint


class PiperExpertActionV3(PiperHumanJoint):
    """PIPER shared-pose + joint data without 30-to-100 interpolation."""

    ACTION_HORIZON = 30
    CHUNK_LENGTH = 30

