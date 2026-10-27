"""Native 30-step pi0.5 contract with front, wrist and fixed side images."""

from __future__ import annotations

from egomimic.rldb.embodiment.piper_human_joint_three_camera import (
    PiperHumanJointThreeCamera,
)


class PiperExpertActionV3ThreeCamera(PiperHumanJointThreeCamera):
    ACTION_HORIZON = 30
    CHUNK_LENGTH = 30

