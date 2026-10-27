import numpy as np
import pytest

from egomimic.robot.piper.gripper_utils import (
    binarize_gripper_commands,
    update_binary_gripper_state,
)


def test_binarize_gripper_commands_uses_hysteresis() -> None:
    commands = np.array([[1.0], [0.7], [0.4], [0.2], [0.5], [0.8], [0.6]])

    result = binarize_gripper_commands(commands)

    np.testing.assert_array_equal(
        result,
        np.array([[1.0], [1.0], [1.0], [0.0], [0.0], [1.0], [1.0]], dtype=np.float32),
    )


def test_binarize_gripper_commands_initializes_from_midpoint() -> None:
    np.testing.assert_array_equal(
        binarize_gripper_commands(np.array([0.4, 0.3, 0.8])),
        np.array([0.0, 0.0, 1.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        binarize_gripper_commands(np.array([0.6, 0.4, 0.2])),
        np.array([1.0, 1.0, 0.0], dtype=np.float32),
    )


def test_update_binary_gripper_state_rejects_invalid_thresholds() -> None:
    with pytest.raises(ValueError, match="close_threshold"):
        update_binary_gripper_state(0.5, close_threshold=0.8, open_threshold=0.2)


def test_binarize_gripper_commands_rejects_out_of_range_values() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        binarize_gripper_commands(np.array([0.0, 1.1]))
