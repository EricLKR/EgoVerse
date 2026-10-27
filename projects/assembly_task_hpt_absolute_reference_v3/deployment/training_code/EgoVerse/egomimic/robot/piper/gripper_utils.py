"""PIPER gripper-command encoding helpers.

The recorded gripper observation is a physical aperture and must remain
continuous: when the fingers close around an object, the measured aperture can
stay well above zero.  Policy targets instead represent the operator's binary
open/close intent.  Values between the two thresholds retain the previous
state so safety-clamped command ramps do not create a third, "half-open" class.
"""

from __future__ import annotations

import numpy as np


def update_binary_gripper_state(
    command: float,
    previous_state: float | None = None,
    *,
    close_threshold: float = 0.2,
    open_threshold: float = 0.8,
) -> float:
    """Apply one open/close hysteresis update to a normalized command.

    ``0`` means closed and ``1`` means open.  If no previous state is
    available, the midpoint between the thresholds initializes the state.
    """
    if not 0.0 <= close_threshold < open_threshold <= 1.0:
        raise ValueError(
            "Expected 0 <= close_threshold < open_threshold <= 1, got "
            f"{close_threshold=} and {open_threshold=}"
        )
    value = float(command)
    if not np.isfinite(value):
        raise ValueError(f"Gripper command must be finite, got {command!r}")
    if value < 0.0 or value > 1.0:
        raise ValueError(f"Gripper command must be in [0, 1], got {value}")

    if previous_state is None:
        previous_state = float(value >= (close_threshold + open_threshold) / 2.0)
    elif previous_state not in (0, 0.0, 1, 1.0):
        raise ValueError(f"previous_state must be binary, got {previous_state!r}")

    if value <= close_threshold:
        return 0.0
    if value >= open_threshold:
        return 1.0
    return float(previous_state)


def binarize_gripper_commands(
    commands: np.ndarray,
    *,
    close_threshold: float = 0.2,
    open_threshold: float = 0.8,
    initial_state: float | None = None,
) -> np.ndarray:
    """Convert a time-ordered normalized command stream to binary intent.

    The returned array has the same shape as ``commands`` and ``float32``
    dtype.  The input may be ``(T,)`` or ``(T, 1)``.
    """
    values = np.asarray(commands, dtype=np.float64)
    if values.ndim not in (1, 2) or (values.ndim == 2 and values.shape[1] != 1):
        raise ValueError(f"Expected gripper commands with shape (T,) or (T, 1), got {values.shape}")
    if values.shape[0] == 0:
        raise ValueError("Gripper command sequence must not be empty")

    flat = values.reshape(-1)
    output = np.empty(flat.shape, dtype=np.float32)
    state = initial_state
    for index, value in enumerate(flat):
        state = update_binary_gripper_state(
            float(value),
            state,
            close_threshold=close_threshold,
            open_threshold=open_threshold,
        )
        output[index] = state
    return output.reshape(values.shape)
