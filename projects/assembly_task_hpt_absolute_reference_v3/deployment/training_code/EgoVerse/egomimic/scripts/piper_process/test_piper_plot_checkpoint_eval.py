import numpy as np

from egomimic.scripts.piper_process.piper_plot_checkpoint_eval import (
    _metrics,
    _reconstruct_receding_horizon,
    _rotation_error_deg,
)


def _identity_pose(n: int) -> np.ndarray:
    pose = np.zeros((n, 9), dtype=np.float32)
    pose[:, 3] = 1.0
    pose[:, 7] = 1.0
    return pose


def test_rotation_error_identity_is_zero():
    pose = _identity_pose(3)
    np.testing.assert_allclose(_rotation_error_deg(pose, pose), 0.0, atol=1e-8)


def test_receding_horizon_matches_rollout_rounding():
    chunk0 = np.repeat(np.arange(100, dtype=np.float32)[:, None], 10, axis=1)
    chunk4 = chunk0 + 1000.0
    pred, source = _reconstruct_receding_horizon(
        {0: chunk0, 4: chunk4}, episode_frames=7, query_frequency=4, action_horizon_frames=10
    )
    np.testing.assert_array_equal(source, [0, 0, 0, 0, 4, 4, 4])
    np.testing.assert_array_equal(pred[:, 0], [0, 10, 20, 30, 1000, 1010, 1020])


def test_metrics_perfect_prediction():
    gt = np.concatenate([_identity_pose(5), np.array([[1], [1], [0], [0], [1]])], axis=1)
    result = _metrics(gt.copy(), gt)
    assert result["frames"] == 5
    assert result["position_l2_mean_m"] == 0.0
    assert result["rotation_error_mean_deg"] == 0.0
    assert result["gripper_command_accuracy"] == 1.0
    assert result["gripper_raw_brier"] == 0.0
