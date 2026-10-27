import numpy as np
import pytest
import torch

from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset


def _dataset(*, reject_outliers: bool, error_on_nan_inf: bool = True) -> MultiDataset:
    stats = {
        "mean": np.zeros((2, 2), dtype=np.float32),
        "std": np.ones((2, 2), dtype=np.float32),
        "min": np.zeros((2, 2), dtype=np.float32),
        "max": np.ones((2, 2), dtype=np.float32),
        "median": np.full((2, 2), 0.5, dtype=np.float32),
        "quantile_1": np.zeros((2, 2), dtype=np.float32),
        "quantile_99": np.ones((2, 2), dtype=np.float32),
        "quantile_0_01": np.zeros((2, 2), dtype=np.float32),
        "quantile_99_99": np.ones((2, 2), dtype=np.float32),
    }
    state = {
        "norm_mode": "quantile",
        "embodiments": [7],
        "key_types": {7: {"actions_cartesian": "action_keys"}},
        "zarr_keys": {7: {"actions_cartesian": "actions_cartesian"}},
        "shapes": {7: {"actions_cartesian": (2, 2)}},
        "norm_stats": {7: {"actions_cartesian": stats}},
    }
    return MultiDataset(
        state=state,
        reject_outliers=reject_outliers,
        error_on_nan_inf=error_on_nan_inf,
    )


def test_outlier_rejection_can_be_disabled_without_disabling_normalization():
    sample = {
        "embodiment": 7,
        "actions_cartesian": torch.tensor([[0.2, 1.2], [0.5, 0.5]]),
    }
    rejecting = _dataset(reject_outliers=True)
    preserving = _dataset(reject_outliers=False)

    assert rejecting._check_bounds(sample, None, 4, "ep") is not None
    assert preserving._check_bounds(sample, None, 4, "ep") is None
    normalized = preserving.normalize(sample, 7)["actions_cartesian"]
    assert normalized[0, 1] > 1.0


def test_nan_inf_raises_even_when_outlier_rejection_is_disabled():
    sample = {
        "embodiment": 7,
        "actions_cartesian": torch.tensor([[0.2, float("nan")], [0.5, 0.5]]),
    }
    dataset = _dataset(reject_outliers=False, error_on_nan_inf=True)
    with pytest.raises(ValueError, match="NaN/Inf"):
        dataset._check_bounds(sample, None, 2, "ep")
