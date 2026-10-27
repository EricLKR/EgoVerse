from collections import OrderedDict

import pytest
import torch

from egomimic.algo import pi_egoverse_multihead_v2 as multihead
from egomimic.models.preprocess_pi_obs import _SimpleObservation


def _observation(history: int) -> _SimpleObservation:
    batch = 2
    return _SimpleObservation(
        images=OrderedDict(
            base_0_rgb=torch.zeros(batch, history, 3, 8, 8),
            left_wrist_0_rgb=torch.ones(batch, history, 3, 8, 8),
        ),
        image_masks=OrderedDict(
            base_0_rgb=torch.ones(batch, dtype=torch.bool),
            left_wrist_0_rgb=torch.ones(batch, dtype=torch.bool),
        ),
        state=torch.zeros(batch, 7),
        tokenized_prompt=torch.zeros(batch, 4, dtype=torch.long),
        tokenized_prompt_mask=torch.ones(batch, 4, dtype=torch.bool),
        token_ar_mask=torch.zeros(batch, 4, dtype=torch.bool),
        token_loss_mask=torch.zeros(batch, 4, dtype=torch.bool),
    )


def test_single_frame_preprocess_returns_openpi_five_tuple(monkeypatch):
    monkeypatch.setattr(
        multihead.openpi_preprocessing,
        "preprocess_observation_pytorch",
        lambda observation, train: observation,
    )
    output = multihead._single_frame_preprocess_observation(
        object(), _observation(1), train=True
    )
    assert isinstance(output, tuple) and len(output) == 5
    images, masks, prompt, prompt_mask, state = output
    assert [tuple(image.shape) for image in images] == [
        (2, 1, 3, 8, 8),
        (2, 1, 3, 8, 8),
    ]
    assert [tuple(mask.shape) for mask in masks] == [(2,), (2,)]
    assert prompt.shape == prompt_mask.shape == (2, 4)
    assert state.shape == (2, 7)


def test_native_bchw_preprocess_also_returns_openpi_five_tuple(monkeypatch):
    monkeypatch.setattr(
        multihead.openpi_preprocessing,
        "preprocess_observation_pytorch",
        lambda observation, train: observation,
    )
    observation = _observation(1)
    observation.images = OrderedDict(
        (key, image[:, 0]) for key, image in observation.images.items()
    )
    output = multihead._single_frame_preprocess_observation(
        object(), observation, train=False
    )
    assert isinstance(output, tuple) and len(output) == 5
    assert [tuple(image.shape) for image in output[0]] == [
        (2, 3, 8, 8),
        (2, 3, 8, 8),
    ]


def test_single_frame_preprocess_rejects_history(monkeypatch):
    monkeypatch.setattr(
        multihead.openpi_preprocessing,
        "preprocess_observation_pytorch",
        lambda observation, train: observation,
    )
    with pytest.raises(ValueError, match="requires T=1"):
        multihead._single_frame_preprocess_observation(
            object(), _observation(2), train=False
        )
