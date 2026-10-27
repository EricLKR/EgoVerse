"""Four-frame deployment wrapper layered on the audited joint-space policy.

The output/queue/filter/gripper behavior is inherited unchanged.  This class
only injects a chronological observation history immediately before checkpoint
normalization, while retaining the current frame under the legacy keys used by
fresh-feedback safety filtering.
"""

from __future__ import annotations

import types

import numpy as np
import torch

from lerobot.policies.egoverse_hpt.configuration_egoverse_hpt_joint_history_v2 import (
    EgoVerseHPTJointHistoryV2Config,
)
from lerobot.policies.egoverse_hpt.modeling_egoverse_hpt_joint import (
    EgoVerseHPTJointPolicy,
    prepare_rgb_chw_image,
)


def prepare_rgb_history(
    value,
    *,
    history: int,
    expected_height: int,
    expected_width: int,
    resize_mismatch: bool,
    name: str,
) -> torch.Tensor:
    """Convert one chronological image history to finite ``[T,C,H,W]``."""
    array = value.detach().cpu() if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if array.ndim == 5:
        if array.shape[0] != 1:
            raise ValueError(f"{name} batch dimension must be 1, got {tuple(array.shape)}")
        array = array[0]
    if array.ndim != 4 or array.shape[0] != history:
        raise ValueError(f"{name} must be [1,T,...] or [T,...] with T={history}, got {tuple(array.shape)}")
    frames = [
        prepare_rgb_chw_image(
            array[index],
            expected_height=expected_height,
            expected_width=expected_width,
            resize_mismatch=resize_mismatch,
            name=f"{name}[{index}]",
        )
        for index in range(history)
    ]
    return torch.stack(frames, dim=0)


class EgoVerseHPTJointHistoryV2Policy(EgoVerseHPTJointPolicy):
    config_class = EgoVerseHPTJointHistoryV2Config
    name = "egoverse_hpt_joint_history_v2"

    def _validate_checkpoint_contract(self) -> None:
        super()._validate_checkpoint_contract()
        model_history = int(getattr(self._algo, "observation_history", -1))
        trunk_history = int(
            getattr(self._algo.nets["policy"], "observation_horizon", -1)
        )
        expected = int(self.config.observation_history)
        if model_history != expected or trunk_history != expected:
            raise RuntimeError(
                "checkpoint is not a matching History V2 model: "
                f"adapter={model_history}, trunk={trunk_history}, expected={expected}"
            )
        self.checkpoint_report.update(
            {
                "observation_contract": "causal_rgb_proprio_history_v2",
                "observation_history": expected,
                "observation_history_step_ms": float(
                    self.config.observation_history_step_ms
                ),
            }
        )

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        history = int(self.config.observation_history)
        state_key = "observation.history_v2.state"
        front_history_key = (
            f"observation.history_v2.images.{self.config.front_camera_key}"
        )
        wrist_history_key = (
            f"observation.history_v2.images.{self.config.wrist_camera_key}"
        )
        missing = [
            key for key in (state_key, front_history_key, wrist_history_key) if key not in batch
        ]
        if missing:
            raise KeyError(f"History V2 inference batch lacks {missing}")

        raw_states = batch[state_key]
        raw_states = (
            raw_states.detach().cpu().numpy()
            if isinstance(raw_states, torch.Tensor)
            else np.asarray(raw_states)
        )
        if raw_states.ndim == 3:
            if raw_states.shape[0] != 1:
                raise ValueError(f"History V2 state batch must be one, got {raw_states.shape}")
            raw_states = raw_states[0]
        if raw_states.shape != (history, 7) or not np.isfinite(raw_states).all():
            raise ValueError(
                f"History V2 raw states must be finite ({history},7), got {raw_states.shape}"
            )
        joint_history = raw_states[:, :6] * 1e-3 * np.pi / 180.0
        gripper_history = np.clip(
            raw_states[:, 6:7] * 1e-3 / self.config.gripper_stroke_mm, 0.0, 1.0
        )
        physical_history = np.concatenate(
            [joint_history, gripper_history], axis=-1
        ).astype(np.float32)
        front_history = prepare_rgb_history(
            batch[front_history_key],
            history=history,
            expected_height=self.config.expected_image_height,
            expected_width=self.config.expected_image_width,
            resize_mismatch=self.config.resize_mismatched_images,
            name=front_history_key,
        )
        wrist_history = prepare_rgb_history(
            batch[wrist_history_key],
            history=history,
            expected_height=self.config.expected_image_height,
            expected_width=self.config.expected_image_width,
            resize_mismatch=self.config.resize_mismatched_images,
            name=wrist_history_key,
        )

        norm_stats = self._algo.norm_stats
        original_normalize = norm_stats.normalize

        def normalize_with_history(_stats_self, hpt_batch, embodiment_id):
            enriched = dict(hpt_batch)
            device = torch.device(self.config.device)
            enriched["observations.state.joints"] = torch.from_numpy(
                physical_history
            ).unsqueeze(0).to(device)
            enriched["observations.images.front_img_1"] = front_history.unsqueeze(
                0
            ).to(device)
            enriched["observations.images.wrist_img"] = wrist_history.unsqueeze(
                0
            ).to(device)
            return original_normalize(enriched, embodiment_id)

        norm_stats.normalize = types.MethodType(normalize_with_history, norm_stats)
        try:
            result = super().predict_action_chunk(batch, **kwargs)
        finally:
            norm_stats.normalize = original_normalize

        if self.last_prediction_report is not None:
            selection_ages = batch.get("observation.history_v2.selection_ages_ms")
            if isinstance(selection_ages, torch.Tensor):
                selection_ages = selection_ages.detach().cpu().reshape(-1).tolist()
            self.last_prediction_report["observation_history_v2"] = {
                "history": history,
                "raw_state_shape": list(raw_states.shape),
                "front_shape": list(front_history.shape),
                "wrist_shape": list(wrist_history.shape),
                "current_raw_state": raw_states[-1].tolist(),
                "selected_ages_ms": selection_ages,
            }
        return result
