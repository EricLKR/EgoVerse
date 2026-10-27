"""Causal observation-history reader for EgoVerse pi0.5 MultiHead V2.

This is an opt-in dataset class.  A key-map entry may declare ``history`` and
``history_stride``; action ``horizon`` semantics remain unchanged.  Declaring
both on one key is rejected to prevent future-frame leakage into observations.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import simplejpeg
import torch

from egomimic.rldb.embodiment.embodiment import get_embodiment_id
from egomimic.rldb.zarr.zarr_dataset_multi import (
    LocalEpisodeResolver,
    ZarrDataset,
    get_fallback_idx,
)

logger = logging.getLogger(__name__)


def causal_history_indices(index: int, history: int, stride: int) -> list[int]:
    if index < 0 or history < 1 or stride < 1:
        raise ValueError(
            f"invalid causal history index={index}, history={history}, stride={stride}"
        )
    return [
        max(0, index - stride * offset)
        for offset in range(history - 1, -1, -1)
    ]


class ZarrPi05MultiHeadV2Dataset(ZarrDataset):
    @staticmethod
    def _history_spec(info: dict) -> tuple[int | None, int]:
        history = info.get("history")
        stride = int(info.get("history_stride", 1))
        if history is None:
            return None, stride
        if info.get("horizon") is not None:
            raise ValueError("a key cannot define both history and horizon")
        history = int(history)
        if history < 1 or stride < 1:
            raise ValueError("history and history_stride must be positive")
        return history, stride

    def _read_history(self, zarr_key: str, indices: list[int]):
        array = self.episode_reader._store[zarr_key]
        return [array[index : index + 1][0] for index in indices]

    def __getitem__(
        self,
        idx: int,
        _fallback_origin: int | None = None,
        _attempts: int | None = None,
    ) -> dict[str, torch.Tensor]:
        origin = _fallback_origin if _fallback_origin is not None else idx
        attempts = _attempts

        def next_index(reason: str, key: str = "") -> int:
            nonlocal attempts
            candidate, attempts = get_fallback_idx(
                idx=idx,
                candidates=range(self.total_frames),
                _attempts=attempts,
                max_attempts=self.total_frames,
                exhausted_error=f"no valid history sample in {self.episode_path}",
            )
            logger.warning(
                "%s ep=%s frame=%d key=%s attempt=%d -> %d",
                reason,
                Path(self.episode_path).name,
                idx,
                key,
                attempts,
                candidate,
            )
            return candidate

        while True:
            data = {}
            retry = False
            for key, info in self.key_map.items():
                zarr_key = info["zarr_key"]
                key_type = info.get("key_type")
                horizon = info.get("horizon")
                history, history_stride = self._history_spec(info)
                if key_type == "annotation_keys":
                    if history is not None:
                        raise ValueError("annotation keys cannot have observation history")
                    data[key] = self._annotation_text_for_frame(idx)
                    continue
                if history is not None:
                    indices = causal_history_indices(idx, history, history_stride)
                    values = self._read_history(zarr_key, indices)
                    if zarr_key in self._image_keys:
                        decoded = []
                        try:
                            for jpeg in values:
                                rgb = simplejpeg.decode_jpeg(jpeg, colorspace="RGB")
                                decoded.append(
                                    np.transpose(rgb, (2, 0, 1)).astype(np.float32)
                                    / 255.0
                                )
                        except Exception:
                            idx = next_index("history JPEG decode failed", key)
                            retry = True
                            break
                        data[key] = np.stack(decoded, axis=0)
                    elif zarr_key in self._json_keys:
                        data[key] = [self._decode_json_entry(v) for v in values]
                    else:
                        data[key] = np.stack(
                            [np.asarray(value) for value in values], axis=0
                        )
                    continue

                interval = (
                    (idx, self._chunk_end_idx(idx, horizon, key_type))
                    if horizon is not None
                    else (idx, None)
                )
                raw = self.episode_reader.read({zarr_key: interval})
                self._pad_sequences(raw, horizon)
                value = raw[zarr_key]
                if zarr_key in self._image_keys:
                    try:
                        value = simplejpeg.decode_jpeg(value, colorspace="RGB")
                    except Exception:
                        idx = next_index("JPEG decode failed", key)
                        retry = True
                        break
                    value = np.transpose(value, (2, 0, 1)).astype(np.float32) / 255.0
                elif zarr_key in self._json_keys:
                    value = (
                        [self._decode_json_entry(v) for v in value]
                        if isinstance(value, np.ndarray)
                        else self._decode_json_entry(value)
                    )
                data[key] = value
            if retry:
                continue
            for transform in self.transform or []:
                data = transform.transform(data)
            for key, value in data.items():
                if isinstance(value, np.ndarray):
                    data[key] = torch.from_numpy(value).float()
            data["embodiment"] = get_embodiment_id(self.embodiment)
            intrinsics = self.intrinsics
            if isinstance(intrinsics, dict):
                intrinsics = next(
                    (v for k, v in intrinsics.items() if "front" in str(k).lower()),
                    next(iter(intrinsics.values()), None) if intrinsics else None,
                )
            intrinsics = np.asarray(intrinsics, dtype=np.float32) if intrinsics is not None else None
            if intrinsics is not None and intrinsics.shape == (3, 3):
                intrinsics = np.concatenate(
                    (intrinsics, np.zeros((3, 1), dtype=np.float32)), axis=1
                )
            if intrinsics is None or intrinsics.shape != (3, 4):
                intrinsics = np.full((3, 4), np.nan, dtype=np.float32)
            data["intrinsics"] = torch.from_numpy(np.ascontiguousarray(intrinsics))
            episode_name = Path(self.episode_path).name
            data["episode_hash"] = (
                episode_name[:-5] if episode_name.endswith(".zarr") else episode_name
            )
            _ = origin
            return data


class LocalPi05MultiHeadV2EpisodeResolver(LocalEpisodeResolver):
    _dataset_class = ZarrPi05MultiHeadV2Dataset
