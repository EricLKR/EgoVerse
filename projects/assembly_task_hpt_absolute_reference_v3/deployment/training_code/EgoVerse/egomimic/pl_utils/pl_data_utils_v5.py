"""Episode-balanced loaders for Mixed CamFrame V5.

The legacy loader shuffles a flattened frame index, so long robot episodes
contribute more samples than short human episodes.  V5 first samples each
episode a bounded number of times and only then chooses query frames.  The two
domains are still delivered separately by Lightning's CombinedLoader.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterator, Mapping

import torch
from lightning import LightningDataModule
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from torch.utils.data import DataLoader, Sampler

from egomimic.pl_utils.pl_data_utils import annotation_collate


class EpisodeUniformWindowSampler(Sampler[int]):
    """Sample a fixed number of full-horizon windows from every episode."""

    def __init__(
        self,
        dataset,
        *,
        samples_per_episode: int,
        future_horizon: int = 30,
        min_query_spacing: int = 1,
        seed: int = 42,
    ) -> None:
        self.dataset = dataset
        self.samples_per_episode = int(samples_per_episode)
        self.future_horizon = int(future_horizon)
        self.min_query_spacing = int(min_query_spacing)
        self.seed = int(seed)
        self.epoch = 0
        if self.samples_per_episode <= 0:
            raise ValueError("samples_per_episode must be positive")
        if self.future_horizon <= 0 or self.min_query_spacing <= 0:
            raise ValueError("future_horizon and min_query_spacing must be positive")
        groups = getattr(dataset, "_global_indices_by_dataset", None)
        if not isinstance(groups, Mapping) or not groups:
            raise TypeError("EpisodeUniformWindowSampler requires a MultiDataset")
        self._valid_by_episode: dict[str, list[int]] = {}
        for episode, global_indices in sorted(groups.items()):
            valid_count = len(global_indices) - self.future_horizon + 1
            if valid_count <= 0:
                raise ValueError(
                    f"episode {episode!r} has {len(global_indices)} frames, "
                    f"shorter than horizon {self.future_horizon}"
                )
            self._valid_by_episode[episode] = list(global_indices[:valid_count])

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _sample_episode(self, candidates: list[int], rng: random.Random) -> list[int]:
        target = min(self.samples_per_episode, len(candidates))
        order = list(candidates)
        rng.shuffle(order)
        selected: list[int] = []
        for candidate in order:
            if all(
                abs(candidate - previous) >= self.min_query_spacing
                for previous in selected
            ):
                selected.append(candidate)
                if len(selected) == target:
                    break
        if len(selected) < target:
            remaining = [value for value in order if value not in selected]
            selected.extend(remaining[: target - len(selected)])
        return selected

    def _all_indices(self) -> list[int]:
        rng = random.Random(self.seed + self.epoch)
        indices: list[int] = []
        for candidates in self._valid_by_episode.values():
            indices.extend(self._sample_episode(candidates, rng))
        rng.shuffle(indices)
        return indices

    def __iter__(self) -> Iterator[int]:
        indices = self._all_indices()
        # Lightning does not guarantee that LightningDataModule epoch hooks are
        # invoked for custom samplers. Advance here so every newly constructed
        # DataLoader iterator gets a deterministic but refreshed window set.
        self.epoch += 1
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
            indices = indices[rank::world_size]
        return iter(indices)

    def __len__(self) -> int:
        total = sum(
            min(self.samples_per_episode, len(candidates))
            for candidates in self._valid_by_episode.values()
        )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
            return max(0, math.ceil((total - rank) / world_size))
        return total


class FullHorizonSequentialSampler(Sampler[int]):
    """Iterate every query with a complete future horizon, without padding."""

    def __init__(self, dataset, *, future_horizon: int = 30) -> None:
        self.dataset = dataset
        self.future_horizon = int(future_horizon)
        if self.future_horizon <= 0:
            raise ValueError("future_horizon must be positive")
        groups = getattr(dataset, "_global_indices_by_dataset", None)
        if not isinstance(groups, Mapping) or not groups:
            raise TypeError("FullHorizonSequentialSampler requires a MultiDataset")
        self.indices: list[int] = []
        for episode, global_indices in sorted(groups.items()):
            valid_count = len(global_indices) - self.future_horizon + 1
            if valid_count <= 0:
                raise ValueError(
                    f"episode {episode!r} is shorter than horizon {self.future_horizon}"
                )
            self.indices.extend(global_indices[:valid_count])

    def __iter__(self) -> Iterator[int]:
        indices = self.indices
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            indices = indices[
                torch.distributed.get_rank() :: torch.distributed.get_world_size()
            ]
        return iter(indices)

    def __len__(self) -> int:
        total = len(self.indices)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
            return max(0, math.ceil((total - rank) / world_size))
        return total


class EpisodeBalancedMultiDataModuleWrapper(LightningDataModule):
    """Additive V5 data module; legacy MultiDataModuleWrapper is untouched."""

    def __init__(
        self,
        train_datasets: dict,
        valid_datasets: dict,
        train_dataloader_params: dict,
        valid_dataloader_params: dict,
        train_sampler_params: dict,
        valid_future_horizon: int = 30,
    ) -> None:
        super().__init__()
        self.train_datasets = {
            key: value for key, value in train_datasets.items() if value is not None
        }
        self.valid_datasets = {
            key: value for key, value in valid_datasets.items() if value is not None
        }
        self.train_dataloader_params = train_dataloader_params
        self.valid_dataloader_params = valid_dataloader_params
        self.train_sampler_params = train_sampler_params
        self.valid_future_horizon = int(valid_future_horizon)
        self._train_samplers: dict[str, EpisodeUniformWindowSampler] = {}

    def train_dataloader(self):
        loaders = {}
        self._train_samplers = {}
        for name, dataset in self.train_datasets.items():
            loader_params = dict(self.train_dataloader_params[name])
            loader_params.pop("shuffle", None)
            sampler_cfg = dict(self.train_sampler_params[name])
            sampler = EpisodeUniformWindowSampler(dataset, **sampler_cfg)
            self._train_samplers[name] = sampler
            loaders[name] = DataLoader(
                dataset,
                sampler=sampler,
                shuffle=False,
                collate_fn=annotation_collate,
                **loader_params,
            )
        return CombinedLoader(loaders, "max_size_cycle")

    def val_dataloader(self):
        loaders = {}
        for name, dataset in self.valid_datasets.items():
            loader_params = dict(self.valid_dataloader_params[name])
            loader_params.pop("shuffle", None)
            sampler = FullHorizonSequentialSampler(
                dataset, future_horizon=self.valid_future_horizon
            )
            loaders[name] = DataLoader(
                dataset,
                sampler=sampler,
                shuffle=False,
                collate_fn=annotation_collate,
                **loader_params,
            )
        return CombinedLoader(loaders, "max_size_cycle")
