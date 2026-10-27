"""Shared conditioning modules for pi0.5 Mixed CamFrame V5.

V5 keeps the tested V4 action contract (30 native ticks, shared 9-D pose,
PIPER-only bounded 6-D joint latent, separate binary gripper), and adds an
explicit learned conditioning token for the two different proprio spaces.

The state adapters are intentionally domain-specific: a seven-dimensional
PIPER state must never be interpreted as the same physical quantity as a
nine-dimensional human camera-frame hand pose.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Pi05MixedDomainConditioningV5(nn.Module):
    """Map normalized per-domain proprio into one prefix token.

    Domain index 0 is PIPER and index 1 is Human.  The returned token already
    includes the learned embodiment embedding, while ``embodiment`` can also
    be added to visual tokens to make camera/domain identity explicit.
    """

    contract_name = "egoverse_pi05_mixed_camframe_v5"
    contract_version = 5

    def __init__(
        self,
        *,
        prefix_dim: int,
        hidden_dim: int = 512,
        piper_state_dim: int = 7,
        human_state_dim: int = 9,
        init_std: float = 0.02,
    ) -> None:
        super().__init__()
        self.prefix_dim = int(prefix_dim)
        self.state_dims = (int(piper_state_dim), int(human_state_dim))
        if self.prefix_dim <= 0 or hidden_dim <= 0:
            raise ValueError("prefix_dim and hidden_dim must be positive")
        if any(dimension <= 0 for dimension in self.state_dims):
            raise ValueError("state dimensions must be positive")

        self.state_adapters = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(state_dim),
                    nn.Linear(state_dim, int(hidden_dim)),
                    nn.GELU(),
                    nn.Linear(int(hidden_dim), self.prefix_dim),
                )
                for state_dim in self.state_dims
            ]
        )
        self.embodiment_embeddings = nn.Parameter(
            torch.empty(2, self.prefix_dim)
        )
        nn.init.normal_(self.embodiment_embeddings, std=float(init_std))

    def forward(
        self, normalized_state: torch.Tensor, domain_index: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        domain_index = int(domain_index)
        if domain_index not in (0, 1):
            raise ValueError("domain_index must be 0 (PIPER) or 1 (Human)")
        if normalized_state.ndim != 2:
            raise ValueError("normalized_state must be [B,D]")
        expected = self.state_dims[domain_index]
        if normalized_state.shape[-1] != expected:
            raise ValueError(
                f"domain {domain_index} requires state dim {expected}, "
                f"got {normalized_state.shape[-1]}"
            )
        if not torch.isfinite(normalized_state).all():
            raise ValueError("normalized_state contains NaN/Inf")
        state_token = self.state_adapters[domain_index](normalized_state.float())
        embodiment = self.embodiment_embeddings[domain_index].to(state_token)
        return state_token + embodiment[None], embodiment


def human_ramp_weight(step: int, *, final_weight: float, ramp_steps: int) -> float:
    """Linear Human-loss warm-up used by V5 and its unit tests."""
    if step < 0:
        raise ValueError("step must be non-negative")
    if final_weight < 0:
        raise ValueError("final_weight must be non-negative")
    if ramp_steps <= 0:
        return float(final_weight)
    return float(final_weight) * min(float(step) / float(ramp_steps), 1.0)

