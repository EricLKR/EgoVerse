import torch
from torch import nn

from egomimic.models.pi05_mixed_lora_v1 import (
    Pi05MixedLoraConfig,
    configure_pi05_mixed_lora,
)


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(8, 8, bias=False)
        self.k_proj = nn.Linear(8, 8, bias=False)
        self.v_proj = nn.Linear(8, 8, bias=False)
        self.o_proj = nn.Linear(8, 8, bias=False)
        self.gate_proj = nn.Linear(8, 16, bias=False)
        self.up_proj = nn.Linear(8, 16, bias=False)
        self.down_proj = nn.Linear(16, 8, bias=False)


class _VisionAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(8, 8, bias=False)
        self.k_proj = nn.Linear(8, 8, bias=False)
        self.v_proj = nn.Linear(8, 8, bias=False)
        self.out_proj = nn.Linear(8, 8, bias=False)


class _VisionMlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16, bias=False)
        self.fc2 = nn.Linear(16, 8, bias=False)


class _VisionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _VisionAttention()
        self.mlp = _VisionMlp()


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.paligemma = nn.Module()
        self.paligemma.model = nn.Module()
        self.paligemma.model.language_model = nn.Module()
        self.paligemma.model.language_model.layers = nn.ModuleList([_Block()])
        self.paligemma.model.vision_tower = nn.Module()
        self.paligemma.model.vision_tower.vision_model = nn.Module()
        self.paligemma.model.vision_tower.vision_model.encoder = nn.Module()
        self.paligemma.model.vision_tower.vision_model.encoder.layers = nn.ModuleList(
            [_VisionBlock(), _VisionBlock()]
        )
        self.paligemma.model.multi_modal_projector = nn.Linear(8, 8)
        self.gemma_expert = nn.Module()
        self.gemma_expert.model = nn.Module()
        self.gemma_expert.model.layers = nn.ModuleList([_Block()])
        self.action_in_proj = nn.Linear(8, 8)
        self.action_out_proj = nn.Linear(8, 8)
        self.egoverse_camera_slot_embeddings = nn.Parameter(torch.zeros(3, 8))


def test_mixed_lora_freezes_base_and_has_gradients():
    model = _Model()
    summary = configure_pi05_mixed_lora(
        model,
        Pi05MixedLoraConfig(vision_last_n_blocks=1, train_multimodal_projector=False),
    )
    assert summary.language_modules == 7
    assert summary.action_expert_modules == 7
    assert summary.vision_modules == 6
    assert model.paligemma.model.language_model.layers[0].q_proj.weight.requires_grad is False
    assert model.action_in_proj.weight.requires_grad is True
    assert model.egoverse_camera_slot_embeddings.requires_grad is True

    result = model.paligemma.model.language_model.layers[0].q_proj(torch.randn(2, 8)).sum()
    result.backward()
    target = model.paligemma.model.language_model.layers[0].q_proj
    assert target.egoverse_lora_b.grad is not None
    assert torch.isfinite(target.egoverse_lora_b.grad).all()
