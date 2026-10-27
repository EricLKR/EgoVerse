from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

import egomimic.algo.pi as pi_module
from egomimic.algo.pi import PI
from egomimic.rldb.embodiment.embodiment import get_embodiment_id


class _StubNormStats:
    def __init__(self, viz_img_keys):
        self._viz_img_keys = viz_img_keys

    def viz_img_key(self):
        return self._viz_img_keys


class _DummyPartialFinetuneModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.paligemma_with_expert = torch.nn.Module()
        self.paligemma_with_expert.paligemma = torch.nn.Module()
        self.paligemma_with_expert.paligemma.vision_tower = torch.nn.Linear(2, 2)
        self.paligemma_with_expert.paligemma.language_model = torch.nn.Linear(2, 2)
        self.paligemma_with_expert.paligemma.multi_modal_projector = torch.nn.Linear(2, 2)
        self.paligemma_with_expert.gemma_expert = torch.nn.Linear(2, 2)
        self.action_in_proj = torch.nn.Linear(2, 2)
        self.action_out_proj = torch.nn.Linear(2, 2)
        self.time_mlp_in = torch.nn.Linear(2, 2)
        self.time_mlp_out = torch.nn.Linear(2, 2)


def _make_transform(name):
    return SimpleNamespace(
        extrinsics={"name": f"{name}_extrinsics"},
        intrinsics={"name": f"{name}_intrinsics"},
    )


def _make_pi(camera_transforms, domains):
    pi = object.__new__(PI)
    pi.domains = domains
    pi.camera_transforms = camera_transforms
    pi.is_6dof = False
    pi.ac_keys = {get_embodiment_id(domain): "actions_cartesian" for domain in domains}
    pi.norm_stats = _StubNormStats(
        {get_embodiment_id(domain): "front_img_1" for domain in domains}
    )
    return pi


def _make_batch(embodiment_name):
    embodiment_id = get_embodiment_id(embodiment_name)
    return {
        "embodiment": torch.tensor([embodiment_id]),
        "front_img_1": torch.zeros(1, 3, 4, 4),
        "actions_cartesian": torch.zeros(1, 2, 6),
    }


def _make_predictions(embodiment_name):
    return {f"{embodiment_name}_actions_cartesian": torch.ones(1, 2, 6)}


def test_visualize_preds_supports_single_transform_object(monkeypatch):
    shared_transform = _make_transform("shared")
    pi = _make_pi(shared_transform, ["human_bimanual"])

    draw_calls = []

    def fake_draw_actions(
        im, ac_type, color, actions, extrinsics, intrinsics, arm="both", **kwargs
    ):
        draw_calls.append((extrinsics, intrinsics))
        return im

    monkeypatch.setattr(pi_module, "draw_actions", fake_draw_actions)

    ims = pi.visualize_preds(
        _make_predictions("human_bimanual"), _make_batch("human_bimanual")
    )

    assert ims.shape == (1, 4, 4, 3)
    assert len(draw_calls) == 2
    assert all(
        extrinsics is shared_transform.extrinsics for extrinsics, _ in draw_calls
    )
    assert all(
        intrinsics is shared_transform.intrinsics for _, intrinsics in draw_calls
    )


def test_visualize_preds_raises_clear_error_for_missing_embodiment():
    pi = _make_pi(
        {"human_bimanual": _make_transform("aria")},
        ["human_bimanual", "eva_bimanual"],
    )

    with pytest.raises(KeyError) as exc_info:
        pi.visualize_preds(
            _make_predictions("eva_bimanual"), _make_batch("eva_bimanual")
        )

    assert "Missing camera transform for embodiment 'eva_bimanual'" in str(
        exc_info.value
    )
    assert "human_bimanual" in str(exc_info.value)


def test_visualize_preds_rejects_invalid_camera_transform_shape():
    pi = _make_pi({"human_bimanual": {"extrinsics": {}}}, ["human_bimanual"])

    with pytest.raises(TypeError) as exc_info:
        pi.visualize_preds(
            _make_predictions("human_bimanual"), _make_batch("human_bimanual")
        )

    assert "camera_transforms must be a CameraTransforms instance or a mapping" in str(
        exc_info.value
    )


def test_visualize_preds_uses_embodiment_specific_camera_transform(monkeypatch):
    aria_transform = _make_transform("aria")
    eva_transform = _make_transform("eva")
    pi = _make_pi(
        {"human_bimanual": aria_transform, "eva_bimanual": eva_transform},
        ["human_bimanual", "eva_bimanual"],
    )

    draw_calls = []

    def fake_draw_actions(
        im, ac_type, color, actions, extrinsics, intrinsics, arm="both", **kwargs
    ):
        draw_calls.append(
            {
                "ac_type": ac_type,
                "color": color,
                "extrinsics": extrinsics,
                "intrinsics": intrinsics,
                "arm": arm,
                "shape": tuple(actions.shape),
            }
        )
        return im

    monkeypatch.setattr(pi_module, "draw_actions", fake_draw_actions)

    ims = pi.visualize_preds(
        _make_predictions("human_bimanual"), _make_batch("human_bimanual")
    )

    assert ims.shape == (1, 4, 4, 3)
    assert len(draw_calls) == 2
    assert all(call["extrinsics"] is aria_transform.extrinsics for call in draw_calls)
    assert all(call["intrinsics"] is aria_transform.intrinsics for call in draw_calls)
    assert all(call["arm"] == "both" for call in draw_calls)
    assert all(call["shape"] == (2, 6) for call in draw_calls)
    assert all(
        call["extrinsics"] is not eva_transform.extrinsics for call in draw_calls
    )


def test_camera_slot_map_is_per_embodiment_and_ordered():
    pi = object.__new__(PI)
    pi.camera_slot_maps = {
        "piper_left_arm": {
            "base_0_rgb": "front_img_1",
            "left_wrist_0_rgb": "wrist_img",
            "right_wrist_0_rgb": None,
        }
    }
    slots = ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]

    resolved = pi._camera_slot_map_for("piper_left_arm", slots)

    assert list(resolved) == slots
    assert resolved["base_0_rgb"] == "front_img_1"
    assert resolved["left_wrist_0_rgb"] == "wrist_img"
    assert resolved["right_wrist_0_rgb"] is None


def test_camera_alias_resolves_canonical_zarr_batch_key():
    batch = {
        "observations.images.front_img_1": torch.zeros(2, 3, 224, 224),
        "observations.images.wrist_img": torch.zeros(2, 3, 224, 224),
    }

    assert (
        PI._resolve_image_batch_key(batch, "front_img_1")
        == "observations.images.front_img_1"
    )
    assert (
        PI._resolve_image_batch_key(batch, "observations.images.wrist_img")
        == "observations.images.wrist_img"
    )
    assert PI._resolve_image_batch_key(batch, "missing") is None
    assert PI._resolve_image_batch_key(batch, None) is None


def test_action_expert_finetune_freezes_only_pretrained_backbones():
    model = _DummyPartialFinetuneModel()

    stats = PI._configure_finetune_parameters(
        model,
        finetune_mode="action_expert",
        train_multimodal_projector=True,
    )

    paligemma = model.paligemma_with_expert.paligemma
    assert not any(p.requires_grad for p in paligemma.vision_tower.parameters())
    assert not any(p.requires_grad for p in paligemma.language_model.parameters())
    assert all(p.requires_grad for p in paligemma.multi_modal_projector.parameters())
    assert all(
        p.requires_grad
        for p in model.paligemma_with_expert.gemma_expert.parameters()
    )
    assert all(p.requires_grad for p in model.action_out_proj.parameters())
    assert stats["trainable"] > 0
    assert stats["frozen"] > 0


def test_full_finetune_restores_every_parameter_to_trainable():
    model = _DummyPartialFinetuneModel()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    stats = PI._configure_finetune_parameters(
        model,
        finetune_mode="full",
        train_multimodal_projector=True,
    )

    assert all(parameter.requires_grad for parameter in model.parameters())
    assert stats["trainable"] == stats["total"]
    assert stats["frozen"] == 0


def test_masked_flow_loss_ignores_padding_and_applies_weights():
    raw = torch.zeros(1, 2, 32)
    weights = torch.zeros_like(raw)
    raw[..., 0] = 2.0
    raw[..., 6] = 4.0
    raw[..., 20:] = 1000.0
    weights[..., 0] = 1.0
    weights[..., 6] = 3.0

    loss = PI._masked_flow_loss(raw, weights)

    assert loss.item() == pytest.approx((2.0 + 3.0 * 4.0) / 4.0)


def test_pi05_state_tokens_use_per_embodiment_canonical_proprio():
    pi = object.__new__(PI)
    pi.proprio_keys_for_prompt = {
        "piper_left_arm": ["observations.state.joints"],
        "human_left_arm": ["observations.state.ee_pose"],
    }
    normalized_model_batch = {
        "observations.state.joints": torch.tensor(
            [[[0.25, -0.5, 1.5], [0.5, -0.25, -1.5]]]
        )
    }

    state = pi._state_for_sample(
        normalized_model_batch, sample_idx=0, embodiment_name="piper_left_arm"
    )

    assert state.tolist() == pytest.approx([0.5, -0.25, -1.0])


def test_pi05_accepts_hydra_dictconfig_for_per_embodiment_proprio():
    configured = OmegaConf.create(
        {"piper_left_arm": ["observations.state.joints"]}
    )
    pi = object.__new__(PI)
    # Mirror the constructor's normalization of Hydra mappings.
    pi.proprio_keys_for_prompt = {
        str(name): list(keys) for name, keys in configured.items()
    }

    state = pi._state_for_sample(
        {"observations.state.joints": torch.zeros(1, 1, 7)},
        sample_idx=0,
        embodiment_name="piper_left_arm",
    )

    assert state.shape == (7,)
