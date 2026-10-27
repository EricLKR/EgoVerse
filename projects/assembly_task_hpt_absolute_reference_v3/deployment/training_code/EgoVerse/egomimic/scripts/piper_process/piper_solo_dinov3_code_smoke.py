#!/usr/bin/env python3
"""GPU code smoke for online Piper-only DINO-front and DINO-both variants."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import hydra
import torch

from egomimic.models.hpt_nets import FrozenDINOv3Encoder
from egomimic.rldb.embodiment.piper import Piper
from egomimic.utils.hydra_utils import load_config


VARIANTS = {
    "dinov3_front": (
        "piper_solo_teddy_0912_session_split_base_dinov3_front",
        "hpt_bc_piper_solo_dinov3_front",
        1,
    ),
    "dinov3_both": (
        "piper_solo_teddy_0912_session_split_base_dinov3_both",
        "hpt_bc_piper_solo_dinov3_both",
        2,
    ),
}


def _check_raw_keymap() -> None:
    keymap = Piper.get_keymap(keymap_mode="cartesian_base")
    assert keymap["observations.images.front_img_1"]["zarr_key"] == "images.front_1"
    assert keymap["observations.images.wrist_img"]["zarr_key"] == "images.wrist_1"
    assert not any(value["zarr_key"].startswith("dino.") for value in keymap.values())


def _instantiate_encoder(spec):
    encoder = hydra.utils.instantiate(spec).cuda().eval()
    assert isinstance(encoder, FrozenDINOv3Encoder)
    assert not encoder.state_dict(), "Frozen external DINO weights leaked into state_dict"
    image = torch.rand(1, 1, 1, 3, 480, 640, device="cuda")
    output = encoder(image)
    assert tuple(output.shape) == (1, 196, 1024), tuple(output.shape)
    assert output.dtype == torch.float32
    assert torch.isfinite(output).all()
    return encoder


def _check_model_config(config_name: str, expected_dino_encoders: int) -> None:
    cfg = load_config(f"model/{config_name}")
    model = cfg.robomimic_model
    assert list(model.domains) == ["piper_left_arm"]
    assert list(model.shared_obs_keys) == ["front_img_1"]
    assert model.shared_stem_specs.front_img_1.input_dim == 1024
    assert model.shared_stem_specs.front_img_1.output_dim == 256
    assert model.stem_specs.piper_left_arm.state_ee_pose.input_dim == 10
    assert "wrist_img" in model.stem_specs.piper_left_arm
    assert model.head_specs.shared.action_horizon == 100
    assert model.head_specs.shared.num_inference_steps == 10
    assert model.head_specs.shared.early_horizon == 30
    assert model.head_specs.shared.early_weight == 2.0
    assert model.head_specs.shared.rotation_geodesic_weight == 0.1
    assert model.head_specs.shared.model.act_dim == 10
    assert model.head_specs.shared.model.act_seq == 100
    dino_specs = [
        spec
        for spec in model.encoder_specs.values()
        if str(spec._target_).endswith("FrozenDINOv3Encoder")
    ]
    assert len(dino_specs) == expected_dino_encoders
    if expected_dino_encoders == 1:
        assert model.encoder_specs.wrist_img.normalize_input is True

    encoders = [_instantiate_encoder(spec) for spec in dino_specs]
    if len(encoders) == 2:
        assert encoders[0]._encoder is encoders[1]._encoder

    front_stem = hydra.utils.instantiate(model.shared_stem_specs.front_img_1)
    front_stem.init_cross_attn(model.shared_stem_specs.front_img_1.specs.cross_attn)
    out = front_stem.compute_latent(torch.randn(1, 196, 1024))
    assert tuple(out.shape) == (1, 16, 256), tuple(out.shape)


def _check_train_composition() -> None:
    for variant, (data_config, model_config, _) in VARIANTS.items():
        cfg = load_config(
            "train_zarr_cartesian",
            overrides=[
                f"data={data_config}",
                f"model={model_config}",
                "logger=debug",
                "+trainer.max_steps=10",
                "trainer.max_epochs=-1",
                "trainer.min_epochs=null",
                "trainer.check_val_every_n_epoch=null",
                "+trainer.val_check_interval=5",
                "+model.robomimic_model.reverse_kl_samples=0",
                "+callbacks.model_checkpoint.monitor=Valid/action_loss",
                "+callbacks.model_checkpoint.mode=min",
                "callbacks.model_checkpoint.save_top_k=1",
                "callbacks.model_checkpoint.every_n_epochs=null",
                "callbacks.model_checkpoint.filename='best_step_{step}'",
            ],
        )
        assert list(cfg.data.train_datasets) == ["piper_left_arm"]
        assert list(cfg.data.valid_datasets) == ["piper_left_arm"]
        for split in (cfg.data.train_datasets, cfg.data.valid_datasets):
            dataset = split.piper_left_arm
            assert dataset.resolver.transform_list.mode == "cartesian_base"
            assert str(dataset.resolver.key_map._target_).endswith("Piper.get_keymap")
            assert "variant" not in dataset.resolver.key_map
        assert cfg.callbacks.model_checkpoint.monitor == "Valid/action_loss"
        assert cfg.callbacks.model_checkpoint.mode == "min"
        assert cfg.callbacks.model_checkpoint.save_top_k == 1
        assert cfg.trainer.val_check_interval == 5
        assert cfg.model.robomimic_model.reverse_kl_samples == 0
        assert cfg.description == "test", variant


def _check_held_out_configs() -> None:
    configs = {
        "dinov3_front": "piper_solo_teddy_0912_test_base_dinov3_front",
        "dinov3_both": "piper_solo_teddy_0912_test_base_dinov3_both",
    }
    for variant, config_name in configs.items():
        cfg = load_config(f"data/{config_name}")
        assert not cfg.train_datasets, variant
        assert list(cfg.valid_datasets) == ["piper_left_arm"]
        dataset = cfg.valid_datasets.piper_left_arm
        assert dataset.resolver.transform_list.mode == "cartesian_base"
        assert str(dataset.resolver.key_map._target_).endswith("Piper.get_keymap")
        assert "teddy_0912_7" in dataset.filters.filter_lambdas[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This smoke test requires a GPU")
    os.environ["DINO_MODEL_PATH"] = str(args.model_dir.resolve())
    _check_raw_keymap()
    for _, model_config, expected_dino_encoders in VARIANTS.values():
        _check_model_config(model_config, expected_dino_encoders)
    _check_train_composition()
    _check_held_out_configs()
    print("online Piper-only DINOv3 code smoke: PASS")


if __name__ == "__main__":
    main()
