#!/usr/bin/env python3
"""GPU code smoke test for the DINOv3 HPT vision-ablation wiring.

This does not read or modify a dataset.  It validates both new Hydra model
configs, keymaps, the local DINOv3 checkpoint, one real ViT forward pass, and
the 1024->256 HPT vision stem shape.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import hydra
import numpy as np
import torch

from egomimic.rldb.embodiment.dinov3_keymaps import (
    get_human_dinov3_keymap,
    get_piper_dinov3_keymap,
)
from egomimic.utils.hydra_utils import load_config


def _check_keymaps() -> None:
    piper_front = get_piper_dinov3_keymap(variant="front")
    piper_both = get_piper_dinov3_keymap(variant="both")
    human = get_human_dinov3_keymap()
    assert piper_front["dino_front_img"]["zarr_key"] == "dino.front_1"
    assert "dino_wrist_img" not in piper_front
    assert piper_both["dino_wrist_img"]["zarr_key"] == "dino.wrist_1"
    assert human["dino_front_img"]["key_type"] == "camera_keys"
    assert "dino_front_img" not in get_piper_dinov3_keymap(
        variant="both", norm_mode=True
    )
    assert "dino_front_img" not in get_human_dinov3_keymap(norm_mode=True)


def _check_model_config(config_name: str, expected_wrist: str) -> None:
    cfg = load_config(f"model/{config_name}")
    model = cfg.robomimic_model
    assert list(model.shared_obs_keys) == ["dino_front_img"]
    assert model.shared_stem_specs.dino_front_img.input_dim == 1024
    assert model.shared_stem_specs.dino_front_img.output_dim == 256
    assert model.head_specs.shared.model.act_dim == 10
    assert model.head_specs.shared.model.act_seq == 100
    piper_stems = model.stem_specs.piper_left_arm
    assert expected_wrist in piper_stems
    if expected_wrist == "wrist_img":
        assert list(model.encoder_specs) == ["wrist_img"]
    else:
        assert not model.encoder_specs

    stem = hydra.utils.instantiate(model.shared_stem_specs.dino_front_img)
    stem.init_cross_attn(model.shared_stem_specs.dino_front_img.specs.cross_attn)
    fake_tokens = torch.randn(1, 1, 1, 196, 1024)
    out = stem.compute_latent(fake_tokens)
    assert tuple(out.shape) == (1, 16, 256), tuple(out.shape)


def _check_train_composition() -> None:
    variants = {
        "resnet18": (
            "piper_human_cotrain_aria0919",
            "hpt_cotrain_piper_human",
        ),
        "dinov3_front": (
            "piper_human_cotrain_aria0919_dinov3_front",
            "hpt_cotrain_piper_human_dinov3_front",
        ),
        "dinov3_both": (
            "piper_human_cotrain_aria0919_dinov3_both",
            "hpt_cotrain_piper_human_dinov3_both",
        ),
    }
    for variant, (data_config, model_config) in variants.items():
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
        assert cfg.callbacks.model_checkpoint.monitor == "Valid/action_loss"
        assert cfg.callbacks.model_checkpoint.save_top_k == 1
        assert cfg.trainer.val_check_interval == 5
        assert cfg.model.robomimic_model.reverse_kl_samples == 0
        assert cfg.description == "test", variant


@torch.inference_mode()
def _check_checkpoint(model_dir: Path) -> None:
    from egomimic.scripts.embedding_process.dinov3_embedding import (
        DINOv3ImageEmbedding,
    )

    embedding = DINOv3ImageEmbedding(
        input_keys=["images.front_1"],
        model_name=str(model_dir),
        batch_size=1,
        device="cuda",
        dtype=torch.bfloat16,
        output_dtype="float16",
        local_files_only=True,
    )
    embedding.setup()
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    patches = embedding._embed_batch([image])
    assert tuple(patches.shape) == (1, 196, 1024), tuple(patches.shape)
    assert patches.dtype == np.float16
    assert np.isfinite(patches).all()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This smoke test requires a GPU")
    _check_keymaps()
    _check_model_config("hpt_cotrain_piper_human_dinov3_front", "wrist_img")
    _check_model_config("hpt_cotrain_piper_human_dinov3_both", "dino_wrist_img")
    _check_train_composition()
    _check_checkpoint(args.model_dir)
    print("vision ablation code smoke: PASS")


if __name__ == "__main__":
    main()
