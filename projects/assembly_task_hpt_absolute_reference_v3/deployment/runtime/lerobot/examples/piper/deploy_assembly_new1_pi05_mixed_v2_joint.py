#!/usr/bin/env python
"""Audited joint-only deployment adapter for Mixed CamFrame V5.

This additive entry point reconstructs the exact training Lightning
``ModelWrapper`` from the frozen pi0.5 base, the immutable V5 Hydra config,
and the two-domain normalization statistics.  It then overlays the compact
V5 parameter-and-buffer checkpoint and delegates inference to V5's own
``forward_joint_deploy`` method.

The executable PIPER contract is the native 30-step output produced during
training: six absolute joint angles in radians plus one binary gripper state.
There is no deployment-only head, 30<->100 interpolation, FK, or IK.
"""

from __future__ import annotations

import copy
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


V5_ROOT = Path(__file__).resolve().parents[1]
V5_TRAINING_ROOT = V5_ROOT / "training_code/EgoVerse"
V5_OPENPI_SRC = V5_TRAINING_ROOT / "external/openpi/src"
V5_OPENPI_CLIENT_SRC = V5_TRAINING_ROOT / "external/openpi/packages/openpi-client/src"

# Insert the V5 snapshot before importing the reusable V4 adapter.  The V4
# module itself inserts its historical source tree, so the V5 paths are moved
# to the front again immediately after import below.
for _path in (V5_TRAINING_ROOT, V5_OPENPI_SRC, V5_OPENPI_CLIENT_SRC):
    if _path.is_dir() and str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import deploy_egoverse_pi05_piper_v3_v4_joint as _v4_base

for _old in (
    getattr(_v4_base, "_TRAINING_ROOT", None),
    getattr(_v4_base, "_OPENPI_SRC", None),
):
    if _old is not None:
        while str(_old) in sys.path:
            sys.path.remove(str(_old))
for _path in (V5_TRAINING_ROOT, V5_OPENPI_SRC, V5_OPENPI_CLIENT_SRC):
    while str(_path) in sys.path:
        sys.path.remove(str(_path))
    sys.path.insert(0, str(_path))


legacy = _v4_base.legacy
pi05_legacy = _v4_base.pi05_legacy
adjacent_joint_rms = _v4_base.adjacent_joint_rms
smooth_joint_action_chunk = _v4_base.smooth_joint_action_chunk
logger = logging.getLogger("deploy_egoverse_pi05_mixed_camframe_v5_joint")

V5_CHECKPOINT_PROFILES: dict[str, dict[str, object]] = {
    "assembly-new1-pi05-mixed-v2-30k": {
        "checkpoint": str(
            V5_ROOT
            / "checkpoints/assembly_new1_pi05_mixed_v2_step30000_trainable.ckpt"
        ),
        "sha256": "7e747a7f0e447153b5e045114ed78614bcb64a6de5283db2abdede7de2d8c6f2",
        "model_action_horizon": 30,
        "description": (
            "Assembly New1 three-camera pi0.5 Mixed V2 LoRA, step 30000"
        ),
        # Fully composed config captured by the formal Hydra run.  The raw
        # model group intentionally omits inherited image augmentations and
        # therefore is not sufficient to reconstruct ModelWrapper by itself.
        "model_config": str(V5_ROOT / "pi05_runtime/formal_30k_config.yaml"),
        "model_config_sha256": "baa4c1e3914dca159d8f6565f027fe5356d0856282594642f0363c36ae812f7a",
        "norm_stats": str(V5_ROOT / "pi05_runtime/norm_stats.json"),
        "norm_stats_sha256": "c56eaff63abdb5cea21cbbed9f0181aaeff7ef1dbe700a848090ce45a6056811",
        "pytorch_weights": str(V5_ROOT / "pi05_runtime/pi05_base_pytorch"),
        "expected_model_target": (
            "egomimic.algo.pi_egoverse_mixed_camframe_v5."
            "PIEgoVerseMixedCamFrameV5LoRA"
        ),
        # The executable action decoder deliberately remains the audited V4
        # bounded auxiliary.  V5 is the separate prefix/domain-conditioning
        # contract checked below.
        "expected_contract": "egoverse_pi05_expert_action_v4_bounded",
        "expected_contract_version": 4,
        "expected_domain_contract": "egoverse_pi05_mixed_camframe_v5",
        "expected_domain_contract_version": 5,
        "expected_step": 30000,
        "expected_state_key_count": 563,
        "expected_stored_tensor_element_count": 34098788,
    },
}


def _profile_for_args(args):
    try:
        return V5_CHECKPOINT_PROFILES[args.checkpoint_profile]
    except KeyError as exc:
        raise RuntimeError(
            "Mixed CamFrame V5 deployment requires an audited "
            "--checkpoint-profile; arbitrary checkpoints are disabled"
        ) from exc


def _persistent_wrapper_state_names(wrapper, model_state: dict[str, torch.Tensor]) -> set[str]:
    required = {
        name for name, parameter in wrapper.named_parameters() if parameter.requires_grad
    }
    required.update(name for name, _ in wrapper.named_buffers() if name in model_state)
    return required


def load_compact_model_wrapper(args):
    """Reconstruct the full V5 wrapper and load its exact compact state."""
    profile = _profile_for_args(args)
    model_config = Path(str(profile["model_config"]))
    norm_stats = Path(str(profile["norm_stats"]))
    pytorch_weights = Path(str(profile["pytorch_weights"]))
    for path in (model_config, norm_stats):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not pytorch_weights.is_dir():
        raise FileNotFoundError(pytorch_weights)
    if not V5_TRAINING_ROOT.is_dir() or not V5_OPENPI_SRC.is_dir():
        raise FileNotFoundError(
            f"isolated Mixed CamFrame V5 source snapshot is unavailable: {V5_TRAINING_ROOT}"
        )
    if legacy.sha256_file(model_config) != profile["model_config_sha256"]:
        raise RuntimeError("V5 training model config SHA-256 mismatch")
    if legacy.sha256_file(norm_stats) != profile["norm_stats_sha256"]:
        raise RuntimeError("V5 normalization SHA-256 mismatch")

    os.environ.setdefault(
        "OPENPI_PALIGEMMA_TOKENIZER_MODEL", str(args.pi05_tokenizer_model)
    )
    from omegaconf import OmegaConf
    from egomimic.pl_utils.pl_model import ModelWrapper

    full_cfg = OmegaConf.load(model_config)
    actual_target = str(full_cfg.model.robomimic_model._target_)
    if actual_target != profile["expected_model_target"]:
        raise RuntimeError(
            "V5 training model target mismatch: "
            f"expected={profile['expected_model_target']} actual={actual_target}"
        )

    model_cfg = copy.deepcopy(full_cfg.model)
    model_cfg.robomimic_model.config.pytorch_weight_path = str(pytorch_weights)
    if "norm_stats" in model_cfg.robomimic_model:
        model_cfg.robomimic_model.norm_stats = None
    config_tree = OmegaConf.create({"model": model_cfg})
    wrapper = ModelWrapper(
        config_tree=config_tree,
        norm_stats_state=_v4_base._norm_state_from_json(norm_stats),
        optimizer=None,
        scheduler=None,
        scheduler_interval="step",
        scheduler_frequency=1,
        evaluator=None,
        enable_grad_norm=False,
    )

    payload = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False, mmap=True
    )
    if payload.get("checkpoint_format") != "egomimic_trainable_state_and_buffers_v5":
        raise RuntimeError("checkpoint is not an audited Mixed CamFrame V5 compact state")
    if int(payload.get("global_step", -1)) != int(profile["expected_step"]):
        raise RuntimeError("V5 compact checkpoint step does not match its profile")
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, dict) or not state_dict:
        raise RuntimeError("V5 compact checkpoint has no state_dict")
    if set(payload.get("state_keys", ())) != set(state_dict):
        raise RuntimeError("V5 compact checkpoint key manifest is inconsistent")
    if len(state_dict) != int(profile["expected_state_key_count"]):
        raise RuntimeError("V5 compact checkpoint state-key count mismatch")
    tensor_elements = sum(value.numel() for value in state_dict.values())
    if tensor_elements != int(profile["expected_stored_tensor_element_count"]):
        raise RuntimeError("V5 compact checkpoint tensor-element count mismatch")
    if int(payload.get("stored_tensor_element_count", -1)) != tensor_elements:
        raise RuntimeError("V5 compact checkpoint element manifest is inconsistent")

    model_state = wrapper.state_dict()
    missing_from_model = sorted(set(state_dict).difference(model_state))
    shape_mismatches = {
        name: (tuple(value.shape), tuple(model_state[name].shape))
        for name, value in state_dict.items()
        if name in model_state and tuple(value.shape) != tuple(model_state[name].shape)
    }
    if missing_from_model or shape_mismatches:
        raise RuntimeError(
            "V5 compact checkpoint is incompatible with reconstructed model: "
            f"unknown={missing_from_model[:10]} shapes={shape_mismatches}"
        )
    nonfinite = [
        name for name, value in state_dict.items() if not torch.isfinite(value).all()
    ]
    if nonfinite:
        raise RuntimeError(f"V5 compact checkpoint contains NaN/Inf: {nonfinite[:10]}")

    required_names = _persistent_wrapper_state_names(wrapper, model_state)
    if required_names != set(state_dict):
        raise RuntimeError(
            "V5 compact state is not the exact reconstructed trainable+buffer set: "
            f"missing={sorted(required_names - set(state_dict))[:10]} "
            f"extra={sorted(set(state_dict) - required_names)[:10]}"
        )
    incompatible = wrapper.load_state_dict(state_dict, strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(
            f"unexpected V5 compact checkpoint keys: {incompatible.unexpected_keys}"
        )

    wrapper = wrapper.to(args.device).eval()
    wrapper.model.device = torch.device(args.device)
    wrapper.compact_deployment_report = {
        "checkpoint_format": payload["checkpoint_format"],
        "global_step": int(payload["global_step"]),
        "loaded_state_keys": len(state_dict),
        "loaded_tensor_elements": tensor_elements,
        "frozen_state_keys_from_base": len(incompatible.missing_keys),
        "model_config": str(model_config),
        "norm_stats": str(norm_stats),
        "pytorch_weights": str(pytorch_weights),
        "model_target": actual_target,
    }
    del payload, state_dict, model_state
    return wrapper


# Reuse the audited camera/state conversion, fixed rollout noise, decoded joint
# clipping, gripper diagnostics and queue interface.  Its global loader/profile
# references are deliberately replaced by the strict V5 implementations above.
_v4_base._TRAINING_ROOT = V5_TRAINING_ROOT
_v4_base._OPENPI_SRC = V5_OPENPI_SRC
_v4_base.PIPER_V3_V4_CHECKPOINT_PROFILES = V5_CHECKPOINT_PROFILES
_v4_base._profile_for_args = _profile_for_args
_v4_base.load_compact_model_wrapper = load_compact_model_wrapper


class EgoVersePI05MixedCamFrameV5JointPolicy(
    _v4_base.EgoVersePI05PiperV3V4JointPolicy
):
    """V5 Gemma action expert exposed through the audited 30-step queue."""

    name = "egoverse_pi05_mixed_camframe_v5_joint"

    def __init__(self, args, joint_limits_deg, gripper_stroke_mm):
        super().__init__(args, joint_limits_deg, gripper_stroke_mm)
        profile = _profile_for_args(args)
        conditioning = getattr(self._algo.model, "egoverse_v5_conditioning", None)
        if conditioning is None:
            raise RuntimeError("V5 model lacks domain-conditioning module")
        if getattr(conditioning, "contract_name", None) != profile[
            "expected_domain_contract"
        ]:
            raise RuntimeError("V5 domain-conditioning contract mismatch")
        if int(getattr(conditioning, "contract_version", -1)) != int(
            profile["expected_domain_contract_version"]
        ):
            raise RuntimeError("V5 domain-conditioning contract version mismatch")
        self.checkpoint_report.update(
            {
                "architecture": "egoverse_pi05_mixed_camframe_v5",
                "action_decoder_contract": "egoverse_pi05_expert_action_v4_bounded",
                "domain_conditioning_contract": conditioning.contract_name,
                "domain_conditioning_contract_version": conditioning.contract_version,
                "training_domains": ["piper_left_arm", "human_right_arm"],
                "piper_camera_slots": {
                    "base_0_rgb": "front_img_1",
                    "left_wrist_0_rgb": "wrist_img",
                    "right_wrist_0_rgb": "side_img",
                },
                "domain_conditioning": "normalized_proprio_plus_embodiment_embedding",
                "checkpoint_state": "trainable_parameters_plus_persistent_buffers",
                "deployment_domain": "piper_left_arm_only",
            }
        )


# Compatibility alias used by the reusable HPT-style RTC controller.
EgoVersePI05PiperV3V4JointPolicy = EgoVersePI05MixedCamFrameV5JointPolicy


def parse_args():
    return _v4_base.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = parse_args()
    if args.list_checkpoint_profiles:
        return
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    actual_sha256 = legacy.sha256_file(args.checkpoint)
    if actual_sha256 != args.checkpoint_sha256:
        raise RuntimeError(
            f"checkpoint SHA-256 mismatch: expected {args.checkpoint_sha256}, "
            f"got {actual_sha256}"
        )
    joint_limits, gripper_stroke_mm = legacy.load_calibration(args.calibration_json)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    policy = EgoVersePI05MixedCamFrameV5JointPolicy(
        args, joint_limits, gripper_stroke_mm
    )
    if args.mode == "checkpoint":
        print(
            "PI05_MIXED_CAMFRAME_V5_CHECKPOINT_OK "
            f"profile={args.checkpoint_profile} sha256={actual_sha256} "
            "robot_constructed=false camera_opened=false can_opened=false"
        )
        return

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log = legacy.JsonlLogger(
        args.log_dir / f"pi05_mixed_camframe_v5_{args.mode}_{timestamp}.jsonl"
    )
    try:
        log.write(
            {
                "event": "start",
                "mode": args.mode,
                "checkpoint_profile": args.checkpoint_profile,
                "checkpoint": str(args.checkpoint),
                "checkpoint_sha256": actual_sha256,
                "checkpoint_contract": policy.checkpoint_report,
                "query_frequency": args.query_frequency,
                "inference_steps": args.inference_steps,
                "seed": args.seed,
                "command_sent": False,
            }
        )
        if args.mode == "offline":
            legacy.run_offline(args, policy, log, device)
        else:
            legacy.run_live(
                args,
                policy,
                log,
                device,
                joint_limits,
                gripper_stroke_mm,
                timestamp,
            )
    finally:
        log.close()
        logger.info("wrote Mixed CamFrame V5 audit log to %s", log.path)


if __name__ == "__main__":
    main()
