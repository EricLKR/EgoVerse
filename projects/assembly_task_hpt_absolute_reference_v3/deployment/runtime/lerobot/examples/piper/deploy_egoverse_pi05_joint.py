#!/usr/bin/env python
"""Guarded PIPER deployment for the EgoVerse pi0.5 joint checkpoints.

The robot/camera/command/gripper implementation is deliberately shared with
``deploy_egoverse_hpt_joint.py``.  This file only replaces the model adapter:
it loads an EgoMimic Lightning pi0.5 checkpoint, recreates the training prompt
and quantile normalization contract, decodes OpenPI's 32-D output through the
PIPER 7-D layout, and exposes the same audited joint-policy queue interface.

No FK or IK is used.  ``checkpoint``, ``offline`` and ``live-dry-run`` never
enable robot motion.  ``execute`` still requires the legacy script's explicit
``--allow-motion`` flag and continuity gate.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

# The private runtime contains the OpenPI-compatible transformers patch.  Put
# it ahead of the shared environment before importing any LeRobot module that
# could import transformers transitively.  A custom runtime argument is still
# validated later, but formal profiles intentionally use this immutable path.
_BOOTSTRAP_RUNTIME = Path("/mnt/wd10/kerun/pi05_runtime")
if _BOOTSTRAP_RUNTIME.is_dir() and str(_BOOTSTRAP_RUNTIME) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_RUNTIME))

import numpy as np
import torch
import torch.nn.functional as F

import deploy_egoverse_hpt_joint as legacy
from lerobot.policies.egoverse_hpt.modeling_egoverse_hpt_joint import (
    EgoVerseHPTJointPolicy,
    adjacent_joint_rms,
    model_indices_for_control_ticks,
    prepare_rgb_chw_image,
    smooth_joint_action_chunk,
)

logger = logging.getLogger("deploy_egoverse_pi05_joint")

PI05_CHECKPOINT_PROFILES: dict[str, dict[str, str]] = {
    "pi05-piper": {
        "checkpoint": "/mnt/wd10/kerun/checkpoints/pi05_piper_best_step30000.ckpt",
        "sha256": "fedbed3eab131c4d76772f25d7a5cc93e717c5b90be02fb4e152496cf0e4c42d",
        "description": "EgoVerse pi0.5 PIPER-only, held-out-validation selected step 30000",
    },
    "pi05-mixed": {
        "checkpoint": "/mnt/wd10/kerun/checkpoints/pi05_mixed_best_step30000.ckpt",
        "sha256": "8d98ea889b3c7c1a1c23b7ef169cb82965a796bd2e3356b203dadfcf71369fb9",
        "description": "EgoVerse pi0.5 PIPER+human mixed, held-out-validation selected step 30000",
    },
}

BUNDLE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PI05_RUNTIME_DIR = BUNDLE_ROOT / "pi05_runtime"
DEFAULT_TOKENIZER_MODEL = DEFAULT_PI05_RUNTIME_DIR / "paligemma_tokenizer.model"
DEFAULT_PROMPT = "Complete the assembly task."
PIPER_CONTROL_MODE = (
    "six absolute PIPER joint angles in radians and one binary gripper command"
)


class OpenPISentencePieceTokenizer:
    """The exact pi0.5 prompt template/tokenization used during training."""

    def __init__(self, model_path: Path, max_length: int = 128):
        if not model_path.is_file():
            raise FileNotFoundError(
                f"pi0.5 tokenizer model is missing: {model_path}. "
                "Run setup_pi05_robot_runtime.sh first."
            )
        try:
            import sentencepiece
        except ImportError as exc:
            raise RuntimeError(
                "sentencepiece is absent from the private pi0.5 runtime; "
                "run setup_pi05_robot_runtime.sh"
            ) from exc
        self.max_length = int(max_length)
        self.processor = sentencepiece.SentencePieceProcessor(
            model_proto=model_path.read_bytes()
        )

    def tokenize(self, prompt: str, normalized_state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        state = np.asarray(normalized_state, dtype=np.float32).reshape(-1)
        if state.shape != (7,) or not np.isfinite(state).all():
            raise ValueError(f"pi0.5 normalized PIPER state must be finite (7,), got {state}")
        state = np.clip(state, -1.0, 1.0)
        bins = np.linspace(-1.0, 1.0, 257, dtype=np.float32)[:-1]
        discretized = np.digitize(state, bins=bins) - 1
        cleaned = prompt.strip().replace("_", " ").replace("\n", " ")
        full_prompt = (
            f"Task: {cleaned}, Embodiment: piper left arm, "
            f"Control mode: {PIPER_CONTROL_MODE}, State: "
            f"{' '.join(map(str, discretized.tolist()))};\nAction: "
        )
        tokens = self.processor.encode(full_prompt, add_bos=True)
        tokens = tokens[: self.max_length]
        mask = [True] * len(tokens)
        padding = self.max_length - len(tokens)
        if padding:
            tokens.extend([0] * padding)
            mask.extend([False] * padding)
        return np.asarray(tokens, dtype=np.int64), np.asarray(mask, dtype=np.bool_)


def _stats_array(stats: dict, name: str, device: torch.device) -> torch.Tensor:
    if name not in stats:
        raise KeyError(f"checkpoint normalization statistics lack {name!r}")
    return torch.as_tensor(stats[name], dtype=torch.float32, device=device)


def quantile_normalize(value: torch.Tensor, stats: dict) -> torch.Tensor:
    q1 = _stats_array(stats, "quantile_1", value.device)
    q99 = _stats_array(stats, "quantile_99", value.device)
    return 2.0 * ((value - q1) / (q99 - q1 + 1e-6)) - 1.0


def quantile_unnormalize(value: torch.Tensor, stats: dict) -> torch.Tensor:
    q1 = _stats_array(stats, "quantile_1", value.device)
    q99 = _stats_array(stats, "quantile_99", value.device)
    return (value + 1.0) * 0.5 * (q99 - q1 + 1e-6) + q1


def resize_with_pad_224(image_chw: torch.Tensor) -> torch.Tensor:
    """Match OpenPI's aspect-preserving 224x224 resize and -1 padding."""
    if image_chw.shape[0] != 3:
        raise ValueError(f"expected RGB CHW image, got {tuple(image_chw.shape)}")
    image = image_chw.unsqueeze(0)
    _, _, height, width = image.shape
    ratio = max(width / 224.0, height / 224.0)
    resized_height = int(height / ratio)
    resized_width = int(width / ratio)
    image = F.interpolate(
        image, size=(resized_height, resized_width), mode="bilinear", align_corners=False
    )
    image = image * 2.0 - 1.0
    pad_h0, remainder_h = divmod(224 - resized_height, 2)
    pad_w0, remainder_w = divmod(224 - resized_width, 2)
    return F.pad(
        image,
        (pad_w0, pad_w0 + remainder_w, pad_h0, pad_h0 + remainder_h),
        value=-1.0,
    )


def _find_piper_norm_stats(norm_state: dict) -> tuple[dict, dict]:
    if norm_state.get("norm_mode") != "quantile":
        raise RuntimeError(
            f"pi0.5 deployment requires quantile normalization, got {norm_state.get('norm_mode')!r}"
        )
    candidates = []
    for embodiment, per_key in norm_state.get("norm_stats", {}).items():
        if "observations.state.joints" in per_key and "actions_joints" in per_key:
            candidates.append((embodiment, per_key))
    if len(candidates) != 1:
        raise RuntimeError(
            "checkpoint must contain exactly one PIPER normalization entry; "
            f"found {[item[0] for item in candidates]}"
        )
    _, per_key = candidates[0]
    return per_key["observations.state.joints"], per_key["actions_joints"]


class EgoVersePI05JointPolicy(EgoVerseHPTJointPolicy):
    """pi0.5 backend with the audited HPT joint queue/filter interface."""

    name = "egoverse_pi05_joint"

    def __init__(
        self,
        args: argparse.Namespace,
        joint_limits_deg: list[tuple[float, float]],
        gripper_stroke_mm: float,
    ) -> None:
        # Do not call EgoVerseHPTJointPolicy.__init__: that constructor loads an
        # HPT checkpoint.  Initialize the shared nn.Module/queue state explicitly.
        torch.nn.Module.__init__(self)
        self.config = SimpleNamespace(
            device=str(args.device),
            chunk_size=100,
            action_horizon_frames=30,
            query_frequency=int(args.query_frequency),
            num_inference_steps=int(args.inference_steps),
            inference_seed=int(args.seed),
            gripper_threshold=float(args.gripper_threshold),
            joint_limit_margin_rad=float(np.radians(args.joint_limit_margin_deg)),
            max_joint_command_step=float(args.max_joint_command_step),
            max_joint_command_accel=float(args.max_joint_command_accel),
            temporal_ensemble=bool(args.temporal_ensemble),
            temporal_replan_interval=int(args.temporal_replan_interval),
            temporal_ensemble_decay=float(args.temporal_ensemble_decay),
            chunk_smoothing=bool(args.chunk_smoothing),
            chunk_smoothing_window=int(args.chunk_smoothing_window),
            chunk_smoothing_sigma=float(args.chunk_smoothing_sigma),
            gripper_stroke_mm=float(gripper_stroke_mm),
            front_camera_key="435if",
            wrist_camera_key="405",
            expected_image_height=480,
            expected_image_width=640,
            resize_mismatched_images=False,
        )
        self._joint_low_rad = np.radians(
            np.asarray([pair[0] for pair in joint_limits_deg], dtype=np.float64)
        )
        self._joint_high_rad = np.radians(
            np.asarray([pair[1] for pair in joint_limits_deg], dtype=np.float64)
        )
        self._action_queue: deque[torch.Tensor] = deque()
        self._chunk_sequence_id = 0
        self._active_chunk_size = 0
        self._active_chunk_consumed = 0
        self._last_planned_velocity_driver = np.zeros(6, dtype=np.float64)
        self._pending_planned_velocity_driver: np.ndarray | None = None
        self._temporal_plans: list[tuple[int, int, np.ndarray]] = []
        self._policy_tick = 0
        self._last_temporal_replan_tick: int | None = None
        self.last_ensemble_report: dict | None = None
        self.last_prediction_report: dict | None = None

        runtime_dir = Path(args.pi05_runtime_dir).resolve()
        if not runtime_dir.is_dir():
            raise FileNotFoundError(
                f"private pi0.5 runtime is missing: {runtime_dir}; "
                "run setup_pi05_robot_runtime.sh"
            )
        if str(runtime_dir) not in sys.path:
            sys.path.insert(0, str(runtime_dir))

        # Load with mmap on CPU, then assign directly into a meta-initialized
        # model.  This avoids holding two 7.5-GB model copies in host memory.
        checkpoint = torch.load(
            args.checkpoint, map_location="cpu", weights_only=False, mmap=True
        )
        if "state_dict" not in checkpoint:
            raise RuntimeError("pi0.5 checkpoint has no Lightning state_dict")
        hparams = checkpoint.get("hyper_parameters", {})
        norm_state = hparams.get("norm_stats_state")
        if norm_state is None:
            raise RuntimeError("pi0.5 checkpoint lacks embedded norm_stats_state")
        self._state_stats, self._action_stats = _find_piper_norm_stats(norm_state)

        from lerobot.policies.pi05.configuration_pi05 import PI05Config
        from lerobot.policies.pi05.modeling_pi05 import PI05Pytorch

        core_config = PI05Config(
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m",
            dtype="bfloat16",
            chunk_size=100,
            n_action_steps=100,
            max_state_dim=32,
            max_action_dim=32,
            num_inference_steps=int(args.inference_steps),
            tokenizer_max_length=128,
            gradient_checkpointing=False,
            compile_model=False,
            device=str(args.device),
        )
        with torch.device("meta"):
            core_model = PI05Pytorch(core_config, rtc_processor=None)
        prefix = "nets.policy."
        core_state = {
            key[len(prefix) :]: value
            for key, value in checkpoint["state_dict"].items()
            if key.startswith(prefix)
        }
        if not core_state:
            raise RuntimeError("checkpoint contains no nets.policy.* pi0.5 weights")
        incompatible = core_model.load_state_dict(core_state, strict=True, assign=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(f"pi0.5 state mismatch: {incompatible}")
        meta_parameters = [
            name for name, parameter in core_model.named_parameters() if parameter.is_meta
        ]
        if meta_parameters:
            raise RuntimeError(
                "checkpoint did not materialize pi0.5 parameters: "
                f"{meta_parameters[:10]}"
            )
        # Transformers owns a few non-persistent rotary/cache buffers that are
        # intentionally absent from state_dict.  Materialize only those small
        # buffers; all learned parameters above must come from the checkpoint.
        for module in core_model.modules():
            for buffer_name, buffer in tuple(module._buffers.items()):
                if buffer is not None and buffer.is_meta:
                    module._buffers[buffer_name] = torch.zeros_like(
                        buffer, device="cpu"
                    )
        self._pi_model = core_model.to(args.device).eval()
        del checkpoint, core_state

        self._tokenizer = OpenPISentencePieceTokenizer(
            Path(args.pi05_tokenizer_model), max_length=128
        )
        self._prompt = str(args.task_prompt)
        self._generator = torch.Generator(device=args.device)
        self._generator.manual_seed(int(args.seed))
        self.checkpoint_report = {
            "architecture": "EgoVerse_pi0.5_PyTorch",
            "checkpoint_profile": args.checkpoint_profile or "custom",
            "action_space": "OpenPI_32D_to_PIPER_7D_absolute_joint_radians_binary_gripper",
            "action_horizon": 100,
            "native_control_horizon": 30,
            "normalization": "checkpoint_quantile_1_99",
            "prompt": self._prompt,
            "command_generation": "direct_absolute_joint_radians_no_ik",
            "strict_state_dict": True,
        }

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        del kwargs
        device = torch.device(self.config.device)
        raw_state_value = batch["observation.state"]
        raw_state = (
            raw_state_value.detach().cpu().numpy()
            if isinstance(raw_state_value, torch.Tensor)
            else np.asarray(raw_state_value)
        )
        raw_state = raw_state.reshape(-1)[:7].astype(np.float64)
        if raw_state.shape != (7,) or not np.isfinite(raw_state).all():
            raise RuntimeError(f"invalid raw PIPER state: {raw_state}")
        obs_joint = (raw_state[:6] * 1e-3 * np.pi / 180.0).astype(np.float32)
        obs_gripper = np.clip(
            raw_state[6:7] * 1e-3 / self.config.gripper_stroke_mm, 0.0, 1.0
        ).astype(np.float32)
        proprio_physical = np.concatenate([obs_joint, obs_gripper]).astype(np.float32)
        state_tensor = torch.from_numpy(proprio_physical).unsqueeze(0).to(device)
        normalized_state = quantile_normalize(state_tensor, self._state_stats)
        if not torch.isfinite(normalized_state).all():
            raise RuntimeError("normalized pi0.5 PIPER state contains NaN/Inf")

        front_key = f"observation.images.{self.config.front_camera_key}"
        wrist_key = f"observation.images.{self.config.wrist_camera_key}"
        front = prepare_rgb_chw_image(
            batch[front_key],
            expected_height=480,
            expected_width=640,
            resize_mismatch=False,
            name=front_key,
        )
        wrist = prepare_rgb_chw_image(
            batch[wrist_key],
            expected_height=480,
            expected_width=640,
            resize_mismatch=False,
            name=wrist_key,
        )
        front_224 = resize_with_pad_224(front).to(device)
        wrist_224 = resize_with_pad_224(wrist).to(device)
        images = [front_224, wrist_224, front_224.clone()]
        image_masks = [
            torch.ones(1, dtype=torch.bool, device=device),
            torch.ones(1, dtype=torch.bool, device=device),
            torch.zeros(1, dtype=torch.bool, device=device),
        ]
        token_ids, token_mask = self._tokenizer.tokenize(
            self._prompt, normalized_state[0].detach().cpu().numpy()
        )
        tokens = torch.from_numpy(token_ids).unsqueeze(0).to(device)
        masks = torch.from_numpy(token_mask).unsqueeze(0).to(device)
        noise = torch.randn(
            (1, 100, 32), generator=self._generator, device=device, dtype=torch.float32
        )

        started = time.perf_counter()
        pred32 = self._pi_model.sample_actions(
            images,
            image_masks,
            tokens,
            masks,
            noise=noise,
            num_steps=self.config.num_inference_steps,
        )
        pred_norm = pred32[..., :7]
        pred = quantile_unnormalize(pred_norm, self._action_stats)[0]
        inference_ms = (time.perf_counter() - started) * 1e3
        pred_norm_np = pred_norm[0].float().cpu().numpy()
        pred = pred.float().cpu().numpy()
        if pred.shape != (100, 7) or not np.isfinite(pred).all():
            raise RuntimeError(f"pi0.5 produced malformed PIPER actions: {pred.shape}")

        pred_for_control = (
            smooth_joint_action_chunk(
                pred, self.config.chunk_smoothing_window, self.config.chunk_smoothing_sigma
            )
            if self.config.chunk_smoothing
            else pred.copy()
        )
        raw_adjacent_rms = adjacent_joint_rms(pred)
        smoothed_adjacent_rms = adjacent_joint_rms(pred_for_control)
        selected_ticks = (
            self.config.action_horizon_frames
            if self.config.temporal_ensemble
            else self.config.query_frequency
        )
        query_indices = model_indices_for_control_ticks(100, 30, selected_ticks)
        full_control_indices = model_indices_for_control_ticks(100, 30, 30)
        full_gripper_raw = pred[full_control_indices, 6].copy()
        full_gripper_pct = np.where(
            full_gripper_raw >= self.config.gripper_threshold, 100.0, 0.0
        ).astype(np.float32)
        selected_unsmoothed = pred[query_indices].copy()
        selected = pred_for_control[query_indices].copy()
        margin = self.config.joint_limit_margin_rad
        low = self._joint_low_rad + margin
        high = self._joint_high_rad - margin
        if np.any(high <= low):
            raise RuntimeError("joint margin collapses a calibrated range")
        before_clip = selected[:, :6].copy()
        selected[:, :6] = np.clip(selected[:, :6], low, high)
        clipped_mask = np.abs(selected[:, :6] - before_clip) > 1e-9
        selected[:, 6] = (
            selected[:, 6] >= self.config.gripper_threshold
        ).astype(np.float32)

        raw_driver_actions = np.empty_like(selected, dtype=np.float32)
        raw_driver_actions[:, :6] = np.stack(
            [self._joint_rad_to_driver(row) for row in selected[:, :6]], axis=0
        )
        raw_driver_actions[:, 6] = selected[:, 6] * 100.0
        current_driver = self._joint_rad_to_driver(obs_joint)
        preview = self._preview_filtered_chunk(raw_driver_actions, current_driver)

        self._chunk_sequence_id += 1
        self.last_prediction_report = {
            "policy_space": "absolute_joint_radians_plus_binary_gripper",
            "checkpoint_contract": self.checkpoint_report,
            "output_head": "pi05_channels_0_6",
            "prompt": self._prompt,
            "proprio_physical": proprio_physical.tolist(),
            "proprio_model_input": normalized_state[0].float().cpu().tolist(),
            "model_chunk_shape": [100, 7],
            "query_frequency": self.config.query_frequency,
            "selected_control_ticks": selected_ticks,
            "temporal_ensemble": self.config.temporal_ensemble,
            "temporal_replan_interval": self.config.temporal_replan_interval,
            "temporal_ensemble_decay": self.config.temporal_ensemble_decay,
            "chunk_smoothing": self.config.chunk_smoothing,
            "chunk_smoothing_window": self.config.chunk_smoothing_window,
            "chunk_smoothing_sigma": self.config.chunk_smoothing_sigma,
            "chunk_raw_adjacent_joint_rms_rad_mean": raw_adjacent_rms[0],
            "chunk_raw_adjacent_joint_rms_rad_max": raw_adjacent_rms[1],
            "chunk_smoothed_adjacent_joint_rms_rad_mean": smoothed_adjacent_rms[0],
            "chunk_smoothed_adjacent_joint_rms_rad_max": smoothed_adjacent_rms[1],
            "query_indices": query_indices,
            "full_control_indices": full_control_indices,
            "full_control_gripper_raw": full_gripper_raw.tolist(),
            "full_control_gripper_commands_pct": full_gripper_pct.tolist(),
            "num_inference_steps": self.config.num_inference_steps,
            "inference_seed": self.config.inference_seed,
            "model_inference_ms": inference_ms,
            "joint_limit_clipped_values": int(clipped_mask.sum()),
            "joint_limit_clipped_steps": int(clipped_mask.any(axis=1).sum()),
            "full_model_chunk_columns": [
                "joint_1_rad", "joint_2_rad", "joint_3_rad", "joint_4_rad",
                "joint_5_rad", "joint_6_rad", "gripper_open",
            ],
            "full_model_chunk_normalized": pred_norm_np.tolist(),
            "full_model_chunk_unnormalized": pred.tolist(),
            "full_model_chunk_smoothed": pred_for_control.tolist(),
            "selected_physical_actions_unsmoothed": selected_unsmoothed.tolist(),
            "selected_physical_actions": selected.tolist(),
            "selected_raw_driver_actions": raw_driver_actions.tolist(),
            "selected_driver_actions": preview.tolist(),
            "trajectory_filter": {
                "max_joint_command_step": self.config.max_joint_command_step,
                "max_joint_command_accel": self.config.max_joint_command_accel,
                "current_driver_anchor": current_driver.tolist(),
            },
        }
        return torch.from_numpy(raw_driver_actions)


def parse_args() -> argparse.Namespace:
    pi_parser = argparse.ArgumentParser(add_help=False)
    pi_parser.add_argument("--pi05-runtime-dir", type=Path, default=DEFAULT_PI05_RUNTIME_DIR)
    pi_parser.add_argument("--pi05-tokenizer-model", type=Path, default=DEFAULT_TOKENIZER_MODEL)
    pi_parser.add_argument("--task-prompt", default=DEFAULT_PROMPT)
    pi_args, remaining = pi_parser.parse_known_args()
    old_profiles = legacy.CHECKPOINT_PROFILES
    old_argv = sys.argv
    try:
        legacy.CHECKPOINT_PROFILES = PI05_CHECKPOINT_PROFILES
        sys.argv = [old_argv[0], *remaining]
        args = legacy.parse_args()
    finally:
        legacy.CHECKPOINT_PROFILES = old_profiles
        sys.argv = old_argv
    for key, value in vars(pi_args).items():
        setattr(args, key, value)
    return args


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    if args.list_checkpoint_profiles:
        return
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    actual_sha256 = legacy.sha256_file(args.checkpoint)
    if actual_sha256 != args.checkpoint_sha256:
        raise RuntimeError(
            f"checkpoint SHA-256 mismatch: expected {args.checkpoint_sha256}, got {actual_sha256}"
        )
    joint_limits, gripper_stroke_mm = legacy.load_calibration(args.calibration_json)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    policy = EgoVersePI05JointPolicy(args, joint_limits, gripper_stroke_mm)
    if args.mode == "checkpoint":
        logger.info(
            "pi0.5 checkpoint loaded without constructing a robot: sha256=%s profile=%s contract=%s",
            actual_sha256,
            args.checkpoint_profile or "custom",
            policy.checkpoint_report,
        )
        return

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log = legacy.JsonlLogger(args.log_dir / f"pi05_{args.mode}_{timestamp}.jsonl")
    try:
        log.write(
            {
                "event": "start",
                "mode": args.mode,
                "checkpoint_profile": args.checkpoint_profile or "custom",
                "checkpoint_description": args.checkpoint_description,
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
        logger.info("wrote pi0.5 audit log to %s", log.path)


if __name__ == "__main__":
    main()
