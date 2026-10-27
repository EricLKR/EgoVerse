"""EgoVerse + pi0.5 Expert-Action V3 training and deployment wrapper.

V3 keeps the existing PI and MultiHead V2 implementations intact.  Unlike V2,
the deployable six-joint trajectory is carried by Gemma action-expert channels
9..14; there is no prefix-only joint decoder.  Both embodiments supervise the
same camera-frame pose channels, while exact masks prevent human samples from
updating robot-only action channels.
"""

from __future__ import annotations

import math
import types
from collections import OrderedDict
from collections.abc import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks

from egomimic.algo.pi import PI
from egomimic.algo.pi_egoverse_multihead_v2 import (
    PIEgoVerseMultiHeadV2,
    _single_frame_preprocess_observation,
)
from egomimic.models.pi05_expert_action_v3 import (
    ACTION_DIM,
    JOINT_SLICE,
    POSE_SLICE,
    Pi05ExpertActionAuxiliaryV3,
    build_expert_action_target,
    clean_action_from_velocity,
)
from egomimic.rldb.embodiment.embodiment import get_embodiment, get_embodiment_id


def _single_frame_embed_prefix_v3(policy, images, img_masks, lang_tokens, lang_masks):
    """Embed one frame and add explicit, learned camera-slot identities."""
    embeddings = []
    pad_masks = []
    attention_pattern: list[int] = []
    slot_embeddings = policy.egoverse_camera_slot_embeddings
    if len(images) > slot_embeddings.shape[0]:
        raise ValueError("more image slots than configured camera embeddings")

    for slot_index, (image, image_mask) in enumerate(
        zip(images, img_masks, strict=True)
    ):
        if image.ndim == 5:
            batch, history = image.shape[:2]
            if history != 1:
                raise ValueError(f"Expert-Action V3 requires T=1, got {history}")
            flat = image.reshape(batch * history, *image.shape[2:])
            image_emb = policy._apply_checkpoint(
                policy.paligemma_with_expert.embed_image, flat
            )
            patches, width = image_emb.shape[1:]
            image_emb = image_emb.reshape(batch, history, patches, width)[:, 0]
        else:
            image_emb = policy._apply_checkpoint(
                policy.paligemma_with_expert.embed_image, image
            )
            batch, patches = image_emb.shape[:2]
        slot = slot_embeddings[slot_index].to(image_emb)[None, None]
        image_emb = image_emb + slot
        embeddings.append(image_emb)
        pad_masks.append(image_mask[:, None].expand(batch, patches))
        attention_pattern += [0] * patches

    def embed_language(tokens):
        result = policy.paligemma_with_expert.embed_language_tokens(tokens)
        return result * math.sqrt(result.shape[-1])

    language_emb = policy._apply_checkpoint(embed_language, lang_tokens)
    embeddings.append(language_emb)
    pad_masks.append(lang_masks)
    attention_pattern += [0] * language_emb.shape[1]
    embeddings = torch.cat(embeddings, dim=1)
    pad_masks = torch.cat(pad_masks, dim=1)
    attention = torch.tensor(
        attention_pattern, dtype=torch.bool, device=pad_masks.device
    )[None].expand(pad_masks.shape[0], -1)
    return embeddings, pad_masks, attention


def _flow_forward_with_expert_hidden_v3(
    policy, observation, actions, noise=None, time=None
):
    """OpenPI flow forward that also returns the action-expert hidden states."""
    images, masks, lang, lang_masks, state = policy._preprocess_observation(
        observation, train=True
    )
    if noise is None:
        noise = policy.sample_noise(actions.shape, actions.device)
    if time is None:
        time = policy.sample_time(actions.shape[0], actions.device)
    expanded_time = time[:, None, None]
    noisy_action = expanded_time * noise + (1.0 - expanded_time) * actions
    target_velocity = noise - actions

    prefix, prefix_pad, prefix_attention = policy.embed_prefix(
        images, masks, lang, lang_masks
    )
    suffix, suffix_pad, suffix_attention, adarms = policy.embed_suffix(
        state, noisy_action, time
    )
    if (
        policy.paligemma_with_expert.paligemma.language_model.layers[0]
        .self_attn.q_proj.weight.dtype
        == torch.bfloat16
    ):
        prefix = prefix.to(torch.bfloat16)
        suffix = suffix.to(torch.bfloat16)
    pad = torch.cat((prefix_pad, suffix_pad), dim=1)
    attention = torch.cat((prefix_attention, suffix_attention), dim=1)
    attention_4d = policy._prepare_attention_masks_4d(
        make_att_2d_masks(pad, attention)
    )
    position_ids = torch.cumsum(pad, dim=1) - 1

    def expert_forward(prefix_tokens, suffix_tokens, mask, positions, condition):
        (_, suffix_output), _ = policy.paligemma_with_expert.forward(
            attention_mask=mask,
            position_ids=positions,
            past_key_values=None,
            inputs_embeds=[prefix_tokens, suffix_tokens],
            use_cache=False,
            adarms_cond=[None, condition],
        )
        return suffix_output

    suffix_hidden = policy._apply_checkpoint(
        expert_forward, prefix, suffix, attention_4d, position_ids, adarms
    )[:, -policy.config.action_horizon :].float()
    velocity = policy._apply_checkpoint(policy.action_out_proj, suffix_hidden)
    elementwise = F.mse_loss(target_velocity, velocity, reduction="none")
    predicted_clean = clean_action_from_velocity(noisy_action, time, velocity)
    return elementwise, suffix_hidden, predicted_clean


def _denoise_step_with_hidden_v3(
    policy,
    state,
    prefix_pad_masks,
    past_key_values,
    noisy_action,
    timestep,
):
    suffix, suffix_pad, suffix_attention, adarms = policy.embed_suffix(
        state, noisy_action, timestep
    )
    suffix_len = suffix_pad.shape[1]
    batch_size = prefix_pad_masks.shape[0]
    prefix_len = prefix_pad_masks.shape[1]
    prefix_mask = prefix_pad_masks[:, None, :].expand(
        batch_size, suffix_len, prefix_len
    )
    suffix_mask = make_att_2d_masks(suffix_pad, suffix_attention)
    full_mask = torch.cat((prefix_mask, suffix_mask), dim=2)
    prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
    position_ids = prefix_offsets + torch.cumsum(suffix_pad, dim=1) - 1
    attention_4d = policy._prepare_attention_masks_4d(full_mask)
    policy.paligemma_with_expert.gemma_expert.model.config._attn_implementation = (
        "eager"
    )
    outputs, _ = policy.paligemma_with_expert.forward(
        attention_mask=attention_4d,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=[None, suffix],
        use_cache=False,
        adarms_cond=[None, adarms],
    )
    hidden = outputs[1][:, -policy.config.action_horizon :].float()
    return policy.action_out_proj(hidden), hidden


@torch.no_grad()
def _sample_actions_with_hidden_v3(
    policy,
    *,
    device,
    observation,
    noise=None,
    num_steps=10,
):
    """Match OpenPI Euler sampling and retain the final expert hidden state."""
    if int(num_steps) <= 0:
        raise ValueError("num_steps must be positive")
    batch_size = observation.state.shape[0]
    if noise is None:
        noise = policy.sample_noise(
            (batch_size, policy.config.action_horizon, policy.config.action_dim),
            device,
        )
    images, masks, lang, lang_masks, state = policy._preprocess_observation(
        observation, train=False
    )
    prefix, prefix_pad, prefix_attention = policy.embed_prefix(
        images, masks, lang, lang_masks
    )
    prefix_mask = make_att_2d_masks(prefix_pad, prefix_attention)
    prefix_position_ids = torch.cumsum(prefix_pad, dim=1) - 1
    policy.paligemma_with_expert.paligemma.language_model.config._attn_implementation = (
        "eager"
    )
    _, cache = policy.paligemma_with_expert.forward(
        attention_mask=policy._prepare_attention_masks_4d(prefix_mask),
        position_ids=prefix_position_ids,
        past_key_values=None,
        inputs_embeds=[prefix, None],
        use_cache=True,
    )

    delta_time = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=device)
    noisy_action = noise
    time = torch.tensor(1.0, dtype=torch.float32, device=device)
    final_hidden = None
    while time >= -delta_time / 2:
        velocity, final_hidden = _denoise_step_with_hidden_v3(
            policy,
            state,
            prefix_pad,
            cache,
            noisy_action,
            time.expand(batch_size),
        )
        noisy_action = noisy_action + delta_time * velocity
        time += delta_time
    if final_hidden is None:
        raise RuntimeError("flow sampler produced no action-expert hidden state")
    return noisy_action, final_hidden


class PIEgoVerseExpertActionV3(PI):
    """Single-frame, native-horizon pi0.5 policy with expert joint channels."""

    # Reuse only V2's tested data/prompt plumbing; no V2 joint module is built.
    _state_for_sample = PIEgoVerseMultiHeadV2._state_for_sample
    _robomimic_to_pi_data = PIEgoVerseMultiHeadV2._robomimic_to_pi_data
    _unnormalize = PIEgoVerseMultiHeadV2._unnormalize
    _piper_joint_inputs = PIEgoVerseMultiHeadV2._piper_joint_inputs

    def __init__(
        self,
        *args,
        observation_history: int = 1,
        auxiliary_joint_key: str = "actions_joints",
        piper_embodiment: str = "piper_left_arm",
        shared_pose_loss_weight: float = 1.0,
        piper_joint_flow_loss_weight: float = 1.0,
        piper_auxiliary_loss_weight: float = 1.0,
        piper_pose_flow_weight: float = 0.5,
        human_pose_flow_weight: float = 1.0,
        camera_slot_init_std: float = 0.02,
        auxiliary: Mapping | None = None,
        **kwargs,
    ) -> None:
        self.observation_history = int(observation_history)
        if self.observation_history != 1:
            raise ValueError("Expert-Action V3 requires one current observation")
        self.auxiliary_joint_key = str(auxiliary_joint_key)
        self.piper_embodiment_id = get_embodiment_id(piper_embodiment)
        self.shared_pose_loss_weight = float(shared_pose_loss_weight)
        self.piper_joint_flow_loss_weight = float(piper_joint_flow_loss_weight)
        self.piper_auxiliary_loss_weight = float(piper_auxiliary_loss_weight)
        self.piper_pose_flow_weight = float(piper_pose_flow_weight)
        self.human_pose_flow_weight = float(human_pose_flow_weight)
        super().__init__(*args, **kwargs)

        if int(self.config.model.action_dim) != ACTION_DIM:
            raise ValueError("Expert-Action V3 requires action_dim=32")
        if int(self.config.model.action_horizon) != 30:
            raise ValueError("Expert-Action V3 requires action_horizon=30")

        prefix_dim = int(
            self.model.paligemma_with_expert.paligemma.config.text_config.hidden_size
        )
        expert_dim = int(self.model.action_in_proj.out_features)
        camera_slots = len(self.pi_cam_keys)
        self.model.egoverse_camera_slot_embeddings = nn.Parameter(
            torch.empty(camera_slots, prefix_dim)
        )
        nn.init.normal_(
            self.model.egoverse_camera_slot_embeddings,
            std=float(camera_slot_init_std),
        )
        self.model.egoverse_observation_history = 1
        self.model.embed_prefix = types.MethodType(
            _single_frame_embed_prefix_v3, self.model
        )
        self.model._preprocess_observation = types.MethodType(
            _single_frame_preprocess_observation, self.model
        )

        auxiliary_cfg = dict(auxiliary or {})
        auxiliary_cfg.setdefault("expert_dim", expert_dim)
        auxiliary_cfg.setdefault("action_horizon", 30)
        self.nets["auxiliary"] = Pi05ExpertActionAuxiliaryV3(**auxiliary_cfg)
        self.joint_residual_scale_rad = self.nets[
            "auxiliary"
        ].joint_residual_scale_rad

        # Robomimic algorithms are containers, not nn.Module subclasses.  All
        # registered torch modules live in ``self.nets``.
        total = sum(parameter.numel() for parameter in self.nets.parameters())
        trainable = sum(
            parameter.numel()
            for parameter in self.nets.parameters()
            if parameter.requires_grad
        )
        self.finetune_parameter_stats = {
            "total": total,
            "trainable": trainable,
            "frozen": total - trainable,
            "trainable_percent": 100.0 * trainable / total,
        }

    @staticmethod
    def _masked_loss(elementwise: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if elementwise.shape != mask.shape:
            raise ValueError("flow loss and mask shapes differ")
        denominator = mask.sum()
        if denominator.detach().item() <= 0:
            raise ValueError("flow loss mask is empty")
        return (elementwise * mask.to(elementwise)).sum() / denominator

    def _flow_targets(self, model_batch, embodiment_id, pose32, pose_weights):
        current_raw = None
        target_joint_raw = None
        if embodiment_id == self.piper_embodiment_id:
            _, current_raw, target_joint_raw = self._piper_joint_inputs(
                model_batch, embodiment_id
            )
        target32, weights32 = build_expert_action_target(
            pose32,
            pose_weights,
            pose_weight=1.0,
            target_joint_raw=target_joint_raw,
            current_raw=current_raw,
            joint_residual_scale_rad=self.joint_residual_scale_rad,
            joint_weight=1.0,
        )
        pose_mask = torch.zeros_like(weights32)
        pose_mask[..., POSE_SLICE] = weights32[..., POSE_SLICE]
        joint_mask = torch.zeros_like(weights32)
        if target_joint_raw is not None:
            joint_mask[..., JOINT_SLICE] = weights32[..., JOINT_SLICE]
        return target32, pose_mask, joint_mask, current_raw, target_joint_raw

    def forward_training(self, batch):
        predictions = OrderedDict()
        for embodiment_id, model_batch in batch.items():
            ac_key = self.ac_keys[embodiment_id]
            name = get_embodiment(embodiment_id).lower()
            observation, pose32, pose_weights = self._robomimic_to_pi_data(
                model_batch,
                self.camera_keys.get(embodiment_id, self.pi_cam_keys),
                self.proprio_keys[embodiment_id],
                self.lang_keys[embodiment_id],
                ac_key,
                name,
            )
            (
                target32,
                pose_mask,
                joint_mask,
                current_raw,
                target_joint_raw,
            ) = self._flow_targets(
                model_batch, embodiment_id, pose32, pose_weights
            )
            elementwise, suffix_hidden, predicted_clean = (
                _flow_forward_with_expert_hidden_v3(
                    self.nets["policy"], observation, target32
                )
            )
            pose_weight = (
                self.piper_pose_flow_weight
                if embodiment_id == self.piper_embodiment_id
                else self.human_pose_flow_weight
            )
            predictions[f"{name}_pose_flow_loss"] = (
                pose_weight * self._masked_loss(elementwise, pose_mask)
            )

            if embodiment_id == self.piper_embodiment_id:
                predictions["piper_joint_flow_loss"] = self._masked_loss(
                    elementwise, joint_mask
                )
                converter = self.action_registry.get(embodiment_id, ac_key)
                predicted_pose_norm = converter.from32(predicted_clean)
                predicted_pose_raw = self._unnormalize(
                    predicted_pose_norm, ac_key, embodiment_id
                )
                predictions["piper_auxiliary_loss"] = self.nets[
                    "auxiliary"
                ].compute_loss(
                    suffix_hidden=suffix_hidden,
                    predicted_clean_action32=predicted_clean,
                    current_raw=current_raw,
                    target_joint_raw=target_joint_raw,
                    predicted_pose_camera_raw=predicted_pose_raw,
                )
        return predictions

    def forward_eval(self, batch):
        outputs = {}
        with torch.no_grad():
            for embodiment_id, model_batch in batch.items():
                ac_key = self.ac_keys[embodiment_id]
                name = get_embodiment(embodiment_id).lower()
                observation, pose32, pose_weights = self._robomimic_to_pi_data(
                    model_batch,
                    self.camera_keys.get(embodiment_id, self.pi_cam_keys),
                    self.proprio_keys[embodiment_id],
                    self.lang_keys[embodiment_id],
                    ac_key,
                    name,
                )
                (
                    target32,
                    pose_mask,
                    joint_mask,
                    current_raw,
                    target_joint_raw,
                ) = self._flow_targets(
                    model_batch, embodiment_id, pose32, pose_weights
                )
                elementwise, _, _ = _flow_forward_with_expert_hidden_v3(
                    self.nets["policy"], observation, target32
                )
                outputs[f"{name}_loss"] = self._masked_loss(
                    elementwise, pose_mask + joint_mask
                )
                sampled, hidden = _sample_actions_with_hidden_v3(
                    self.nets["policy"],
                    device=self.device,
                    observation=observation,
                    num_steps=self.num_steps,
                )
                converter = self.action_registry.get(embodiment_id, ac_key)
                pose_norm = converter.from32(sampled)
                outputs[f"{name}_{ac_key}"] = self._unnormalize(
                    pose_norm, ac_key, embodiment_id
                )
                if embodiment_id == self.piper_embodiment_id:
                    actions, logits = self.nets["auxiliary"].decode_actions(
                        sampled_action32=sampled,
                        suffix_hidden=hidden,
                        current_raw=current_raw,
                    )
                    outputs[f"{name}_{self.auxiliary_joint_key}"] = actions
                    outputs[f"{name}_gripper_logits"] = logits
        return outputs

    @torch.no_grad()
    def forward_joint_deploy(
        self,
        *,
        front_image: torch.Tensor,
        wrist_image: torch.Tensor,
        current_raw: torch.Tensor,
        prompt: str | None = None,
        noise: torch.Tensor | None = None,
        num_steps: int | None = None,
    ) -> dict[str, torch.Tensor | list[str]]:
        """Run the exact training wrapper and return native PIPER commands."""
        embodiment_id = self.piper_embodiment_id
        state_key = next(
            (
                key
                for key in self.proprio_keys[embodiment_id]
                if key.endswith(".joints")
            ),
            None,
        )
        if state_key is None:
            raise RuntimeError("checkpoint has no PIPER joint proprio key")
        current_raw = current_raw.to(self.device, dtype=torch.float32)
        if current_raw.ndim != 2 or current_raw.shape[1] != 7:
            raise ValueError("current_raw must be [B,7]")
        if not torch.isfinite(current_raw).all():
            raise ValueError("current_raw contains NaN/Inf")
        batch_size = current_raw.shape[0]
        if front_image.shape[0] != batch_size or wrist_image.shape[0] != batch_size:
            raise ValueError("image and proprio batch sizes differ")

        stats = self.norm_stats.norm_stats.get(embodiment_id, {}).get(state_key)
        if stats is None:
            raise RuntimeError(f"checkpoint lacks normalization stats for {state_key}")
        normalized_state = self.norm_stats._apply_norm_one(current_raw, stats)
        name = get_embodiment(embodiment_id).lower()
        if prompt is None:
            prompts = self._build_prompts({}, name, batch_size)
        else:
            embodiment_label = name.replace("_", " ")
            prompts = []
            for _ in range(batch_size):
                blocks = [str(prompt)]
                if self.embodiment_label:
                    blocks.append(f"Embodiment: {embodiment_label}")
                if self.control_mode:
                    blocks.append(
                        f"Control mode: {self._control_mode_for(embodiment_label)}"
                    )
                prompts.append(", ".join(blocks))
        states = (
            [row.detach().cpu().numpy() for row in normalized_state]
            if self.proprio_in_prompt
            else [None] * batch_size
        )
        tokenized = {
            key: value.to(self.device)
            for key, value in self._tokenize_prompts(prompts, states).items()
        }
        model_batch = {
            "observations.images.front_img_1": front_image.to(self.device),
            "observations.images.wrist_img": wrist_image.to(self.device),
            state_key: normalized_state[:, None],
            **tokenized,
        }
        ac_key = self.ac_keys[embodiment_id]
        model_batch[ac_key] = torch.zeros(
            batch_size, 30, 9, device=self.device, dtype=torch.float32
        )
        observation, _, _ = self._robomimic_to_pi_data(
            model_batch,
            self.camera_keys.get(embodiment_id, self.pi_cam_keys),
            self.proprio_keys[embodiment_id],
            self.lang_keys[embodiment_id],
            ac_key,
            name,
        )
        sampled, hidden = _sample_actions_with_hidden_v3(
            self.nets["policy"],
            device=self.device,
            observation=observation,
            noise=noise,
            num_steps=self.num_steps if num_steps is None else int(num_steps),
        )
        actions, gripper_logits = self.nets["auxiliary"].decode_actions(
            sampled_action32=sampled,
            suffix_hidden=hidden,
            current_raw=current_raw,
        )
        return {
            "actions": actions,
            "action32": sampled,
            "gripper_logits": gripper_logits,
            "normalized_state": normalized_state,
            "prompts": prompts,
        }

    def compute_losses(self, predictions, batch):
        del batch
        pose_losses = [
            value
            for key, value in predictions.items()
            if key.endswith("_pose_flow_loss")
        ]
        if not pose_losses:
            raise ValueError("Expert-Action V3 received no pose loss")
        pose = torch.stack(pose_losses).mean()
        total = self.shared_pose_loss_weight * pose
        losses = OrderedDict(shared_pose_flow_loss=pose)
        if "piper_joint_flow_loss" in predictions:
            joint_flow = predictions["piper_joint_flow_loss"]
            total = total + self.piper_joint_flow_loss_weight * joint_flow
            losses["piper_joint_flow_loss"] = joint_flow
        if "piper_auxiliary_loss" in predictions:
            auxiliary = predictions["piper_auxiliary_loss"]
            total = total + self.piper_auxiliary_loss_weight * auxiliary
            losses["piper_auxiliary_loss"] = auxiliary
            for name, value in self.nets["auxiliary"].last_loss_components.items():
                losses[f"piper_{name}"] = value
        losses["action_loss"] = total
        return losses
