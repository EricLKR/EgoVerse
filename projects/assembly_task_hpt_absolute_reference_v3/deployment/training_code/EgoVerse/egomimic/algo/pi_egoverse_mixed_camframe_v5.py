"""Single-frame pi0.5 mixed PIPER/Human CamFrame V5.

This is an additive policy.  It retains Expert-Action V4's tested native
30-step action semantics and bounded PIPER joint decoder, while adding:

* explicit learned embodiment and domain-specific proprio prefix tokens;
* PIPER wrist-camera dropout during training only;
* a checkpointed linear warm-up for the Human pose loss;
* domain losses combined by sum rather than an episode/frame-count average.

Human samples have exact-zero masks on joint, gripper, and reserved outputs.
Deployment executes only decoded PIPER joints and the binary gripper head.
"""

from __future__ import annotations

import math
import types
from collections import OrderedDict
from collections.abc import Mapping

import torch
import torch.nn as nn

from egomimic.algo.pi_egoverse_expert_action_v3 import (
    _flow_forward_with_expert_hidden_v3,
    _sample_actions_with_hidden_v3,
)
from egomimic.algo.pi_egoverse_expert_action_v4_bounded import (
    PIEgoVerseExpertActionV4Bounded,
)
from egomimic.models.pi05_expert_action_v3 import JOINT_SLICE, POSE_SLICE
from egomimic.models.pi05_expert_action_v4_bounded import (
    build_expert_action_target_v4,
)
from egomimic.models.pi05_mixed_camframe_v5 import (
    Pi05MixedDomainConditioningV5,
    human_ramp_weight,
)
from egomimic.rldb.embodiment.embodiment import get_embodiment


def _single_frame_embed_prefix_v5(
    policy, images, img_masks, lang_tokens, lang_masks
):
    """Embed images/language plus one explicit domain proprio token."""
    active_state = getattr(policy, "egoverse_v5_active_state", None)
    active_domain = getattr(policy, "egoverse_v5_active_domain_index", None)
    if active_state is None or active_domain is None:
        raise RuntimeError("V5 domain context was not activated before prefix embedding")
    state_token, embodiment = policy.egoverse_v5_conditioning(
        active_state, int(active_domain)
    )

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
                raise ValueError(f"V5 requires T=1, got {history}")
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
        image_emb = image_emb + slot_embeddings[slot_index].to(image_emb)[None, None]
        image_emb = image_emb + embodiment.to(image_emb)[None, None]
        embeddings.append(image_emb)
        pad_masks.append(image_mask[:, None].expand(batch, patches))
        attention_pattern += [0] * patches

    embeddings.append(state_token[:, None].to(embeddings[0]))
    pad_masks.append(
        torch.ones(
            state_token.shape[0], 1, dtype=torch.bool, device=state_token.device
        )
    )
    attention_pattern.append(0)

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


class PIEgoVerseMixedCamFrameV5(PIEgoVerseExpertActionV4Bounded):
    """V5 mixed policy, initialized from official pi0.5 plus new V5 modules."""

    def __init__(
        self,
        *args,
        human_embodiment: str = "human_left_arm",
        human_pose_final_weight: float = 0.30,
        human_pose_ramp_steps: int = 2500,
        human_update_interval: int = 2,
        piper_wrist_dropout_probability: float = 0.35,
        domain_conditioning: Mapping | None = None,
        **kwargs,
    ) -> None:
        self.human_embodiment_name = str(human_embodiment).upper()
        self.human_pose_final_weight = float(human_pose_final_weight)
        self.human_pose_ramp_steps = int(human_pose_ramp_steps)
        self.human_update_interval = int(human_update_interval)
        self.piper_wrist_dropout_probability = float(
            piper_wrist_dropout_probability
        )
        if not 0.0 <= self.piper_wrist_dropout_probability < 1.0:
            raise ValueError("piper_wrist_dropout_probability must be in [0,1)")
        if self.human_pose_final_weight < 0 or self.human_pose_ramp_steps < 0:
            raise ValueError("Human loss schedule must be non-negative")
        if self.human_update_interval <= 0:
            raise ValueError("human_update_interval must be positive")
        super().__init__(*args, **kwargs)

        prefix_dim = int(
            self.model.paligemma_with_expert.paligemma.config.text_config.hidden_size
        )
        conditioning_cfg = dict(domain_conditioning or {})
        conditioning_cfg.setdefault("prefix_dim", prefix_dim)
        self.model.egoverse_v5_conditioning = Pi05MixedDomainConditioningV5(
            **conditioning_cfg
        )
        self.model.register_buffer(
            "egoverse_v5_training_step",
            torch.zeros((), dtype=torch.long),
            persistent=True,
        )
        self.model.embed_prefix = types.MethodType(
            _single_frame_embed_prefix_v5, self.model
        )

    def _domain_index(self, embodiment_id: int) -> int:
        return 0 if embodiment_id == self.piper_embodiment_id else 1

    def _activate_context(self, observation, embodiment_id: int, *, train: bool) -> None:
        domain_index = self._domain_index(embodiment_id)
        expected = self.model.egoverse_v5_conditioning.state_dims[domain_index]
        if observation.state.ndim != 2 or observation.state.shape[-1] != expected:
            raise ValueError(
                f"V5 domain {domain_index} expects normalized proprio [B,{expected}], "
                f"got {tuple(observation.state.shape)}"
            )
        self.model.egoverse_v5_active_state = observation.state
        self.model.egoverse_v5_active_domain_index = domain_index

        if train and domain_index == 0 and self.piper_wrist_dropout_probability:
            wrist_key = "left_wrist_0_rgb"
            mask = observation.image_masks.get(wrist_key)
            if mask is not None:
                keep = torch.rand(mask.shape, device=mask.device) >= (
                    self.piper_wrist_dropout_probability
                )
                observation.image_masks[wrist_key] = mask.bool() & keep

    def _flow_targets(self, model_batch, embodiment_id, pose32, pose_weights):
        current_raw = None
        target_joint_raw = None
        if embodiment_id == self.piper_embodiment_id:
            _, current_raw, target_joint_raw = self._piper_joint_inputs(
                model_batch, embodiment_id
            )
        target32, weights32 = build_expert_action_target_v4(
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
        step = int(self.model.egoverse_v5_training_step.item())
        current_human_weight = human_ramp_weight(
            step,
            final_weight=self.human_pose_final_weight,
            ramp_steps=self.human_pose_ramp_steps,
        )
        predictions["human_pose_schedule_weight"] = torch.tensor(
            current_human_weight, device=self.device
        )

        for embodiment_id, model_batch in batch.items():
            if (
                embodiment_id != self.piper_embodiment_id
                and step % self.human_update_interval != 0
            ):
                continue
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
            self._activate_context(observation, embodiment_id, train=True)
            target32, pose_mask, joint_mask, current_raw, target_joint_raw = (
                self._flow_targets(model_batch, embodiment_id, pose32, pose_weights)
            )
            elementwise, suffix_hidden, predicted_clean = (
                _flow_forward_with_expert_hidden_v3(
                    self.nets["policy"], observation, target32
                )
            )
            pose_weight = (
                self.piper_pose_flow_weight
                if embodiment_id == self.piper_embodiment_id
                else current_human_weight
            )
            predictions[f"{name}_pose_flow_loss"] = pose_weight * self._masked_loss(
                elementwise, pose_mask
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
        self.model.egoverse_v5_training_step.add_(1)
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
                self._activate_context(observation, embodiment_id, train=False)
                target32, pose_mask, joint_mask, current_raw, target_joint_raw = (
                    self._flow_targets(
                        model_batch, embodiment_id, pose32, pose_weights
                    )
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
        side_image: torch.Tensor,
        current_raw: torch.Tensor,
        prompt: str | None = None,
        noise: torch.Tensor | None = None,
        num_steps: int | None = None,
    ):
        """Run the exact Assembly New1 three-camera training contract.

        The inherited V4 deployment helper only accepted front and wrist
        frames.  Assembly New1 trained ``right_wrist_0_rgb`` from ``side_img``;
        omitting it changes the SigLIP prefix and is therefore forbidden.
        """
        embodiment_id = self.piper_embodiment_id
        state_key = next(
            key
            for key in self.proprio_keys[embodiment_id]
            if key.endswith(".joints")
        )
        stats = self.norm_stats.norm_stats.get(embodiment_id, {}).get(state_key)
        if stats is None:
            raise RuntimeError(f"checkpoint lacks normalization stats for {state_key}")
        current_device = current_raw.to(self.device, dtype=torch.float32)
        normalized_state = self.norm_stats._apply_norm_one(current_device, stats)
        self.model.egoverse_v5_active_state = normalized_state
        self.model.egoverse_v5_active_domain_index = 0
        batch_size = current_device.shape[0]
        images = (front_image, wrist_image, side_image)
        if any(image.shape[0] != batch_size for image in images):
            raise ValueError("three camera and proprio batch sizes differ")

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
            "observations.images.side_img": side_image.to(self.device),
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
            current_raw=current_device,
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
        losses = OrderedDict()
        pose_terms = []
        for key, value in predictions.items():
            if key.endswith("_pose_flow_loss"):
                losses[key] = value
                pose_terms.append(value)
        if not pose_terms:
            raise ValueError("V5 received no pose-flow loss")
        pose_total = torch.stack(pose_terms).sum()
        total = self.shared_pose_loss_weight * pose_total
        losses["shared_pose_flow_loss"] = pose_total
        losses["human_pose_schedule_weight"] = predictions[
            "human_pose_schedule_weight"
        ]
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


class PIEgoVerseMixedCamFrameV5LoRA(PIEgoVerseMixedCamFrameV5):
    """Official pi0.5 base + audited LoRA + fully trained V5 interfaces."""

    def __init__(self, *args, lora: Mapping | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        from egomimic.models.pi05_mixed_lora_v1 import (
            Pi05MixedLoraConfig,
            configure_pi05_mixed_lora,
        )

        config = Pi05MixedLoraConfig(**dict(lora or {}))
        summary = configure_pi05_mixed_lora(self.nets["policy"], config)
        for parameter in self.model.egoverse_v5_conditioning.parameters():
            parameter.requires_grad_(True)
        auxiliary_trainable = sum(
            parameter.numel()
            for parameter in self.nets["auxiliary"].parameters()
            if parameter.requires_grad
        )
        conditioning_trainable = sum(
            parameter.numel()
            for parameter in self.model.egoverse_v5_conditioning.parameters()
            if parameter.requires_grad
        )
        total = sum(parameter.numel() for parameter in self.nets.parameters())
        trainable = sum(
            parameter.numel()
            for parameter in self.nets.parameters()
            if parameter.requires_grad
        )
        self.lora_config = config
        self.lora_summary = summary.to_dict()
        self.finetune_parameter_stats = {
            "total": total,
            "trainable": trainable,
            "frozen": total - trainable,
            "trainable_percent": 100.0 * trainable / total,
            "policy_lora_and_interfaces": summary.trainable_parameters,
            "v5_domain_conditioning": conditioning_trainable,
            "auxiliary_heads": auxiliary_trainable,
        }
