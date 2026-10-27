"""EgoVerse + pi0.5 MultiHead V2 training adapter.

The existing :class:`egomimic.algo.pi.PI` implementation remains untouched.
This opt-in subclass fixes the mixed-training semantic collision by using:

* one shared 9-D camera-frame pose flow head for PIPER and human data;
* one deterministic 7-D PIPER residual-joint head for deployment;
* one synchronized image/proprio observation, matching the deployment clock;
* physical FK and camera-frame agreement losses, without IK.
"""

from __future__ import annotations

import math
import types
from collections import OrderedDict
from collections.abc import Mapping

import torch
import torch.nn.functional as F
from openpi.models_pytorch import preprocessing_pytorch as openpi_preprocessing
from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
from openpi.shared.image_tools import resize_with_pad_torch

from egomimic.algo.pi import PI
from egomimic.models.pi05_egoverse_multihead_v2 import (
    Pi05PrefixJointHeadV2,
    ensure_btchw,
)
from egomimic.models.preprocess_pi_obs import (
    _SimpleObservation,
    _empty_lang_placeholders,
    _to_minus1_1,
)
from egomimic.rldb.embodiment.embodiment import get_embodiment, get_embodiment_id


def _single_frame_preprocess_observation(policy, observation, *, train=True):
    """Return OpenPI's five-item tuple while retaining a length-one image axis."""
    first = next(iter(observation.images.values()))
    if first.ndim != 5:
        processed = openpi_preprocessing.preprocess_observation_pytorch(
            observation, train=train
        )
        return (
            list(processed.images.values()),
            list(processed.image_masks.values()),
            processed.tokenized_prompt,
            processed.tokenized_prompt_mask,
            processed.state,
        )
    batch, history = first.shape[:2]
    if history != 1:
        raise ValueError(f"single-frame V2 requires T=1, got T={history}")
    flat_images = {}
    flat_masks = {}
    for key, image in observation.images.items():
        if image.ndim != 5 or image.shape[:2] != (batch, history):
            raise ValueError("all MultiHead V2 images must share [B,T]")
        flat_images[key] = image.reshape(batch * history, *image.shape[2:])
        flat_masks[key] = observation.image_masks[key].repeat_interleave(history)
    flat_observation = _SimpleObservation(
        images=flat_images,
        image_masks=flat_masks,
        state=torch.zeros(batch * history, 0, device=first.device),
        tokenized_prompt=observation.tokenized_prompt,
        tokenized_prompt_mask=observation.tokenized_prompt_mask,
        token_ar_mask=observation.token_ar_mask,
        token_loss_mask=observation.token_loss_mask,
    )
    processed = openpi_preprocessing.preprocess_observation_pytorch(
        flat_observation, train=train
    )
    images = {
        key: image.reshape(batch, history, *image.shape[1:])
        for key, image in processed.images.items()
    }
    return (
        list(images.values()),
        list(observation.image_masks.values()),
        observation.tokenized_prompt,
        observation.tokenized_prompt_mask,
        observation.state,
    )


def _single_frame_embed_prefix(policy, images, img_masks, lang_tokens, lang_masks):
    """Embed the current frame with the standard pi0.5 visual token layout."""
    embeddings = []
    pad_masks = []
    attention_pattern: list[int] = []
    for image, image_mask in zip(images, img_masks, strict=True):
        if image.ndim == 5:
            batch, history = image.shape[:2]
            flat = image.reshape(batch * history, *image.shape[2:])
            image_emb = policy._apply_checkpoint(
                policy.paligemma_with_expert.embed_image, flat
            )
            patches, width = image_emb.shape[1:]
            image_emb = image_emb.reshape(batch, history, patches, width)
            if history != 1:
                raise ValueError(f"single-frame V2 requires T=1, got T={history}")
            # Exact pretrained SigLIP feature: no temporal embedding/fusion.
            image_emb = image_emb[:, 0]
        else:
            image_emb = policy._apply_checkpoint(
                policy.paligemma_with_expert.embed_image, image
            )
            batch, patches = image_emb.shape[:2]
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


def _flow_forward_with_prefix_features(policy, observation, actions, noise=None, time=None):
    """Equivalent PI flow loss that also exposes aligned PaliGemma prefix tokens."""
    images, masks, lang, lang_masks, state = policy._preprocess_observation(
        observation, train=True
    )
    if noise is None:
        noise = policy.sample_noise(actions.shape, actions.device)
    if time is None:
        time = policy.sample_time(actions.shape[0], actions.device)
    expanded_time = time[:, None, None]
    x_t = expanded_time * noise + (1.0 - expanded_time) * actions
    target_velocity = noise - actions

    prefix, prefix_pad, prefix_attention = policy.embed_prefix(
        images, masks, lang, lang_masks
    )
    suffix, suffix_pad, suffix_attention, adarms = policy.embed_suffix(
        state, x_t, time
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

    def joint_forward(prefix_tokens, suffix_tokens, mask, positions, condition):
        (prefix_output, suffix_output), _ = policy.paligemma_with_expert.forward(
            attention_mask=mask,
            position_ids=positions,
            past_key_values=None,
            inputs_embeds=[prefix_tokens, suffix_tokens],
            use_cache=False,
            adarms_cond=[None, condition],
        )
        return prefix_output, suffix_output

    prefix_output, suffix_output = policy._apply_checkpoint(
        joint_forward, prefix, suffix, attention_4d, position_ids, adarms
    )
    suffix_output = suffix_output[:, -policy.config.action_horizon :].float()
    velocity = policy._apply_checkpoint(policy.action_out_proj, suffix_output)
    flow_loss = F.mse_loss(target_velocity, velocity, reduction="none")
    return flow_loss, prefix_output.float(), prefix_pad


@torch.no_grad()
def _extract_prefix_features(policy, observation):
    images, masks, lang, lang_masks, _ = policy._preprocess_observation(
        observation, train=False
    )
    prefix, prefix_pad, prefix_attention = policy.embed_prefix(
        images, masks, lang, lang_masks
    )
    if (
        policy.paligemma_with_expert.paligemma.language_model.layers[0]
        .self_attn.q_proj.weight.dtype
        == torch.bfloat16
    ):
        prefix = prefix.to(torch.bfloat16)
    attention_4d = policy._prepare_attention_masks_4d(
        make_att_2d_masks(prefix_pad, prefix_attention)
    )
    position_ids = torch.cumsum(prefix_pad, dim=1) - 1
    # The robot's pinned Transformers runtime otherwise selects SDPA, whose
    # BF16 query path rejects OpenPI's FP32 additive mask.  Eager attention is
    # numerically the same prefix operation used by validation.
    policy.paligemma_with_expert.paligemma.language_model.config._attn_implementation = (
        "eager"
    )
    (prefix_output, _), _ = policy.paligemma_with_expert.forward(
        attention_mask=attention_4d,
        position_ids=position_ids,
        past_key_values=None,
        inputs_embeds=[prefix, None],
        use_cache=False,
    )
    return prefix_output.float(), prefix_pad


class PIEgoVerseMultiHeadV2(PI):
    """pi0.5 shared pose flow plus a deterministic PIPER joint head."""

    def __init__(
        self,
        *args,
        observation_history: int = 1,
        auxiliary_joint_key: str = "actions_joints",
        piper_embodiment: str = "piper_left_arm",
        shared_pose_loss_weight: float = 1.0,
        piper_joint_loss_weight: float = 1.0,
        joint_head: Mapping | None = None,
        **kwargs,
    ) -> None:
        self.observation_history = int(observation_history)
        if self.observation_history != 1:
            raise ValueError("MultiHead V2 is a single-frame model; observation_history must be 1")
        self.auxiliary_joint_key = str(auxiliary_joint_key)
        self.piper_embodiment_id = get_embodiment_id(piper_embodiment)
        self.shared_pose_loss_weight = float(shared_pose_loss_weight)
        self.piper_joint_loss_weight = float(piper_joint_loss_weight)
        super().__init__(*args, **kwargs)

        prefix_dim = int(
            self.model.paligemma_with_expert.paligemma.config.text_config.hidden_size
        )
        self.model.egoverse_observation_history = self.observation_history
        self.model._preprocess_observation = types.MethodType(
            _single_frame_preprocess_observation, self.model
        )
        self.model.embed_prefix = types.MethodType(_single_frame_embed_prefix, self.model)

        joint_cfg = dict(joint_head or {})
        joint_cfg.setdefault("prefix_dim", prefix_dim)
        joint_cfg.setdefault("action_horizon", self.config.model.action_horizon)
        self.nets["joint_head"] = Pi05PrefixJointHeadV2(**joint_cfg)

    def _state_for_sample(
        self, batch, sample_idx: int, embodiment_name: str
    ):
        """Tokenize the configured proprio observation (one frame in this version)."""
        parts = []
        keys = self.proprio_keys_for_prompt
        if isinstance(keys, Mapping):
            keys = next(
                (value for name, value in keys.items() if name.lower() == embodiment_name.lower()),
                None,
            )
            if keys is None:
                raise ValueError(f"no proprio prompt keys for {embodiment_name!r}")
        for key in keys:
            if key not in batch:
                continue
            value = batch[key][sample_idx]
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            parts.append(value.reshape(-1))
        if not parts:
            return None
        import numpy as np

        return np.clip(np.concatenate(parts).astype(np.float32), -1.0, 1.0)

    def _robomimic_to_pi_data(
        self, batch, cam_keys, proprio_keys, lang_keys, ac_key, embodiment
    ):
        del lang_keys
        if ac_key not in batch:
            raise KeyError(f"missing shared pose action {ac_key!r}")
        action = batch[ac_key].to(self.device)
        embodiment_id = get_embodiment_id(embodiment)
        converter = self.action_registry.get(embodiment_id, ac_key)
        action32 = converter.to32(action)
        action_weights = converter.loss_weights32(action)

        required_slots = list(self.pi_cam_keys)
        slot_map = self._camera_slot_map_for(embodiment, required_slots)
        sources = {
            slot: self._resolve_image_batch_key(batch, source)
            for slot, source in slot_map.items()
        }
        present = [source for source in sources.values() if source]
        if not present:
            raise ValueError(f"no camera observation found for {embodiment}")
        seed = ensure_btchw(batch[present[0]].to(self.device))
        if seed.shape[1] != self.observation_history:
            raise ValueError(
                f"camera time-axis mismatch: expected {self.observation_history}, got {seed.shape[1]}"
            )
        images = {}
        flags = {}
        resolution = tuple(getattr(self, "image_resolution", (224, 224)))
        for slot in required_slots:
            source = sources[slot]
            flags[slot] = source is not None
            image = ensure_btchw(batch[source].to(self.device)) if source else seed.clone()
            if image.shape[:2] != seed.shape[:2]:
                raise ValueError("all camera observations must share B and T")
            image = _to_minus1_1(image)
            batch_size, history = image.shape[:2]
            flat = image.reshape(batch_size * history, *image.shape[2:])
            if flat.shape[2:] != resolution:
                flat = resize_with_pad_torch(flat, *resolution)
            images[slot] = flat.reshape(batch_size, history, *flat.shape[1:])

        state_parts = []
        for key in proprio_keys:
            if key in batch:
                value = batch[key].to(self.device)
                if value.ndim == 2:
                    value = value[:, None]
                if value.ndim != 3 or value.shape[1] != self.observation_history:
                    raise ValueError(f"proprio time-axis contract failed for {key}: {value.shape}")
                state_parts.append(value)
        if state_parts:
            state_history = torch.cat(state_parts, dim=-1)
            state = state_history[:, -1]
        else:
            state = torch.zeros(seed.shape[0], 0, device=self.device)
        batch_size = seed.shape[0]
        image_masks = {
            slot: (
                torch.ones(batch_size, dtype=torch.bool, device=self.device)
                if flags[slot]
                else torch.zeros(batch_size, dtype=torch.bool, device=self.device)
            )
            for slot in required_slots
        }
        if "tokenized_prompt" in batch and batch["tokenized_prompt"].numel() > 0:
            prompt = batch["tokenized_prompt"].to(self.device)
            prompt_mask = batch["tokenized_mask"].to(self.device)
            ar_mask = batch["token_ar_mask"].to(self.device)
            loss_mask = batch["token_loss_mask"].to(self.device)
        else:
            prompt, prompt_mask, ar_mask, loss_mask = _empty_lang_placeholders(
                batch_size, self.device
            )
        observation = _SimpleObservation(
            images=images,
            image_masks=image_masks,
            state=state,
            tokenized_prompt=prompt,
            tokenized_prompt_mask=prompt_mask,
            token_ar_mask=ar_mask,
            token_loss_mask=loss_mask,
        )
        return observation, action32, action_weights

    def _unnormalize(self, value: torch.Tensor, key: str, embodiment_id: int):
        return self.norm_stats.unnormalize({key: value}, embodiment_id)[key]

    def _piper_joint_inputs(self, batch, embodiment_id: int):
        if self.auxiliary_joint_key not in batch:
            raise KeyError(
                f"PIPER sample lacks {self.auxiliary_joint_key!r}; available={sorted(batch)}"
            )
        state_key = next(
            (key for key in self.proprio_keys[embodiment_id] if key.endswith(".joints")),
            None,
        )
        if state_key is None or state_key not in batch:
            raise KeyError("PIPER MultiHead V2 requires observations.state.joints")
        normalized_state = batch[state_key]
        if normalized_state.ndim == 2:
            normalized_state = normalized_state[:, None]
        state_raw = self._unnormalize(normalized_state, state_key, embodiment_id)
        joint_raw = self._unnormalize(
            batch[self.auxiliary_joint_key], self.auxiliary_joint_key, embodiment_id
        )
        return normalized_state, state_raw[:, -1], joint_raw

    def forward_training(self, batch):
        predictions = OrderedDict()
        for embodiment_id, model_batch in batch.items():
            ac_key = self.ac_keys[embodiment_id]
            embodiment_name = get_embodiment(embodiment_id).lower()
            observation, action, action_weights = self._robomimic_to_pi_data(
                model_batch,
                self.camera_keys.get(embodiment_id, self.pi_cam_keys),
                self.proprio_keys[embodiment_id],
                self.lang_keys[embodiment_id],
                ac_key,
                embodiment_name,
            )
            elementwise, prefix, prefix_mask = _flow_forward_with_prefix_features(
                self.nets["policy"], observation, action
            )
            predictions[f"{embodiment_name}_pose_loss"] = self._masked_flow_loss(
                elementwise, action_weights
            )
            if embodiment_id == self.piper_embodiment_id:
                state_norm, current_raw, target_joint_raw = self._piper_joint_inputs(
                    model_batch, embodiment_id
                )
                target_pose_raw = self._unnormalize(
                    model_batch[ac_key], ac_key, embodiment_id
                )
                predictions["piper_joint_loss"] = self.nets["joint_head"].compute_loss(
                    prefix_features=prefix,
                    prefix_pad_mask=prefix_mask,
                    normalized_state_history=state_norm,
                    current_raw=current_raw,
                    target_joint_raw=target_joint_raw,
                    target_pose_camera_raw=target_pose_raw,
                )
        return predictions

    def forward_eval(self, batch):
        output = {}
        with torch.no_grad():
            for embodiment_id, model_batch in batch.items():
                ac_key = self.ac_keys[embodiment_id]
                embodiment_name = get_embodiment(embodiment_id).lower()
                observation, action, action_weights = self._robomimic_to_pi_data(
                    model_batch,
                    self.camera_keys.get(embodiment_id, self.pi_cam_keys),
                    self.proprio_keys[embodiment_id],
                    self.lang_keys[embodiment_id],
                    ac_key,
                    embodiment_name,
                )
                elementwise = self.nets["policy"].forward(observation, action)
                output[f"{embodiment_name}_loss"] = self._masked_flow_loss(
                    elementwise, action_weights
                )
                sampled = self.nets["policy"].sample_actions(
                    device=self.device,
                    observation=observation,
                    num_steps=self.num_steps,
                )
                converter = self.action_registry.get(embodiment_id, ac_key)
                pose = converter.from32(sampled)[
                    :, : model_batch[ac_key].shape[1], : model_batch[ac_key].shape[2]
                ]
                pose = self.norm_stats.unnormalize({ac_key: pose}, embodiment_id)[ac_key]
                output[f"{embodiment_name}_{ac_key}"] = pose
                if embodiment_id == self.piper_embodiment_id:
                    prefix, prefix_mask = _extract_prefix_features(
                        self.nets["policy"], observation
                    )
                    state_norm, current_raw, _ = self._piper_joint_inputs(
                        model_batch, embodiment_id
                    )
                    output[
                        f"{embodiment_name}_{self.auxiliary_joint_key}"
                    ] = self.nets["joint_head"].predict(
                        prefix_features=prefix,
                        prefix_pad_mask=prefix_mask,
                        normalized_state_history=state_norm,
                        current_raw=current_raw,
                    )
        return output

    @torch.no_grad()
    def forward_joint_deploy(
        self,
        *,
        front_image: torch.Tensor,
        wrist_image: torch.Tensor,
        current_raw: torch.Tensor,
        prompt: str | None = None,
    ) -> dict[str, torch.Tensor | list[str]]:
        """Run the exact training wrapper's PIPER joint-only inference path.

        Images are RGB tensors in ``[0,1]`` with shape BCHW/BHWC.  ``current_raw``
        contains six physical joint radians plus gripper-open fraction.  The
        shared 9-D flow head is deliberately not sampled here.
        """
        embodiment_id = self.piper_embodiment_id
        state_key = next(
            (key for key in self.proprio_keys[embodiment_id] if key.endswith(".joints")),
            None,
        )
        if state_key is None:
            raise RuntimeError("checkpoint has no PIPER observations.state.joints key")
        current_raw = current_raw.to(self.device, dtype=torch.float32)
        if current_raw.ndim != 2 or current_raw.shape[1] != 7:
            raise ValueError(f"current_raw must be [B,7], got {tuple(current_raw.shape)}")
        if not torch.isfinite(current_raw).all():
            raise ValueError("current_raw contains NaN or Inf")
        batch_size = current_raw.shape[0]
        if front_image.shape[0] != batch_size or wrist_image.shape[0] != batch_size:
            raise ValueError("image and proprio batch sizes do not match")

        stats = self.norm_stats.norm_stats.get(embodiment_id, {}).get(state_key)
        if stats is None:
            raise RuntimeError(f"checkpoint has no normalization stats for {state_key!r}")
        normalized_state = self.norm_stats._apply_norm_one(current_raw, stats)
        normalized_state_history = normalized_state[:, None]

        embodiment_name = get_embodiment(embodiment_id).lower()
        prompts = (
            [str(prompt)] * batch_size
            if prompt is not None
            else self._build_prompts({}, embodiment_name, batch_size)
        )
        if prompt is not None:
            # A caller override still receives the exact training suffix blocks.
            raw_batch = {self.annotation_key: [[str(prompt)]] * batch_size} if self.annotation_key else {}
            if self.annotation_key:
                prompts = self._build_prompts(raw_batch, embodiment_name, batch_size)
            else:
                any_block = self.proprio_in_prompt or self.embodiment_label or bool(self.control_mode)
                if any_block:
                    emb_label = embodiment_name.replace("_", " ")
                    prompts = []
                    for _ in range(batch_size):
                        blocks = [str(prompt)]
                        if self.embodiment_label:
                            blocks.append(f"Embodiment: {emb_label}")
                        if self.control_mode:
                            blocks.append(f"Control mode: {self._control_mode_for(emb_label)}")
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
            state_key: normalized_state_history,
            **tokenized,
        }
        # Observation construction is shared verbatim with training.  The
        # converter requires an action-shaped tensor, but joint-only inference
        # never reads or samples the shared pose output.
        ac_key = self.ac_keys[embodiment_id]
        model_batch[ac_key] = torch.zeros(
            batch_size,
            int(self.config.model.action_horizon),
            9,
            dtype=torch.float32,
            device=self.device,
        )
        observation, _, _ = self._robomimic_to_pi_data(
            model_batch,
            self.camera_keys.get(embodiment_id, self.pi_cam_keys),
            self.proprio_keys[embodiment_id],
            self.lang_keys[embodiment_id],
            ac_key,
            embodiment_name,
        )
        prefix_features, prefix_pad_mask = _extract_prefix_features(
            self.nets["policy"], observation
        )
        prediction, gripper_logits = self.nets["joint_head"].predict_with_details(
            prefix_features=prefix_features,
            prefix_pad_mask=prefix_pad_mask,
            normalized_state_history=normalized_state_history,
            current_raw=current_raw,
        )
        return {
            "actions": prediction,
            "gripper_logits": gripper_logits,
            "normalized_state": normalized_state,
            "prefix_features": prefix_features,
            "prefix_pad_mask": prefix_pad_mask,
            "prompts": prompts,
        }

    def compute_losses(self, predictions, batch):
        del batch
        pose_losses = [
            value for key, value in predictions.items() if key.endswith("_pose_loss")
        ]
        if not pose_losses:
            raise ValueError("MultiHead V2 received no shared pose losses")
        pose_loss = torch.stack(pose_losses).mean()
        total = self.shared_pose_loss_weight * pose_loss
        losses = OrderedDict(shared_pose_loss=pose_loss)
        if "piper_joint_loss" in predictions:
            joint_loss = predictions["piper_joint_loss"]
            total = total + self.piper_joint_loss_weight * joint_loss
            losses["piper_joint_loss"] = joint_loss
            for name, value in self.nets["joint_head"].last_loss_components.items():
                losses[f"piper_{name}"] = value
        losses["action_loss"] = total
        return losses
