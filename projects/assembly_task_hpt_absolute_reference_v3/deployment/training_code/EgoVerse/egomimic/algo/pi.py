import logging
import os
import random
from collections import OrderedDict
from collections.abc import Mapping
from typing import Literal

import numpy as np
import openpi
import openpi.models.tokenizer
import openpi.models_pytorch.pi0_pytorch
import safetensors
import torch
import torch.nn as nn
from openpi.shared.image_tools import resize_with_pad_torch
from overrides import override

from egomimic.algo.algo import Algo
from egomimic.models.preprocess_pi_obs import (
    _concat_proprio,
    _empty_lang_placeholders,
    _ensure_bchw,
    _SimpleObservation,
    _to_minus1_1,
)
from egomimic.rldb.embodiment.embodiment import get_embodiment, get_embodiment_id
from egomimic.utils.action_utils import ConverterRegistry

logger = logging.getLogger(__name__)
# Ensure logger propagates to root logger and has appropriate level
# Child loggers inherit from parent, but we explicitly set level to ensure INFO messages appear
logger.setLevel(logging.INFO)
logger.propagate = True  # Explicitly enable propagation (default, but ensures it works)


class PI(Algo):
    """ """

    def __init__(
        self,
        norm_stats,
        camera_transforms,
        domains,
        # ---------------------------
        # Image augmentations
        # ---------------------------
        train_image_augs,
        eval_image_augs,
        # ---------------------------
        # Model params
        # ---------------------------
        config,
        # ---------------------------
        ac_keys,
        action_converters,
        # ---------------------------
        # Prompt / tokenization (moved from data config). The PI algo owns
        # the tokenizer because the prompt template is model-specific
        # (paligemma + pi0.5 anchor). See ``process_batch_for_training`` for
        # how these knobs assemble the prompt and produce ``tokenized_*``.
        # ---------------------------
        tokenizer_model_name: str = "google/paligemma-3b-mix-224",
        tokenizer_max_length: int = 128,
        sampling_mode: Literal["first", "random"] = "random",
        annotation_key: str | None = None,
        default_prompt: str = "",
        proprio_in_prompt: bool = False,
        embodiment_label: bool = False,
        state_num_bins: int = 256,
        control_mode: dict[str, str] | None = None,
        proprio_keys_for_prompt: list[str] | dict[str, list[str]] | None = None,
        camera_slot_maps: dict[str, dict[str, str | None]] | None = None,
        gradient_checkpointing: bool = False,
        finetune_mode: Literal["full", "action_expert"] = "full",
        train_multimodal_projector: bool = True,
        **kwargs,
    ):
        self.nets = nn.ModuleDict()
        self.norm_stats = norm_stats

        # ---- Prompt assembly + tokenization (was in collate_fn) ----
        # Use OpenPI's own public SentencePiece tokenizer.  This keeps token IDs
        # exactly aligned with the checkpoint and avoids a gated Hugging Face
        # PaliGemma download during cluster setup.
        self.tokenizer = openpi.models.tokenizer.PaligemmaTokenizer(
            max_len=tokenizer_max_length
        )
        self.tokenizer_model_name = tokenizer_model_name  # legacy config metadata
        self.tokenizer_max_length = tokenizer_max_length
        self.sampling_mode = sampling_mode
        self.annotation_key = annotation_key
        self.default_prompt = default_prompt
        self.proprio_in_prompt = proprio_in_prompt
        self.embodiment_label = embodiment_label
        self.state_num_bins = state_num_bins
        if self.state_num_bins != 256:
            raise ValueError(
                "OpenPI's PaligemmaTokenizer uses exactly 256 state bins; "
                f"got state_num_bins={self.state_num_bins}"
            )
        self.control_mode = control_mode
        # Default to the canonical concat key produced by each embodiment's
        # transform_list (ConcatKeys with delete_old_keys removes per-arm zarr keys).
        if proprio_keys_for_prompt is None:
            self.proprio_keys_for_prompt = ["observations.state.ee_pose"]
        elif isinstance(proprio_keys_for_prompt, Mapping):
            self.proprio_keys_for_prompt = {
                str(name): list(keys)
                for name, keys in proprio_keys_for_prompt.items()
            }
        else:
            self.proprio_keys_for_prompt = list(proprio_keys_for_prompt)

        self.camera_transforms = camera_transforms
        self.train_image_augs = train_image_augs
        self.eval_image_augs = eval_image_augs
        if "image_resolution" in kwargs:
            self.image_resolution = kwargs["image_resolution"]
        self.pi_cam_keys = kwargs.get(
            "pi_cam_keys", ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]
        )
        self.camera_slot_maps = camera_slot_maps or {}
        self.config = config

        self.ac_keys = ac_keys

        self.domains = domains

        self.device = None

        self.camera_keys = {}
        self.proprio_keys = {}
        self.lang_keys = {}

        for embodiment in self.domains:
            embodiment_id = get_embodiment_id(embodiment)
            self.camera_keys[embodiment_id] = []
            self.proprio_keys[embodiment_id] = []
            self.lang_keys[embodiment_id] = []
            for key in norm_stats.keys_of_type("action_keys", embodiment_id):
                if (
                    norm_stats.is_key_with_embodiment(key, embodiment_id)
                    and key == self.ac_keys[embodiment]
                ):
                    self.ac_keys[embodiment_id] = key
            for key in norm_stats.keys_of_type("camera_keys", embodiment_id):
                if norm_stats.is_key_with_embodiment(key, embodiment_id):
                    self.camera_keys[embodiment_id].append(key)
            for key in norm_stats.keys_of_type("proprio_keys", embodiment_id):
                if norm_stats.is_key_with_embodiment(key, embodiment_id):
                    self.proprio_keys[embodiment_id].append(key)
            for key in norm_stats.keys_of_type("lang_keys", embodiment_id):
                if norm_stats.is_key_with_embodiment(key, embodiment_id):
                    self.lang_keys[embodiment_id].append(key)

        self.num_steps = getattr(self.config, "num_sampling_steps", 10)
        self.is_6dof = kwargs.get("is_6dof", True)

        self.action_converters = action_converters

        self.action_registry = ConverterRegistry()

        arcfg = self.action_converters
        default_ac_key = getattr(arcfg, "ac_key", "actions_cartesian")

        for emb_name, conv_obj in arcfg.rules.items():
            emb_id = get_embodiment_id(emb_name)
            self.action_registry.register(emb_id, self.ac_keys[emb_id], conv_obj)

        fb_obj = arcfg.fallback
        self.action_registry.register("*", default_ac_key, fb_obj)
        self.action_registry.register("*", "*", fb_obj)

        # Create the model
        model_cfg = openpi.models_pytorch.pi0_pytorch.TorchPi0Config(
            dtype=self.config.pytorch_training_precision,
            action_dim=self.config.model.action_dim,
            action_horizon=self.config.model.action_horizon,
            max_token_len=self.config.model.max_token_len,
            paligemma_variant=getattr(
                self.config.model, "paligemma_variant", "gemma_2b"
            ),
            action_expert_variant=getattr(
                self.config.model, "action_expert_variant", "gemma_300m"
            ),
            pi05=getattr(config.model, "pi05", False),
            pytorch_compile_mode=getattr(
                config.model, "pytorch_compile_mode", None
            ),
        )

        self.model = openpi.models_pytorch.pi0_pytorch.PI0Pytorch(model_cfg)

        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        target = (
            self.model.module
            if isinstance(self.model, torch.nn.parallel.DistributedDataParallel)
            else self.model
        )
        if self.config.pytorch_weight_path is not None:
            model_path = os.path.join(
                self.config.pytorch_weight_path, "model.safetensors"
            )
            if not os.path.isfile(model_path):
                raise FileNotFoundError(
                    f"Pretrained weight file not found: {model_path}"
                )
            safetensors.torch.load_model(target, model_path)
            logger.info(
                "Loaded pretrained weights from %s (%d parameters)",
                model_path,
                sum(p.numel() for p in target.parameters()),
            )
        else:
            logger.warning("No pytorch_weight_path specified — training from scratch")

        self.finetune_mode = finetune_mode
        self.train_multimodal_projector = train_multimodal_projector
        self.finetune_parameter_stats = self._configure_finetune_parameters(
            target,
            finetune_mode=finetune_mode,
            train_multimodal_projector=train_multimodal_projector,
        )
        logger.info(
            "PI finetune mode=%s trainable=%d/%d parameters (%.2f%%); "
            "multimodal_projector_trainable=%s",
            finetune_mode,
            self.finetune_parameter_stats["trainable"],
            self.finetune_parameter_stats["total"],
            self.finetune_parameter_stats["trainable_percent"],
            train_multimodal_projector,
        )
        self.nets = nn.ModuleDict()
        self.nets["policy"] = self.model

    @staticmethod
    def _configure_finetune_parameters(
        model: nn.Module,
        *,
        finetune_mode: str,
        train_multimodal_projector: bool,
    ) -> dict[str, float | int]:
        """Apply the explicitly supported full or action-expert tuning policy.

        ``action_expert`` preserves the pretrained SigLIP and PaliGemma
        language backbones.  The Gemma action expert, action/time projections,
        and (by default) multimodal projector remain trainable.  This is a
        better fit for a small, single-task robot dataset than updating all
        3.6B parameters with a tiny physical batch.
        """
        if finetune_mode not in {"full", "action_expert"}:
            raise ValueError(
                f"Unsupported PI finetune_mode={finetune_mode!r}; expected "
                "'full' or 'action_expert'"
            )

        for parameter in model.parameters():
            parameter.requires_grad_(True)

        if finetune_mode == "action_expert":
            try:
                paligemma = model.paligemma_with_expert.paligemma
                vision_tower = paligemma.vision_tower
                language_model = paligemma.language_model
                multimodal_projector = paligemma.multi_modal_projector
                action_expert = model.paligemma_with_expert.gemma_expert
            except AttributeError as exc:
                raise RuntimeError(
                    "OpenPI PyTorch model structure changed; refusing to apply a "
                    "partial-freeze policy to unknown modules"
                ) from exc

            for module in (vision_tower, language_model):
                for parameter in module.parameters():
                    parameter.requires_grad_(False)
            if not train_multimodal_projector:
                for parameter in multimodal_projector.parameters():
                    parameter.requires_grad_(False)

            # Fail fast if an upstream refactor accidentally freezes the action
            # expert that this mode exists to optimize.
            if not any(parameter.requires_grad for parameter in action_expert.parameters()):
                raise RuntimeError("Action expert has no trainable parameters")

        total = sum(parameter.numel() for parameter in model.parameters())
        trainable = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        if trainable <= 0:
            raise RuntimeError("PI finetune policy selected zero trainable parameters")
        return {
            "total": total,
            "trainable": trainable,
            "frozen": total - trainable,
            "trainable_percent": 100.0 * trainable / total,
        }

    def _control_mode_for(self, emb_name: str | None) -> str:
        if self.control_mode and emb_name is not None:
            for key, val in self.control_mode.items():
                if key.lower() in emb_name:
                    return val
        raise ValueError(
            f"control_mode has no entry matching embodiment {emb_name!r} "
            f"(declared keys: {sorted(self.control_mode or {})}). Control modes "
            "are declared per-embodiment in the model yaml; there is no fallback."
        )

    def _state_for_sample(
        self, _batch, sample_idx: int, embodiment_name: str
    ) -> np.ndarray | None:
        """Return the latest normalized proprio vector for OpenPI tokenization."""
        parts = []
        proprio_keys = self.proprio_keys_for_prompt
        if isinstance(proprio_keys, Mapping):
            selected = None
            for name, keys in proprio_keys.items():
                if name.lower() == embodiment_name.lower():
                    selected = keys
                    break
            if selected is None:
                raise ValueError(
                    f"proprio_keys_for_prompt has no entry for {embodiment_name!r}"
                )
            proprio_keys = selected
        for k in proprio_keys:
            if k not in _batch:
                continue
            v = _batch[k]
            if isinstance(v, torch.Tensor):
                v = v[sample_idx].detach().cpu().numpy()
            else:
                v = np.asarray(v)[sample_idx]
            v = np.asarray(v, dtype=np.float32)
            while v.ndim > 1:
                v = v[-1]
            parts.append(v.reshape(-1))
        if not parts:
            return None
        state = np.concatenate(parts, axis=-1)
        return np.clip(state, -1.0, 1.0)

    def _build_prompts(
        self, _batch, embodiment_name: str, batch_size: int
    ) -> list[str]:
        """Sample one prompt per item from the raw annotation lists and
        splice in any of the active blocks. Returns ``batch_size`` strings.

        Prompt assembly lives here rather than in a collate_fn because the
        template is model-specific. Embodiment is known per-batch (one
        DataLoader per embodiment), so we don't re-derive it per sample.
        """
        if self.annotation_key is None or self.annotation_key not in _batch:
            prompts = [self.default_prompt] * batch_size
        else:
            prompts = []
            for sample in _batch[self.annotation_key]:
                if not sample:
                    prompts.append(self.default_prompt)
                elif self.sampling_mode == "random":
                    prompts.append(sample[random.randint(0, len(sample) - 1)])
                else:  # "first"
                    prompts.append(sample[0])

        any_block_active = (
            self.proprio_in_prompt or self.embodiment_label or bool(self.control_mode)
        )
        if not any_block_active:
            return prompts

        emb_name = embodiment_name.lower().replace("_", " ")
        spliced = []
        for i, prompt in enumerate(prompts):
            blocks = [prompt]
            if self.embodiment_label:
                blocks.append(f"Embodiment: {emb_name}")
            if self.control_mode:
                blocks.append(f"Control mode: {self._control_mode_for(emb_name)}")
            spliced.append(", ".join(blocks))
        return spliced

    def _tokenize_prompts(
        self, prompts: list[str], states: list[np.ndarray | None]
    ) -> dict:
        tokenized = [
            self.tokenizer.tokenize(prompt, state)
            for prompt, state in zip(prompts, states, strict=True)
        ]
        input_ids = torch.from_numpy(np.stack([item[0] for item in tokenized])).long()
        attention_mask = torch.from_numpy(
            np.stack([item[1] for item in tokenized])
        ).bool()
        token_loss_mask = attention_mask.clone()
        token_loss_mask[:, -1] = False
        return {
            "tokenized_prompt": input_ids.requires_grad_(False),
            "tokenized_mask": attention_mask.requires_grad_(False),
            "token_loss_mask": token_loss_mask.requires_grad_(False),
            "token_ar_mask": attention_mask.clone().requires_grad_(False),
        }

    @override
    def process_batch_for_training(self, batch):
        """
        Processes input batch from a data loader to filter out
        relevant information and prepare the batch for training.
        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader
        Returns:
            batch (dict): processed dict of batchs that works with pi0.
        """
        processed_batch = {}

        for embodiment_name, _batch in batch.items():
            embodiment_id = get_embodiment_id(embodiment_name)
            processed_batch[embodiment_id] = {}
            for key, value in _batch.items():
                key_name = self.norm_stats.zarr_key_to_keyname(key, embodiment_id)
                if key_name is not None:
                    processed_batch[embodiment_id][key_name] = value

            # Carry forward episode_hash from the data dict (other keys are
            # built inline below via _tokenize_prompts).
            if "episode_hash" in _batch:
                processed_batch[embodiment_id]["episode_hash"] = _batch["episode_hash"]

            ac_key = self.ac_keys[embodiment_id]
            if ac_key not in processed_batch[embodiment_id]:
                raise KeyError(
                    f"Missing action key '{ac_key}' for embodiment {embodiment_id}. "
                    f"Incoming keys were: {list(_batch.keys())}"
                )
            if len(processed_batch[embodiment_id][ac_key].shape) != 3:
                raise ValueError("Action shape in batch is not 2")

            B, S, _ = processed_batch[embodiment_id][ac_key].shape

            # Build prompts + tokenize. Text annotations stay in the raw collated
            # batch, while pi0.5 state tokens must use the same canonical,
            # normalized proprio tensors that are passed to the policy.
            prompts = self._build_prompts(_batch, embodiment_name, B)
            model_batch = processed_batch[embodiment_id]
            states = (
                [
                    self._state_for_sample(model_batch, i, embodiment_name)
                    for i in range(B)
                ]
                if self.proprio_in_prompt
                else [None] * B
            )
            if self.proprio_in_prompt and any(state is None for state in states):
                raise KeyError(
                    f"No configured proprio key was found for {embodiment_name!r}; "
                    "pi0.5 requires normalized state tokens. "
                    f"Available canonical keys: {sorted(model_batch)}"
                )
            processed_batch[embodiment_id]["sampled_prompt"] = prompts
            processed_batch[embodiment_id].update(
                self._tokenize_prompts(prompts, states)
            )
            processed_batch[embodiment_id]["pad_mask"] = torch.ones(
                B, S, 1, device=self.device
            )
            # Samples are already normalized by MultiDataset.__getitem__.
            processed_batch[embodiment_id]["embodiment"] = torch.tensor(
                [embodiment_id], device=self.device, dtype=torch.int64
            )

            for key, value in processed_batch[embodiment_id].items():
                if isinstance(value, torch.Tensor):
                    value = value.to(self.device)
                    if value.is_floating_point():
                        value = value.float()
                    processed_batch[embodiment_id][key] = value

        if not processed_batch:
            raise ValueError(
                f"No valid embodiments found in batch. Batch contained: {list(batch.keys())}, "
                f"but ac_keys only has: {list(self.ac_keys.keys())}"
            )

        return processed_batch

    @override
    def forward_training(self, batch):
        """
        One iteration of training. Sequentially, forward pass loss, Compute forward pass and compute losses.  Return predictions dictionary.  HPT also calculates loss here.
        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training (see docstring for expected keys/shapes)
        Returns:
            predictions (dict): {ac_key: torch.Tensor (B, Seq, D), loss_key_name: torch.Tensor (1)}
        """
        # self.nets["policy"].train()
        predictions = OrderedDict()
        for embodiment_id, _batch in batch.items():
            proprio_keys = self.proprio_keys[embodiment_id]
            lang_keys = self.lang_keys[embodiment_id]
            ac_key = self.ac_keys[embodiment_id]
            camera_keys = self.camera_keys.get(embodiment_id, self.pi_cam_keys)
            embodiment_name = get_embodiment(embodiment_id).lower()
            processed_obs, action, action_weights = self._robomimic_to_pi_data(
                _batch,
                camera_keys,
                proprio_keys,
                lang_keys,
                ac_key,
                embodiment_name,
            )

            losses = self.nets["policy"].forward(processed_obs, action)

            if isinstance(losses, list | tuple):
                losses = torch.stack(losses)
            elif not isinstance(losses, torch.Tensor):
                losses = torch.tensor(losses, device=action.device, dtype=torch.float32)

            loss = self._masked_flow_loss(losses, action_weights)

            predictions[f"{embodiment_name}_{ac_key}"] = _batch[ac_key]
            predictions[f"{embodiment_name}_loss"] = loss

        return predictions

    @override
    def forward_eval(self, batch):
        """
        Compute forward pass and return network outputs in @predictions dict.
        Unnormalize data here.
        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training (see docstring for expected keys/shapes)
        Returns:
            unnorm_preds (dict): {
                <embodiment_name>_<ac_key>: torch.Tensor (B, Seq, D),
                <embodiment_name>_loss: torch.Tensor (1),  # flow-matching val loss
            }
        """
        unnorm_preds = {}
        with torch.no_grad():
            for embodiment_id, _batch in batch.items():
                proprio_keys = self.proprio_keys[embodiment_id]
                lang_keys = self.lang_keys[embodiment_id]
                ac_key = self.ac_keys[embodiment_id]
                camera_keys = self.camera_keys.get(embodiment_id, self.pi_cam_keys)
                embodiment_name = get_embodiment(embodiment_id).lower()
                processed_obs, action, action_weights = self._robomimic_to_pi_data(
                    _batch,
                    camera_keys,
                    proprio_keys,
                    lang_keys,
                    ac_key,
                    embodiment_name,
                )

                # Flow-matching val loss — same call as forward_training.
                losses = self.nets["policy"].forward(processed_obs, action)
                if isinstance(losses, (list, tuple)):
                    losses = torch.stack(losses)
                elif not isinstance(losses, torch.Tensor):
                    losses = torch.tensor(
                        losses, device=action.device, dtype=torch.float32
                    )
                unnorm_preds[f"{embodiment_name}_loss"] = self._masked_flow_loss(
                    losses, action_weights
                )

                pred_actions = self.nets["policy"].sample_actions(
                    device=self.device,
                    observation=processed_obs,
                    noise=None,
                    num_steps=self.num_steps,
                )

                predictions = OrderedDict()
                ref = _batch[ac_key]
                B, T, D = ref.shape

                converter = self.action_registry.get(embodiment_id, ac_key)
                pred_actions_orig = converter.from32(pred_actions)

                pred = pred_actions_orig[:, :T, :D]
                predictions[ac_key] = pred

                unnorm_actions = self.norm_stats.unnormalize(predictions, embodiment_id)
                for key in unnorm_actions:
                    unnorm_preds[f"{embodiment_name}_{key}"] = unnorm_actions[key]

        return unnorm_preds

    @override
    def compute_losses(self, predictions, batch):
        """
        Compute losses based on network outputs in @predictions dict, using reference labels in @batch.
        Args:
            predictions (dict): dictionary containing network outputs, from @forward_training
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training (see docstring for expected keys/shapes)
        Returns:
            losses (dict): dictionary of losses computed over the batch
                loss_key_name: torch.Tensor (1)
        """
        loss_dict = OrderedDict()
        total_action_loss = None

        for embodiment_id, _batch in batch.items():
            embodiment_name = get_embodiment(embodiment_id).lower()
            bc_loss = predictions[f"{embodiment_name}_loss"]
            if total_action_loss is None:
                total_action_loss = torch.tensor(0.0, device=bc_loss.device)
            total_action_loss += bc_loss
            loss_dict[f"{embodiment_name}_loss"] = bc_loss  # for logging

        # in the case we put all embodiments in one batch, get rid of this norm.
        loss_dict["action_loss"] = total_action_loss / len(self.domains)

        return loss_dict

    @override
    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.
        Args:
            info (dict): dictionary of losses returned by compute_losses
                losses:
                    loss_key_name: torch.Tensor (1)
        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = OrderedDict()
        log["Loss"] = info["losses"]["action_loss"].item()
        for loss_key, loss in info["losses"].items():
            log[loss_key] = loss.item()
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        return log

    def _robomimic_to_pi_data(
        self, batch, cam_keys, proprio_keys, lang_keys, ac_key, embodiment
    ):
        """ """
        if ac_key not in batch:
            raise KeyError(f"Missing action key '{ac_key}' in batch")

        device = self.device
        action = batch[ac_key].to(device)
        image_resolution = getattr(self, "image_resolution", (224, 224))
        required_cam_keys = list(getattr(self, "pi_cam_keys", cam_keys))

        emb_id = get_embodiment_id(embodiment)  # embodiment is a name string
        converter = self.action_registry.get(emb_id, ac_key)
        action32 = converter.to32(action)
        action_weights = converter.loss_weights32(action)
        if action32.shape != action_weights.shape:
            raise ValueError(
                f"Action converter returned incompatible action/weight shapes: "
                f"{tuple(action32.shape)} vs {tuple(action_weights.shape)}"
            )

        # OpenPI expects three fixed, ordered image slots. Map the canonical
        # EgoMimic camera keys into those slots per embodiment, then duplicate
        # one real view for absent slots while masking the duplicates out.
        slot_map = self._camera_slot_map_for(embodiment, required_cam_keys)
        resolved_sources = {
            slot: self._resolve_image_batch_key(batch, source)
            for slot, source in slot_map.items()
        }
        present_sources = [source for source in resolved_sources.values() if source]
        if not present_sources:
            raise ValueError(
                f"No usable camera for embodiment {embodiment!r}; slot map={slot_map}, "
                f"batch keys={sorted(batch)}"
            )
        seed_img = _ensure_bchw(batch[present_sources[0]].to(device))

        # ---- Images (dict[str, Tensor]) ----
        images = {}
        present_flags = {}
        for slot in required_cam_keys:
            source = resolved_sources[slot]
            is_present = source is not None
            present_flags[slot] = is_present
            img = _ensure_bchw(batch[source].to(device)) if is_present else seed_img.clone()
            img = _to_minus1_1(img)
            if img.shape[2:] != tuple(image_resolution):
                img = resize_with_pad_torch(img, *image_resolution)
            if img.ndim != 4:
                raise ValueError(
                    f"Expected 4D BCHW image for slot '{slot}', got shape {tuple(img.shape)}"
                )
            images[slot] = img

        if not images:
            raise ValueError("No camera tensors found for the provided cam_keys.")

        # ---- Proprio -> state [B, D] ----
        state = _concat_proprio(batch, proprio_keys, device)
        if state.numel() == 0:
            B = next(iter(images.values())).shape[0]
            state = torch.zeros(B, 0, device=device)
        else:
            B = state.shape[0]

        # ---- Masks for duplicated images + empty language fields ----
        image_masks = {
            k: (
                torch.ones(B, dtype=torch.bool, device=device)
                if present_flags[k]
                else torch.zeros(B, dtype=torch.bool, device=device)
            )
            for k in images.keys()
        }

        has_lang = "tokenized_prompt" in batch and batch["tokenized_prompt"].numel() > 0
        if has_lang:
            tokenized_prompt = batch["tokenized_prompt"].to(device)
            tokenized_prompt_mask = batch["tokenized_mask"].to(device)
            token_ar_mask = batch["token_ar_mask"].to(device)
            token_loss_mask = batch["token_loss_mask"].to(device)
        else:
            tokenized_prompt, tokenized_prompt_mask, token_ar_mask, token_loss_mask = (
                _empty_lang_placeholders(B, device)
            )

        # ---- Wrap into simple observation (helpers) ----
        observation = _SimpleObservation(
            images=images,
            image_masks=image_masks,
            state=state,
            tokenized_prompt=tokenized_prompt,
            tokenized_prompt_mask=tokenized_prompt_mask,
            token_ar_mask=token_ar_mask,
            token_loss_mask=token_loss_mask,
        )

        # Do NOT call _preprocessing here; the PI model does it internally.
        return observation, action32, action_weights

    @staticmethod
    def _resolve_image_batch_key(
        batch: Mapping, source: str | None
    ) -> str | None:
        """Resolve camera aliases against raw and canonical dataloader keys.

        Older EgoMimic configs use short names such as ``front_img_1`` while
        the zarr dataloader exposes ``observations.images.front_img_1`` after
        preprocessing.  Accept both spellings, but only return a key holding a
        BCHW tensor so an unrelated metadata field cannot be selected.
        """
        if source is None:
            return None
        candidates = [str(source)]
        if not str(source).startswith("observations.images."):
            candidates.append(f"observations.images.{source}")
        for candidate in candidates:
            value = batch.get(candidate)
            if isinstance(value, torch.Tensor) and value.ndim == 4:
                return candidate
        return None

    def _camera_slot_map_for(
        self, embodiment: str, required_cam_keys: list[str]
    ) -> dict[str, str | None]:
        """Resolve one embodiment's canonical cameras into OpenPI slots."""
        selected = None
        embodiment_lower = embodiment.lower()
        for name, mapping in self.camera_slot_maps.items():
            if str(name).lower() == embodiment_lower:
                selected = mapping
                break
        if selected is None:
            # Backwards-compatible identity mapping for existing EVA/ARIA PI
            # configs whose dataset keys already use OpenPI slot names.
            return {slot: slot for slot in required_cam_keys}

        unknown = set(selected) - set(required_cam_keys)
        if unknown:
            raise ValueError(
                f"Unknown OpenPI camera slots for {embodiment!r}: {sorted(unknown)}; "
                f"expected only {required_cam_keys}"
            )
        return {slot: selected.get(slot) for slot in required_cam_keys}

    @staticmethod
    def _masked_flow_loss(losses: torch.Tensor, action_weights: torch.Tensor) -> torch.Tensor:
        """Reduce OpenPI's elementwise flow loss over real action channels only."""
        if losses.ndim == 0:
            raise ValueError(
                "OpenPI returned an already-reduced scalar loss; masked 32-D action "
                "training requires an elementwise [B,T,32] loss tensor"
            )
        if losses.shape != action_weights.shape:
            raise ValueError(
                f"OpenPI loss shape {tuple(losses.shape)} does not match action loss "
                f"weights {tuple(action_weights.shape)}"
            )
        weights = action_weights.to(device=losses.device, dtype=losses.dtype)
        denominator = weights.sum()
        if denominator.item() <= 0:
            raise ValueError("Action loss mask contains no active channels")
        return (losses * weights).sum() / denominator

    def _clone_batch(self, batch):
        """Recursively clones all tensors inside a nested dictionary."""
        if isinstance(batch, dict):
            return {key: self._clone_batch(val) for key, val in batch.items()}
        elif isinstance(batch, torch.Tensor):
            return batch.clone()
        else:
            return batch  # Return as is for non-tensor types

    def _extract_xyz(self, x):
        """
        Extract xyz (3D position) and rotation from 6DoF or 6DoF+gripper actions.

        Supports:
        - 6: 6DoF (single arm)
        - 7: 6DoF + gripper (single arm)
        - 12: 2 arms × 6DoF
        - 14: 2 arms × (6DoF + gripper)

        Returns:
            xyz: Tensor with only xyz per arm (shape: ..., 3) or (..., 6) for dual-arm.
            rot: Tensor with only rotation per arm (shape: ..., 3) or (..., 6) for dual-arm.
        """
        if x.shape[-1] == 6:
            return x[..., :3], x[..., 3:6]
        elif x.shape[-1] == 7:
            return x[..., :3], x[..., 3:6]
        elif x.shape[-1] == 12:
            xyz_right = x[..., :3]
            rot_right = x[..., 3:6]
            xyz_left = x[..., 6:9]
            rot_left = x[..., 9:12]
            return torch.cat([xyz_right, xyz_left], dim=-1), torch.cat(
                [rot_right, rot_left], dim=-1
            )
        elif x.shape[-1] == 14:
            xyz_right = x[..., :3]
            rot_right = x[..., 3:6]
            xyz_left = x[..., 7:10]
            rot_left = x[..., 10:13]
            return torch.cat([xyz_right, xyz_left], dim=-1), torch.cat(
                [rot_right, rot_left], dim=-1
            )
        else:
            raise ValueError(f"Unexpected shape for 6DoF input: {x.shape}")
