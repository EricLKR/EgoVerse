import copy

import numpy as np
import torch
from torchmetrics import MeanSquaredError

from egomimic.eval.eval_video import EvalVideo
from egomimic.rldb.embodiment.embodiment import Embodiment, get_embodiment
from egomimic.utils.metrics import frechet_gaussian_over_time, reverse_kl_from_samples
from egomimic.utils.pose_utils import _xyzwxyz_to_matrix


def _joint_space_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict:
    """Joint, binary-gripper, and FK-derived cmd_ee_pose validation metrics."""
    if prediction.shape[-1] != 7 or target.shape[-1] != 7:
        raise ValueError(
            f"Joint-space evaluation expects (..., 7), got {prediction.shape} and {target.shape}"
        )
    pred = prediction.detach().cpu().float()
    gt = target.detach().cpu().float()
    joint_error = pred[..., :6] - gt[..., :6]
    metrics = {
        "joint_mae_deg": torch.rad2deg(joint_error.abs()).mean(),
        "joint_rmse_deg": torch.rad2deg(joint_error.square().mean().sqrt()),
    }
    for index in range(6):
        metrics[f"joint_{index + 1}_mae_deg"] = torch.rad2deg(
            joint_error[..., index].abs()
        ).mean()

    pred_open = pred[..., 6] >= 0.5
    gt_open = gt[..., 6] >= 0.5
    tp = (pred_open & gt_open).sum().float()
    fp = (pred_open & ~gt_open).sum().float()
    fn = (~pred_open & gt_open).sum().float()
    precision = tp / (tp + fp).clamp_min(1.0)
    recall = tp / (tp + fn).clamp_min(1.0)
    metrics.update(
        {
            "gripper_accuracy": (pred_open == gt_open).float().mean(),
            "gripper_open_precision": precision,
            "gripper_open_recall": recall,
            "gripper_open_f1": 2.0 * precision * recall / (precision + recall).clamp_min(1e-8),
        }
    )

    # Convert both commanded joint trajectories through exactly the same DH FK
    # used to create the stored cmd_ee_pose signal.  This exposes whether a
    # small joint-space error amplifies into a meaningful task-space error.
    from egomimic.robot.piper.piper_kinematics import PiperKinematicsSolver

    solver = PiperKinematicsSolver(device="cpu")
    pred_pose = solver.fk(pred[..., :6].reshape(-1, 6).numpy())
    gt_pose = solver.fk(gt[..., :6].reshape(-1, 6).numpy())
    position_error = np.linalg.norm(pred_pose[:, :3] - gt_pose[:, :3], axis=-1)
    pred_rot = _xyzwxyz_to_matrix(pred_pose)[:, :3, :3]
    gt_rot = _xyzwxyz_to_matrix(gt_pose)[:, :3, :3]
    relative = np.einsum("nij,njk->nik", np.transpose(pred_rot, (0, 2, 1)), gt_rot)
    cosine = np.clip((np.trace(relative, axis1=1, axis2=2) - 1.0) * 0.5, -1.0, 1.0)
    rotation_error_deg = np.rad2deg(np.arccos(cosine))
    metrics.update(
        {
            "cmd_ee_pose_position_mae_m": float(position_error.mean()),
            "cmd_ee_pose_position_rmse_m": float(np.sqrt(np.mean(position_error**2))),
            "cmd_ee_pose_rotation_mae_deg": float(rotation_error_deg.mean()),
            "cmd_ee_pose_rotation_rmse_deg": float(
                np.sqrt(np.mean(rotation_error_deg**2))
            ),
        }
    )
    metrics["physical_selection_score"] = (
        metrics["joint_mae_deg"]
        + 50.0 * metrics["cmd_ee_pose_position_mae_m"]
        + 0.25 * metrics["cmd_ee_pose_rotation_mae_deg"]
        + 10.0 * (1.0 - metrics["gripper_accuracy"])
    )
    return metrics


class HPTEvalVideo(EvalVideo):
    """
    Eval class for HPT models. Per embodiment, computes:
      - val loss (BC loss, same as training; also aggregated as ``Valid/action_loss``)
      - paired/final MSE + Frechet over time for the main / shared / auxiliary heads
      - paired/final MSE in cam frame on the main ``ac_key``, when a
        ``transform_lists`` entry is configured
      - optional Reverse KL from samples
    The revert transform is applied once and reused by both the cam-frame MSE
    and the viz video.
    """

    def compute_metrics_and_viz(self, batch):
        algo = self.model
        preds = algo.forward_eval(batch)

        metrics = {}
        images_dict = {}
        mse = MeanSquaredError()
        total_loss = None
        n_loss_embodiments = 0
        for embodiment_id, _batch in batch.items():
            _batch = algo.norm_stats.unnormalize(_batch, embodiment_id)
            embodiment_name = get_embodiment(embodiment_id).lower()
            ac_key = algo.ac_keys[embodiment_id]

            loss_key = f"{embodiment_name}_loss"
            if loss_key in preds:
                loss_val = preds[loss_key]
                metrics[f"Valid/{loss_key}"] = loss_val
                if total_loss is None:
                    total_loss = torch.zeros_like(loss_val)
                total_loss = total_loss + loss_val
                n_loss_embodiments += 1

            for component_name in (
                "joint_flow_loss",
                "joint_position_loss",
                "gripper_bce",
                "joint_delta_loss",
                "joint_acceleration_loss",
                "joint_total_loss",
            ):
                component_key = f"{embodiment_name}_{component_name}"
                if component_key in preds:
                    metrics[f"Valid/{component_key}"] = preds[component_key]

            if f"{embodiment_name}_{ac_key}" in preds and ac_key != algo.shared_ac_key:
                metrics[f"Valid/{embodiment_name}_{ac_key}_paired_mse_avg"] = mse(
                    preds[f"{embodiment_name}_{ac_key}"].cpu(), _batch[ac_key].cpu()
                )
                metrics[f"Valid/{embodiment_name}_{ac_key}_final_mse_avg"] = mse(
                    preds[f"{embodiment_name}_{ac_key}"][:, -1].cpu(),
                    _batch[ac_key][:, -1].cpu(),
                )
                fd = frechet_gaussian_over_time(
                    preds[f"{embodiment_name}_{ac_key}"], _batch[ac_key]
                )
                metrics[f"Valid/{embodiment_name}_{ac_key}_frechet_gauss_avg"] = (
                    fd.mean().item()
                )
                metrics[f"Valid/{embodiment_name}_{ac_key}_frechet_gauss_min"] = (
                    fd.min().item()
                )
                metrics[f"Valid/{embodiment_name}_{ac_key}_frechet_gauss_max"] = (
                    fd.max().item()
                )

            if embodiment_name in algo.auxiliary_ac_keys:
                for aux_key in algo.auxiliary_ac_keys[embodiment_name]:
                    pred_key = f"{embodiment_name}_{aux_key}"
                    if pred_key in preds:
                        metrics[f"Valid/{pred_key}_paired_mse_avg"] = mse(
                            preds[pred_key].cpu(), _batch[aux_key].cpu()
                        )
                        metrics[f"Valid/{pred_key}_final_mse_avg"] = mse(
                            preds[pred_key][:, -1].cpu(), _batch[aux_key][:, -1].cpu()
                        )
                        fd = frechet_gaussian_over_time(
                            preds[pred_key], _batch[aux_key]
                        )
                        metrics[f"Valid/{pred_key}_frechet_gauss_avg"] = (
                            fd.mean().item()
                        )
                        metrics[f"Valid/{pred_key}_frechet_gauss_min"] = fd.min().item()
                        metrics[f"Valid/{pred_key}_frechet_gauss_max"] = fd.max().item()
                        if aux_key == "actions_joints":
                            for metric_name, metric_value in _joint_space_metrics(
                                preds[pred_key], _batch[aux_key]
                            ).items():
                                metrics[f"Valid/{pred_key}_{metric_name}"] = metric_value

            if (
                algo.shared_ac_key
                and f"{embodiment_name}_{algo.shared_ac_key}" in preds
            ):
                pred_key = f"{embodiment_name}_{algo.shared_ac_key}"
                metrics[f"Valid/{pred_key}_paired_mse_avg"] = mse(
                    preds[pred_key].cpu(), _batch[algo.shared_ac_key].cpu()
                )
                metrics[f"Valid/{pred_key}_final_mse_avg"] = mse(
                    preds[pred_key][:, -1].cpu(),
                    _batch[algo.shared_ac_key][:, -1].cpu(),
                )
                fd = frechet_gaussian_over_time(
                    preds[pred_key], _batch[algo.shared_ac_key]
                )
                metrics[f"Valid/{pred_key}_frechet_gauss_avg"] = fd.mean().item()
                metrics[f"Valid/{pred_key}_frechet_gauss_min"] = fd.min().item()
                metrics[f"Valid/{pred_key}_frechet_gauss_max"] = fd.max().item()

                if algo.shared_ac_key == "actions_joints":
                    for metric_name, metric_value in _joint_space_metrics(
                        preds[pred_key], _batch[algo.shared_ac_key]
                    ).items():
                        metrics[f"Valid/{pred_key}_{metric_name}"] = metric_value

            if algo.rkl_samples and algo.rkl_samples > 1:
                hpt_batch = {
                    "domain": embodiment_name,
                    "data": algo._robomimic_to_hpt_data(
                        batch[embodiment_id],
                        algo.camera_keys[embodiment_id],
                        algo.proprio_keys[embodiment_id],
                        algo.lang_keys[embodiment_id],
                        ac_key,
                        algo.auxiliary_ac_keys.get(embodiment_name, []),
                    ),
                }
                rkl_targets = []

                if (
                    f"{embodiment_name}_{ac_key}" in preds
                    and ac_key != algo.shared_ac_key
                ):
                    rkl_targets.append(
                        (
                            f"{embodiment_name}_{ac_key}",
                            _batch[ac_key].to(algo.device),
                            embodiment_name,
                        )
                    )

                if embodiment_name in algo.auxiliary_ac_keys:
                    for aux_key in algo.auxiliary_ac_keys[embodiment_name]:
                        aux_pred_key = f"{embodiment_name}_{aux_key}"
                        if aux_pred_key in preds:
                            rkl_targets.append(
                                (
                                    aux_pred_key,
                                    _batch[aux_key].to(algo.device),
                                    aux_key,
                                )
                            )

                if algo.shared_ac_key:
                    shared_pred_key = f"{embodiment_name}_{algo.shared_ac_key}"
                    if shared_pred_key in preds:
                        rkl_targets.append(
                            (
                                shared_pred_key,
                                _batch[algo.shared_ac_key].to(algo.device),
                                "shared",
                            )
                        )

                M = int(algo.rkl_samples)
                for pred_key_name, gt_tensor, head_key in rkl_targets:
                    samples = self._collect_policy_samples(
                        hpt_batch, ref=gt_tensor, key_name=head_key, M=M
                    )
                    rkl = reverse_kl_from_samples(samples, gt_tensor)
                    metrics[f"Valid/{pred_key_name}_reverse_kl_M{M}"] = rkl.item()

            transform_list = self.transform_lists.get(embodiment_name)
            main_pred_key = f"{embodiment_name}_{ac_key}"
            gt_batch_viz = _batch
            preds_for_viz = preds
            if transform_list is not None and main_pred_key in preds:
                pred_batch = copy.deepcopy(_batch)
                pred_batch[ac_key] = preds[main_pred_key]
                gt_t = Embodiment.apply_transform(_batch, transform_list)
                pred_t = Embodiment.apply_transform(pred_batch, transform_list)
                # apply_transform drops keys whose shape[0] != batch_size
                # (e.g. ``embodiment``, ``annotations``). Merge to preserve them.
                gt_batch_viz = {**_batch, **gt_t}
                pred_batch_viz = {**_batch, **pred_t}

                # ``.contiguous()`` because ``apply_transform`` returns CPU tensors,
                # so ``.cpu()`` here is a no-op and ``[:, -1]`` leaves a non-contiguous
                # view that torchmetrics' MSE doesn't accept.
                metrics[f"Valid/{main_pred_key}_cam_paired_mse_avg"] = mse(
                    pred_batch_viz[ac_key].cpu().contiguous(),
                    gt_batch_viz[ac_key].cpu().contiguous(),
                )
                metrics[f"Valid/{main_pred_key}_cam_final_mse_avg"] = mse(
                    pred_batch_viz[ac_key][:, -1].cpu().contiguous(),
                    gt_batch_viz[ac_key][:, -1].cpu().contiguous(),
                )

                preds_for_viz = dict(preds)
                preds_for_viz[main_pred_key] = pred_batch_viz[ac_key]

            ims = self._visualize_preds(preds_for_viz, gt_batch_viz)
            images_dict[embodiment_id] = ims

        if total_loss is not None and n_loss_embodiments > 0:
            metrics["Valid/action_loss"] = total_loss / n_loss_embodiments

        return metrics, images_dict

    def _visualize_preds(self, predictions, batch):
        if self.viz_func is None:
            raise ValueError("viz_func is not set")
        embodiment_id = batch["embodiment"][0].item()
        embodiment_name = get_embodiment(embodiment_id).lower()
        return self.viz_func[embodiment_name](predictions, batch)

    @torch.no_grad()
    def _collect_policy_samples(self, hpt_batch, ref, key_name, M):
        """Collect policy samples for Reverse KL."""
        algo = self.model
        B, T, D = ref.shape
        samples = []
        was_training = algo.nets.training
        algo.nets.eval()
        for _ in range(M):
            out = algo.nets["policy"].forward(
                hpt_batch["domain"], algo._clone_batch(hpt_batch["data"])
            )
            if key_name in out:
                pred = out[key_name]
            else:
                pred = out[hpt_batch["domain"]]

            pred = pred[:, :T, :D]
            samples.append(pred.unsqueeze(0))
        if was_training:
            algo.nets.train()
        return torch.cat(samples, dim=0)
