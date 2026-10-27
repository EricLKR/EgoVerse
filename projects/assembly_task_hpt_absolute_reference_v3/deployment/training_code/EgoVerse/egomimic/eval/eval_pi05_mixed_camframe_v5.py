"""Independent validation metrics for pi0.5 Mixed CamFrame V5."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from egomimic.eval.eval_video import EvalVideo
from egomimic.models.pi05_egoverse_multihead_v2 import rotation_6d_to_matrix
from egomimic.rldb.embodiment.embodiment import get_embodiment


def _rotation_error_deg(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    predicted_matrix = rotation_6d_to_matrix(predicted)
    target_matrix = rotation_6d_to_matrix(target)
    relative = predicted_matrix.transpose(-1, -2) @ target_matrix
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5).clamp(
        -1.0, 1.0
    )
    return torch.rad2deg(torch.acos(cosine))


class PI05MixedCamFrameV5Eval(EvalVideo):
    """Evaluate each domain without stateful metric accumulation."""

    def compute_metrics_and_viz(self, batch):
        algo = self.model
        predictions = algo.forward_eval(batch)
        metrics = {}
        images = {}
        flow_losses = []

        for embodiment_id, normalized_batch in batch.items():
            raw_batch = algo.norm_stats.unnormalize(normalized_batch, embodiment_id)
            name = get_embodiment(embodiment_id).lower()
            loss_key = f"{name}_loss"
            if loss_key in predictions:
                metrics[f"Valid/{loss_key}"] = predictions[loss_key]
                flow_losses.append(predictions[loss_key])

            pose_key = algo.ac_keys[embodiment_id]
            pose_prediction = predictions[f"{name}_{pose_key}"].cpu()
            pose_target = raw_batch[pose_key].cpu()
            if pose_prediction.shape != pose_target.shape:
                raise ValueError(
                    f"V5 pose prediction shape mismatch for {name}: "
                    f"{tuple(pose_prediction.shape)} != {tuple(pose_target.shape)}"
                )
            position_error = torch.linalg.vector_norm(
                pose_prediction[..., :3] - pose_target[..., :3], dim=-1
            )
            rotation_error = _rotation_error_deg(
                pose_prediction[..., 3:9], pose_target[..., 3:9]
            )
            metrics[f"Valid/{name}_pose_mse"] = F.mse_loss(
                pose_prediction, pose_target
            )
            metrics[f"Valid/{name}_position_mae_cm"] = position_error.mean() * 100.0
            metrics[f"Valid/{name}_rotation_mae_deg"] = rotation_error.mean()

            if embodiment_id == algo.piper_embodiment_id:
                joint_key = algo.auxiliary_joint_key
                predicted = predictions[f"{name}_{joint_key}"].cpu()
                target = raw_batch[joint_key].cpu()
                if predicted.shape[1:] != (30, 7) or target.shape[1:] != (30, 7):
                    raise ValueError("V5 requires native [B,30,7] PIPER actions")
                error_deg = torch.rad2deg(predicted[..., :6] - target[..., :6])
                velocity_error = torch.rad2deg(
                    torch.diff(predicted[..., :6], dim=1)
                    - torch.diff(target[..., :6], dim=1)
                ).abs().mean()
                acceleration_error = torch.rad2deg(
                    torch.diff(predicted[..., :6], n=2, dim=1)
                    - torch.diff(target[..., :6], n=2, dim=1)
                ).abs().mean()
                jerk_error = torch.rad2deg(
                    torch.diff(predicted[..., :6], n=3, dim=1)
                    - torch.diff(target[..., :6], n=3, dim=1)
                ).abs().mean()
                predicted_open = predicted[..., 6] >= 0.5
                target_open = target[..., 6] >= 0.5
                accuracy = (predicted_open == target_open).float().mean()
                true_close = ((~predicted_open) & (~target_open)).sum().float()
                false_close = ((~predicted_open) & target_open).sum().float()
                missed_close = (predicted_open & (~target_open)).sum().float()
                close_f1 = 2.0 * true_close / (
                    2.0 * true_close + false_close + missed_close
                ).clamp_min(1.0)
                mae = error_deg.abs().mean()
                metrics[f"Valid/{name}_joint_mae_deg"] = mae
                metrics[f"Valid/{name}_joint_rmse_deg"] = error_deg.square().mean().sqrt()
                metrics[f"Valid/{name}_velocity_mae_deg"] = velocity_error
                metrics[f"Valid/{name}_acceleration_mae_deg"] = acceleration_error
                metrics[f"Valid/{name}_jerk_mae_deg"] = jerk_error
                metrics[f"Valid/{name}_gripper_accuracy"] = accuracy
                metrics[f"Valid/{name}_gripper_close_f1"] = close_f1
                metrics[f"Valid/{name}_physical_selection_score"] = (
                    mae
                    + 0.25 * velocity_error
                    + 0.10 * acceleration_error
                    + 0.03 * jerk_error
                    + 10.0 * (1.0 - accuracy)
                )

            images[embodiment_id] = self._visualize_preds(predictions, raw_batch)

        if flow_losses:
            metrics["Valid/action_loss"] = torch.stack(flow_losses).mean()
        return metrics, images

    def _visualize_preds(self, predictions, batch):
        embodiment_id = int(batch["embodiment"].reshape(-1)[0].item())
        name = get_embodiment(embodiment_id).lower()
        return self.viz_func[name](predictions, batch)
