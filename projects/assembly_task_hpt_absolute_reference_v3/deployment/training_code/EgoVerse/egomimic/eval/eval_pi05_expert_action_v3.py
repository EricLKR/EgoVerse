"""Validation metrics for native-horizon pi0.5 Expert-Action V3."""

from __future__ import annotations

import torch
from torchmetrics import MeanSquaredError

from egomimic.eval.eval_video import EvalVideo
from egomimic.rldb.embodiment.embodiment import get_embodiment


class PI05ExpertActionV3Eval(EvalVideo):
    """Evaluate shared pose, expert joint trajectory, smoothness and gripper."""

    def compute_metrics_and_viz(self, batch):
        algo = self.model
        predictions = algo.forward_eval(batch)
        metrics = {}
        images = {}
        mse = MeanSquaredError()
        losses = []
        for embodiment_id, normalized_batch in batch.items():
            raw_batch = algo.norm_stats.unnormalize(normalized_batch, embodiment_id)
            name = get_embodiment(embodiment_id).lower()
            loss_key = f"{name}_loss"
            if loss_key in predictions:
                metrics[f"Valid/{loss_key}"] = predictions[loss_key]
                losses.append(predictions[loss_key])

            pose_key = algo.ac_keys[embodiment_id]
            pose_prediction_key = f"{name}_{pose_key}"
            if pose_prediction_key in predictions:
                pose_prediction = predictions[pose_prediction_key].cpu()
                pose_target = raw_batch[pose_key].cpu()
                metrics[f"Valid/{pose_prediction_key}_paired_mse_avg"] = mse(
                    pose_prediction, pose_target
                )
                metrics[f"Valid/{pose_prediction_key}_final_mse_avg"] = mse(
                    pose_prediction[:, -1], pose_target[:, -1]
                )

            if embodiment_id == algo.piper_embodiment_id:
                joint_key = algo.auxiliary_joint_key
                prediction_key = f"{name}_{joint_key}"
                if prediction_key not in predictions or joint_key not in raw_batch:
                    raise KeyError("V3 validation requires expert joint predictions")
                predicted = predictions[prediction_key].cpu()
                target = raw_batch[joint_key].cpu()
                if predicted.shape[1:] != (30, 7) or target.shape[1:] != (30, 7):
                    raise ValueError("V3 evaluation requires native [B,30,7] actions")
                error_deg = torch.rad2deg(predicted[..., :6] - target[..., :6])
                mae = error_deg.abs().mean()
                rmse = error_deg.square().mean().sqrt()
                velocity_error = torch.rad2deg(
                    torch.diff(predicted[..., :6], dim=1)
                    - torch.diff(target[..., :6], dim=1)
                ).abs().mean()
                acceleration_error = torch.rad2deg(
                    torch.diff(predicted[..., :6], n=2, dim=1)
                    - torch.diff(target[..., :6], n=2, dim=1)
                ).abs().mean()
                predicted_gripper = predicted[..., 6] >= 0.5
                target_gripper = target[..., 6] >= 0.5
                accuracy = (predicted_gripper == target_gripper).float().mean()
                true_close = ((~predicted_gripper) & (~target_gripper)).sum().float()
                false_close = ((~predicted_gripper) & target_gripper).sum().float()
                missed_close = (predicted_gripper & (~target_gripper)).sum().float()
                close_f1 = 2.0 * true_close / (
                    2.0 * true_close + false_close + missed_close
                ).clamp_min(1.0)
                metrics[f"Valid/{prediction_key}_joint_mae_deg"] = mae
                metrics[f"Valid/{prediction_key}_joint_rmse_deg"] = rmse
                metrics[f"Valid/{prediction_key}_velocity_mae_deg"] = velocity_error
                metrics[f"Valid/{prediction_key}_acceleration_mae_deg"] = acceleration_error
                metrics[f"Valid/{prediction_key}_gripper_accuracy"] = accuracy
                metrics[f"Valid/{prediction_key}_gripper_close_f1"] = close_f1
                metrics[f"Valid/{prediction_key}_physical_selection_score"] = (
                    mae
                    + 0.25 * velocity_error
                    + 0.10 * acceleration_error
                    + 10.0 * (1.0 - accuracy)
                )

            images[embodiment_id] = self._visualize_preds(predictions, raw_batch)
        if losses:
            metrics["Valid/action_loss"] = torch.stack(losses).mean()
        return metrics, images

    def _visualize_preds(self, predictions, batch):
        embodiment_id = int(batch["embodiment"].reshape(-1)[0].item())
        name = get_embodiment(embodiment_id).lower()
        return self.viz_func[name](predictions, batch)
