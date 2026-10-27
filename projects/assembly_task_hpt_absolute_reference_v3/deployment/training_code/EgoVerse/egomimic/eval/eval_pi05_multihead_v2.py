"""Validation metrics and plots for EgoVerse pi0.5 MultiHead V2."""

from __future__ import annotations

import torch
from torchmetrics import MeanSquaredError

from egomimic.eval.eval_video import EvalVideo
from egomimic.rldb.embodiment.embodiment import get_embodiment


class PI05MultiHeadV2Eval(EvalVideo):
    """Score the shared pose head and deployable PIPER joint head separately."""

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
                metrics[f"Valid/{pose_prediction_key}_paired_mse_avg"] = mse(
                    predictions[pose_prediction_key].cpu(), raw_batch[pose_key].cpu()
                )
                metrics[f"Valid/{pose_prediction_key}_final_mse_avg"] = mse(
                    predictions[pose_prediction_key][:, -1].cpu(),
                    raw_batch[pose_key][:, -1].cpu(),
                )

            if embodiment_id == algo.piper_embodiment_id:
                joint_key = algo.auxiliary_joint_key
                prediction_key = f"{name}_{joint_key}"
                if prediction_key not in predictions or joint_key not in raw_batch:
                    raise KeyError("PIPER validation requires deployable joint predictions")
                predicted = predictions[prediction_key].cpu()
                target = raw_batch[joint_key].cpu()
                error_deg = torch.rad2deg(predicted[..., :6] - target[..., :6])
                mae = error_deg.abs().mean()
                rmse = error_deg.square().mean().sqrt()
                predicted_gripper = predicted[..., 6] >= 0.5
                target_gripper = target[..., 6] >= 0.5
                accuracy = (predicted_gripper == target_gripper).float().mean()
                metrics[f"Valid/{prediction_key}_joint_mae_deg"] = mae
                metrics[f"Valid/{prediction_key}_joint_rmse_deg"] = rmse
                metrics[f"Valid/{prediction_key}_gripper_accuracy"] = accuracy
                metrics[f"Valid/{prediction_key}_physical_selection_score"] = (
                    mae + 10.0 * (1.0 - accuracy)
                )

            images[embodiment_id] = self._visualize_preds(predictions, raw_batch)
        if losses:
            metrics["Valid/action_loss"] = torch.stack(losses).mean()
        return metrics, images

    def _visualize_preds(self, predictions, batch):
        embodiment_id = int(batch["embodiment"].reshape(-1)[0].item())
        name = get_embodiment(embodiment_id).lower()
        # Existing visualizers expect one RGB frame per sample.  The adapter
        # keeps a length-one time axis internally, so unwrap it if present.
        viz_batch = dict(batch)
        for key, value in list(viz_batch.items()):
            if key.startswith("observations.images.") and getattr(value, "ndim", 0) == 5:
                viz_batch[key] = value[:, -1]
        return self.viz_func[name](predictions, viz_batch)
