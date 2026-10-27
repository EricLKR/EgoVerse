"""Offline pi0.5 evaluator that persists aggregate metrics as JSON.

This is intentionally separate from the training evaluator.  It is used only
after validation has selected a checkpoint, on the held-out test split.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import torch

from egomimic.eval.eval_pi import PIEvalVideo


class PI05OfflineEvalVideo(PIEvalVideo):
    """PI evaluator with videos plus a machine-readable metric summary."""

    def __init__(self, *args, summary_path: str, **kwargs):
        super().__init__(*args, **kwargs)
        self.summary_path = Path(summary_path)
        self._metric_sums: dict[str, float] = defaultdict(float)
        self._metric_counts: dict[str, int] = defaultdict(int)

    def compute_metrics_and_viz(self, batch):
        metrics, images = super().compute_metrics_and_viz(batch)
        for key, value in metrics.items():
            if torch.is_tensor(value):
                value = value.detach().float().mean().cpu().item()
            value = float(value)
            self._metric_sums[key] += value
            self._metric_counts[key] += 1
        return metrics, images

    def on_validation_end(self):
        super().on_validation_end()
        if not self.trainer.is_global_zero:
            return
        payload = {
            "aggregation": "mean_of_batches",
            "metrics": {
                key: self._metric_sums[key] / self._metric_counts[key]
                for key in sorted(self._metric_sums)
            },
            "batch_counts": dict(sorted(self._metric_counts.items())),
        }
        self.summary_path.parent.mkdir(parents=True, exist_ok=True)
        self.summary_path.write_text(json.dumps(payload, indent=2) + "\n")

