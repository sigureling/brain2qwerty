# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
from pathlib import Path

import lightning.pytorch as pl

from .utils import compute_ctc_sample_metrics


class PredictionJSONCallback(pl.Callback):
    """Save per-sentence CTC predictions to ``predictions_test.json``.

    Only written at test time (not every validation epoch). The rows accumulated by
    the Lightning module are gathered across ranks first, which is a no-op for the
    single-process ``eval`` and keeps the training-end test complete on multi-GPU.
    """

    def __init__(self, save_dir: str):
        super().__init__()
        self.save_dir = Path(save_dir)

    @staticmethod
    def _gather_rows(trainer, rows: list[dict]) -> list[dict]:
        if trainer.world_size <= 1:
            return rows
        import torch.distributed as dist

        gathered: list[list[dict] | None] = [None] * trainer.world_size
        dist.all_gather_object(gathered, rows)
        if trainer.global_rank == 0:
            return [r for rank_rows in gathered for r in rank_rows]
        return []

    def _save(self, trainer, rows, filename):
        rows = self._gather_rows(trainer, rows)
        if not rows or trainer.global_rank != 0:
            return
        has_segment_meta = any(r.get("subject") for r in rows)
        rows_with_metrics = compute_ctc_sample_metrics(
            [r["typed_text"] for r in rows],
            [r["ctc_text"] for r in rows],
        )
        if has_segment_meta:
            rows_with_metrics = [
                {
                    "sentence_UID": row_raw.get("sentence_UID", ""),
                    "subject": row_raw.get("subject", ""),
                    **row_m,
                }
                for row_m, row_raw in zip(rows_with_metrics, rows)
            ]
        self.save_dir.mkdir(parents=True, exist_ok=True)
        path = self.save_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rows_with_metrics, f, ensure_ascii=False, indent=2)
            f.write("\n")
        print(f"Saved {len(rows_with_metrics)} predictions to {path}")

    def on_test_epoch_end(self, trainer, pl_module):
        rows = getattr(pl_module, "_test_predictions", [])
        self._save(trainer, rows, "predictions_test.json")
