# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging

import lightning.pytorch as pl
import torch
import torch.nn as nn
from torch import optim

from .augmentations import Preprocess, PreprocessConfig
from .losses import CtcLoss
from .metrics import CharacterErrorRate
from .utils import compute_output_lens, ctc_greedy_decode, label_to_text

log = logging.getLogger(__name__)


def minimum_ctc_frames(
    targets: torch.Tensor, target_sizes: torch.Tensor
) -> torch.Tensor:
    """Minimum CTC frames, including blanks between adjacent repeats."""
    if targets.shape[1] < 2:
        return target_sizes
    positions = torch.arange(targets.shape[1] - 1, device=targets.device)
    valid = positions.unsqueeze(0) < (target_sizes - 1).unsqueeze(1)
    repeats = (targets[:, :-1] == targets[:, 1:]) & valid
    return target_sizes + repeats.sum(dim=1)


class NeuroCTCModule(pl.LightningModule):
    """Lightning module for the sentence-level CTC decoder."""

    def __init__(
        self,
        *,
        network: nn.Module,
        loss_alpha: float = 0.7,
        optimizer_config: dict | None = None,
        scheduler_config: dict | None = None,
        preprocess_config: dict | None = None,
        encoder_lr: float | None = None,
    ):
        super().__init__()
        self.network = network
        self.optimizer_config = optimizer_config or {}
        self.scheduler_config = scheduler_config or {}
        self.preprocess = Preprocess(
            **PreprocessConfig(**(preprocess_config or {})).model_dump()
        )

        self.ctc_loss = CtcLoss()
        self.loss_alpha = loss_alpha
        self.val_ctc_cer = CharacterErrorRate()
        self.test_ctc_cer = CharacterErrorRate()
        self.encoder_lr = encoder_lr

        self._test_predictions: list[dict] = []

    # --- core step -----------------------------------------------------
    def _run_step(self, batch, batch_idx, step_name):
        data = batch.data
        if step_name == "train":
            data = self.preprocess(data)

        batch_size = data["neuros"].shape[0]
        model_out = self.network(
            data["neuros"],
            data["days"],
            data["pos"],
            data["sensor_type"],
        )
        ctc_logits = model_out["c_out"]

        out_lens = compute_output_lens(self.network, data["neuro_sizes"])
        out_lens = torch.clamp(out_lens, min=0, max=ctc_logits.shape[1])

        required_lens = minimum_ctc_frames(data["phonemes"], data["phoneme_sizes"])
        invalid = out_lens < required_lens
        if invalid.any():
            rows = []
            for idx in invalid.nonzero(as_tuple=False).flatten().tolist():
                extra = getattr(batch.segments[idx].trigger, "extra", {}) or {}
                rows.append(
                    f"{extra.get('sentence_UID', idx)}: "
                    f"output={int(out_lens[idx])}, required={int(required_lens[idx])}"
                )
            raise RuntimeError("Invalid CTC alignment: " + "; ".join(rows))

        loss_final = self.ctc_loss(
            ctc_logits, data["phonemes"], out_lens, data["phoneme_sizes"]
        )
        if "z_aux" in model_out:
            loss_aux = self.ctc_loss(
                model_out["z_aux"],
                data["phonemes"],
                out_lens,
                data["phoneme_sizes"],
            )
            ctc_loss = (1 - self.loss_alpha) * loss_final + self.loss_alpha * loss_aux
        else:
            ctc_loss = loss_final
        self.log(
            f"{step_name}/loss_ctc",
            ctc_loss,
            on_step=(step_name == "train"),
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )

        if step_name != "train":
            metric = self.val_ctc_cer if step_name == "val" else self.test_ctc_cer
            metric.update(ctc_logits, data["phonemes"], out_lens, data["phoneme_sizes"])

        if step_name == "test":
            ctc_texts = ctc_greedy_decode(ctc_logits.detach(), out_lens)
            for i, seg in enumerate(batch.segments):
                extra = getattr(seg.trigger, "extra", {}) or {}
                typed_ids = data["phonemes"][i, : data["phoneme_sizes"][i]]
                self._test_predictions.append(
                    {
                        "typed_text": label_to_text(typed_ids.detach().cpu().tolist()),
                        "ctc_text": ctc_texts[i],
                        "subject": extra.get("subject", ""),
                        "sentence_UID": extra.get("sentence_UID", ""),
                    }
                )

        self.log(
            f"{step_name}/loss",
            ctc_loss,
            on_step=(step_name == "train"),
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        return ctc_loss, ctc_logits, data["phonemes"]

    def training_step(self, batch, batch_idx):
        return self._run_step(batch, batch_idx, "train")[0]

    def validation_step(self, batch, batch_idx):
        _, y_pred, y_true = self._run_step(batch, batch_idx, "val")
        return y_pred, y_true

    def test_step(self, batch, batch_idx):
        _, y_pred, y_true = self._run_step(batch, batch_idx, "test")
        return y_pred, y_true

    def on_validation_epoch_end(self) -> None:
        self.log("val/cer_epo", self.val_ctc_cer.compute(), prog_bar=True)
        self.val_ctc_cer.reset()

    def on_test_epoch_end(self) -> None:
        self.log("test/cer_epo", self.test_ctc_cer.compute(), prog_bar=True)
        self.test_ctc_cer.reset()
        self._test_predictions = []

    # --- optimiser -----------------------------------------------------
    def configure_optimizers(self):
        backbone = getattr(self.network, "backbone", self.network)
        backbone_ids = {id(p) for p in backbone.parameters()}
        network_ids = {id(p) for p in self.network.parameters()}
        backbone_trainable = [p for p in backbone.parameters() if p.requires_grad]
        head_trainable = [
            p
            for p in self.network.parameters()
            if id(p) not in backbone_ids and p.requires_grad
        ]
        other_trainable = [
            p
            for p in self.parameters()
            if id(p) not in network_ids and p.requires_grad
        ]
        base_lr = self.optimizer_config.get("lr", 4e-4)
        enc_lr = self.encoder_lr or base_lr
        wd = self.optimizer_config.get("weight_decay", 1e-3)

        groups = []
        if head_trainable or other_trainable:
            groups.append({"params": head_trainable + other_trainable, "lr": base_lr})
        if backbone_trainable:
            groups.append({"params": backbone_trainable, "lr": enc_lr})
        optimizer = optim.AdamW(
            groups or [{"params": list(self.parameters())}],
            lr=base_lr,
            weight_decay=wd,
        )

        # Warmup (linear) followed by cosine decay, as in the paper.
        total_steps = self.trainer.estimated_stepping_batches
        if "warmup_ratio" in self.scheduler_config:
            warmup_steps = max(
                1, int(total_steps * self.scheduler_config["warmup_ratio"])
            )
        else:
            warmup_steps = self.scheduler_config.get("warmup_steps", 500)
        cosine_steps = max(total_steps - warmup_steps, 1)
        warmup = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps
        )
        cosine = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.scheduler_config.get("T_max", cosine_steps),
            eta_min=self.scheduler_config.get("eta_min", 1e-6),
        )
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps]
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
