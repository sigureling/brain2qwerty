# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import typing as tp
from pathlib import Path

import lightning.pytorch as pl
import pydantic
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
from torch.utils.data import DataLoader

import neuralset as ns
from neuralset.events.study import EventsTransform
from neuraltrain.models.base import BaseModelConfig
from neuraltrain.utils import WandbLoggerConfig

from . import models as _models  # noqa: F401  (registers the BrainOmni encoder)
from . import transforms as _transforms  # noqa: F401  (registers SpanishBCBL adapter)
from .callbacks import PredictionJSONCallback
from .data import SentenceDataset
from .pl_module import NeuroCTCModule
from .utils import BrainOmniSensorMetadata, accelerator, build_events

log = logging.getLogger(__name__)


class Data(pydantic.BaseModel):
    """Sentence-level dataloaders for Brain2Qwerty V2 with BrainOmni.

    Runs the study, applies the preprocessing and split transforms, extends each
    sentence window by a small random tail, and builds one padded dataloader per
    split (train-time MEG onset jitter is applied inside the dataset).
    """

    model_config = pydantic.ConfigDict(extra="ignore", arbitrary_types_allowed=True)

    study: ns.events.Study
    transforms: list[EventsTransform] = pydantic.Field(default_factory=list)
    neuro: ns.extractors.BaseExtractor
    extractor: ns.extractors.BaseExtractor

    start: float = -0.4
    duration: float | None = None
    jitter: bool = True
    num_classes: int = 29
    tail_min: float = 0.4  # extend each sentence window by a random tail (seconds)
    tail_max: float = 0.5

    batch_size: int = 32
    val_batch_size: int = 128
    test_batch_size: int = 8
    num_workers: int = 0
    pin_memory: bool = False
    persistent_workers: bool = False

    def build(self) -> dict[str, DataLoader]:
        events = build_events(self.study, self.transforms, (self.tail_min, self.tail_max))
        self.neuro.prepare(events)
        self.extractor.prepare(events)

        subject_encoder = ns.extractors.LabelEncoder(
            event_types="Meg", event_field="subject"
        )
        subject_encoder.prepare(events)
        sensor_metadata = BrainOmniSensorMetadata(neuro=self.neuro)
        sensor_metadata.prepare(events)

        extractors = {
            "neuros": self.neuro,
            "phonemes": self.extractor,
            "days": subject_encoder,
            "sensor_metadata": sensor_metadata,
        }
        batch_sizes = {
            "train": self.batch_size,
            "val": self.val_batch_size,
            "test": self.test_batch_size,
        }
        loaders: dict[str, DataLoader] = {}
        for split, batch_size in batch_sizes.items():
            mask = (events.split == split) & (events.type == "Sentence")
            segments = ns.segments.list_segments(
                events, mask, start=self.start, duration=self.duration
            )
            if not segments:
                continue
            dataset = SentenceDataset(
                extractors,
                segments,
                jitter=(self.jitter and split == "train"),
                remove_incomplete_segments=True,
            )
            loaders[split] = DataLoader(
                dataset,
                collate_fn=dataset.collate_fn,
                batch_size=batch_size,
                shuffle=(split == "train"),
                drop_last=(split == "train"),
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=self.persistent_workers,
            )
        return loaders


class Experiment(pydantic.BaseModel):
    """Train and evaluate the Brain2Qwerty V2 BrainOmni CTC pipeline."""

    model_config = pydantic.ConfigDict(extra="ignore", arbitrary_types_allowed=True)

    data: Data
    brain_model_config: BaseModelConfig
    num_classes: int = 29

    seed: int = 123
    max_epochs: int = 275
    stage1_epochs: int = 20
    precision: str = "16-mixed"
    gradient_clip_val: float | None = 1.0
    accumulate_gradient_batches: int = 4
    devices: int = 4
    output_dir: str = "."

    # auxiliary/final CTC loss weighting
    loss_alpha: float = 0.7
    encoder_lr: float | None = None

    optimizer_config: dict = pydantic.Field(
        default_factory=lambda: {"lr": 8e-4, "weight_decay": 1e-3}
    )
    scheduler_config: dict = pydantic.Field(
        default_factory=lambda: {"name": "OneCycleLR", "pct_start": 0.3}
    )
    preprocess_config: dict = pydantic.Field(default_factory=dict)

    save_checkpoints: bool = True
    eval_only: bool = False
    ckpt_path: str | None = None
    resume_ckpt: str | None = None  # resume training (trainer state) from this ckpt
    wandb_config: WandbLoggerConfig | None = None

    _trainer: pl.Trainer | None = None
    _module: NeuroCTCModule | None = None

    def model_post_init(self, log__: tp.Any) -> None:
        pl.seed_everything(self.seed, workers=True)
        torch.set_float32_matmul_precision("medium")

    def _build_module(
        self, loaders: dict, scheduler_config: dict | None = None
    ) -> NeuroCTCModule:
        n_in_channels = loaders["train"].dataset[0].data["neuros"].shape[1]
        subject_encoder = loaders["train"].dataset.extractors["days"]
        network = self.brain_model_config.build(
            n_in_channels=n_in_channels,
            n_outputs=self.num_classes,
            n_subjects=subject_encoder._n_classes,
        )

        module = NeuroCTCModule(
            network=network,
            loss_alpha=self.loss_alpha,
            encoder_lr=self.encoder_lr,
            optimizer_config=self.optimizer_config,
            scheduler_config=scheduler_config or self.scheduler_config,
            preprocess_config=self.preprocess_config,
        )

        # materialise lazy params (channel merger) before DDP wraps the model
        sample = loaders["train"].dataset[0]
        module.eval()
        with torch.no_grad():
            module.network(
                sample.data["neuros"].transpose(1, 2),
                torch.zeros(1, dtype=torch.long),
                sample.data["sensor_metadata"][..., :6],
                sample.data["sensor_metadata"][..., 6].long(),
            )
        module.train()
        for mod in module.modules():
            for pname, p in list(getattr(mod, "_parameters", {}).items()):
                if isinstance(p, torch.nn.UninitializedParameter):
                    mod._parameters[pname] = nn.Parameter(torch.empty(1))
        return module

    def _trainer_setup(
        self,
        *,
        output_dir: str,
        max_epochs: int,
        eval_only: bool = False,
    ) -> pl.Trainer:
        accel, devices = accelerator(self.devices)
        if eval_only:
            # Evaluate in a single process so the prediction callback captures the
            # whole test split without DDP sharding to reconcile.
            devices = 1
        elif self.devices > 1:
            # Full training targets exactly 4 GPUs: refuse to silently run on
            # fewer so the effective batch (batch_size x 4 x accumulate=32) stays
            # equivalent to the paper's 8-GPU x accumulate=2 setup.
            assert devices == 4, f"training requires exactly 4 GPUs, got {devices}"
        callbacks: list[pl.Callback] = []
        if eval_only or output_dir == self.output_dir:
            callbacks.append(PredictionJSONCallback(save_dir=self.output_dir))
        if self.save_checkpoints:
            callbacks.append(
                ModelCheckpoint(
                    dirpath=output_dir,
                    filename="best_ctc",
                    save_last=True,
                    save_top_k=1,
                    monitor="val/cer_epo",
                    mode="min",
                )
            )
        loggers: list = [TensorBoardLogger(output_dir, name="logs")]
        if self.wandb_config is not None:
            loggers.append(self._build_wandb_logger())
        return pl.Trainer(
            accelerator=accel,
            devices=devices,
            strategy=DDPStrategy(find_unused_parameters=True) if devices > 1 else "auto",
            max_epochs=max_epochs,
            gradient_clip_val=self.gradient_clip_val,
            accumulate_grad_batches=self.accumulate_gradient_batches,
            precision=self.precision,
            callbacks=callbacks,
            logger=loggers,
            log_every_n_steps=2,
        )

    def _build_wandb_logger(self):
        try:
            xp_config = self.model_dump(mode="json")
        except Exception:
            xp_config = None
        return self.wandb_config.build(save_dir=self.output_dir, xp_config=xp_config)

    def run(self) -> None:
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        loaders = self.data.build()
        if self.eval_only:
            self._module = self._build_module(loaders)
            self._trainer = self._trainer_setup(
                output_dir=self.output_dir,
                max_epochs=self.max_epochs,
                eval_only=True,
            )
        elif self.resume_ckpt is not None or self.stage1_epochs == 0:
            self._module = self._build_module(loaders)
            self._trainer = self._trainer_setup(
                output_dir=self.output_dir, max_epochs=self.max_epochs
            )
            self._trainer.fit(
                self._module,
                loaders["train"],
                loaders.get("val"),
                ckpt_path=self.resume_ckpt,
            )
        else:
            stage1_scheduler = dict(self.scheduler_config)
            stage1_scheduler.pop("warmup_steps", None)
            stage1_scheduler["warmup_ratio"] = 0.1
            stage1_module = self._build_module(
                loaders, scheduler_config=stage1_scheduler
            )
            stage1_module.network.freeze_backbone()
            stage1_dir = str(Path(self.output_dir) / "stage1")
            stage1_trainer = self._trainer_setup(
                output_dir=stage1_dir, max_epochs=self.stage1_epochs
            )
            stage1_trainer.fit(
                stage1_module,
                loaders["train"],
                loaders.get("val"),
            )

            self._module = self._build_module(loaders)
            best_stage1 = (
                stage1_trainer.checkpoint_callback.best_model_path
                if self.save_checkpoints
                else ""
            )
            if best_stage1:
                checkpoint = torch.load(
                    best_stage1, map_location="cpu", weights_only=False
                )
                self._module.load_state_dict(checkpoint["state_dict"], strict=True)
            else:
                self._module.load_state_dict(stage1_module.state_dict(), strict=True)
            self._module.network.unfreeze_backbone()
            self._trainer = self._trainer_setup(
                output_dir=self.output_dir, max_epochs=self.max_epochs
            )
            self._trainer.fit(
                self._module,
                loaders["train"],
                loaders.get("val"),
            )
        if "test" in loaders:
            self._trainer.test(
                self._module,
                dataloaders=loaders["test"],
                ckpt_path=self.ckpt_path if self.eval_only else None,
            )


def main(argv: list[str] | None = None) -> None:
    """Run the experiment in a given mode:

    ``python -m brain2qwerty_v2_brainomni.main {debug,train,eval,cache}`` — every command is
    the same ``Experiment``, only the config (and eval/cache mode) differs.
    """
    import argparse

    import studies  # noqa: F401  (registers the SpanishBCBL study)

    from .cli import add_wandb_args, wandb_config
    from .config.xp_config import debug_config, experiment_config

    parser = argparse.ArgumentParser(prog="brain2qwerty_v2_brainomni")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("debug", help="1-timeline smoke test (default debug config)")
    p_train = sub.add_parser("train", help="full training (1 node, 4 GPUs)")
    p_train.add_argument("--resume", default=None, help="checkpoint to resume from")
    p_train.add_argument("--seed", type=int, default=None, help="override the seed")
    p_eval = sub.add_parser("eval", help="evaluate a checkpoint on the test split")
    p_eval.add_argument("--ckpt", required=True, help="checkpoint to evaluate")
    p_cache = sub.add_parser("cache", help="pre-warm the feature cache")
    p_cache.add_argument("--debug", action="store_true", help="only the debug subset")
    for p in (sub.choices["debug"], p_train, p_eval):
        add_wandb_args(p)
    args = parser.parse_args(argv)

    if args.command == "cache":
        cfg = debug_config() if args.debug else experiment_config()
        print("[brain2qwerty_v2_brainomni] pre-warming the feature cache...")
        Experiment(**cfg).data.build()
        print("[brain2qwerty_v2_brainomni] cache warmed.")
        return

    cfg = debug_config() if args.command == "debug" else experiment_config()
    if args.command == "eval":
        cfg["eval_only"] = True
        cfg["ckpt_path"] = args.ckpt
    if getattr(args, "resume", None):
        cfg["resume_ckpt"] = args.resume
    if getattr(args, "seed", None) is not None:
        cfg["seed"] = args.seed
    wandb = wandb_config(args, args.command, cfg.get("seed", 0))
    if wandb is not None:
        cfg["wandb_config"] = wandb
    print(
        f"[brain2qwerty_v2_brainomni] running in "
        f"'{args.command}' mode (seed={cfg.get('seed')})"
    )
    Experiment(**cfg).run()


if __name__ == "__main__":
    main()
