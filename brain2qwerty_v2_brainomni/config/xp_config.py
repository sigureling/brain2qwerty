# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from pathlib import Path

from .model_config import ENCODER

STUDY_PATH = os.environ.get(
    "BRAIN2QWERTY_STUDIES", str(Path.home() / "brain2qwerty_data" / "studies")
)
CACHE = os.environ.get(
    "BRAIN2QWERTY_CACHE",
    str(Path(__file__).resolve().parents[2] / ".cache" / "spanishbcbl_meg_v2_brainomni"),
)
RESULTS = os.environ.get("BRAIN2QWERTY_RESULTS", str(Path(CACHE) / "results"))


def experiment_config() -> dict:
    """Full V2 CTC configuration using the SpanishBCBL MEG dataset."""
    return {
        "output_dir": RESULTS,
        "seed": 123,
        "max_epochs": 275,
        "data": {
            "study": {
                "name": "Pinet2024Meg",
                "path": STUDY_PATH,
                "infra": {"folder": CACHE},
                "infra_timelines": {"folder": CACHE, "cluster": None},
            },
            "transforms": [
                {"name": "SpanishBCBLV2Preprocessing"},
                {
                    "name": "Brain2QwertyV2Splitter",
                    "deterministic_splitter": {
                        "ratios": {"train": 0.8, "val": 0.1, "test": 0.1}
                    },
                },
            ],
            "neuro": {
                "name": "MegExtractor",
                "frequency": 256,
                "filter": (0.5, 85.0),
                "scaler": None,
                "apply_proj": False,
                "clamp": None,
                "picks": "meg",
                "notch_filter": [50.0, 60.0],
                "allow_maxshield": True,
                "infra": {"folder": CACHE, "cluster": None},
            },
            "extractor": {
                "name": "SentenceKeySeq",
                "mode": "typed_label",
                # A segment is triggered by its Sentence event.  Selecting the
                # trigger avoids collisions when the random tail overlaps the
                # preceding Sentence event.
                "aggregation": "trigger",
                "infra": {"folder": CACHE},
            },
            "batch_size": 8,
            "val_batch_size": 8,
            "test_batch_size": 8,
            "num_workers": 16,
            "pin_memory": True,
            "persistent_workers": True,
        },
        # MEG augmentation (on-device, train only): per-channel offset + SpecAugment
        # masking + time-stretch (no white noise), matching the paper.
        "preprocess_config": {
            "whiteNoiseSD": 0.0,
            "constantOffsetSD": 0.3,
            "time_mask_param": 50,
            "p_time_mask": 0.2,
            "freq_mask_param": 400,
            "time_stretch": True,
        },
        "brain_model_config": ENCODER,
        # Blend the auxiliary and final CTC heads.
        "loss_alpha": 0.7,
        "stage1_epochs": 20,
        "optimizer_config": {"lr": 8e-4, "weight_decay": 1e-3},
        "scheduler_config": {
            "name": "WarmupCosine",
            "warmup_steps": 500,
            "eta_min": 1e-6,
        },
        "accumulate_gradient_batches": 32,
        "precision": "bf16-mixed",
    }


def debug_config() -> dict:
    """Smoke-test config: one timeline, single GPU."""
    cfg = experiment_config()
    cfg["data"]["study"]["query"] = "timeline_index == 0"
    cfg["data"]["batch_size"] = 4
    cfg["data"]["val_batch_size"] = 4
    cfg["data"]["test_batch_size"] = 4
    cfg["max_epochs"] = 2
    cfg["stage1_epochs"] = 1
    cfg["devices"] = 1
    cfg["save_checkpoints"] = False
    return cfg
