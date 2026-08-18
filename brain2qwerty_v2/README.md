<!--
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
-->

# Brain2Qwerty V2

Official implementation of [**Accurate Decoding of Natural Sentences from Non-Invasive Brain Recordings**](https://facebookresearch.github.io/brain2qwerty/assets/brain2qwerty_v2.pdf) (under review, 2026).

Brain2Qwerty V2 decodes whole typed sentences from a single continuous MEG window — *asynchronously*, without segmenting the recording around individual keystrokes. A convolutional + Conformer encoder is trained with character-level CTC objectives.

<p align="center">
  <img src="resources/approach.png" alt="Asynchronous decoding of a continuous MEG response window into text, versus the synchronous keystroke-windowed decoding of Brain2Qwerty V1." width="100%">
</p>

## This folder contains

- The end-to-end CTC decoder (Conv + Conformer encoder and CTC heads) with training and evaluation using PyTorch Lightning
- The model experiment configuration
- The evaluation pipeline

## Installation

**Requirements:** Python 3.10+, CUDA-capable GPU.

```bash
pip install -r requirements.lock   # pinned dependencies
pip install -e . --no-deps         # the brain2qwerty package
```

## Data

The default V2 configuration in this repository uses the public SpanishBCBL
dataset (`Pinet2024Meg`) through a minimal sentence-level adapter. Download the
dataset from [Hugging Face](https://huggingface.co/datasets/bcbl190626/SpanishBCBL)
and point `BRAIN2QWERTY_STUDIES` to its root directory.

See [SPANISHBCBL_ADAPTER.md](SPANISHBCBL_ADAPTER.md) for the dataset-specific
event handling, unchanged V2 numerical/training settings, and validation notes.

## Quickstart

Each step is its own mode of `main` (the same set of commands as V1). Training uses one node (8 GPUs by default) and automatically falls back to a single GPU.

```bash
# (optional) pre-warm the feature cache (--debug for the 1-timeline subset)
python -m brain2qwerty_v2.main cache

# short single-timeline CTC run on 1 GPU (sanity check)
python -m brain2qwerty_v2.main debug

# full CTC training
python -m brain2qwerty_v2.main train

# evaluate a checkpoint on the test split
python -m brain2qwerty_v2.main eval --ckpt $BRAIN2QWERTY_RESULTS/best_ctc.ckpt
```

The full configuration lives in [`config/xp_config.py`](config/xp_config.py) (experiment) and [`config/model_config.py`](config/model_config.py) (architecture).

## Result extraction and analysis

The typical end-to-end workflow, from raw data to the final per-subject numbers:

**1. Pre-warm the cache** (once; CPU-bound feature extraction):

```bash
python -m brain2qwerty_v2.main cache
```

**2. Train** — trains the CTC decoder and saves `best_ctc.ckpt` (best validation CER):

```bash
python -m brain2qwerty_v2.main train
```

**3. Evaluate the checkpoint** on the test split — writes `predictions_test.csv` (true text, CTC text and CTC CER per sentence):

```bash
python -m brain2qwerty_v2.main eval --ckpt $BRAIN2QWERTY_RESULTS/best_ctc.ckpt
```

**4. Compute the per-subject metrics** from that CSV (sentence-wise and averaged per subject, with the standard error across subjects):

```bash
python -m brain2qwerty_v2.scripts.extract_predictions \
    --input $BRAIN2QWERTY_RESULTS/predictions_test.csv --split test
```

## Citing

Mingfang (Lucy) Zhang\* and Jarod Lévy\* contributed equally.

```bibtex
@article{brain2qwertyv2,
  title={Accurate Decoding of Natural Sentences from Non-Invasive Brain Recordings},
  author={Zhang, Mingfang and L{\'e}vy, Jarod and Rommel, Cedric and Rapin, J{\'e}r{\'e}my and Bel, Corentin and Bonnaire, Julie and Nieto, Daniel and Bourdillon, Pierre and Pinet, Svetlana and d'Ascoli, St{\'e}phane and Moreau, Thomas and King, Jean-R{\'e}mi},
  year={2026}
}
```
