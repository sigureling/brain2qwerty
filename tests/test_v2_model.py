# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Fast (synthetic, CPU) tests for the V2 encoder and CTC decode."""

import copy

import torch

import brain2qwerty_v2.models  # noqa: F401  (registers the ConvConformer encoder)
from brain2qwerty_v2.config.model_config import ENCODER
from brain2qwerty_v2.utils import ctc_greedy_decode
from neuraltrain.models.base import BaseModelConfig

N_CH = 306
N_CLASSES = 29
DIM = 32


def _tiny_encoder():
    cfg = copy.deepcopy(ENCODER)
    cfg["dim"] = DIM
    cfg["encoder_config"].update(hidden=64, depth=2, initial_linear=16)
    cfg["encoder_config"]["merger_config"].update(n_virtual_channels=16)
    cfg["encoder_config"]["merger_config"]["fourier_emb_config"].update(total_dim=DIM)
    cfg["transformer_config"].update(ffn_dim=DIM, num_heads=2, num_layers=1)
    return BaseModelConfig(**cfg).build(n_in_channels=N_CH, n_outputs=N_CLASSES)


def test_encoder_forward_returns_final_and_aux_ctc():
    """The ConvConformer exposes final and auxiliary CTC logits."""
    torch.manual_seed(0)
    enc = _tiny_encoder()
    b, t = 2, 120
    neuros = torch.randn(b, t, N_CH)  # (B, T, C) sentence layout
    days = torch.zeros(b, dtype=torch.long)
    chan_pos = torch.rand(b, N_CH, 2)

    out = enc(neuros, days, chan_pos)
    assert set(("z", "c_out", "z_aux")).issubset(out.keys())
    assert out["c_out"].shape[2] == N_CLASSES
    assert out["z_aux"].shape[2] == N_CLASSES
    assert out["z_aux"].shape[1] == out["c_out"].shape[1]


def test_greedy_decode_returns_one_string_per_item():
    torch.manual_seed(0)
    b, t = 2, 40
    ctc_logits = torch.randn(b, t, N_CLASSES)
    texts = ctc_greedy_decode(ctc_logits)
    assert len(texts) == b and all(isinstance(s, str) for s in texts)
