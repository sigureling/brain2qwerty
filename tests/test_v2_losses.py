# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Fast (synthetic, CPU) tests for the V2 losses and CTC decoding."""

import torch

from brain2qwerty_v2.losses import CtcLoss
from brain2qwerty_v2.utils import ctc_greedy_decode, label_to_text


def test_ctc_loss_backward():
    # The character-level CTC loss (stage-1 objective) must be finite and produce
    # clean gradients for variable-length targets — the basic guarantee that the
    # encoder can be trained at all.
    torch.manual_seed(0)
    B, T, C = 2, 20, 29
    logits = torch.randn(B, T, C, requires_grad=True)
    targets = torch.randint(1, C, (B, 5))  # class 0 is the CTC blank
    in_lens = torch.full((B,), T, dtype=torch.long)
    tgt_lens = torch.full((B,), 5, dtype=torch.long)
    loss = CtcLoss()(logits, targets, in_lens, tgt_lens)
    loss.backward()
    assert loss.item() > 0 and logits.grad is not None
    assert not torch.isnan(logits.grad).any()


def test_ctc_greedy_decode_and_label_to_text():
    # CTC greedy decoding semantics: collapse repeats and drop the blank (class 0),
    # then map indices to characters. The crafted logits spell [1,2,3] after
    # collapsing the interleaved blanks; label 27 ("&") must render as a space.
    B, T, C = 1, 10, 29
    logits = torch.full((B, T, C), -10.0)
    seq = [1, 0, 2, 0, 3, 0, 0, 0, 0, 0]  # collapses to [1, 2, 3] -> "abc"
    for t, c in enumerate(seq):
        logits[0, t, c] = 10.0
    assert ctc_greedy_decode(logits) == ["abc"]
    assert label_to_text([1, 2, 3]) == "abc"
    assert label_to_text([1, 27, 2]) == "a b"  # 27 -> "&" -> space
