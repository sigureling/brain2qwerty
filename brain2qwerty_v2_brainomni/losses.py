# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch


class CtcLoss:
    """Character-level CTC loss on the encoder logits (blank=0)."""

    def __init__(self) -> None:
        self.loss = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)

    def __call__(self, phoneme_pred, phoneme_true, x_len_norm, y_len):
        log_probs = phoneme_pred.log_softmax(2).permute(1, 0, 2)
        return self.loss(log_probs, phoneme_true, x_len_norm, y_len)
