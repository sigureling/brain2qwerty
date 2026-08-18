# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from edit_distance import SequenceMatcher
from torchmetrics import Metric


class CharacterErrorRate(Metric):
    """CTC greedy character error rate on the encoder logits (blank=0, collapse repeats).

    Monitors the CTC head during training, validation and test.
    """

    def __init__(self) -> None:
        super().__init__()
        for name in ("total_edit_distance", "total_length"):
            self.add_state(name, default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, y_pred, y_true, adjusted_x_len, y_len):
        for idx in range(y_pred.shape[0]):
            decoded = torch.argmax(y_pred[idx, : adjusted_x_len[idx], :], dim=-1)
            decoded = torch.unique_consecutive(decoded, dim=-1)
            decoded = [x.item() for x in decoded if x.item() != 0]
            true_seq = y_true[idx, : y_len[idx]]
            self.total_edit_distance += SequenceMatcher(
                a=true_seq.tolist(), b=decoded
            ).distance()
            self.total_length += len(true_seq)

    def compute(self):
        if self.total_length == 0:
            return torch.tensor(0.0)
        return self.total_edit_distance.float() / self.total_length
