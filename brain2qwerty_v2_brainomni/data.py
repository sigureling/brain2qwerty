# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp

import torch
from torch.nn.utils.rnn import pad_sequence

import neuralset as ns
from neuralset.dataloader import Batch, SegmentDataset
from neuralset.extractors import BaseExtractor
from neuralset.extractors.neuro import MegExtractor

from .utils import apply_jitter


def channel_zscore(data: torch.Tensor) -> torch.Tensor:
    data = data - data.mean(dim=-1, keepdim=True)
    return data / (data.std(dim=-1, keepdim=True) + 1e-6)


class SentenceDataset(SegmentDataset):
    """Sentence-mode dataset with per-item MEG onset jitter (train only) and
    padded collation of variable-length sentences."""

    def __init__(
        self,
        extractors: tp.Mapping[str, BaseExtractor],
        segments: tp.Sequence[ns.segments.Segment],
        jitter: bool = False,
        *,
        remove_incomplete_segments: bool = False,
    ) -> None:
        super().__init__(
            extractors=extractors,
            segments=segments,
            remove_incomplete_segments=remove_incomplete_segments,
        )
        self.jitter = jitter

    def __getitem__(self, idx: int) -> Batch:
        if not isinstance(idx, int):
            raise ValueError(f"idx must be int, got {type(idx)}")
        seg = self.segments[idx]
        out: dict[str, torch.Tensor] = {}
        for name, extractor in self.extractors.items():
            data = extractor(
                seg.ns_events, start=seg.start, duration=seg.duration, trigger=seg.trigger
            )
            if self.jitter and isinstance(extractor, MegExtractor):
                data = apply_jitter(data, seg, extractor)
            if isinstance(extractor, MegExtractor):
                data = channel_zscore(data)
            out[name] = data[None, ...]
        return Batch(data=out, segments=[seg])

    def collate_fn(self, batches: list[Batch]) -> Batch:
        if not batches or not batches[0].data:
            return Batch(data={}, segments=[])
        batch_data = [b.data for b in batches]
        out: dict[str, tp.Any] = {}

        # Extractors add a leading batch dimension in __getitem__.  Remove
        # only that dimension: squeeze() would turn a one-character target
        # into a scalar and make pad_sequence fail on mixed-length batches.
        phonemes = [d["phonemes"].squeeze(0) for d in batch_data]
        out["phoneme_sizes"] = torch.tensor(
            [p.shape[0] for p in phonemes], dtype=torch.long
        )
        out["phonemes"] = pad_sequence(phonemes, batch_first=True, padding_value=0)

        neuros = [d["neuros"].squeeze(0).T for d in batch_data]  # -> (T, C)
        out["neuro_sizes"] = torch.tensor([n.shape[0] for n in neuros], dtype=torch.long)
        out["neuros"] = pad_sequence(neuros, batch_first=True, padding_value=0)

        out["days"] = torch.cat([d["days"].reshape(-1) for d in batch_data])
        metadata = torch.cat([d["sensor_metadata"] for d in batch_data])
        out["pos"] = metadata[..., :6]
        out["sensor_type"] = metadata[..., 6].long()
        # sentence texts/segments stay on Batch.segments so Batch.to(device) only
        # moves tensors
        return Batch(data=out, segments=[b.segments[0] for b in batches])
