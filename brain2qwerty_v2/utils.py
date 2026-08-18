# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp

import numpy as np
import torch

import neuralset as ns
from neuralset.extractors import BaseExtractor
from neuralset.extractors.base import BaseStatic
from neuralset.extractors.neuro import ChannelPositions as _ChannelPositions

# Character vocabulary for the CTC head: a..z plus space ("&"); the blank
# symbol ("-") is class 0.
key_to_int = {
    "a": 1,
    "b": 2,
    "c": 3,
    "d": 4,
    "e": 5,
    "f": 6,
    "g": 7,
    "h": 8,
    "i": 9,
    "j": 10,
    "k": 11,
    "l": 12,
    "m": 13,
    "n": 14,
    "o": 15,
    "p": 16,
    "q": 17,
    "r": 18,
    "s": 19,
    "t": 20,
    "u": 21,
    "v": 22,
    "w": 23,
    "x": 24,
    "y": 25,
    "z": 26,
    "&": 27,
}
letters_withblank = ["-"] + list(key_to_int.keys())

def compute_output_lens(
    network: torch.nn.Module, neuro_sizes: torch.Tensor
) -> torch.Tensor:
    """Map input MEG lengths to encoder output lengths (post temporal downsampling)."""
    if hasattr(network, "compute_output_lens"):
        return network.compute_output_lens(neuro_sizes)
    conv = network.temporal_downsampling.agg
    return (neuro_sizes - conv.kernel_size[0]) // conv.stride[0] + 1


def apply_jitter(
    data: torch.Tensor, seg: ns.segments.Segment, feat: BaseExtractor
) -> torch.Tensor:
    """Drop a random prefix (up to the pre-trigger window) to jitter sentence onset."""
    seg_start = seg.trigger.start - seg.start
    jitter_amount = np.random.uniform(0, seg_start * feat.frequency)
    return data[:, int(jitter_amount) :]


def prediction_fieldnames(has_segment_meta: bool = False) -> list[str]:
    """Return the columns used by the CTC prediction CSV."""
    cols: list[str] = []
    if has_segment_meta:
        cols += ["sentence_UID", "subject"]
    cols += ["true_text", "ctc_text", "CTC_CER"]
    return cols


def compute_ctc_sample_metrics(
    true_texts: list[str], ctc_texts: list[str]
) -> list[dict]:
    """Compute per-sentence CTC CER for the predictions CSV."""
    from torchmetrics.text import CharErrorRate

    cer_fn = CharErrorRate()
    rows: list[dict] = []
    for tgt, ctc_text in zip(true_texts, ctc_texts):
        row: dict[str, tp.Any] = {
            "true_text": tgt,
            "ctc_text": ctc_text,
            "CTC_CER": cer_fn([ctc_text], [tgt]).item(),
        }
        rows.append(row)
    return rows


# --- CTC label <-> text helpers --------------------------------------------
def label_to_text(ids: list[int]) -> str:
    """Map a CTC target id sequence to text ('&' -> space)."""
    chars = [letters_withblank[i] for i in ids if 0 < i < len(letters_withblank)]
    return "".join(" " if c == "&" else c for c in chars)


def ctc_greedy_decode(
    ctc_logits: torch.Tensor, output_lens: torch.Tensor | None = None
) -> list[str]:
    """Greedy CTC decode (blank=0, collapse repeats, '&' -> space)."""
    preds = ctc_logits.argmax(dim=-1)
    texts: list[str] = []
    for b in range(preds.shape[0]):
        chars: list[str] = []
        prev = 0
        end = preds.shape[1] if output_lens is None else int(output_lens[b].item())
        for t in range(end):
            c = preds[b, t].item()
            if c != prev and c != 0 and c < len(letters_withblank):
                ch = letters_withblank[c]
                chars.append(" " if ch == "&" else ch)
            prev = c
        texts.append("".join(chars))
    return texts


# --- Channel positions -----------------------------------------------------
class ChannelPositions2D(_ChannelPositions):
    """Re-enable 2D channel positions for MEG to match the paper."""

    def model_post_init(self, log__: tp.Any) -> None:
        BaseStatic.model_post_init(self, log__)
        if self.neuro is not None:
            if self.event_types not in {"MneRaw", self.neuro.event_types}:
                raise ValueError(
                    f"event_types={self.event_types} must match "
                    f"neuro.event_types={self.neuro.event_types}."
                )
            self._neuro = self.neuro


# --- Data / experiment helpers ---------------------------------------------
def accelerator(devices: int) -> tuple[str, int]:
    """Return (accelerator, n_devices), capped to the available GPUs."""
    if torch.cuda.is_available():
        return "gpu", max(1, min(devices, torch.cuda.device_count()))
    return "cpu", 1


def build_events(study, transforms, tail_range: tuple[float, float] = (0.4, 0.5)):
    """Run the study and its transforms, then extend each sentence window by a
    random tail (so a segment never ends exactly on the last keystroke)."""
    events = study.run()
    for transform in transforms:
        events = transform.run(events)
    events = ns.events.standardize_events(events)
    sentences = events[events.type == "Sentence"]
    events.loc[sentences.index, "duration"] = sentences.duration + np.random.uniform(
        tail_range[0], tail_range[1], len(sentences)
    )
    return events
