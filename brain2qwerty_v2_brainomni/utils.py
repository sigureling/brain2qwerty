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


def build_brainomni_temporal_mask(
    network: torch.nn.Module,
    neuros: torch.Tensor,
    neuro_sizes: torch.Tensor,
) -> torch.Tensor:
    """Convert collated MEG padding lengths to a temporal patch mask.

    ``neuros`` is the collated ``(B, T, C)`` tensor and ``neuro_sizes`` contains
    the valid input length for each item.  BrainOmni's temporal encoder operates
    on ``W`` patch steps, so the returned boolean mask has shape ``(B, W)``.

    ``W`` is computed from the exact Conv1d geometry used by
    ``BrainOmniBackbone.spatial_forward``.  Input lengths are not rounded up to
    a patch-size multiple, so the effective per-sample length follows the
    floor-based Conv1d formula.
    """
    if neuros.ndim != 3:
        raise ValueError(f"neuros must have shape (B, T, C), got {tuple(neuros.shape)}")
    if neuro_sizes.ndim != 1 or neuro_sizes.shape[0] != neuros.shape[0]:
        raise ValueError(
            "neuro_sizes must have shape (B,) matching neuros, got "
            f"{tuple(neuro_sizes.shape)} for batch size {neuros.shape[0]}"
        )
    if neuros.shape[1] == 0:
        raise ValueError("neuros must contain at least one time sample")

    # Lightning normally keeps the network as a child of the module, but
    # unwrapping makes this helper safe when it is called with DDP/DataParallel
    # wrappers as well.
    model = network
    while hasattr(model, "module") and isinstance(model.module, torch.nn.Module):
        model = model.module
    backbone = getattr(model, "backbone", None)
    if backbone is None:
        raise TypeError(
            "build_brainomni_temporal_mask requires a BrainOmni network"
        )

    patch_stride = int(backbone.patch_stride)
    if int(backbone.patch_size) <= 0 or patch_stride <= 0:
        raise ValueError(
            "unsupported BrainOmni patch geometry: "
            f"patch_size={backbone.patch_size}, patch_stride={patch_stride}"
        )

    device = neuros.device
    sizes = neuro_sizes.to(device=device, dtype=torch.long)
    valid_steps = compute_output_lens(model, sizes)

    batch_length = torch.tensor([neuros.shape[1]], device=device, dtype=torch.long)
    padded_steps = int(compute_output_lens(model, batch_length).item())
    padded_steps = max(padded_steps, 0)
    valid_steps = valid_steps.clamp(min=0, max=padded_steps)

    step_index = torch.arange(padded_steps, device=device).unsqueeze(0)
    valid = step_index < valid_steps.unsqueeze(1)
    return valid


def apply_jitter(
    data: torch.Tensor, seg: ns.segments.Segment, feat: BaseExtractor
) -> torch.Tensor:
    """Drop a random prefix (up to the pre-trigger window) to jitter sentence onset."""
    seg_start = seg.trigger.start - seg.start
    jitter_amount = np.random.uniform(0, seg_start * feat.frequency)
    return data[:, int(jitter_amount) :]


def compute_subject_cer_summary(rows: list[dict]) -> dict:
    """Mean sentence CER per subject, plus the overall mean across subjects.

    Aggregates the per-sentence ``CTC_CER`` already stored in ``rows``, so each
    subject value is exactly the average of that subject's sentence CERs and the
    across-subject value is the unweighted mean over subjects.
    """
    by_subject: dict[str, list[float]] = {}
    for r in rows:
        cer = r.get("CTC_CER")
        if cer is None or cer != cer:  # skip missing / NaN
            continue
        by_subject.setdefault(r.get("subject", ""), []).append(float(cer))
    subject_cer = {s: sum(vs) / len(vs) for s, vs in sorted(by_subject.items())}
    across_subject_cer = (
        sum(subject_cer.values()) / len(subject_cer) if subject_cer else float("nan")
    )
    return {"subject_cer": subject_cer, "across_subject_cer": across_subject_cer}


def compute_ctc_sample_metrics(
    typed_texts: list[str], ctc_texts: list[str]
) -> list[dict]:
    """Compute per-sentence CTC CER against the participant's typed target."""
    from torchmetrics.text import CharErrorRate

    cer_fn = CharErrorRate()
    rows: list[dict] = []
    for typed_text, ctc_text in zip(typed_texts, ctc_texts):
        row: dict[str, tp.Any] = {
            "typed_text": typed_text,
            "ctc_text": ctc_text,
            "CTC_CER": cer_fn([ctc_text], [typed_text]).item(),
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


class BrainOmniSensorMetadata(_ChannelPositions):
    """Return BrainOmni's six position/direction values and sensor type."""

    n_spatial_dims: tp.Literal[3] = 3

    def _compute_positions(self, ta) -> torch.Tensor:
        ch_locs = np.asarray(ta.header["ch_locs"])
        channel_idx = self._neuro._get_channels(ta.ch_names)
        out = torch.zeros((len(self._neuro._channels), 7), dtype=torch.float32)

        for source_idx, (target_idx, ch_type) in enumerate(
            zip(channel_idx, ta.ch_types)
        ):
            sensor_type = {"mag": 1, "grad": 2}[ch_type]
            direction_idx = 3 if ch_type == "mag" else 1
            loc = ch_locs[source_idx]
            out[target_idx, :3] = torch.from_numpy(loc[:3]).float()
            out[target_idx, 3:6] = torch.from_numpy(
                loc[3 * direction_idx : 3 * (direction_idx + 1)]
            ).float()
            out[target_idx, 6] = sensor_type
        return out


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
