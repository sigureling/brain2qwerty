# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import typing as tp

import numpy as np
import pandas as pd
from exca import MapInfra
from tqdm import tqdm

from neuralset.events.study import EventsTransform
from neuralset.events.transforms.utils import DeterministicSplitter
from neuralset.extractors.text import BaseText

from studies.spanishbcbl_participants import select_participants

from .utils import key_to_int

logger = logging.getLogger(__name__)


class EnglishBCBLPreprocessing(EventsTransform):
    """Clean the raw EnglishBCBL events and build the integer CTC target.

    Drops practice trials, normalises buttons (``<space>`` -> ``&``), removes
    unmapped/perception events, and stores the per-sentence ``typed_label`` (the
    space-separated key ids used as the CTC target) plus a stable ``button_UID``.
    """

    def _run(self, events: pd.DataFrame) -> pd.DataFrame:
        # drop the four practice trials of each session
        events = events[~events.trial_id.isin([0, 1, 2, 3])]
        events["sentence_UID"] = (
            events["trial_id"].astype(str) + "_" + events["timeline"].astype(str)
        )

        # normalise buttons: space -> "&", drop special/number tokens
        events = events[~events.button.isin(["<special>", "<number>"])]
        events.loc[events.button == "<space>", "button"] = "&"

        # drop keystrokes whose button is outside the CTC vocabulary
        unmapped = (events.type == "Keystroke") & ~events.button.isin(key_to_int)
        if unmapped.any():
            logger.info("Dropping %d keystroke(s) with unmapped buttons", unmapped.sum())
            events = events[~unmapped]

        # integer key id per keystroke
        button_events = events[events.type == "Keystroke"]
        events["typed_key_int"] = -1
        events["typed_key_int"] = events["typed_key_int"].astype(int)
        events.loc[button_events.index, "typed_key_int"] = button_events.button.map(
            key_to_int
        )

        # build the space-separated CTC target per sentence (skip near-empty ones)
        uids_to_drop: list[str] = []
        label_by_uid: dict[str, str] = {}
        for uid, group in tqdm(events.groupby("sentence_UID"), desc="Typed labels"):
            if "nan" in uid:
                continue
            buttons = group[group.type == "Keystroke"]
            if len(buttons) == 0 or len(buttons) < 0.5 * len(group):
                uids_to_drop.append(uid)
                continue
            typed_seq_ids = [int(i) for i in buttons.typed_key_int.values]
            assert sum(i == -1 for i in typed_seq_ids) == 0, f"Unmapped keys in {uid}"
            label_by_uid[uid] = " ".join(str(i) for i in typed_seq_ids)
        events["typed_label"] = events["sentence_UID"].map(label_by_uid)
        if uids_to_drop:
            logger.info(
                "Dropping %d sentences with too few keystrokes", len(uids_to_drop)
            )
            events = events[~events.sentence_UID.isin(uids_to_drop)]

        # keep MEG + keystrokes + sentences that carry a label; drop perception rows
        events = events[~(events.is_percep.eq(True) & (events.type != "Meg"))]
        events = events[events.type.isin(["Sentence", "Keystroke", "Meg"])]
        is_sentence = events["type"] == "Sentence"
        has_label = events["typed_label"].notna() & (events["typed_label"] != "")
        events = events[~is_sentence | has_label]

        # stable per-keystroke id (ordered within each sentence)
        keystroke_mask = events["type"] == "Keystroke"
        ks = events.loc[keystroke_mask].sort_values("start")
        if len(ks) > 0:
            counter = ks.groupby("sentence_UID").cumcount() + 1
            events.loc[keystroke_mask, "button_UID"] = (
                ks["sentence_UID"] + "_button_" + counter.astype(str)
            )
        return events


class SpanishBCBLV2Preprocessing(EventsTransform):
    """Adapt SpanishBCBL events to the sentence-level V2 CTC pipeline.

    The transform keeps V2's target vocabulary and sentence-windowed training
    contract while applying the participant and recording corrections used by
    V1. Unlike V1 preprocessing, it preserves the source Sentence events and
    expands their boundaries only when a retained keystroke falls outside them.
    """

    bad_sentence_uid: str = "65.0_Pinet2024Meg_subject-S1_session-1_task-block1"

    def _run(self, events: pd.DataFrame) -> pd.DataFrame:
        events = events.copy()
        events["type"] = events["type"].replace(
            {"Button": "Keystroke", "DetectedButton": "Keystroke"}
        )

        # SpanishBCBL has two practice trials per block.
        if "trial_id" in events.columns:
            events = events[~events["trial_id"].isin([0, 1])]

        # Reuse V1's participant exclusions and duplicate-recording merges, but
        # keep string subject ids: V2's LabelEncoder performs the integer coding.
        events = select_participants(events)

        if "sentence_UID" not in events.columns:
            events["sentence_UID"] = (
                events["trial_id"].astype(str)
                + "_"
                + events["timeline"].astype(str)
            )
        events = events[events["sentence_UID"] != self.bad_sentence_uid]

        task_type = events["type"].isin(["Sentence", "Keystroke"])
        if "is_percep" in events.columns:
            production = events["is_percep"].eq(False)  # noqa: E712
            keep = events["type"].eq("Meg") | (task_type & production)
        else:
            keep = events["type"].eq("Meg") | task_type
        events = events[keep].copy()

        # A sentence UID must resolve to exactly one non-empty source Sentence.
        sentences = events[events["type"] == "Sentence"]
        sentence_counts = sentences.groupby("sentence_UID").size()
        valid_uids = set(sentence_counts[sentence_counts == 1].index)
        text_ok = sentences["text"].notna() & (
            sentences["text"].astype(str).str.strip().ne("")
        )
        valid_uids &= set(sentences.loc[text_ok, "sentence_UID"])
        invalid_sentence_count = int(
            sentences["sentence_UID"].nunique() - len(valid_uids)
        )
        if invalid_sentence_count:
            logger.info(
                "Dropping %d SpanishBCBL sentence(s) without one source Sentence",
                invalid_sentence_count,
            )
        events = events[
            events["type"].eq("Meg") | events["sentence_UID"].isin(valid_uids)
        ].copy()

        sentence_mask = events["type"] == "Sentence"
        events.loc[sentence_mask, "text"] = (
            events.loc[sentence_mask, "text"].astype(str).str.lower().str.strip()
        )

        # Use V2's a-z + space vocabulary. Invalid/special keystrokes are not CTC
        # targets and therefore do not influence the adapted sentence boundary.
        keystroke_mask = events["type"] == "Keystroke"
        events.loc[keystroke_mask & events["button"].eq("<space>"), "button"] = "&"
        unmapped = keystroke_mask & ~events["button"].isin(key_to_int)
        if unmapped.any():
            logger.info(
                "Dropping %d SpanishBCBL keystroke(s) outside the V2 vocabulary",
                int(unmapped.sum()),
            )
            events = events[~unmapped].copy()

        # Sort before constructing targets so labels follow typing chronology.
        events = events.reset_index(drop=True)
        keystrokes = events[events["type"] == "Keystroke"].sort_values(
            ["sentence_UID", "start"], kind="stable"
        )
        events["typed_key_int"] = -1
        events["typed_key_int"] = events["typed_key_int"].astype(int)
        events.loc[keystrokes.index, "typed_key_int"] = (
            keystrokes["button"].map(key_to_int).astype(int)
        )
        keystrokes = events.loc[keystrokes.index]
        label_by_uid = (
            keystrokes.groupby("sentence_UID", sort=False)["typed_key_int"]
            .apply(lambda values: " ".join(str(int(value)) for value in values))
            .to_dict()
        )

        valid_target_uids = set(label_by_uid)
        empty_target_uids = valid_uids - valid_target_uids
        if empty_target_uids:
            logger.info(
                "Dropping %d SpanishBCBL sentence(s) without a valid V2 target",
                len(empty_target_uids),
            )
        events = events[
            events["type"].eq("Meg") | events["sentence_UID"].isin(valid_target_uids)
        ].copy()
        events["typed_label"] = events["sentence_UID"].map(label_by_uid)

        # Expand, but never shrink, source sentence windows to cover every valid
        # target keystroke. This preserves source timing whenever it is already
        # well formed and repairs the early-key cases present in SpanishBCBL.
        keystrokes = events[events["type"] == "Keystroke"].copy()
        fallback_key_stop = keystrokes["start"] + keystrokes["duration"]
        key_stop = (
            keystrokes["stop"].fillna(fallback_key_stop)
            if "stop" in keystrokes
            else fallback_key_stop
        )
        key_bounds = keystrokes.assign(_key_stop=key_stop).groupby("sentence_UID").agg(
            _key_start=("start", "min"), _key_stop=("_key_stop", "max")
        )
        sentence_mask = events["type"] == "Sentence"
        for idx in events.index[sentence_mask]:
            uid = events.at[idx, "sentence_UID"]
            original_start = float(events.at[idx, "start"])
            if "stop" in events.columns and pd.notna(events.at[idx, "stop"]):
                original_stop = float(events.at[idx, "stop"])
            else:
                original_stop = original_start + float(events.at[idx, "duration"])
            start = min(original_start, float(key_bounds.at[uid, "_key_start"]))
            stop = max(original_stop, float(key_bounds.at[uid, "_key_stop"]))
            events.at[idx, "start"] = start
            events.at[idx, "duration"] = stop - start
            if "stop" in events.columns:
                events.at[idx, "stop"] = stop

        # Stable per-keystroke id, ordered within sentence.
        keystrokes = events[events["type"] == "Keystroke"].sort_values(
            ["sentence_UID", "start"], kind="stable"
        )
        counter = keystrokes.groupby("sentence_UID", sort=False).cumcount() + 1
        events.loc[keystrokes.index, "button_UID"] = (
            keystrokes["sentence_UID"].astype(str)
            + "_button_"
            + counter.astype(str)
        )
        return events.reset_index(drop=True)


class Brain2QwertyV2Splitter(EventsTransform):
    """Train/val/test split by unique sentence text (no text leakage).

    Each unique Sentence ``text`` is hashed to a split via ``DeterministicSplitter``
    and the assignment is propagated to every row sharing the same ``sentence_UID``.
    """

    deterministic_splitter: DeterministicSplitter

    def _run(self, events: pd.DataFrame) -> pd.DataFrame:
        sents = events[events["type"] == "Sentence"].copy()
        text_to_split = {
            text: self.deterministic_splitter(str(text))
            for text in sents["text"].dropna().unique()
        }
        uid_to_split = {
            row["sentence_UID"]: text_to_split[row["text"]]
            for _, row in sents.iterrows()
            if pd.notna(row["text"]) and row["text"] in text_to_split
        }
        events["split"] = events["sentence_UID"].map(uid_to_split)
        return events


class SentenceKeySeq(BaseText):
    """Turn each sentence into the integer character sequence the CTC head predicts.

    Two ways to build that sequence:
    - ``mode="typed_label"`` uses what the participant actually typed (the integer
      sequence precomputed per sentence in ``event.extra["typed_label"]``).
    - ``mode="sentence_text"`` uses the reference sentence text: lowercase it, map
      spaces to ``&`` and each character to its index via ``key_to_int``.
    """

    event_types: str | tuple[str, ...] = "Sentence"
    mode: tp.Literal["typed_label", "sentence_text"] = "typed_label"

    infra: MapInfra = MapInfra(version="v6")

    @infra.apply(
        item_uid=lambda event: _sentence_key_uid(event),
        exclude_from_cache_uid="method:_exclude_from_cache_uid",
        cache_type="MemmapArrayFile",
    )
    def _get_data(self, events: list[tp.Any]) -> tp.Iterator[np.ndarray]:
        if len(events) > 1:
            events = tqdm(events, desc="Sequence labels")  # type: ignore
        for event in events:
            yield self.get_embedding(event)

    def get_embedding(self, event) -> np.ndarray:
        if self.mode == "typed_label":
            return np.array(
                [int(i) for i in event.extra["typed_label"].split(" ")], dtype=np.int32
            )
        text = str(event.text).lower().replace(" ", "&")
        seq = [key_to_int[ch] for ch in text if ch in key_to_int]
        if not seq:
            raise ValueError(f"Empty target for text={event.text!r}")
        return np.array(seq, dtype=np.int32)


def _sentence_key_uid(event: tp.Any) -> str:
    """Cache identity including the actual typed target, not reference text only."""
    typed_label = event.extra.get("typed_label", "")
    return f"{event.text!s}\0{typed_label!s}"
