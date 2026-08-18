# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace

import pandas as pd

from brain2qwerty_v2.transforms import (
    Brain2QwertyV2Splitter,
    SpanishBCBLV2Preprocessing,
    _sentence_key_uid,
)
from brain2qwerty_v2.utils import key_to_int


def _row(**kwargs) -> dict:
    row = {
        "type": "Button",
        "trial_id": 2.0,
        "timeline": "Pinet2024Meg_subject-S18_session-1_task-block1",
        "subject": "Pinet2024Meg/S18",
        "sentence_UID": "2.0_Pinet2024Meg_subject-S18_session-1_task-block1",
        "button": "h",
        "text": "h",
        "start": 9.2,
        "duration": 0.1,
        "stop": 9.3,
        "is_percep": False,
    }
    row.update(kwargs)
    return row


def _synthetic_events() -> pd.DataFrame:
    uid = "2.0_Pinet2024Meg_subject-S18_session-1_task-block1"
    repeated_uid = "3.0_Pinet2024Meg_subject-S2_session-1_task-block1"
    empty_uid = "4.0_Pinet2024Meg_subject-S2_session-1_task-block1"
    rows = [
        # MEG rows survive and S18 is merged to S1.
        _row(
            type="Meg",
            trial_id=float("nan"),
            sentence_UID=None,
            button=None,
            text=None,
            start=0.0,
            duration=30.0,
            stop=30.0,
            is_percep=None,
        ),
        # Source window [10, 11] must expand to [8.9, 11.6].
        _row(
            type="Sentence",
            button=None,
            text=" Hola Mundo ",
            start=10.0,
            duration=1.0,
            stop=11.0,
        ),
        _row(button="a", text="a", start=11.5, stop=11.6),
        _row(button="o", text="o", start=8.9, stop=9.0),
        _row(button="h", text="h", start=9.2, stop=9.3),
        _row(button="<space>", text="<space>", start=10.5, stop=10.6),
        # Invalid keys are removed and do not expand the sentence boundary.
        _row(button="<special>", text="<special>", start=7.0, stop=7.1),
        # Perception events are removed before target construction.
        _row(button="z", text="z", start=9.4, stop=9.5, is_percep=True),
        # Same reference sentence in another recording, with a different target.
        _row(
            type="Sentence",
            trial_id=3.0,
            timeline="Pinet2024Meg_subject-S2_session-1_task-block1",
            subject="Pinet2024Meg/S2",
            sentence_UID=repeated_uid,
            button=None,
            text="hola mundo",
            start=20.0,
            duration=2.0,
            stop=22.0,
        ),
        _row(
            trial_id=3.0,
            timeline="Pinet2024Meg_subject-S2_session-1_task-block1",
            subject="Pinet2024Meg/S2",
            sentence_UID=repeated_uid,
            button="x",
            text="x",
            start=20.5,
            stop=20.6,
        ),
        # A sentence containing only invalid keys has no V2 target and is dropped.
        _row(
            type="Sentence",
            trial_id=4.0,
            timeline="Pinet2024Meg_subject-S2_session-1_task-block1",
            subject="Pinet2024Meg/S2",
            sentence_UID=empty_uid,
            button=None,
            text="solo especial",
            start=23.0,
            duration=1.0,
            stop=24.0,
        ),
        _row(
            trial_id=4.0,
            timeline="Pinet2024Meg_subject-S2_session-1_task-block1",
            subject="Pinet2024Meg/S2",
            sentence_UID=empty_uid,
            button="<number>",
            text="<number>",
            start=23.5,
            stop=23.6,
        ),
        # Practice and excluded-participant rows are discarded.
        _row(trial_id=0.0, sentence_UID="practice"),
        _row(
            type="Meg",
            subject="Pinet2024Meg/S23",
            timeline="excluded",
            sentence_UID=None,
            button=None,
            text=None,
            start=0.0,
            duration=1.0,
            stop=1.0,
            is_percep=None,
        ),
    ]
    return pd.DataFrame(rows)


def test_spanish_v2_preprocessing_contract():
    out = SpanishBCBLV2Preprocessing().run(_synthetic_events())

    assert set(out["type"]) <= {"Meg", "Sentence", "Keystroke"}
    assert not out["trial_id"].isin([0.0, 1.0]).any()
    assert "Pinet2024Meg/S23" not in set(out["subject"])
    assert "Pinet2024Meg/S1" in set(out["subject"])
    assert "Pinet2024Meg/S18" not in set(out["subject"])

    sentences = out[out["type"] == "Sentence"]
    assert len(sentences) == sentences["sentence_UID"].nunique() == 2
    assert set(sentences["text"]) == {"hola mundo"}

    first = sentences[sentences["sentence_UID"].str.startswith("2.0_")].iloc[0]
    assert first["start"] == 8.9
    assert first["stop"] == 11.6
    assert first["duration"] == 11.6 - 8.9

    first_keys = out[
        (out["type"] == "Keystroke")
        & out["sentence_UID"].eq(first["sentence_UID"])
    ].sort_values("start")
    assert first_keys["button"].tolist() == ["o", "h", "&", "a"]
    expected_ids = [key_to_int[key] for key in ["o", "h", "&", "a"]]
    assert first_keys["typed_key_int"].tolist() == expected_ids
    assert first["typed_label"] == " ".join(map(str, expected_ids))
    assert first_keys["button_UID"].is_unique
    assert first_keys["button_UID"].str.endswith(("1", "2", "3", "4")).all()

    # Every retained key is inside its source/adapted Sentence interval.
    bounds = sentences.set_index("sentence_UID")[["start", "stop"]]
    for row in out[out["type"] == "Keystroke"].itertuples():
        assert bounds.at[row.sentence_UID, "start"] <= row.start
        assert bounds.at[row.sentence_UID, "stop"] >= row.stop


def test_v2_splitter_groups_repeated_reference_text_deterministically():
    events = SpanishBCBLV2Preprocessing().run(_synthetic_events())
    splitter = Brain2QwertyV2Splitter(
        deterministic_splitter={
            "ratios": {"train": 0.8, "val": 0.1, "test": 0.1}
        }
    )
    out = splitter.run(events)
    out_again = splitter.run(events.copy())

    sentence_splits = out[out["type"] == "Sentence"]["split"]
    assert sentence_splits.nunique() == 1
    pd.testing.assert_series_equal(out["split"], out_again["split"])


def test_sentence_key_cache_uid_includes_actual_target():
    first = SimpleNamespace(text="hola mundo", extra={"typed_label": "1 2"})
    second = SimpleNamespace(text="hola mundo", extra={"typed_label": "1 3"})
    assert _sentence_key_uid(first) != _sentence_key_uid(second)

