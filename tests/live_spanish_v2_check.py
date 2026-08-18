# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Opt-in real-data check for the SpanishBCBL adapter used by V2.

Run explicitly (this is intentionally not part of the default pytest suite):

    PYTHONPATH=. python tests/live_spanish_v2_check.py /path/to/SpanishBCBL
"""

import os
import sys

from brain2qwerty_v2.transforms import (
    Brain2QwertyV2Splitter,
    SpanishBCBLV2Preprocessing,
)
from studies.spanishbcbl import Pinet2024Meg


def main(path: str) -> None:
    study = Pinet2024Meg(
        path=path,
        query="timeline_index == 0",
        infra_timelines={"cluster": None},
    )
    events = SpanishBCBLV2Preprocessing().run(study.run())
    events = Brain2QwertyV2Splitter(
        deterministic_splitter={
            "ratios": {"train": 0.8, "val": 0.1, "test": 0.1}
        }
    ).run(events)

    sentences = events[events["type"] == "Sentence"].set_index("sentence_UID")
    keystrokes = events[events["type"] == "Keystroke"]
    joined = keystrokes.join(
        sentences[["start", "stop"]], on="sentence_UID", rsuffix="_sentence"
    )
    uncovered = (joined["start"] < joined["start_sentence"]) | (
        joined["stop"] > joined["stop_sentence"]
    )

    assert len(sentences) > 0, "no production Sentence events"
    assert sentences.index.is_unique, "sentence_UID is not unique"
    assert sentences["typed_label"].notna().all(), "empty CTC target"
    assert not uncovered.any(), "a retained keystroke is outside its Sentence window"
    assert sentences["split"].notna().all(), "Sentence without split"
    print("event types:", events["type"].value_counts().to_dict())
    print("sentence splits:", sentences["split"].value_counts().to_dict())
    print("LIVE SPANISH V2 CHECK PASSED")


if __name__ == "__main__":
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else os.getenv(
        "BRAIN2QWERTY_STUDIES", ""
    )
    if not dataset_path:
        raise SystemExit("pass the SpanishBCBL path or set BRAIN2QWERTY_STUDIES")
    main(dataset_path)
