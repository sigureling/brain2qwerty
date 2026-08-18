# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Shared SpanishBCBL participant exclusions and recording-id merges."""

import pandas as pd


CONTROL_SUBJECTS = {
    "Pinet2024Meg/S11122024",
    "Pinet2024Meg/S12122024",
    "Pinet2024Meg/S26112024",
    "Pinet2024Meg/S27112024",
    "Pinet2024Meg/S28112024",
}
EXCLUDED_SUBJECTS = {"Pinet2024Meg/S23"}
SUBJECT_MERGE = {
    "Pinet2024Meg/S18": "Pinet2024Meg/S1",
    "Pinet2024Meg/S14": "Pinet2024Meg/S4",
    "Pinet2024Meg/S10": "Pinet2024Meg/S5",
    "Pinet2024Meg/S21": "Pinet2024Meg/S5",
}


def select_participants(events: pd.DataFrame) -> pd.DataFrame:
    """Keep 19 unique participants and merge duplicate recording ids."""
    keep = ~events["subject"].isin(CONTROL_SUBJECTS | EXCLUDED_SUBJECTS)
    events = events[keep].copy()
    events["subject"] = events["subject"].replace(SUBJECT_MERGE)
    return events
