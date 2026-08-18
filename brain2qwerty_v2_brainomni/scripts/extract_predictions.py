# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import os

import Levenshtein
import pandas as pd

HEADLINE_METRIC = "CTC_CER"


def compute_cer(true: str, pred: str) -> float:
    true, pred = str(true), str(pred)
    if len(true) == 0:
        return float("nan")
    return Levenshtein.distance(true, pred) / len(true)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute CTC CER from the saved typed target and decoded text."""
    df = df.copy()
    df[HEADLINE_METRIC] = df.apply(
        lambda r: compute_cer(r["typed_text"], r["ctc_text"]), axis=1
    )
    return df


def print_summary(df: pd.DataFrame) -> None:
    cols = [HEADLINE_METRIC]
    has_subject = "subject" in df.columns and df["subject"].notna().any()
    group = df["subject"] if has_subject else pd.Series("all", index=df.index)

    header = f"{'Subject':<22} {'N':>6} " + " ".join(
        f"{c:>9}" for c in cols
    )
    print(f"\n{header}\n" + "-" * len(header))
    for subject in sorted(group.unique()):
        sdf = df[group == subject]
        vals = " ".join(f"{sdf[c].mean():>9.3f}" for c in cols)
        print(f"{str(subject):<22} {len(sdf):>6} {vals}")
    print("-" * len(header))

    sent = " ".join(f"{df[c].mean():>9.3f}" for c in cols)
    print(f"{'Overall (sentence)':<22} {len(df):>6} {sent}")
    if has_subject:
        per_subj = df.groupby("subject")[cols].mean()
        macro = " ".join(f"{per_subj[c].mean():>9.3f}" for c in cols)
        n_subj = df["subject"].nunique()
        print(f"{'Overall (per-subject)':<22} {n_subj:>6} {macro}")
        m = per_subj[HEADLINE_METRIC]
        sem = m.std(ddof=1) / (len(m) ** 0.5)
        print(
            f"\n==> {HEADLINE_METRIC} = {m.mean():.1%} +/- {sem:.1%} (SEM) "
            f"across {n_subj} subjects"
        )
    print()


def main(argv: list[str] | None = None) -> None:
    """Summarize the CTC prediction JSON written by ``PredictionJSONCallback``."""
    parser = argparse.ArgumentParser(description="Summarize CTC predictions JSON")
    parser.add_argument("--input", required=True, help="predictions JSON or its directory")
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--output", default=None, help="optional per-subject summary CSV")
    args = parser.parse_args(argv)

    json_path = (
        os.path.join(args.input, f"predictions_{args.split}.json")
        if os.path.isdir(args.input)
        else args.input
    )
    if not os.path.exists(json_path):
        raise FileNotFoundError(json_path)

    print(f"Reading {args.split} predictions from {json_path}")
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    rows = data["rows"] if isinstance(data, dict) else data
    df = summarize(pd.DataFrame(rows))
    n_subj = df["subject"].nunique() if "subject" in df.columns else 1
    print(
        f"Scoring {len(df)} sentences across {n_subj} subjects "
        f"(headline metric: {HEADLINE_METRIC})"
    )
    print_summary(df)
    if args.output and "subject" in df.columns:
        df.groupby("subject")[[HEADLINE_METRIC]].mean().to_csv(args.output)
        print(f"Saved per-subject summary to {args.output}")


if __name__ == "__main__":
    main()
