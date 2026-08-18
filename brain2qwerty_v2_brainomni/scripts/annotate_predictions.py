# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Offline tool: add the per-subject / across-subject CER summary to an existing
``predictions_test.json`` written before the summary keys existed (plain array),
rewriting it in the current dict format (``rows`` + summary)."""

import argparse
import json
from pathlib import Path

from brain2qwerty_v2_brainomni.utils import compute_subject_cer_summary


def annotate(json_path: Path) -> None:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    rows = data["rows"] if isinstance(data, dict) else data
    output = {"rows": rows, **compute_subject_cer_summary(rows)}
    json_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Annotated {len(rows)} rows in {json_path}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        required=True,
        help="predictions JSON file, or a directory containing predictions_test.json files",
    )
    args = parser.parse_args(argv)

    path = Path(args.input)
    files = [path] if path.is_file() else sorted(path.rglob("predictions_test.json"))
    if not files:
        raise FileNotFoundError(f"no predictions_test.json found under {path}")
    for f in files:
        annotate(f)


if __name__ == "__main__":
    main()
