#!/usr/bin/env python3
"""Run the two-stage BrainOmni sweep on both rjob clusters.

The launcher deliberately uses the shared result directory as its only source
of completion information.  It never asks rjob for job status: a stage-A
experiment is complete only when its ``predictions_test.json`` is complete and
contains a finite ``across_subject_cer`` value.

Typical invocations::

    python script/brain2qwerty_v2_brainomni/rjob_train.py --auto
    python script/brain2qwerty_v2_brainomni/rjob_train.py --dry-run
    python script/brain2qwerty_v2_brainomni/rjob_train.py \
        --resume /path/to/workflow.json

``--auto`` submits the four stage-A head/aux configurations to both clusters,
waits locally for their prediction files, selects the best head/aux pair, and
then submits the twelve stage-B LR/WD configurations to both clusters.  Stage
B is intentionally not monitored by this process.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT = REPO / "script/brain2qwerty_v2_brainomni/train.sh"

# These values match the defaults in train.sh.  They are also sent explicitly
# to rjob so that both clusters use exactly the same shared paths.
DEFAULT_CACHE = REPO / ".cache" / "spanishbcbl_meg_v2_brainomni"
DEFAULT_MANIFEST_NAME = "workflow.json"

IMAGE = (
    "registry.h.pjlab.org.cn/ailab-brainllm/"
    "xiaoqinfan-workspace:brainomni-deepspeed-20260727"
)
STUDIES_MOUNT = "gpfs://gpfs1/xiaoqinfan:/mnt/shared-storage-user/xiaoqinfan"
WORKSPACE_MOUNT = "gpfs://gpfs1/brainllm-share:/mnt/shared-storage-user/brainllm-share"

DEFAULT_SEED = 42
STAGE_A_LR = 8e-4
STAGE_A_WEIGHT_DECAY = 1e-3

# The stage-A experiment is a 2 x 2 logical comparison.  Keep the baseline
# first: it is the deterministic representative used only for --dry-run's
# stage-B command preview, where no measured best result exists yet.
STAGE_A_SETTINGS = (
    # name, aux_prediction, classifier_head, lr, weight_decay
    ("baseline", True, "rms_linear", STAGE_A_LR, STAGE_A_WEIGHT_DECAY),
    ("linear_no_aux", False, "rms_linear", STAGE_A_LR, STAGE_A_WEIGHT_DECAY),
    ("conv_head_aux", True, "rms_conv", STAGE_A_LR, STAGE_A_WEIGHT_DECAY),
    ("conv_head_no_aux", False, "rms_conv", STAGE_A_LR, STAGE_A_WEIGHT_DECAY),
)

# Twelve stage-B experiments: three learning rates crossed with four weight
# decays.  The baseline (8e-4, 1e-3) is included to make the comparison
# reproducible and to provide a direct reference for the stage-A result.
STAGE_B_LRS = (4e-4, 8e-4, 1.6e-3)
STAGE_B_WEIGHT_DECAYS = (1e-4, 5e-4, 1e-3, 2e-3)


@dataclass(frozen=True)
class Setting:
    """The four model/optimiser values passed to train.sh."""

    name: str
    aux_prediction: bool
    classifier_head: str
    lr: float
    weight_decay: float


@dataclass(frozen=True)
class Cluster:
    """rjob routing values for one execution cluster."""

    name: str
    namespace: str
    charged_group: str


CLUSTERS = (
    Cluster("brainllm", "ailab-brainllm", "brainllm_gpu"),
    Cluster("speechllm", "ailab-speechllm", "speechllm_gpu"),
)


def _stage_a_settings() -> tuple[Setting, ...]:
    return tuple(Setting(*values) for values in STAGE_A_SETTINGS)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slug(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-._")
    if not value:
        raise ValueError("setting name must contain at least one alphanumeric character")
    return value


def _short_float(value: float) -> str:
    """Keep floating-point values compact and unambiguous in run names."""
    if not math.isfinite(value):
        raise ValueError(f"numeric run-name value must be finite, got {value!r}")
    mantissa, exponent = f"{value:.6e}".split("e")
    mantissa = mantissa.rstrip("0").rstrip(".")
    exponent = int(exponent)
    return mantissa if exponent == 0 else f"{mantissa}e{exponent}"


def _experiment_name(
    setting: Setting,
    seed: int,
    *,
    stage: str = "stage-a",
    prefix: str | None = None,
) -> str:
    """Build the shared ``EXPERIMENT_NAME`` used by both clusters."""
    label = setting.name if prefix is None else f"{prefix}-{setting.name}"
    return (
        f"{_slug(stage)}-{_slug(label)}-lr{_short_float(setting.lr)}"
        f"-wd{_short_float(setting.weight_decay)}-seed{seed}"
    )


def _cache_path(value: str | None) -> Path:
    raw = value or os.environ.get("BRAIN2QWERTY_CACHE")
    path = Path(raw) if raw else DEFAULT_CACHE
    return path.expanduser().resolve()


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _job_command(
    setting: Setting,
    seed: int,
    timestamp: str,
    cluster: Cluster = CLUSTERS[0],
    cache: Path = DEFAULT_CACHE,
    *,
    experiment_name: str | None = None,
) -> list[str]:
    """Create one concrete ``rjob submit`` command.

    ``experiment_name`` is optional for compatibility with the old helper and
    is supplied by the workflow so stage A/B can use distinct names while the
    two cluster commands for one logical experiment remain identical in every
    training-related environment variable.
    """
    experiment_name = experiment_name or _experiment_name(setting, seed)
    result_dir = cache / "results" / experiment_name
    job_name = f"brain2qwerty-v2-brainomni-{experiment_name}-{cluster.name}-{timestamp}"
    return [
        "rjob",
        "submit",
        f"--name={job_name}",
        "--gpu=4",
        "--memory=480000",
        "--cpu=64",
        f"--charged-group={cluster.charged_group}",
        f"--namespace={cluster.namespace}",
        "--private-machine=group",
        "--store-host-nvme",
        "--custom-resources",
        "rdma/mlnx_shared=8",
        "--custom-resources",
        "brainpp.cn/fuse=1",
        "--custom-resources",
        "mellanox.com/mlnx_rdma=1",
        "-e",
        f"GROUP={cluster.charged_group}",
        "-e",
        "DISTRIBUTED_JOB=true",
        "-e",
        f"SEED={seed}",
        "-e",
        f"EXPERIMENT_NAME={experiment_name}",
        "-e",
        f"BRAIN2QWERTY_CACHE={cache}",
        "-e",
        f"BRAIN2QWERTY_RESULTS={result_dir}",
        "-e",
        f"AUX_PREDICTION={str(setting.aux_prediction).lower()}",
        "-e",
        f"CLASSIFIER_HEAD={setting.classifier_head}",
        "-e",
        f"LR={setting.lr}",
        "-e",
        f"WEIGHT_DECAY={setting.weight_decay}",
        "--termination-grace-period-seconds",
        "600",
        f"--image={IMAGE}",
        f"--mount={WORKSPACE_MOUNT}",
        f"--mount={STUDIES_MOUNT}",
        "--",
        "bash",
        "-exc",
        str(TRAIN_SCRIPT),
    ]


def _experiment_entry(
    *,
    setting: Setting,
    seed: int,
    cache: Path,
    stage: str,
    timestamp: str,
    extra_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    stage_name = stage.lower()
    experiment_name = _experiment_name(setting, seed, stage=stage_name)
    command_records: dict[str, dict[str, Any]] = {}
    for cluster in CLUSTERS:
        command = _job_command(
            setting,
            seed,
            timestamp,
            cluster,
            cache,
            experiment_name=experiment_name,
        )
        command_records[cluster.name] = {
            "cluster": cluster.name,
            "namespace": cluster.namespace,
            "charged_group": cluster.charged_group,
            "job_name": next(
                arg.split("=", 1)[1]
                for arg in command
                if arg.startswith("--name=")
            ),
            "command": command,
            "status": "pending",
            "success": None,
        }
    config = {
        "name": setting.name,
        "aux_prediction": setting.aux_prediction,
        "classifier_head": setting.classifier_head,
        "lr": setting.lr,
        "weight_decay": setting.weight_decay,
        "seed": seed,
        "stage": stage.upper(),
    }
    if extra_config:
        config.update(extra_config)
    return {
        "name": experiment_name,
        "config": config,
        "result_path": str(cache / "results" / experiment_name / "predictions_test.json"),
        "submissions": command_records,
        "result": None,
    }


def _new_workflow(cache: Path, seed: int, manifest_path: Path) -> dict[str, Any]:
    stage_a_entries = [
        _experiment_entry(
            setting=setting,
            seed=seed,
            cache=cache,
            stage="stage-a",
            timestamp=_timestamp(),
        )
        for setting in _stage_a_settings()
    ]
    return {
        "schema_version": 1,
        "workflow": "brain2qwerty-v2-brainomni-dual-cluster",
        "created_at": _now(),
        "updated_at": _now(),
        "manifest_path": str(manifest_path),
        "cache": str(cache),
        "results_root": str(cache / "results"),
        "seed": seed,
        "clusters": [asdict(cluster) for cluster in CLUSTERS],
        "stage_a": {
            "status": "pending",
            "experiments": stage_a_entries,
            "best": None,
        },
        "best": None,
        "best_head_aux": None,
        "stage_b": {
            "status": "not_started",
            "experiments": [],
        },
    }


def _save_manifest(workflow: dict[str, Any], path: Path) -> None:
    """Persist a manifest atomically after each meaningful state change."""
    path.parent.mkdir(parents=True, exist_ok=True)
    workflow["updated_at"] = _now()
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(workflow, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        workflow = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(f"manifest does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"manifest is not valid JSON: {path}: {exc}") from exc
    if not isinstance(workflow, dict):
        raise SystemExit(f"manifest must contain a JSON object: {path}")
    if workflow.get("schema_version") != 1:
        raise SystemExit(f"unsupported manifest schema_version: {workflow.get('schema_version')!r}")
    if workflow.get("workflow") != "brain2qwerty-v2-brainomni-dual-cluster":
        raise SystemExit("manifest is not a dual-cluster BrainOmni workflow")
    return workflow


def _submission_succeeded(record: dict[str, Any] | None) -> bool:
    if not record:
        return False
    return record.get("status") == "submitted" or record.get("success") is True


def _all_submissions_succeeded(entry: dict[str, Any]) -> bool:
    submissions = entry.get("submissions", {})
    return all(
        _submission_succeeded(submissions.get(cluster.name)) for cluster in CLUSTERS
    )


def _submit_entries(
    entries: list[dict[str, Any]],
    *,
    stage: str,
    workflow: dict[str, Any],
    manifest_path: Path,
) -> bool:
    """Submit missing cluster records and return whether all succeeded.

    A failed submission is recorded and does not prevent the other cluster or
    the remaining experiments from being attempted.  This is what makes a
    later ``--resume`` able to fill only the missing stage-B records.
    """
    for entry in entries:
        for cluster in CLUSTERS:
            record = entry.setdefault("submissions", {}).setdefault(
                cluster.name,
                {"cluster": cluster.name, "status": "pending", "success": None},
            )
            if _submission_succeeded(record):
                continue
            command = record.get("command")
            if not isinstance(command, list) or not command:
                record.update(
                    {
                        "status": "failed",
                        "success": False,
                        "error": "submission command missing from manifest",
                        "failed_at": _now(),
                    }
                )
                _save_manifest(workflow, manifest_path)
                continue
            print(
                f"[rjob] submitting {stage} {entry.get('name', '<unknown>')} "
                f"to {cluster.name}",
                flush=True,
            )
            time.sleep(5)
            try:
                subprocess.run(command, check=True)
            except Exception as exc:  # keep the other cluster submissions going
                record.update(
                    {
                        "status": "failed",
                        "success": False,
                        "error": f"{type(exc).__name__}: {exc}",
                        "failed_at": _now(),
                    }
                )
                print(f"[rjob] submission failed: {exc}", flush=True)
            else:
                record.update(
                    {
                        "status": "submitted",
                        "success": True,
                        "submitted_at": _now(),
                    }
                )
            _save_manifest(workflow, manifest_path)
    return all(_all_submissions_succeeded(entry) for entry in entries)


def _prediction_summary(path: Path) -> tuple[bool, dict[str, Any] | None, str]:
    """Validate one local prediction file without relying on rjob metadata."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return False, None, "file does not exist"
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return False, None, f"JSON is not complete/readable: {exc}"

    if not isinstance(payload, dict):
        return False, None, "top-level JSON value is not an object"
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        return False, None, "rows is missing or empty"
    metric = payload.get("across_subject_cer")
    if isinstance(metric, bool) or not isinstance(metric, (int, float)):
        return False, None, "across_subject_cer is not numeric"
    metric = float(metric)
    if not math.isfinite(metric):
        return False, None, "across_subject_cer is not finite"
    return (
        True,
        {
            "rows": len(rows),
            "across_subject_cer": metric,
            "path": str(path),
        },
        "valid",
    )


def _wait_for_stage_a(
    workflow: dict[str, Any],
    manifest_path: Path,
    *,
    poll_interval: float,
    timeout_hours: float,
) -> bool:
    """Wait only for stage-A local prediction JSON files."""
    entries = workflow["stage_a"]["experiments"]
    deadline = time.monotonic() + timeout_hours * 3600.0
    last_reason: dict[str, str] = {}

    workflow["stage_a"]["status"] = "waiting_for_local_results"
    _save_manifest(workflow, manifest_path)
    while True:
        all_valid = True
        for entry in entries:
            path = Path(entry["result_path"])
            valid, summary, reason = _prediction_summary(path)
            if valid:
                old_result = entry.get("result")
                result = {
                    "status": "valid",
                    "path": str(path),
                    "rows": summary["rows"],
                    "across_subject_cer": summary["across_subject_cer"],
                    "checked_at": _now(),
                }
                entry["result"] = result
                if old_result is None or old_result.get("status") != "valid":
                    print(
                        f"[stage A] valid result for {entry['name']}: "
                        f"across_subject_cer={summary['across_subject_cer']}",
                        flush=True,
                    )
            else:
                all_valid = False
                previous = last_reason.get(entry["name"])
                if previous != reason:
                    print(f"[stage A] waiting for {entry['name']}: {reason}", flush=True)
                    last_reason[entry["name"]] = reason
                entry["result"] = {
                    "status": "pending",
                    "path": str(path),
                    "reason": reason,
                    "checked_at": _now(),
                }

        _save_manifest(workflow, manifest_path)
        if all_valid:
            workflow["stage_a"]["status"] = "complete"
            _save_manifest(workflow, manifest_path)
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            workflow["stage_a"]["status"] = "timed_out"
            _save_manifest(workflow, manifest_path)
            return False
        time.sleep(min(poll_interval, remaining))


def _choose_best(workflow: dict[str, Any]) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for entry in workflow["stage_a"]["experiments"]:
        result = entry.get("result") or {}
        metric = result.get("across_subject_cer")
        if result.get("status") != "valid" or not isinstance(metric, (int, float)):
            continue
        if isinstance(metric, bool) or not math.isfinite(float(metric)):
            continue
        candidates.append(entry)
    if len(candidates) != len(workflow["stage_a"]["experiments"]):
        raise RuntimeError("cannot select a stage-A winner before all results are valid")
    winner = min(
        candidates,
        key=lambda entry: (
            float(entry["result"]["across_subject_cer"]),
            entry["name"],
        ),
    )
    config = winner["config"]
    best = {
        "experiment_name": winner["name"],
        "result_path": winner["result_path"],
        "setting_name": config["name"],
        "classifier_head": config["classifier_head"],
        "aux_prediction": bool(config["aux_prediction"]),
        "across_subject_cer": float(winner["result"]["across_subject_cer"]),
    }
    workflow["stage_a"]["best"] = best
    workflow["best"] = best
    workflow["best_head_aux"] = {
        "classifier_head": best["classifier_head"],
        "aux_prediction": best["aux_prediction"],
        "source_experiment": best["experiment_name"],
        "across_subject_cer": best["across_subject_cer"],
    }
    return best


def _stage_b_entries(
    *,
    best: dict[str, Any],
    seed: int,
    cache: Path,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    source = _slug(str(best["setting_name"]))
    for lr in STAGE_B_LRS:
        for weight_decay in STAGE_B_WEIGHT_DECAYS:
            setting = Setting(
                name=f"best_{source}",
                aux_prediction=bool(best["aux_prediction"]),
                classifier_head=str(best["classifier_head"]),
                lr=lr,
                weight_decay=weight_decay,
            )
            entries.append(
                _experiment_entry(
                    setting=setting,
                    seed=seed,
                    cache=cache,
                    stage="stage-b",
                    timestamp=_timestamp(),
                    extra_config={
                        "selected_from": best["experiment_name"],
                        "selected_across_subject_cer": best["across_subject_cer"],
                    },
                )
            )
    return entries


def _start_stage_b(
    workflow: dict[str, Any],
    manifest_path: Path,
    *,
    best: dict[str, Any],
) -> bool:
    stage_b = workflow.get("stage_b")
    if not isinstance(stage_b, dict):
        stage_b = {"status": "not_started", "experiments": []}
        workflow["stage_b"] = stage_b
    if not stage_b.get("experiments"):
        stage_b["source_stage_a"] = best["experiment_name"]
        stage_b["selected_head_aux"] = {
            "classifier_head": best["classifier_head"],
            "aux_prediction": best["aux_prediction"],
        }
        stage_b["grid"] = {
            "lr": list(STAGE_B_LRS),
            "weight_decay": list(STAGE_B_WEIGHT_DECAYS),
        }
        stage_b["experiments"] = _stage_b_entries(
            best=best,
            seed=int(workflow["seed"]),
            cache=Path(workflow["cache"]),
        )
    stage_b["status"] = "submitting"
    _save_manifest(workflow, manifest_path)
    complete = _submit_entries(
        stage_b["experiments"],
        stage="stage B",
        workflow=workflow,
        manifest_path=manifest_path,
    )
    stage_b["status"] = "submitted" if complete else "partial"
    stage_b["submitted_at"] = _now() if complete else None
    stage_b["failed_experiments"] = [
        entry["name"]
        for entry in stage_b["experiments"]
        if not _all_submissions_succeeded(entry)
    ]
    _save_manifest(workflow, manifest_path)
    if complete:
        print(
            f"[stage B] submitted {len(stage_b['experiments'])} experiments to "
            f"{len(CLUSTERS)} clusters; monitoring ends here.",
            flush=True,
        )
    else:
        print(
            "[stage B] some submissions failed; use --resume with the manifest "
            "to retry only missing cluster records.",
            flush=True,
        )
    return complete


def _default_manifest_path(cache: Path) -> Path:
    return cache / DEFAULT_MANIFEST_NAME


def _run_auto(args: argparse.Namespace) -> int:
    cache = _cache_path(args.cache)
    manifest_path = Path(args.manifest).expanduser().resolve() if args.manifest else _default_manifest_path(cache)
    if manifest_path.exists():
        raise SystemExit(
            f"manifest already exists: {manifest_path}; use --resume to continue it"
        )
    workflow = _new_workflow(cache, args.seed, manifest_path)
    _save_manifest(workflow, manifest_path)
    print(f"[workflow] manifest: {manifest_path}", flush=True)

    stage_a_complete = _submit_entries(
        workflow["stage_a"]["experiments"],
        stage="stage A",
        workflow=workflow,
        manifest_path=manifest_path,
    )
    workflow["stage_a"]["submissions_complete"] = stage_a_complete
    _save_manifest(workflow, manifest_path)

    if not _wait_for_stage_a(
        workflow,
        manifest_path,
        poll_interval=args.poll_interval,
        timeout_hours=args.timeout_hours,
    ):
        print(
            f"[workflow] stage A did not produce four valid prediction files; "
            f"stage B was not submitted. Resume with {manifest_path} if needed.",
            flush=True,
        )
        return 1

    best = _choose_best(workflow)
    _save_manifest(workflow, manifest_path)
    print(
        f"[stage A] best={best['experiment_name']} "
        f"head={best['classifier_head']} aux={best['aux_prediction']} "
        f"across_subject_cer={best['across_subject_cer']}",
        flush=True,
    )
    return 0 if _start_stage_b(workflow, manifest_path, best=best) else 1


def _manifest_cache(workflow: dict[str, Any], requested: str | None) -> Path:
    try:
        cache = Path(workflow["cache"]).expanduser().resolve()
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit("manifest does not contain a valid cache path") from exc
    if requested is not None and _cache_path(requested) != cache:
        raise SystemExit(
            f"--cache does not match the manifest cache: {_cache_path(requested)} != {cache}"
        )
    return cache


def _run_resume(args: argparse.Namespace) -> int:
    manifest_path = Path(args.resume).expanduser().resolve()
    workflow = _load_manifest(manifest_path)
    cache = _manifest_cache(workflow, args.cache)
    if Path(workflow["cache"]).expanduser().resolve() != cache:
        raise SystemExit("manifest cache path could not be normalized")

    stage_b = workflow.get("stage_b")
    if isinstance(stage_b, dict) and stage_b.get("experiments"):
        best = workflow.get("best") or workflow.get("stage_a", {}).get("best")
        if not isinstance(best, dict):
            raise SystemExit("manifest has stage-B submissions but no stage-A best head/aux")
        complete = _start_stage_b(workflow, manifest_path, best=best)
        return 0 if complete else 1

    stage_a = workflow.get("stage_a")
    if not isinstance(stage_a, dict) or not stage_a.get("experiments"):
        raise SystemExit("manifest has no stage-A experiment records")
    if workflow.get("best") or stage_a.get("best"):
        best = workflow.get("best") or stage_a.get("best")
        if not isinstance(best, dict):
            raise SystemExit("manifest contains an invalid stage-A best record")
    else:
        # Resume never resubmits stage A.  It only continues checking the local
        # prediction paths saved in the manifest.
        if not _wait_for_stage_a(
            workflow,
            manifest_path,
            poll_interval=args.poll_interval,
            timeout_hours=args.timeout_hours,
        ):
            print(
                f"[workflow] stage A still has no complete set of valid results; "
                f"stage B was not submitted. Manifest: {manifest_path}",
                flush=True,
            )
            return 1
        best = _choose_best(workflow)
        _save_manifest(workflow, manifest_path)

    # If stage B is absent or empty, create it from the already-recorded best;
    # otherwise _start_stage_b only retries records without a successful
    # submission and does not monitor any stage-B result.
    return 0 if _start_stage_b(workflow, manifest_path, best=best) else 1


def _dry_run(args: argparse.Namespace) -> int:
    cache = _cache_path(args.cache)
    seed = args.seed
    timestamp = _timestamp()
    stage_a_entries = [
        _experiment_entry(
            setting=setting,
            seed=seed,
            cache=cache,
            stage="stage-a",
            timestamp=timestamp,
        )
        for setting in _stage_a_settings()
    ]
    # No result exists during a dry run.  Use the baseline's head/aux pair only
    # to render concrete stage-B commands; --auto chooses the measured winner.
    baseline = stage_a_entries[0]["config"]
    best = {
        "experiment_name": stage_a_entries[0]["name"],
        "setting_name": baseline["name"],
        "classifier_head": baseline["classifier_head"],
        "aux_prediction": baseline["aux_prediction"],
        "across_subject_cer": None,
    }
    stage_b_entries = _stage_b_entries(best=best, seed=seed, cache=cache)

    print(f"# Stage A: {len(stage_a_entries) * len(CLUSTERS)} submissions")
    for entry in stage_a_entries:
        for cluster in CLUSTERS:
            print(
                shlex.join(entry["submissions"][cluster.name]["command"])
            )
    print(
        f"# Stage B: {len(stage_b_entries) * len(CLUSTERS)} submissions "
        "(preview uses the baseline head/aux pair)"
    )
    for entry in stage_b_entries:
        for cluster in CLUSTERS:
            print(
                shlex.join(entry["submissions"][cluster.name]["command"])
            )
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--auto",
        action="store_true",
        help="submit stage A, wait for local JSON results, then submit stage B",
    )
    mode.add_argument(
        "--resume",
        metavar="WORKFLOW_JSON",
        help="resume a saved workflow manifest",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print 8 stage-A and 24 stage-B submission commands without waiting/submitting",
    )
    parser.add_argument(
        "--cache",
        help="shared BRAIN2QWERTY_CACHE (default: env BRAIN2QWERTY_CACHE or repository .cache)",
    )
    parser.add_argument(
        "--manifest",
        help="manifest path for --auto (default: <cache>/workflow.json)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"seed shared by all logical experiments (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=60.0,
        help="seconds between local stage-A JSON checks (default: 60)",
    )
    parser.add_argument(
        "--timeout-hours",
        type=float,
        default=48.0,
        help="maximum local stage-A wait (default: 48 hours)",
    )
    args = parser.parse_args()
    if args.poll_interval <= 0:
        parser.error("--poll-interval must be positive")
    if args.timeout_hours < 0:
        parser.error("--timeout-hours must be non-negative")
    if args.manifest and args.resume:
        parser.error("--manifest is only valid with --auto")
    if args.dry_run and args.resume:
        parser.error("--dry-run cannot be combined with --resume")
    if not args.dry_run and not args.auto and not args.resume:
        parser.error("choose --auto, --resume, or --dry-run")
    return args


def main() -> None:
    args = _parse_args()
    if args.dry_run:
        raise SystemExit(_dry_run(args))
    if args.resume:
        raise SystemExit(_run_resume(args))
    raise SystemExit(_run_auto(args))


if __name__ == "__main__":
    main()
