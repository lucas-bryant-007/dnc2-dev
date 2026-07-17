"""Build a portable integrity and provenance manifest for an experiment bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _git(repo: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _parse_run_commits(values: list[str]) -> dict[str, str]:
    parsed = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected RUN=COMMIT, got {value!r}")
        run, commit = value.split("=", 1)
        run, commit = run.strip(), commit.strip()
        if not run or not commit:
            raise ValueError(f"Expected RUN=COMMIT, got {value!r}")
        parsed[run] = commit
    return parsed


def _json_summary(path: Path, results_root: Path) -> dict[str, Any] | None:
    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    keys = ("method", "dataset", "attribute", "epoch", "split", "tag", "config", "ckpt_path")
    summary = {key: payload.get(key) for key in keys if payload.get(key) is not None}
    if not summary:
        return None
    summary["artifact"] = path.relative_to(results_root).as_posix()
    return summary


def build_manifest(
    results_root: Path,
    output: Path,
    repo_root: Path | None = None,
    run_commits: dict[str, str] | None = None,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a manifest without writing it; ``output`` is excluded from hashing."""
    root = results_root.resolve()
    output = output.resolve()
    artifacts = []
    metric_records = []
    checkpoint_paths = set()
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.resolve() == output:
            continue
        relative = path.relative_to(root).as_posix()
        artifacts.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
        if path.suffix.lower() == ".json" and "metrics" in path.parts:
            summary = _json_summary(path, root)
            if summary:
                metric_records.append(summary)
                if summary.get("ckpt_path"):
                    checkpoint_paths.add(summary["ckpt_path"])

    repository = None
    if repo_root is not None:
        repo = repo_root.resolve()
        status = _git(repo, "status", "--porcelain")
        repository = {
            "path": str(repo),
            "head": _git(repo, "rev-parse", "HEAD"),
            "branch": _git(repo, "branch", "--show-current"),
            "remote": _git(repo, "remote", "get-url", "origin"),
            "working_tree_dirty": bool(status),
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "bundle_id": root.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "integrity_scope": "All bundle files except this manifest",
        "repository_at_manifest_creation": repository,
        "run_commits": run_commits or {},
        "execution_environment": {
            "captured_at_run_time": False,
            "note": (
                "The original server package did not include pip-freeze, CUDA, driver, or "
                "OS snapshots. Do not substitute the manifest-creation machine environment."
            ),
        },
        "checkpoint_references": [
            {
                "path_recorded_by_run": path,
                "included_in_bundle": False,
                "sha256": None,
            }
            for path in sorted(checkpoint_paths)
        ],
        "metric_records": metric_records,
        "provenance": provenance or {},
        "artifacts": artifacts,
    }


def main(args: argparse.Namespace) -> None:
    results_root = Path(args.results_root).expanduser().resolve()
    if not results_root.is_dir():
        raise SystemExit(f"Results root does not exist: {results_root}")
    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else results_root / "reproducibility_manifest.json"
    )
    provenance = None
    if args.provenance:
        provenance_path = Path(args.provenance).expanduser().resolve()
        with provenance_path.open(encoding="utf-8") as handle:
            provenance = json.load(handle)
    try:
        run_commits = _parse_run_commits(args.run_commit)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    manifest = build_manifest(
        results_root,
        output,
        Path(args.repo_root).expanduser() if args.repo_root else None,
        run_commits,
        provenance,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    temporary.replace(output)
    print(f"Wrote manifest for {len(manifest['artifacts'])} artifacts: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", required=True, help="Experiment bundle directory")
    parser.add_argument("--output", default=None, help="Manifest path")
    parser.add_argument("--repo_root", default=None, help="Git repository used to build it")
    parser.add_argument(
        "--run_commit",
        action="append",
        default=[],
        metavar="RUN=COMMIT",
        help="Record the exact source commit for one run (repeatable)",
    )
    parser.add_argument(
        "--provenance",
        default=None,
        help="Optional JSON with commands, repair records, or other run metadata",
    )
    main(parser.parse_args())
