"""IO helpers for exporting B_R / CDNV analysis metrics.

Deliberately free of heavy dependencies (torch, matplotlib) so it can be
unit-tested in isolation and imported cheaply by ``analysis/celeba.py``.

Two artifacts are produced per run:
  * one JSON per evaluated epoch  (``metrics_<stem>.json``) -- the durable,
    re-plottable record of everything computed for that checkpoint;
  * one tidy long-format CSV per (method, attribute, tag) covering every
    epoch evaluated in the run (``metrics_<method>_<attr><tag>.csv``).

Filenames are keyed by method + attribute + epoch (+ optional tag) so that
sweeps over epochs/methods/attributes never overwrite each other.
"""
import os
import csv
import json
import math
from typing import Any, Dict, List, Optional


CSV_FIELDNAMES = [
    "method", "attribute", "tag", "epoch", "k_cap", "k_eff", "r",
    "B_r", "B_r_raw",
    "tilde_V_pred", "tilde_V_obs",
    "V_pred", "V_obs",
    "gram_eig_min", "gram_eig_max", "gram_cond", "gram_fro_error",
    "orig_cdnv", "orig_directional_cdnv",
    "b_metric",
]


def slug(value: Any) -> str:
    """Filesystem-safe, lowercase token for use in filenames."""
    out = str(value).strip().lower()
    for ch in (" ", "/", "\\", ":", "(", ")", ","):
        out = out.replace(ch, "-")
    while "--" in out:
        out = out.replace("--", "-")
    return out.strip("-")


def run_stem(method: str, attribute: str, epoch: Any, tag: str = "") -> str:
    """Per-epoch filename stem, e.g. ``ijepa_male_epoch_1000_twoview``."""
    suffix = f"_{slug(tag)}" if tag else ""
    return f"{slug(method)}_{slug(attribute)}_epoch_{epoch}{suffix}"


def csv_name(method: str, attribute: str, tag: str = "") -> str:
    """Combined-CSV filename for a (method, attribute, tag) run."""
    suffix = f"_{slug(tag)}" if tag else ""
    return f"metrics_{slug(method)}_{slug(attribute)}{suffix}.csv"


def _finite(x: Any) -> Any:
    """Map non-finite floats (inf/nan) to None for portable JSON/CSV."""
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    return x


def sanitize(obj: Any) -> Any:
    """Recursively replace non-finite floats with None."""
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def write_epoch_json(metrics_dir: str, stem: str, payload: Dict[str, Any]) -> str:
    """Write one JSON record for a single epoch; returns the path written."""
    os.makedirs(metrics_dir, exist_ok=True)
    path = os.path.join(metrics_dir, f"metrics_{stem}.json")
    with open(path, "w") as f:
        json.dump(sanitize(payload), f, indent=2)
    return path


def build_csv_rows(
    method: str,
    attribute: str,
    tag: str,
    epoch: Any,
    per_cap_results: Dict[str, Any],
    orig_cdnv: Optional[float],
    orig_directional_cdnv: Optional[float],
    b_metric: str,
) -> List[Dict[str, Any]]:
    """Flatten one epoch's per-cap results into long-format CSV rows.

    ``per_cap_results`` matches the structure built in ``celeba.py``:
    ``{cap_key: {"k_eff", "r_values", "B_r", "B_r_raw", "tilde_V_pred",
    "tilde_V_obs", "V_pred", "V_obs", "gram_stats"}}``.
    """
    rows: List[Dict[str, Any]] = []
    for cap_key, cap in per_cap_results.items():
        k_eff = cap.get("k_eff")
        B_r = cap.get("B_r") or {}
        B_r_raw = cap.get("B_r_raw") or {}
        tV_pred = cap.get("tilde_V_pred") or {}
        tV_obs = cap.get("tilde_V_obs") or {}
        V_pred = cap.get("V_pred") or {}
        V_obs = cap.get("V_obs") or {}
        gram = cap.get("gram_stats") or {}
        for r in cap.get("r_values", []):
            gs = gram.get(r) or {}
            rows.append({
                "method": method,
                "attribute": attribute,
                "tag": tag,
                "epoch": epoch,
                "k_cap": cap_key,
                "k_eff": k_eff,
                "r": r,
                "B_r": _finite(B_r.get(r)),
                "B_r_raw": _finite(B_r_raw.get(r)),
                "tilde_V_pred": _finite(tV_pred.get(r)),
                "tilde_V_obs": _finite(tV_obs.get(r)),
                "V_pred": _finite(V_pred.get(r)),
                "V_obs": _finite(V_obs.get(r)),
                "gram_eig_min": _finite(gs.get("eig_min")),
                "gram_eig_max": _finite(gs.get("eig_max")),
                "gram_cond": _finite(gs.get("cond")),
                "gram_fro_error": _finite(gs.get("fro_error")),
                "orig_cdnv": _finite(orig_cdnv),
                "orig_directional_cdnv": _finite(orig_directional_cdnv),
                "b_metric": b_metric,
            })
    return rows


def write_csv(csv_path: str, rows: List[Dict[str, Any]]) -> str:
    """Write long-format rows to CSV (overwrites); returns the path written."""
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in CSV_FIELDNAMES})
    return csv_path
