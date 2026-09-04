#!/usr/bin/env python3
"""Compatibility input layer for the A2 direct-DOA-map tracker.

The archived two-source frontend writes two independently detected azimuth and
zenith peaks per node/frame.  This module reads those products without
consulting GPS, preserves the rank pairing and peak strengths, and exposes
explicit flags for boundary-valued candidates.  It deliberately does not
call the A1 confidence frontend, whose input path resolves each peak to a
hard global identity before localization.

The returned ``quality`` fields are MUSIC-peak/contrast proxies only.  They
must not be reported as calibrated acoustic SNR or DBN noise precision.
"""
from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Iterable, Mapping


REQUIRED_COLUMNS = (
    "node_id",
    "time_s",
    "time_hhmmss",
    "frame_start_sample",
    "azimuth_1_deg",
    "azimuth_2_deg",
    "zenith_1_deg",
    "zenith_2_deg",
    "azimuth_strength_1",
    "azimuth_strength_2",
    "zenith_strength_1",
    "zenith_strength_2",
)

FILENAME_RE = re.compile(r"dual_doa_node_(?P<node>\d+)(?:_[^/]*)?\.csv$")


def _finite(value: str, field: str, path: Path, row_number: int) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}: row {row_number} field {field!r} is not numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{path}: row {row_number} field {field!r} is not finite")
    return number


def wrap_azimuth(value: float) -> float:
    """Wrap an azimuth to [0, 360), retaining 360 as 0."""
    wrapped = value % 360.0
    return 0.0 if math.isclose(wrapped, 360.0, abs_tol=1e-12) else wrapped


def transformed_angle_pair(
    azimuth_deg: float,
    zenith_deg: float,
    calibration: Mapping[str, float] | None = None,
) -> tuple[float, float]:
    """Apply a frozen orientation calibration without using GPS at runtime."""
    if calibration is None:
        return wrap_azimuth(azimuth_deg), zenith_deg
    az_sign = float(calibration.get("az_sign", 1.0))
    az_offset = float(calibration.get("az_offset_deg", 0.0))
    zen_sign = float(calibration.get("zenith_sign", 1.0))
    zen_offset = float(calibration.get("zenith_offset_deg", 0.0))
    return wrap_azimuth(az_sign * azimuth_deg + az_offset), zen_sign * zenith_deg + zen_offset


@dataclass(frozen=True)
class DOACandidate:
    """One rank-paired azimuth/zenith candidate from one node/frame."""

    peak_index: int
    azimuth_deg: float
    zenith_deg: float
    azimuth_strength: float
    zenith_strength: float
    azimuth_boundary: bool
    zenith_boundary: bool

    @property
    def boundary(self) -> bool:
        return self.azimuth_boundary or self.zenith_boundary

    @property
    def log_geometric_strength(self) -> float:
        return 0.5 * (math.log(max(self.azimuth_strength, 1e-12)) + math.log(max(self.zenith_strength, 1e-12)))

    @property
    def log_peak_ratio(self) -> float:
        # Filled by ``candidate_quality`` when the alternative candidate is
        # available.  This local value is intentionally not used by itself.
        return self.log_geometric_strength


@dataclass(frozen=True)
class FrontendRow:
    node_id: int
    frame_index: int
    time_s: float
    time_hhmmss: int
    frame_start_sample: int
    azimuth_deg: tuple[float, float]
    zenith_deg: tuple[float, float]
    azimuth_strength: tuple[float, float]
    zenith_strength: tuple[float, float]

    def candidates(self, calibration: Mapping[str, float] | None = None) -> tuple[DOACandidate, DOACandidate]:
        values: list[DOACandidate] = []
        for index in range(2):
            azimuth, zenith = transformed_angle_pair(
                self.azimuth_deg[index], self.zenith_deg[index], calibration
            )
            # The historical search returns a clipped boundary value when a
            # second local elevation peak is absent.  Keep it for soft
            # assignment, but expose the fact so A2 can inflate its variance.
            values.append(
                DOACandidate(
                    peak_index=index,
                    azimuth_deg=azimuth,
                    zenith_deg=zenith,
                    azimuth_strength=self.azimuth_strength[index],
                    zenith_strength=self.zenith_strength[index],
                    azimuth_boundary=self.azimuth_deg[index] <= 0.0 or self.azimuth_deg[index] >= 360.0,
                    zenith_boundary=self.zenith_deg[index] <= 0.0 or self.zenith_deg[index] >= 90.0,
                )
            )
        return values[0], values[1]


@dataclass(frozen=True)
class FrozenIdentityPrior:
    """A frozen A1 association anchor, not a runtime identity decision."""

    target1_peak_index: int
    target2_peak_index: int
    matched_label_cost_deg: float
    alternative_label_cost_deg: float

    @property
    def assignment_margin_deg(self) -> float:
        return abs(self.alternative_label_cost_deg - self.matched_label_cost_deg)


def load_frontend_rows(path: Path, expected_node: int | None = None) -> list[FrontendRow]:
    """Load and validate one archived dual-DOA CSV.

    ``frame_index`` is the row order because the historical CSV does not
    contain an explicit frame column.  Time and sample positions are retained
    as written; no GPS or wall-clock lookup occurs here.
    """
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = tuple(reader.fieldnames or ())
        missing = [column for column in REQUIRED_COLUMNS if column not in columns]
        if missing:
            raise ValueError(f"{path}: missing required columns: {', '.join(missing)}")
        rows: list[FrontendRow] = []
        for frame_index, raw in enumerate(reader):
            row_number = frame_index + 2
            try:
                node_id = int(float(raw["node_id"]))
                time_hhmmss = int(float(raw["time_hhmmss"]))
                frame_start_sample = int(float(raw["frame_start_sample"]))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}: row {row_number} has invalid integer metadata") from exc
            if expected_node is not None and node_id != expected_node:
                raise ValueError(f"{path}: row {row_number} node_id={node_id}, expected {expected_node}")
            azimuth = tuple(_finite(raw[f"azimuth_{index}_deg"], f"azimuth_{index}_deg", path, row_number) for index in (1, 2))
            zenith = tuple(_finite(raw[f"zenith_{index}_deg"], f"zenith_{index}_deg", path, row_number) for index in (1, 2))
            az_strength = tuple(max(_finite(raw[f"azimuth_strength_{index}"], f"azimuth_strength_{index}", path, row_number), 1e-12) for index in (1, 2))
            zen_strength = tuple(max(_finite(raw[f"zenith_strength_{index}"], f"zenith_strength_{index}", path, row_number), 1e-12) for index in (1, 2))
            rows.append(
                FrontendRow(
                    node_id=node_id,
                    frame_index=frame_index,
                    time_s=_finite(raw["time_s"], "time_s", path, row_number),
                    time_hhmmss=time_hhmmss,
                    frame_start_sample=frame_start_sample,
                    azimuth_deg=azimuth,
                    zenith_deg=zenith,
                    azimuth_strength=az_strength,
                    zenith_strength=zen_strength,
                )
            )
    if not rows:
        raise ValueError(f"{path}: no data rows")
    times = [row.time_s for row in rows]
    if any(later <= earlier for earlier, later in zip(times, times[1:])):
        raise ValueError(f"{path}: time_s is not strictly increasing")
    return rows


def load_frontend_bundle(
    frontend: Path,
    node_ids: Iterable[int] | None = None,
    pattern: str = "dual_doa_node_*.csv",
    timestamp_tolerance_s: float = 1e-9,
) -> dict[int, list[FrontendRow]]:
    """Load an aligned multi-node frontend bundle without truncating frames.

    The older scripts used ``min(len(rows))`` and silently trimmed longer
    files.  A2 instead fails closed on a frame-count or timestamp mismatch, so
    a missing node-frame cannot be mistaken for a simultaneous observation.
    """
    paths = sorted(frontend.glob(pattern))
    path_by_node: dict[int, Path] = {}
    for path in paths:
        inferred = _node_from_filename(path)
        if inferred is None:
            continue
        if inferred in path_by_node:
            raise ValueError(f"{frontend}: duplicate filename node identifier {inferred}")
        path_by_node[inferred] = path
    if node_ids is not None:
        requested = tuple(sorted(int(node) for node in node_ids))
        missing = [node for node in requested if node not in path_by_node]
        if missing:
            raise FileNotFoundError(f"{frontend}: missing frontend CSV for nodes {missing}")
        path_by_node = {node: path_by_node[node] for node in requested}
    if not path_by_node:
        raise FileNotFoundError(f"{frontend}: no files matching {pattern!r}")

    loaded = {
        node: load_frontend_rows(path, expected_node=node)
        for node, path in sorted(path_by_node.items())
    }
    ordered_nodes = sorted(loaded)
    reference = loaded[ordered_nodes[0]]
    reference_times = [row.time_s for row in reference]
    for node in ordered_nodes[1:]:
        rows = loaded[node]
        if len(rows) != len(reference):
            raise ValueError(
                f"frontend frame-count mismatch: node {node} has {len(rows)}, "
                f"reference has {len(reference)}"
            )
        max_delta = max(abs(row.time_s - ref) for row, ref in zip(rows, reference_times))
        if max_delta > timestamp_tolerance_s:
            raise ValueError(
                f"frontend timestamp mismatch: node {node} max delta {max_delta:.3g}s "
                f"exceeds {timestamp_tolerance_s:.3g}s"
            )
    return loaded


def _node_from_filename(path: Path) -> int | None:
    match = FILENAME_RE.search(path.name)
    return int(match.group("node")) if match else None


def validate_frontend_bundle(
    frontend: Path,
    node_ids: Iterable[int] | None = None,
    pattern: str = "dual_doa_node_*.csv",
    timestamp_tolerance_s: float = 1e-9,
) -> dict[str, object]:
    """Validate a multi-node bundle and return a JSON-serializable manifest."""
    loaded = load_frontend_bundle(frontend, node_ids, pattern, timestamp_tolerance_s)
    ordered_nodes = sorted(loaded)
    reference = loaded[ordered_nodes[0]]
    reference_times = [row.time_s for row in reference]
    deltas = [later - earlier for earlier, later in zip(reference_times, reference_times[1:])]
    boundary_counts = {
        str(node): {
            "candidate1_zenith_boundary": sum(row.zenith_deg[0] <= 0.0 or row.zenith_deg[0] >= 90.0 for row in rows),
            "candidate2_zenith_boundary": sum(row.zenith_deg[1] <= 0.0 or row.zenith_deg[1] >= 90.0 for row in rows),
            "candidate1_azimuth_boundary": sum(row.azimuth_deg[0] <= 0.0 or row.azimuth_deg[0] >= 360.0 for row in rows),
            "candidate2_azimuth_boundary": sum(row.azimuth_deg[1] <= 0.0 or row.azimuth_deg[1] >= 360.0 for row in rows),
        }
        for node, rows in sorted(loaded.items())
    }
    return {
        "frontend_root": str(frontend),
        "nodes": ordered_nodes,
        "files": {
            str(node): str(next(frontend.glob(f"dual_doa_node_{node}_*.csv")))
            for node in ordered_nodes
        },
        "frame_count": len(reference),
        "time_start_s": reference_times[0],
        "time_end_s": reference_times[-1],
        "frame_dt_s": median(deltas) if deltas else None,
        "frame_dt_min_s": min(deltas) if deltas else None,
        "frame_dt_max_s": max(deltas) if deltas else None,
        "timestamp_tolerance_s": timestamp_tolerance_s,
        "gps_used": False,
        "boundary_candidate_counts": boundary_counts,
    }


def candidate_quality(candidate: DOACandidate, alternative: DOACandidate) -> tuple[float, float]:
    """Return log-strength and log-strength-ratio proxies for one candidate."""
    strength = candidate.log_geometric_strength
    ratio = 0.5 * (
        math.log(max(candidate.azimuth_strength, 1e-12) / max(alternative.azimuth_strength, 1e-12))
        + math.log(max(candidate.zenith_strength, 1e-12) / max(alternative.zenith_strength, 1e-12))
    )
    return strength, ratio


def load_frozen_orientation_calibrations(
    association_gate: Path,
    expected_nodes: Iterable[int] | None = None,
) -> tuple[dict[int, dict[str, float]], int]:
    """Read already-fitted calibration constants without opening GPS files."""
    payload = json.loads(association_gate.read_text(encoding="utf-8"))
    raw = payload.get("node_calibrations")
    if not isinstance(raw, dict):
        raise ValueError(f"{association_gate}: missing node_calibrations")
    calibrations: dict[int, dict[str, float]] = {}
    for key, value in raw.items():
        if not isinstance(value, dict):
            raise ValueError(f"{association_gate}: calibration for node {key!r} is not an object")
        node = int(key)
        required = ("az_sign", "az_offset_deg", "zenith_sign", "zenith_offset_deg")
        missing = [field for field in required if field not in value]
        if missing:
            raise ValueError(f"{association_gate}: node {node} missing {missing}")
        calibrations[node] = {field: float(value[field]) for field in required}
    if expected_nodes is not None:
        requested = set(int(node) for node in expected_nodes)
        if set(calibrations) != requested:
            raise ValueError(
                f"{association_gate}: calibration nodes {sorted(calibrations)} do not match "
                f"frontend nodes {sorted(requested)}"
            )
    try:
        delay_s = int(payload["selected_delay_s"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{association_gate}: invalid selected_delay_s") from exc
    return calibrations, delay_s


def angular_distance_deg(first: tuple[float, float], second: tuple[float, float]) -> float:
    """Historical degree-space distance used only to recover an exported label."""
    azimuth_difference = (first[0] - second[0] + 180.0) % 360.0 - 180.0
    return math.hypot(azimuth_difference, first[1] - second[1])


def infer_frozen_identity_prior(
    candidates: tuple[DOACandidate, DOACandidate],
    target1_label: tuple[float, float],
    target2_label: tuple[float, float],
    label_tolerance_deg: float = 1e-6,
) -> FrozenIdentityPrior:
    """Recover the A1 hard label as a soft-A2 prior with an audit margin.

    This function does *not* decide A2 identity.  It merely recovers which
    original rank pair produced each saved A1 label.  The A2 likelihood must
    still evaluate both assignments at runtime.
    """
    direct = angular_distance_deg((candidates[0].azimuth_deg, candidates[0].zenith_deg), target1_label)
    direct += angular_distance_deg((candidates[1].azimuth_deg, candidates[1].zenith_deg), target2_label)
    swapped = angular_distance_deg((candidates[1].azimuth_deg, candidates[1].zenith_deg), target1_label)
    swapped += angular_distance_deg((candidates[0].azimuth_deg, candidates[0].zenith_deg), target2_label)
    if min(direct, swapped) > label_tolerance_deg:
        raise ValueError(
            "saved global association labels do not match either transformed raw candidate "
            f"(best mismatch {min(direct, swapped):.3g} deg)"
        )
    if direct <= swapped:
        return FrozenIdentityPrior(0, 1, direct, swapped)
    return FrozenIdentityPrior(1, 0, swapped, direct)


def load_frozen_identity_priors(
    association: Path,
    frontend_rows: Mapping[int, list[FrontendRow]],
    calibrations: Mapping[int, Mapping[str, float]],
    timestamp_tolerance_s: float = 1e-9,
) -> dict[int, list[FrozenIdentityPrior]]:
    """Read legacy global associations solely as A2 identity prior anchors."""
    output: dict[int, list[FrozenIdentityPrior]] = {}
    for node, rows in sorted(frontend_rows.items()):
        path = association / f"associated_global_node_{node}.csv"
        if not path.exists():
            raise FileNotFoundError(f"missing frozen association CSV: {path}")
        with path.open(encoding="utf-8", newline="") as stream:
            labels = list(csv.DictReader(stream))
        if len(labels) != len(rows):
            raise ValueError(
                f"{path}: {len(labels)} labels for {len(rows)} frontend frames; A2 will not truncate"
            )
        priors: list[FrozenIdentityPrior] = []
        for row, label in zip(rows, labels, strict=True):
            try:
                frame_index = int(float(label["frame_index"]))
                label_node = int(float(label["node_id"]))
                label_time = float(label["time_s"])
                target1 = (float(label["target1_az_deg"]), float(label["target1_zenith_deg"]))
                target2 = (float(label["target2_az_deg"]), float(label["target2_zenith_deg"]))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}: malformed association row {row.frame_index}") from exc
            if frame_index != row.frame_index or label_node != node:
                raise ValueError(f"{path}: association row does not align with node {node}, frame {row.frame_index}")
            if abs(label_time - row.time_s) > timestamp_tolerance_s:
                raise ValueError(f"{path}: time mismatch at node {node}, frame {row.frame_index}")
            priors.append(infer_frozen_identity_prior(row.candidates(calibrations[node]), target1, target2))
        output[node] = priors
    return output
