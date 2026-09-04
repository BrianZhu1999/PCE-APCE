#!/usr/bin/env python3
"""Reproducible audit for the 2017 Baoding ``shuangyuan_4`` segment.

This is deliberately an admission/association audit, not a two-target
PCE/APCE runner.  The archive contains a useful reference-node product
(``gps_doa_49_125540-125900.txt``) with two labelled DOA pairs, while the
other node ``.doa`` products do not carry a target identity.  The audit
therefore reports what is established and blocks a single-target tracker
until multi-node target association is explicit.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path


IP_TO_NODE = {40: 3, 43: 6, 46: 13, 47: 1, 48: 2, 49: 7, 5: 11, 54: 5, 61: 8}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_nod(path: Path) -> dict[int, tuple[float, float, float]]:
    nodes = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = line.split()
        if len(fields) < 6:
            continue
        nodes[int(fields[2])] = (float(fields[3]), float(fields[4]), float(fields[5]))
    return nodes


def parse_gps(path: Path) -> dict[int, tuple[float, float, float]]:
    output = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = line.split("\t")
        if len(fields) < 8:
            continue
        try:
            output[int(float(fields[7]))] = (
                float(fields[4]),
                float(fields[5]),
                float(fields[6]),
            )
        except ValueError:
            continue
    return output


def fuse(a: dict[int, tuple[float, float, float]], b: dict[int, tuple[float, float, float]]) -> dict[int, tuple[float, float, float]]:
    return {
        key: tuple((a[key][i] + b[key][i]) / 2.0 for i in range(3))
        for key in sorted(a.keys() & b.keys())
    }


def angle(position: tuple[float, float, float], node: tuple[float, float, float]) -> tuple[float, float]:
    dx, dy, dz = (position[i] - node[i] for i in range(3))
    azimuth = math.degrees(math.atan2(dy, dx)) % 360.0
    zenith = 90.0 - math.degrees(math.atan2(dz, math.hypot(dx, dy)))
    return azimuth, zenith


def circular_error(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def parse_reference_metadata(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = line.split()
        if len(fields) < 5:
            continue
        try:
            rows.append(
                {
                    "time_hhmmss": int(fields[-1]),
                    "target1_az_deg": float(fields[1]),
                    "target1_zenith_deg": float(fields[2]),
                    "target2_az_deg": float(fields[3]),
                    "target2_zenith_deg": float(fields[4]),
                    "metadata_fields": len(fields),
                }
            )
        except ValueError:
            continue
    return rows


def parse_doa_inventory(segment: Path) -> list[dict]:
    rows = []
    for path in sorted(segment.glob("*.doa")):
        ip = int(path.stem.split("_")[-2])
        node = IP_TO_NODE.get(ip)
        samples = []
        for line in path.read_text(encoding="ascii", errors="replace").splitlines():
            fields = line.split()
            if len(fields) < 6:
                continue
            try:
                azimuth = float(fields[1])
                zenith = float(fields[2])
                timestamp = float(fields[-1])
            except ValueError:
                continue
            if timestamp < 120000.0:
                continue
            samples.append((timestamp, azimuth, zenith))
        seconds = [int(value[0]) for value in samples]
        counts = {}
        for second in seconds:
            counts[second] = counts.get(second, 0) + 1
        rows.append(
            {
                "file": path.name,
                "ip_suffix": ip,
                "node_id": node,
                "sha256": sha256(path),
                "rows_with_absolute_time": len(samples),
                "unique_seconds": len(counts),
                "timestamp_min": min((value[0] for value in samples), default=None),
                "timestamp_max": max((value[0] for value in samples), default=None),
                "rows_per_second_median": statistics.median(counts.values()) if counts else None,
                "explicit_target_id_present": False,
            }
        )
    return rows


def audit(args: argparse.Namespace) -> dict:
    archive = args.remote_root / "20171107保定实验"
    gps_root = archive / "GPS_data"
    segment = archive / "project/20171107baoding/shuangyuan_4"
    nodes = parse_nod(gps_root / "20171107baoding.nod")
    target1 = fuse(parse_gps(gps_root / "GPS1_plane1.gps"), parse_gps(gps_root / "GPS2_plane1.gps"))
    target2 = fuse(parse_gps(gps_root / "GPS3_plane2.gps"), parse_gps(gps_root / "GPS4_plane2to3.gps"))
    metadata = parse_reference_metadata(segment / "gps_doa_49_125540-125900.txt")
    reference_node = nodes[7]

    delay_candidates = []
    for delay in range(-1, 5):
        errors_1, errors_2 = [], []
        for row in metadata:
            t = row["time_hhmmss"] - delay
            if t not in target1 or t not in target2:
                continue
            truth1 = angle(target1[t], reference_node)
            truth2 = angle(target2[t], reference_node)
            errors_1.append(
                math.hypot(
                    circular_error(row["target1_az_deg"], truth1[0]),
                    row["target1_zenith_deg"] - truth1[1],
                )
            )
            errors_2.append(
                math.hypot(
                    circular_error(row["target2_az_deg"], truth2[0]),
                    row["target2_zenith_deg"] - truth2[1],
                )
            )
        joint = errors_1 + errors_2
        delay_candidates.append(
            {
                "delay_s": delay,
                "rows": len(joint) // 2,
                "median_joint_error_deg": statistics.median(joint) if joint else None,
                "p90_joint_error_deg": sorted(joint)[min(len(joint) - 1, int(0.90 * len(joint)))] if joint else None,
            }
        )
    selected_delay = min(
        delay_candidates,
        key=lambda row: (row["median_joint_error_deg"] if row["median_joint_error_deg"] is not None else float("inf"), abs(row["delay_s"])),
    )["delay_s"]
    selected = next(row for row in delay_candidates if row["delay_s"] == selected_delay)

    association_rows = []
    for row in metadata:
        t = row["time_hhmmss"] - selected_delay
        if t not in target1 or t not in target2:
            continue
        truth1 = angle(target1[t], reference_node)
        truth2 = angle(target2[t], reference_node)
        association_rows.append(
            {
                **row,
                "delay_s": selected_delay,
                "target1_truth_az_deg": truth1[0],
                "target1_truth_zenith_deg": truth1[1],
                "target2_truth_az_deg": truth2[0],
                "target2_truth_zenith_deg": truth2[1],
                "target1_joint_error_deg": math.hypot(
                    circular_error(row["target1_az_deg"], truth1[0]),
                    row["target1_zenith_deg"] - truth1[1],
                ),
                "target2_joint_error_deg": math.hypot(
                    circular_error(row["target2_az_deg"], truth2[0]),
                    row["target2_zenith_deg"] - truth2[1],
                ),
            }
        )

    inventory = parse_doa_inventory(segment)
    doas_have_target_ids = any(row["explicit_target_id_present"] for row in inventory)
    result = {
        "task": "2017 Baoding shuangyuan_4 multi-target association audit",
        "archive_segment": str(segment),
        "target_ground_truth_sources": {
            "target1": ["GPS1_plane1.gps", "GPS2_plane1.gps"],
            "target2": ["GPS3_plane2.gps", "GPS4_plane2to3.gps"],
        },
        "network_nodes": sorted(nodes),
        "reference_metadata": {
            "file": str(segment / "gps_doa_49_125540-125900.txt"),
            "reference_node": 7,
            "rows": len(metadata),
            "time_start": metadata[0]["time_hhmmss"] if metadata else None,
            "time_end": metadata[-1]["time_hhmmss"] if metadata else None,
            "two_target_pairs_present": bool(metadata) and all(row["metadata_fields"] >= 5 for row in metadata),
        },
        "delay_candidates": delay_candidates,
        "selected_delay_s": selected_delay,
        "reference_node_association": {
            "rows": selected["rows"],
            "median_joint_error_deg": selected["median_joint_error_deg"],
            "p90_joint_error_deg": selected["p90_joint_error_deg"],
            "target1_rows_below_5_deg": sum(row["target1_joint_error_deg"] < 5 for row in association_rows),
            "target2_rows_below_5_deg": sum(row["target2_joint_error_deg"] < 5 for row in association_rows),
        },
        "doa_inventory": inventory,
        "multi_node_target_association": {
            "all_nodes_have_explicit_target_ids": doas_have_target_ids,
            "status": "resolved" if doas_have_target_ids else "unresolved",
            "reason": (
                "The reference-node text product contains two target-labelled DOA pairs, "
                "but the other node .doa files contain only unlabelled azimuth/zenith rows."
            ),
        },
        "admission": {
            "single_target_pce_apce_allowed": False,
            "two_target_pce_apce_allowed": False,
            "status": "association_audit_only",
            "next_required_artifact": "target-labelled two-peak DOA stream for every participating node, or a reproducible multi-target MUSIC reprocessing from .raw/.wavfm",
        },
        "provenance": {
            "nod_sha256": sha256(gps_root / "20171107baoding.nod"),
            "metadata_sha256": sha256(segment / "gps_doa_49_125540-125900.txt"),
            "gps_sha256": {name: sha256(gps_root / name) for name in (
                "GPS1_plane1.gps",
                "GPS2_plane1.gps",
                "GPS3_plane2.gps",
                "GPS4_plane2to3.gps",
            )},
            "script_sha256": sha256(Path(__file__)),
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "shuangyuan4_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with (args.output / "shuangyuan4_reference_association.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted(association_rows[0]) if association_rows else [])
        if association_rows:
            writer.writeheader()
            writer.writerows(association_rows)
    with (args.output / "shuangyuan4_doa_inventory.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted(inventory[0]) if inventory else [])
        if inventory:
            writer.writeheader()
            writer.writerows(inventory)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote-root", type=Path, default=Path("<PRIVATE_DATA_ROOT>/2017保定实验/2017保定实验"))
    parser.add_argument("--output", type=Path, required=True)
    audit(parser.parse_args())


if __name__ == "__main__":
    main()
