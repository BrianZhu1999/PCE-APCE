"""Build per-target PCE/APCE inputs from the GPS-free raw-MUSIC association.

The observations remain acoustic: corrected MUSIC peaks are selected using the
temporal association output.  GPS is copied only into a separate truth file
for offline scoring and is never used to construct or accept observations.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from solve_three_source_bearing_triangulation import corrected_angles
from summarize_three_source_raw_music_xyz import load_gps, load_nod


RAW_IP_TO_PAPER_NODE = {61: 8, 5: 11, 47: 1, 43: 6, 54: 5, 49: 7, 46: 13, 40: 3}
GPS_FILES = {1: "GPS1_plane1.gps", 2: "GPS3_plane2.gps", 3: "GPS4_plane2to3.gps"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def nearest_xyz(gps: tuple[np.ndarray, np.ndarray], time_s: float) -> np.ndarray:
    xyz, times = gps
    return xyz[int(np.argmin(np.abs(times - time_s)))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--temporal-json", type=Path, required=True)
    parser.add_argument("--json-dir", type=Path, required=True)
    parser.add_argument("--nod", type=Path, required=True)
    parser.add_argument("--gps-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    temporal = json.loads(args.temporal_json.read_text(encoding="utf-8"))
    rows = temporal["rows"]
    raw_nodes = sorted(temporal["protocol"]["nodes"], key=int)
    nod = load_nod(args.nod)
    missing = [node for node in raw_nodes if node not in nod]
    if missing:
        raise RuntimeError(f"missing node geometry: {missing}")
    paper_nodes = {RAW_IP_TO_PAPER_NODE[int(node)]: node for node in raw_nodes}
    raw_to_paper = {raw_node: paper_node for paper_node, raw_node in paper_nodes.items()}
    center = np.mean(
        [[nod[node][key] for key in ("x", "y", "z")] for node in raw_nodes], axis=0
    )
    gps = {
        target: load_gps(args.gps_dir / filename)
        for target, filename in GPS_FILES.items()
    }
    raw = {
        node: {
            int(item["frame"]): item
            for item in json.loads(
                (args.json_dir / f"node{node}_k3_paper512x4_82.json").read_text(encoding="utf-8")
            )["frames"]
        }
        for node in raw_nodes
    }

    args.out.mkdir(parents=True, exist_ok=True)
    target_manifests = {}
    for target in (1, 2, 3):
        target_root = args.out / f"target{target}"
        frontend = target_root / "frontend"
        runs = target_root / "runs"
        frontend.mkdir(parents=True, exist_ok=True)
        runs.mkdir(parents=True, exist_ok=True)
        observations = []
        truth_rows = []
        used_frames = []
        for temporal_row in rows:
            if temporal_row.get("held_prediction"):
                continue
            frame = int(temporal_row["frame"])
            assignments = temporal_row.get("node_assignments", {})
            valid = True
            frame_rows = []
            for raw_node in raw_nodes:
                if raw_node not in assignments:
                    valid = False
                    break
                item = raw[raw_node][frame]
                azimuth, zenith = corrected_angles(item, nod[raw_node])
                assignment = assignments[raw_node]
                index = int(assignment["perm"][target - 1])
                confidence = float(assignment.get("confidence", [1.0, 1.0, 1.0])[target - 1])
                frame_rows.append(
                    {
                        "segment": "sanyuan_shuangyuan_5",
                        "time_s": float(temporal_row["time"]),
                        "node_id": raw_to_paper[raw_node],
                        "raw_node_ip_suffix": int(raw_node),
                        "azimuth_deg": float(azimuth[index]),
                        # The raw MUSIC solver stores the angle from +vertical
                        # (zenith convention); run_baoding.py consumes the
                        # elevation above the horizontal plane.
                        "elevation_deg": float(90.0 - zenith[index]),
                        "concentration": max(confidence, 0.05),
                        "valid": True,
                        "frontend_runtime_s": 0.0,
                        "observation_source": "raw_19_channel_music_temporal_association",
                    }
                )
            if not valid:
                continue
            observations.extend(frame_rows)
            used_frames.append(frame)
            xyz = nearest_xyz(gps[target], float(temporal_row["time"])) - center
            truth_rows.append(
                {
                    "time_s": float(temporal_row["time"]),
                    "px": float(xyz[0]),
                    "py": float(xyz[1]),
                    "pz": float(xyz[2]),
                }
            )
        if not observations:
            raise RuntimeError(f"no observations for target {target}")
        node_manifest = {
            str(paper_node): {
                "x": float(nod[raw_node]["x"]),
                "y": float(nod[raw_node]["y"]),
                "z": float(nod[raw_node]["z"]),
                "raw_node_ip_suffix": int(raw_node),
            }
            for paper_node, raw_node in sorted(paper_nodes.items())
        }
        fields = list(observations[0])
        with (frontend / "observations.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(observations)
        with (frontend / "gps_truth.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(truth_rows[0]))
            writer.writeheader()
            writer.writerows(truth_rows)
        manifest = {
            "claim_status": "raw_music_three_source_acoustic_frontend",
            "target": target,
            "nodes": node_manifest,
            "paper_nodes": sorted(node_manifest, key=int),
            "coordinate_frame": "centered local ENU; origin is mean of the eight active raw-data nodes",
            "frame_count": len(used_frames),
            "frame_indices": used_frames,
            "source_temporal_association": str(args.temporal_json),
            "source_temporal_association_sha256": sha256(args.temporal_json),
            "source_raw_music_dir": str(args.json_dir),
            "gps_source": str(args.gps_dir / GPS_FILES[target]),
            "gps_role": "offline truth scoring only",
            "gps_runtime_filter_correction": False,
            "independent_acoustic_frontend": True,
            "observation_definition": "corrected 19-channel WAVFM MUSIC azimuth peak plus zenith-to-horizontal elevation conversion, selected by GPS-free temporal association",
            "angle_convention": "azimuth=atan2(dy,dx); raw zenith=atan2(horizontal,dz); exported elevation=90deg-zenith for run_baoding.py",
            "concentration_definition": "median-normalized geometric mean of azimuth/elevation MUSIC peak strengths, floored at 0.05",
            "warning": "This is a raw-acoustic downstream PCE/APCE diagnostic input, not the authors' private DBN field implementation or a GPS-derived angle bridge.",
        }
        (frontend / "frontend_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        calibration = {
            "source": "raw MUSIC temporal association",
            "azimuth_sign_and_offset": "already applied from .nod",
            "elevation_sign_and_offset": "already applied from .nod",
            "nodes": sorted(node_manifest, key=int),
        }
        (frontend / "frontend_calibration.json").write_text(
            json.dumps(calibration, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        target_manifests[str(target)] = manifest

    root_manifest = {
        "claim_status": "three_source_raw_acoustic_pce_apce_diagnostic_bundle",
        "source_temporal_association": str(args.temporal_json),
        "source_raw_music_dir": str(args.json_dir),
        "active_raw_nodes": raw_nodes,
        "paper_node_mapping": {str(k): int(v) for k, v in paper_nodes.items()},
        "target_count": 3,
        "target_manifests": target_manifests,
        "gps_runtime_use": False,
        "pce_apce_status": "ready_for_smoke_only",
        "warning": "Downstream PCE/APCE metrics quantify filtering of real raw-MUSIC angular observations after acoustic association; they are not a full reproduction of the private field DBN code.",
    }
    (args.out / "bundle_manifest.json").write_text(
        json.dumps(root_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(root_manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
