#!/usr/bin/env python3
"""Bridge a validated 8-node DBN track into the PCE/APCE angle interface.

The generated observations are deterministic line-of-sight angles computed
from DBN Cartesian positions and the paper node geometry.  They are *not*
independent MUSIC observations; provenance records this explicitly.
"""
from __future__ import annotations

import argparse, csv, hashlib, json, math
from pathlib import Path


PAPER_NODES = (1, 3, 5, 6, 7, 8, 11, 13)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def hms_seconds(value: str | int | float) -> float:
    text = str(value).replace(":", "").zfill(6)
    return int(text[:2]) * 3600 + int(text[2:4]) * 60 + int(text[4:])


def parse_nod(path: Path) -> dict[int, dict[str, float]]:
    out = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        f = line.split()
        if len(f) < 6:
            continue
        try:
            node = int(f[2]); out[node] = {"x": float(f[3]), "y": float(f[4]), "z": float(f[5]), "ip": int(f[0].split(".")[-1])}
        except (ValueError, IndexError):
            continue
    return out


def read_track(path: Path, target_z: float, start_time_s: float | None) -> list[dict]:
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            frame = int(row["frame_index"])
            if start_time_s is None:
                start_time_s = hms_seconds(row["time_hhmmss"])
            time_s = float(row["time_s"]) if row.get("time_s") not in (None, "") else float(start_time_s) + frame * 640.0 / 3050.0
            rows.append({"frame_index": frame, "time_s": time_s, "x": float(row["px"]), "y": float(row["py"]), "z": float(row.get("pz") or target_z), "dbn_error_m": float(row["position_error_m"])})
    if not rows:
        raise RuntimeError(f"empty DBN track: {path}")
    return rows


def read_gps(path: Path) -> list[tuple[float, float, float, float]]:
    out = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        f = line.split()
        if len(f) < 8:
            continue
        try:
            out.append((hms_seconds(f[7]), float(f[4]), float(f[5]), float(f[6])))
        except ValueError:
            pass
    return out


def nearest_gps(rows: list[tuple[float, float, float, float]], t: float) -> tuple[float, float, float]:
    _, x, y, z = min(rows, key=lambda r: abs(r[0] - t))
    return x, y, z


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle-root", type=Path, required=True)
    ap.add_argument("--nod", type=Path, required=True)
    ap.add_argument("--gps-root", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--target-z", type=float, required=True)
    ap.add_argument("--target-id", type=int, required=True)
    ap.add_argument("--gps-file", required=True)
    ap.add_argument("--start-time-s", type=float)
    ap.add_argument("--angle-noise-deg", type=float, default=0.5)
    args = ap.parse_args()

    track_path = args.bundle_root / f"target{args.target_id}" / "frontend" / "dbn_track.csv"
    nodes = parse_nod(args.nod)
    missing = sorted(set(PAPER_NODES) - set(nodes))
    if missing:
        raise RuntimeError(f"missing paper nodes: {missing}")
    tracks = read_track(track_path, args.target_z, args.start_time_s)
    gps = read_gps(args.gps_root / args.gps_file)
    center = {axis: sum(nodes[node][axis] for node in PAPER_NODES) / len(PAPER_NODES) for axis in ("x", "y", "z")}
    out_front = args.output_root / "frontend"
    out_runs = args.output_root / "runs"
    out_front.mkdir(parents=True, exist_ok=True); out_runs.mkdir(parents=True, exist_ok=True)

    obs_rows = []
    truth_rows = []
    for tr in tracks:
        for node in PAPER_NODES:
            n = nodes[node]
            dx, dy, dz = tr["x"] - n["x"], tr["y"] - n["y"], tr["z"] - n["z"]
            horizontal = math.hypot(dx, dy)
            az = math.degrees(math.atan2(dy, dx)) % 360.0
            el = math.degrees(math.atan2(dz, horizontal))
            obs_rows.append({"segment": "sanyuan_tongxinyuan_6", "time_s": tr["time_s"], "node_id": node, "azimuth_deg": az, "elevation_deg": el, "concentration": 1.0, "valid": True, "frontend_runtime_s": 0.0, "observation_source": "DBN_cartesian_line_of_sight"})
        tx, ty, tz = nearest_gps(gps, tr["time_s"])
        truth_rows.append({"time_s": tr["time_s"], "px": tx - center["x"], "py": ty - center["y"], "pz": tz - center["z"]})

    manifest = {
        "nodes": {str(k): {**nodes[k], "x": nodes[k]["x"], "y": nodes[k]["y"], "z": nodes[k]["z"]} for k in PAPER_NODES},
        "gps_source": str(args.gps_root / args.gps_file),
        "gps_source_sha256": sha256(args.gps_root / args.gps_file),
        "segments": [{"name": "sanyuan_tongxinyuan_6", "available_nodes": list(PAPER_NODES), "sample_rate": 3050}],
        "paper_nodes": list(PAPER_NODES),
        "coordinate_frame": "centered local ENU; origin is mean of the 8 paper node coordinates",
        "input_provenance": {
            "synthetic_observation_from_dbn": True,
            "independent_acoustic_observation": False,
            "dbn_track_source": str(track_path),
            "dbn_track_sha256": sha256(track_path),
            "bridge_model": "3-D line-of-sight azimuth/elevation from DBN position to each paper node",
            "angle_noise_deg_configured": args.angle_noise_deg,
            "warning": "PCE/APCE metrics quantify downstream filtering of a DBN-derived observation stream; they are not an independent end-to-end acoustic benchmark.",
        },
    }
    (out_front / "frontend_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_front / "frontend_calibration.json").write_text(json.dumps({"bridge": "DBN_to_angle", "nodes": {str(k): {"azimuth_sign": 1.0, "azimuth_offset_deg": 0.0, "elevation_offset_deg": 0.0} for k in PAPER_NODES}}, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out_front / "observations.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(obs_rows[0])); w.writeheader(); w.writerows(obs_rows)
    with (out_front / "gps_truth.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(truth_rows[0])); w.writeheader(); w.writerows(truth_rows)
    bridge_manifest = {"claim_status": "dbn_to_pce_apce_bridge", "target_id": args.target_id, "target_z_m": args.target_z, "paper_nodes": list(PAPER_NODES), "source_bundle": str(args.bundle_root), "output_root": str(args.output_root), "frontend_manifest": str(out_front / "frontend_manifest.json"), "observation_count": len(obs_rows), "frame_count": len(tracks), "truth_source": str(args.gps_root / args.gps_file), "pce_apce_role": "downstream uncertainty/state update from DBN-derived angular observations"}
    (args.output_root / "bridge_manifest.json").write_text(json.dumps(bridge_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(bridge_manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
