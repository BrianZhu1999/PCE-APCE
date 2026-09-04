#!/usr/bin/env python3
"""Run PCE/APCE smoke tracking on the admitted shuangyuan_4 dual stream.

The global association output is converted into two independent
target-labelled observation bundles.  Each target is then passed through the
existing Baoding PCE/APCE runner without changing its algorithmic code.  GPS
is written only as an evaluation truth stream; it is never placed in the
observation vectors or used to update weights.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path


METHODS = ("pce", "apce")
TARGETS = (1, 2)
SEEDS = (2026081900, 2026081901, 2026081902, 2026081903, 2026081904)
NODES = (1, 2, 3, 5, 6, 7, 8, 11, 13)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hms_seconds(value: str | int | float) -> int:
    text = str(value).replace(":", "").strip().zfill(6)
    return int(text[:2]) * 3600 + int(text[2:4]) * 60 + int(text[4:])


def parse_gps(path: Path) -> dict[int, tuple[float, float, float]]:
    output = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = line.split()
        if len(fields) < 8:
            continue
        try:
            output[hms_seconds(fields[7])] = (
                float(fields[4]), float(fields[5]), float(fields[6])
            )
        except ValueError:
            continue
    return output


def fuse(
    first: dict[int, tuple[float, float, float]],
    second: dict[int, tuple[float, float, float]],
) -> dict[int, tuple[float, float, float]]:
    return {
        key: tuple((first[key][dim] + second[key][dim]) / 2.0 for dim in range(3))
        for key in first.keys() & second.keys()
    }


def load_nodes(path: Path) -> dict[int, dict[str, float]]:
    nodes = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = line.split()
        if len(fields) < 6:
            continue
        try:
            node = int(fields[2])
            nodes[node] = {
                "ip": int(fields[0].split(".")[-1]),
                "node_id": node,
                "network_id": int(fields[1]),
                "x": float(fields[3]),
                "y": float(fields[4]),
                "z": float(fields[5]),
            }
        except (ValueError, IndexError):
            continue
    if sorted(nodes) != list(NODES):
        raise RuntimeError(f"expected nodes {NODES}, got {sorted(nodes)}")
    return nodes


def load_associated(path: Path) -> list[dict]:
    output = []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            for key in (
                "node_id", "frame_index", "time_s", "time_second",
                "target1_az_deg", "target1_zenith_deg",
                "target2_az_deg", "target2_zenith_deg",
            ):
                if key in row:
                    row[key] = float(row[key])
            output.append(row)
    output.sort(key=lambda row: int(row["frame_index"]))
    return output


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def make_frontend_bundle(
    target: int,
    associated_dir: Path,
    bundle: Path,
    nodes: dict[int, dict[str, float]],
    truth: dict[int, tuple[float, float, float]],
    center: tuple[float, float, float],
    source_gate: Path,
    selected_frames: set[int] | None = None,
    window_manifest: Path | None = None,
) -> dict:
    associated = {
        node: associated_dir / f"associated_global_node_{node}.csv"
        for node in NODES
    }
    rows_by_node = {node: load_associated(path) for node, path in associated.items()}
    counts = {node: len(rows) for node, rows in rows_by_node.items()}
    if len(set(counts.values())) != 1:
        raise RuntimeError(f"node frame counts are not paired: {counts}")
    if selected_frames is not None:
        rows_by_node = {
            node: [row for row in rows if int(row["frame_index"]) in selected_frames]
            for node, rows in rows_by_node.items()
        }
    frame_count = len(rows_by_node[NODES[0]])
    if frame_count < 3:
        raise RuntimeError("selected acoustic-quality window has fewer than three frames")
    bundle.mkdir(parents=True, exist_ok=True)
    obs_rows = []
    truth_rows = []
    for index in range(frame_count):
        frame_time = float(rows_by_node[NODES[0]][index]["time_s"])
        truth_second = int(math.floor(frame_time + 1e-9)) + 1
        truth_position = truth.get(truth_second)
        if truth_position is None:
            continue
        truth_rows.append({
            "time_s": frame_time,
            "px": truth_position[0] - center[0],
            "py": truth_position[1] - center[1],
            "pz": truth_position[2] - center[2],
        })
        for node in NODES:
            row = rows_by_node[node][index]
            azimuth = float(row[f"target{target}_az_deg"])
            elevation = 90.0 - float(row[f"target{target}_zenith_deg"])
            obs_rows.append({
                "time_s": frame_time,
                "node_id": node,
                "azimuth_deg": azimuth,
                "elevation_deg": elevation,
                "concentration": 1.0,
                "valid": True,
                "source_frame_index": int(row["frame_index"]),
                "global_assignment_mask": row.get("global_assignment_mask"),
            })
    with (bundle / "observations.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = list(obs_rows[0])
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(obs_rows)
    with (bundle / "gps_truth.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["time_s", "px", "py", "pz"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(truth_rows)
    manifest = {
        "target": target,
        "nodes": {str(node): nodes[node] for node in NODES},
        "frames": frame_count,
        "observation_rows": len(obs_rows),
        "truth_rows": len(truth_rows),
        "truth_delay_protocol": "selected acoustic delay -1 s; truth key = floor(frame_time_s)+1",
        "gps_role": "evaluation truth and alpha-star audit only",
        "source_association_gate": str(source_gate),
        "source_association_gate_sha256": sha256(source_gate),
        "center_xyz": list(center),
        "acoustic_quality_window_manifest": str(window_manifest) if window_manifest else None,
        "acoustic_quality_selected": selected_frames is not None,
    }
    write_json(bundle / "frontend_manifest.json", manifest)
    write_json(bundle / "frontend_calibration.json", {
        "target": target,
        "protocol": "global dual-target association output; no additional GPS correction",
        "gps_used_after_association": False,
        "source_association_gate_sha256": sha256(source_gate),
    })
    return manifest


def aggregate_target(target_dir: Path, target: int) -> list[dict]:
    rows = []
    for path in sorted(target_dir.glob("*_seed_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        records = [
            row for row in payload.get("records", [])
            if row.get("position_error_m") is not None
        ]
        if not records:
            continue
        errors = [float(row["position_error_m"]) for row in records]
        finite = lambda key: [float(row[key]) for row in records if row.get(key) is not None]
        rows.append({
            "target": target,
            "method": payload["method"],
            "seed": payload["seed"],
            "status": payload.get("status"),
            "frames": len(records),
            "position_rmse_m": math.sqrt(sum(x * x for x in errors) / len(errors)),
            "position_median_error_m": statistics.median(errors),
            "position_p90_error_m": sorted(errors)[min(len(errors) - 1, int(0.90 * len(errors)))],
            "crps_position_m": statistics.mean(finite("crps_position_m")),
            "coverage_90": statistics.mean(finite("coverage_90")),
            "interval_width_m": statistics.mean(finite("interval_width_m")),
            "alpha_mae": statistics.mean(finite("alpha_error")) if finite("alpha_error") else None,
            "runtime_s": payload.get("runtime_s"),
        })
    return rows


def install_real_data_stability_layer(run_baoding) -> dict:
    """Install bounded numerical guards around the unchanged PCE/APCE core.

    The synthetic Figure 2/3 runners are untouched.  On this real archive the
    nonlinear angle geometry and occasional bad peaks can make a raw EnKF
    covariance solve produce an unphysical state increment.  We retain the
    original forecast/evidence/analysis structure but add:

    * covariance diagonal jitter;
    * conservative analysis-update damping;
    * finite-value sanitization;
    * centered position and velocity bounds.

    The bounds are diagnostics for numerical stability, not data-derived
    truth constraints.
    """
    import torch

    original_propagate = run_baoding.propagate

    def physical_project(states: torch.Tensor, max_range_m: float = 5000.0, max_speed_mps: float = 250.0) -> torch.Tensor:
        output = torch.nan_to_num(states, nan=0.0, posinf=0.0, neginf=0.0).clone()
        position_norm = torch.linalg.vector_norm(output[..., :3], dim=-1, keepdim=True).clamp_min(1.0)
        velocity_norm = torch.linalg.vector_norm(output[..., 3:6], dim=-1, keepdim=True).clamp_min(1.0)
        output[..., :3] = output[..., :3] * torch.clamp(max_range_m / position_norm, max=1.0)
        output[..., 3:6] = output[..., 3:6] * torch.clamp(max_speed_mps / velocity_norm, max=1.0)
        return output

    def stable_propagate(states, alpha, dt, noise, q_min, q_max):
        propagated = original_propagate(states, alpha, dt, noise, q_min, q_max)
        return physical_project(propagated)

    def stable_enkf_update(states, observation, node_xyz, node_ids, obs_std):
        predicted = run_baoding.predict_angles(states, node_xyz, node_ids)
        mean_state = states.mean(dim=-2)
        mean_obs = torch.atan2(torch.sin(predicted).mean(dim=-2), torch.cos(predicted).mean(dim=-2))
        state_anomaly = states - mean_state.unsqueeze(-2)
        obs_anomaly = run_baoding.angle_residual(predicted, mean_obs.unsqueeze(-2))
        denominator = max(states.shape[-2] - 1, 1)
        covariance = state_anomaly.transpose(-2, -1) @ obs_anomaly / denominator
        obs_cov = obs_anomaly.transpose(-2, -1) @ obs_anomaly / denominator
        jitter = math.radians(float(obs_std)) ** 2 + 1e-4
        obs_cov = obs_cov + torch.eye(
            obs_cov.shape[-1], dtype=obs_cov.dtype, device=obs_cov.device
        ) * jitter
        gain = torch.linalg.solve(
            obs_cov.transpose(-2, -1), covariance.transpose(-2, -1)
        ).transpose(-2, -1)
        increment = torch.einsum(
            "...ij,...nj->...ni",
            gain,
            run_baoding.angle_residual(observation.unsqueeze(-2), predicted),
        )
        # The acoustic front-end is admitted but not perfectly Gaussian.
        # Damping prevents a single ill-conditioned angular update from
        # ejecting all candidates from the local geometric basin.
        updated = states + 0.35 * increment
        return physical_project(updated)

    run_baoding.propagate = stable_propagate
    run_baoding.enkf_update = stable_enkf_update
    return {
        "enabled": True,
        "position_bound_m": 5000.0,
        "velocity_bound_mps": 250.0,
        "analysis_update_damping": 0.35,
        "observation_covariance_jitter_rad2": 1e-4,
        "scope": "shuangyuan_4 real-data adapter only; Figure 2/3 runners unchanged",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote-root", type=Path, default=Path("<PRIVATE_DATA_ROOT>/2017保定实验/2017保定实验"))
    parser.add_argument("--association", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--target", type=int, choices=TARGETS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--window-manifest", type=Path)
    args = parser.parse_args()

    code_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(code_dir))
    import run_baoding
    stability = install_real_data_stability_layer(run_baoding)

    archive = args.remote_root / "20171107保定实验"
    gps_root = archive / "GPS_data"
    nodes = load_nodes(gps_root / "20171107baoding.nod")
    center = tuple(
        sum(nodes[node][axis] for node in NODES) / len(NODES)
        for axis in ("x", "y", "z")
    )
    target_truth = {
        1: fuse(
            parse_gps(gps_root / "GPS1_plane1.gps"),
            parse_gps(gps_root / "GPS2_plane1.gps"),
        ),
        2: fuse(
            parse_gps(gps_root / "GPS3_plane2.gps"),
            parse_gps(gps_root / "GPS4_plane2to3.gps"),
        ),
    }
    gate = args.association / "shuangyuan4_global_association_gate.json"
    gate_payload = json.loads(gate.read_text(encoding="utf-8"))
    if not gate_payload["admission"]["dual_target_position_gate"]:
        raise RuntimeError("dual-target association gate is not admitted")

    if args.prepare:
        selected_frames = None
        if args.window_manifest is not None:
            window = json.loads(args.window_manifest.read_text(encoding="utf-8"))
            start = int(window["selected"]["start_frame"])
            end = int(window["selected"]["end_frame"])
            selected_frames = set(range(start, end + 1))
        for target in TARGETS:
            make_frontend_bundle(
                target,
                args.association,
                args.output / f"target{target}" / "frontend",
                nodes,
                target_truth[target],
                center,
                gate,
                selected_frames,
                args.window_manifest,
            )
        write_json(args.output / "experiment_manifest.json", {
            "task": "2017 Baoding shuangyuan_4 PCE/APCE dual-target smoke",
            "methods": list(METHODS),
            "targets": list(TARGETS),
            "seeds": list(SEEDS),
            "association_gate": str(gate),
            "association_gate_sha256": sha256(gate),
            "gps_role": "truth evaluation and alpha-star audit only",
            "runner": str(Path(__file__).resolve()),
            "runner_sha256": sha256(Path(__file__).resolve()),
            "acoustic_quality_window_manifest": str(args.window_manifest) if args.window_manifest else None,
            "selection_uses_gps_error": False if args.window_manifest else None,
        })
        print(json.dumps({"prepared_targets": list(TARGETS)}, ensure_ascii=False, indent=2))
        return

    if args.aggregate:
        all_rows = []
        for target in TARGETS:
            all_rows.extend(aggregate_target(args.output / f"target{target}" / "runs", target))
        summary = args.output / "paired_summary.csv"
        summary.parent.mkdir(parents=True, exist_ok=True)
        fields = list(all_rows[0]) if all_rows else ["target", "method"]
        with summary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(all_rows)
        write_json(args.output / "aggregate_manifest.json", {
            "expected_runs": len(TARGETS) * len(METHODS) * len(SEEDS),
            "valid_runs": sum(row["status"] == "valid" for row in all_rows),
            "targets": list(TARGETS),
            "methods": list(METHODS),
            "seeds": list(SEEDS),
            "source_association_gate_sha256": sha256(gate),
        })
        print(json.dumps({
            "expected_runs": len(TARGETS) * len(METHODS) * len(SEEDS),
            "valid_runs": sum(row["status"] == "valid" for row in all_rows),
            "summary": str(summary),
        }, ensure_ascii=False, indent=2))
        return

    if args.method is None or args.target is None or args.seed is None:
        raise SystemExit("--method, --target and --seed are required unless --prepare/--aggregate is set")
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    frontend = args.output / f"target{args.target}" / "frontend"
    target_dir = args.output / f"target{args.target}" / "runs"
    payload = run_baoding.run_track(
        cfg,
        frontend,
        target_dir,
        args.method,
        args.seed,
        args.device,
    )
    payload.update({
        "dual_target": True,
        "target_id": args.target,
        "association_gate": str(gate),
        "association_gate_sha256": sha256(gate),
        "wrapper_sha256": sha256(Path(__file__).resolve()),
        "real_data_stability_layer": stability,
    })
    write_json(target_dir / f"{args.method}_seed_{args.seed}.json", payload)
    print(json.dumps({
        "target": args.target,
        "method": args.method,
        "seed": args.seed,
        "status": payload.get("status"),
        "records": len(payload.get("records", [])),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
