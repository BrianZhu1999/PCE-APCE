"""Alpha-beta state filtering for the GPS-free three-source audit.

The input is the acoustic-only temporal-association result.  Each target is
updated independently with a constant-velocity predictor and an innovation
gate.  The reported covariance is a Joseph-form diagnostic covariance built
from the ray residual/condition number; it is not the authors' private DBN
covariance implementation.  GPS is used only to score the final output.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np

from summarize_three_source_raw_music_xyz import load_gps, load_nod


PERMS = list(itertools.permutations(range(3)))


def nearest_xyz(gps: tuple[np.ndarray, np.ndarray], time_s: float) -> np.ndarray:
    xyz, times = gps
    return xyz[int(np.argmin(np.abs(times - time_s)))]


def score_gps(
    positions: list[np.ndarray],
    gps: list[tuple[np.ndarray, np.ndarray]],
    origin: np.ndarray,
    time_s: float,
) -> tuple[float, list[float], tuple[int, ...]]:
    truth = [nearest_xyz(track, time_s) - origin for track in gps]
    best = None
    for permutation in PERMS:
        errors = [
            float(np.linalg.norm(positions[target] - truth[permutation[target]]))
            for target in range(3)
        ]
        score = float(np.mean(errors))
        if best is None or score < best[0]:
            best = (score, errors, permutation)
    assert best is not None
    return best


def transition(dt: float, acceleration_std: float) -> tuple[np.ndarray, np.ndarray]:
    f = np.block(
        [[np.eye(3), dt * np.eye(3)], [np.zeros((3, 3)), np.eye(3)]]
    )
    gain = np.vstack((0.5 * dt * dt * np.eye(3), dt * np.eye(3)))
    q = acceleration_std * acceleration_std * gain @ gain.T
    return f, q


def measurement_sigma(row: dict, target: int) -> float:
    residuals = row.get("line_residual_m", [])
    conditions = row.get("condition_numbers", [])
    residual = float(residuals[target]) if len(residuals) > target else 50.0
    condition = float(conditions[target]) if len(conditions) > target else 50.0
    # Geometry-derived diagnostic uncertainty.  It is intentionally bounded
    # so an ill-conditioned frame is downweighted without exploding R.
    sigma = 15.0 + 0.50 * residual + 0.10 * math.sqrt(max(condition, 0.0))
    # Joint-quality front-end rows carry target-specific angular uncertainty.
    # Convert the angular scale to a conservative Cartesian diagnostic term
    # using the local triangulation residual, without using GPS or truth.
    assignments = row.get("node_assignments", {})
    angle_sigmas = []
    for assignment in assignments.values():
        values = assignment.get("azimuth_sigma_deg", [])
        zenith = assignment.get("zenith_sigma_deg", [])
        if len(values) > target and len(zenith) > target:
            angle_sigmas.append(math.hypot(float(values[target]), float(zenith[target])))
    if angle_sigmas:
        sigma = max(sigma, 2.5 * float(np.median(angle_sigmas)))
    return float(np.clip(sigma, 15.0, 300.0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--nod", type=Path, required=True)
    parser.add_argument("--gps-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gain", type=float, default=0.30)
    parser.add_argument("--beta", type=float, default=0.30)
    parser.add_argument("--innovation-gate-m", type=float, default=500.0)
    parser.add_argument("--acceleration-std-mps2", type=float, default=12.0)
    parser.add_argument("--initial-position-std-m", type=float, default=200.0)
    parser.add_argument("--initial-velocity-std-mps", type=float, default=80.0)
    args = parser.parse_args()

    base = json.loads(args.input.read_text(encoding="utf-8"))
    rows_in = base["rows"]
    if not rows_in:
        raise RuntimeError("input contains no rows")
    nod = load_nod(args.nod)
    nodes = base.get("protocol", {}).get("nodes", sorted(nod))
    origin = np.mean(
        [[nod[node][key] for key in ("x", "y", "z")] for node in nodes], axis=0
    )
    gps = [
        load_gps(args.gps_dir / "GPS1_plane1.gps"),
        load_gps(args.gps_dir / "GPS3_plane2.gps"),
        load_gps(args.gps_dir / "GPS4_plane2to3.gps"),
    ]
    dt = float(base["protocol"]["frame_period_s"])
    f, q = transition(dt, args.acceleration_std_mps2)
    h = np.hstack((np.eye(3), np.zeros((3, 3))))
    identity = np.eye(6)
    states = np.zeros((3, 6), dtype=float)
    states[:, :3] = np.asarray(rows_in[0]["positions"], dtype=float)
    covariances = np.asarray(
        [
            np.diag(
                [args.initial_position_std_m**2] * 3
                + [args.initial_velocity_std_mps**2] * 3
            )
            for _ in range(3)
        ]
    )
    output_rows = []
    updates = np.zeros(3, dtype=int)
    rejected = np.zeros(3, dtype=int)
    step_lengths = []
    for index, source_row in enumerate(rows_in):
        if index == 0:
            predicted_states = states.copy()
            predicted_covariances = covariances.copy()
            innovations = np.zeros((3, 3), dtype=float)
            updated = [False] * 3
        else:
            predicted_states = np.asarray([f @ states[target] for target in range(3)])
            predicted_covariances = np.asarray(
                [f @ covariances[target] @ f.T + q for target in range(3)]
            )
            innovations = np.asarray(source_row["positions"], dtype=float) - predicted_states[:, :3]
            updated = []
            for target in range(3):
                innovation = innovations[target]
                norm = float(np.linalg.norm(innovation))
                if norm > args.innovation_gate_m:
                    states[target] = predicted_states[target]
                    covariances[target] = predicted_covariances[target]
                    rejected[target] += 1
                    updated.append(False)
                    continue
                # Alpha-beta update.  The same gain is used for all Cartesian
                # components; beta controls the velocity correction.
                states[target] = predicted_states[target].copy()
                states[target][:3] += args.gain * innovation
                states[target][3:] += args.beta * innovation / dt
                sigma = measurement_sigma(source_row, target)
                measurement_covariance = np.eye(3) * sigma * sigma
                k = np.vstack((args.gain * np.eye(3), args.beta / dt * np.eye(3)))
                residual = identity - k @ h
                covariances[target] = (
                    residual @ predicted_covariances[target] @ residual.T
                    + k @ measurement_covariance @ k.T
                )
                covariances[target] = 0.5 * (
                    covariances[target] + covariances[target].T
                )
                updates[target] += 1
                updated.append(True)
        if index > 0:
            step_lengths.extend(
                float(np.linalg.norm(states[target, :3] - output_rows[-1]["positions"][target]))
                for target in range(3)
            )
        score, errors, assignment = score_gps(
            [states[target, :3].copy() for target in range(3)], gps, origin, float(source_row["time"])
        )
        row = {
            "frame": int(source_row["frame"]),
            "time": float(source_row["time"]),
            "positions": states[:, :3].tolist(),
            "velocity_mps": states[:, 3:].tolist(),
            "speed_mps": np.linalg.norm(states[:, 3:], axis=1).tolist(),
            "covariance_6x6": covariances.tolist(),
            "innovation_m": np.linalg.norm(innovations, axis=1).tolist(),
            "measurement_updated": updated,
            "measurement_sigma_m": [measurement_sigma(source_row, target) for target in range(3)],
            "condition_numbers": source_row.get("condition_numbers", []),
            "line_residual_m": source_row.get("line_residual_m", []),
            "anchor_nodes": source_row.get("anchor_nodes", []),
            "held_prediction": [not value for value in updated],
            "offline_gps_assignment": list(assignment),
            "offline_gps_error_m": errors,
            "offline_gps_mean_error_m": score,
        }
        output_rows.append(row)

    errors = np.asarray([row["offline_gps_mean_error_m"] for row in output_rows], dtype=float)
    covariance_psd = []
    for row in output_rows:
        for covariance in row["covariance_6x6"]:
            covariance_psd.append(
                bool(np.all(np.linalg.eigvalsh(np.asarray(covariance)) >= -1e-8))
            )
    summary = {
        "frames": len(output_rows),
        "offline_gps_mean_error_m": float(np.mean(errors)),
        "offline_gps_median_error_m": float(np.median(errors)),
        "offline_gps_p90_error_m": float(np.quantile(errors, 0.90)),
        "offline_gps_within_100m_fraction": float(np.mean(errors <= 100.0)),
        "state_step_p90_m": float(np.quantile(step_lengths, 0.90)) if step_lengths else 0.0,
        "state_step_max_m": float(np.max(step_lengths)) if step_lengths else 0.0,
        "target_measurement_update_fraction": (updates / max(len(output_rows) - 1, 1)).tolist(),
        "target_hold_fraction": (rejected / max(len(output_rows) - 1, 1)).tolist(),
        "covariance_psd_fraction": float(np.mean(covariance_psd)),
    }
    payload = {
        "protocol": {
            "estimator": "GPS-free alpha-beta constant-velocity state filter over temporal MUSIC hypotheses",
            "input": str(args.input),
            "gps_role": "offline scoring only",
            "frame_period_s": dt,
            "filter": {
                "gain": args.gain,
                "beta": args.beta,
                "innovation_gate_m": args.innovation_gate_m,
                "acceleration_std_mps2": args.acceleration_std_mps2,
                "initial_position_std_m": args.initial_position_std_m,
                "initial_velocity_std_mps": args.initial_velocity_std_mps,
            },
            "covariance": "Joseph-form diagnostic covariance using geometry-derived measurement sigma",
        },
        "summary": summary,
        "rows": output_rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True))


if __name__ == "__main__":
    main()
