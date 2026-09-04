"""Summarize the Baoding three-source front-end admission experiments.

All inputs are acoustic tracking outputs.  GPS-derived errors already stored
in those outputs are used only for this offline admission report.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def target_metrics(rows: list[dict], target: int, frame_range: tuple[int, int] | None = None) -> dict:
    selected = rows
    if frame_range is not None:
        selected = [row for row in rows if frame_range[0] <= int(row["frame"]) <= frame_range[1]]
    values = np.asarray([float(row["offline_gps_error_m"][target]) for row in selected], dtype=float)
    return {
        "frames": len(values),
        "mean_m": float(np.mean(values)),
        "median_m": float(np.median(values)),
        "p90_m": float(np.quantile(values, 0.90)),
        "within_100m_fraction": float(np.mean(values <= 100.0)),
    }


def identity_consistency(rows: list[dict]) -> float:
    assignments = [tuple(row.get("offline_gps_assignment", [])) for row in rows]
    assignments = [x for x in assignments if x]
    if not assignments:
        return float("nan")
    count = Counter(assignments)
    return float(count.most_common(1)[0][1] / len(assignments))


def node_residuals(rows: list[dict]) -> dict:
    values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        for node, assignment in row.get("node_assignments", {}).items():
            values[str(node)].extend(float(x) for x in assignment.get("errors_deg", []))
    result = {}
    for node, errors in values.items():
        array = np.asarray(errors, dtype=float)
        result[node] = {
            "count": len(array),
            "mean_deg": float(np.mean(array)),
            "median_deg": float(np.median(array)),
            "p90_deg": float(np.quantile(array, 0.90)),
            "over_30deg_fraction": float(np.mean(array > 30.0)),
        }
    return result


def pairing_mismatch(rows: list[dict]) -> float | None:
    flags = []
    for row in rows:
        for assignment in row.get("node_assignments", {}).values():
            az = assignment.get("az_perm")
            el = assignment.get("el_perm")
            if az is not None and el is not None:
                flags.extend(int(a != e) for a, e in zip(az, el))
    return float(np.mean(flags)) if flags else None


def summarize(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["rows"]
    per_frame = [
        {
            "frame": int(row["frame"]),
            "time": float(row["time"]),
            "mean_error_m": float(row["offline_gps_mean_error_m"]),
            "target_error_m": [float(x) for x in row["offline_gps_error_m"]],
        }
        for row in rows
    ]
    worst = sorted(per_frame, key=lambda row: row["mean_error_m"], reverse=True)[:10]
    return {
        "path": str(path),
        "protocol": payload.get("protocol", {}),
        "reported_summary": payload.get("summary", {}),
        "target_metrics": {f"T{target + 1}": target_metrics(rows, target) for target in range(3)},
        "target_metrics_frames_32_41": {
            f"T{target + 1}": target_metrics(rows, target, (32, 41)) for target in range(3)
        },
        "offline_identity_consistency": identity_consistency(rows),
        "node_angular_residuals": node_residuals(rows),
        "azimuth_zenith_pair_mismatch_fraction": pairing_mismatch(rows),
        "worst_frames": worst,
        "per_frame": per_frame,
    }


def admission(summary: dict) -> dict:
    reported = summary["reported_summary"]
    gates = {
        "offline_identity_consistency_ge_0_90": summary["offline_identity_consistency"] >= 0.90,
        "mean_error_le_150m": float(reported.get("offline_gps_mean_error_m", float("inf"))) <= 150.0,
        "state_step_p90_lt_100m": float(reported.get("state_step_p90_m", reported.get("temporal_jump_p90_m", float("inf")))) < 100.0,
        "covariance_psd_ge_0_99": float(reported.get("covariance_psd_fraction", 0.0)) >= 0.99,
    }
    update = reported.get("target_measurement_update_fraction")
    if update is not None:
        gates["all_target_update_fraction_ge_0_90"] = bool(np.all(np.asarray(update, dtype=float) >= 0.90))
    else:
        gates["hold_fraction_le_0_10"] = float(reported.get("hold_fraction", 1.0)) <= 0.10
    return {"gates": gates, "passed": bool(all(gates.values()))}


def benchmark_admission(summary: dict, engineering: dict) -> dict:
    """Stricter gate for a balanced three-target PCE/APCE benchmark.

    The engineering gate uses an all-target mean and can be passed when one
    easy target masks a persistently poor target.  A benchmark input must also
    keep every target's mean error within the same 150 m bound.
    """
    target_means = {
        target: float(metrics["mean_m"])
        for target, metrics in summary["target_metrics"].items()
    }
    gates = {
        "engineering_admission_passed": bool(engineering["passed"]),
        "all_target_mean_error_le_150m": bool(all(value <= 150.0 for value in target_means.values())),
    }
    return {"gates": gates, "target_mean_error_m": target_means, "passed": bool(all(gates.values()))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", nargs=2, metavar=("LABEL", "PATH"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    experiments = {label: summarize(Path(path)) for label, path in args.input}
    for value in experiments.values():
        value["admission"] = admission(value)
        value["benchmark_admission"] = benchmark_admission(value, value["admission"])
    output = {
        "protocol": {
            "purpose": "offline three-source acoustic front-end admission audit",
            "gps_role": "offline scoring only; not used by estimators",
            "gates": {
                "identity_consistency": ">=0.90",
                "mean_position_error_m": "<=150",
                "state_step_p90_m": "<100",
                "covariance_psd_fraction": ">=0.99 for state outputs",
                "measurement_update_fraction": ">=0.90 for all targets; raw paths use hold_fraction<=0.10",
            },
            "benchmark_extension": "engineering admission plus every target mean position error <=150 m",
        },
        "experiments": experiments,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps({label: value["admission"] for label, value in experiments.items()}, ensure_ascii=True))


if __name__ == "__main__":
    main()
