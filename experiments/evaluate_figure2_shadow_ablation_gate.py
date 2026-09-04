from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

CASES = ("wave", "spring", "heat")
FAMILIES = (("pce_shadow", "pce_analysis"), ("apce_shadow", "apce_analysis"))


def read_rows(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def f(row: dict[str, object], key: str) -> float:
    try:
        value = float(row.get(key, "nan"))
    except (TypeError, ValueError):
        return float("nan")
    return value if math.isfinite(value) else float("nan")


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, draws: int = 10000) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    sample = values[rng.integers(0, values.size, size=(draws, values.size))].mean(axis=1)
    return float(np.quantile(sample, 0.025)), float(np.quantile(sample, 0.975))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seed-count", type=int, required=True)
    parser.add_argument("--mode", choices=("smoke", "formal"), required=True)
    args = parser.parse_args()

    aggregate = args.root / "aggregate"
    rows = read_rows(aggregate / "figure2_corrected_formal_run_source_data.csv")
    audit = json.loads((aggregate / "figure2_corrected_formal_aggregate.json").read_text(encoding="utf-8"))["audit"]
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault((str(row["case"]), str(row["method"])), []).append(row)

    rng = np.random.default_rng(2026082401)
    case_results = []
    for case in CASES:
        shadow_erasure = np.asarray([f(row, "mechanism_mean_erasure_ratio") for row in grouped.get((case, "pce_shadow"), [])], dtype=float)
        erasure_lo, erasure_hi = bootstrap_ci(shadow_erasure, rng)
        family_results = []
        for shadow, analysis in FAMILIES:
            shadow_rows = grouped.get((case, shadow), [])
            analysis_rows = {int(row["seed"]): row for row in grouped.get((case, analysis), [])}
            paired = [(row, analysis_rows[int(row["seed"])]) for row in shadow_rows if int(row["seed"]) in analysis_rows]
            if not paired:
                family_results.append({"family": shadow.split("_")[0], "valid": False})
                continue
            dn = np.asarray([f(a, "nrmse") - f(s, "nrmse") for s, a in paired], dtype=float)
            dc = np.asarray([f(a, "crps") - f(s, "crps") for s, a in paired], dtype=float)
            n_shadow = float(np.nanmean([f(s, "nrmse") for s, _ in paired]))
            n_analysis = float(np.nanmean([f(a, "nrmse") for _, a in paired]))
            c_shadow = float(np.nanmean([f(s, "crps") for s, _ in paired]))
            c_analysis = float(np.nanmean([f(a, "crps") for _, a in paired]))
            performance_pass = (
                (n_shadow <= n_analysis or c_shadow <= c_analysis)
                and n_shadow <= n_analysis * 1.02
                and c_shadow <= c_analysis * 1.02
            )
            family_results.append({
                "family": shadow.split("_")[0],
                "n": len(paired),
                "nrmse_shadow_mean": n_shadow,
                "nrmse_analysis_mean": n_analysis,
                "crps_shadow_mean": c_shadow,
                "crps_analysis_mean": c_analysis,
                "delta_analysis_minus_shadow_nrmse": float(np.nanmean(dn)),
                "delta_analysis_minus_shadow_crps": float(np.nanmean(dc)),
                "performance_pass": bool(performance_pass),
            })
        mechanism_pass = bool(np.isfinite(erasure_hi) and (erasure_hi < 1.0 if args.mode == "formal" else np.nanmedian(shadow_erasure) <= 0.90))
        performance_pass = any(item.get("performance_pass", False) for item in family_results)
        case_results.append({
            "case": case,
            "erasure_ratio_median": float(np.nanmedian(shadow_erasure)) if shadow_erasure.size else float("nan"),
            "erasure_ratio_ci95": [erasure_lo, erasure_hi],
            "mechanism_pass": mechanism_pass,
            "performance_pass": performance_pass,
            "case_pass": bool(mechanism_pass and performance_pass),
            "families": family_results,
        })

    passed_cases = sum(int(item["case_pass"]) for item in case_results)
    gate_pass = passed_cases >= 2 and int(audit.get("valid_rows", 0)) == 4 * 3 * args.seed_count and not audit.get("missing") and not audit.get("failed_or_invalid")
    payload = {
        "mode": args.mode,
        "root": str(args.root),
        "expected_valid_rows": 12 * args.seed_count,
        "audit": audit,
        "case_results": case_results,
        "passed_cases": passed_cases,
        "gate_pass": gate_pass,
        "interpretation": "formal_allowed=true only when at least two systems pass both mechanism and performance gates",
    }
    (aggregate / f"shadow_ablation_{args.mode}_gate.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [f"mode={args.mode}", f"gate_pass={gate_pass}", f"passed_cases={passed_cases}", f"valid_rows={audit.get('valid_rows')}" ]
    for item in case_results:
        lines.append(f"{item['case']}: mechanism_pass={item['mechanism_pass']} performance_pass={item['performance_pass']} erasure_median={item['erasure_ratio_median']:.6g}")
    (aggregate / f"shadow_ablation_{args.mode}_gate.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if not gate_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
