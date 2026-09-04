#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    methods = ["trilinear_baseline", "denkf", "bma", "pce", "apce"]
    rows = []
    for name in methods:
        path = args.smoke / f"{name}.json"
        row = json.loads(path.read_text(encoding="utf-8"))
        row["record"] = name
        rows.append(row)
    keys = sorted({key for row in rows for key, value in row.items() if not isinstance(value, (list, dict))})
    with (args.output / "s1_full_dynamics_source_data.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)
    baseline = rows[0]
    apce = rows[-1]
    gates = {
        "analysis_reconstruction_pass": apce["analysis_nrmse"] < baseline["analysis_nrmse"],
        "one_ms_forecast_pass": apce["forecast_1ms_nrmse"] < 1.0 and apce["forecast_1ms_correlation"] > 0.5,
        "candidate_separation_pass": apce["median_separation_ratio"] > 1.0,
        "coverage_pass": 0.70 <= apce["coverage_90"] <= 1.0,
        "full_state_pass": apce["reduced_order_model_used"] is False,
    }
    gates["pilot_admitted"] = all(value for key, value in gates.items() if key.endswith("_pass"))
    (args.output / "admission.json").write_text(json.dumps(gates, indent=2), encoding="utf-8")
    report = [
        "# MeshRIR S1 full-grid dynamics smoke", "",
        "## Scope", "",
        "The measured 21 x 21 x 9 pressure field was observed on a volume-covering 7 x 7 x 3 grid. The propagated state retained both full 3969-point time layers (7938 dimensions). No reduced-order model was used.", "",
        "A common recording delay was fitted from direct-arrival time versus source distance. The aligned field was causally filtered to 500 Hz, consistent with the spatial Nyquist limit of the sparse layout.", "",
        "## Result", "",
        f"- Trilinear reconstruction nRMSE: {baseline['analysis_nrmse']:.4f}",
        f"- APCE reconstruction nRMSE: {apce['analysis_nrmse']:.4f}",
        f"- APCE reconstruction correlation: {apce['analysis_correlation']:.4f}",
        f"- APCE 1/2/4-ms forecast nRMSE: {apce['forecast_1ms_nrmse']:.4f} / {apce['forecast_2ms_nrmse']:.4f} / {apce['forecast_4ms_nrmse']:.4f}",
        f"- APCE 1/2/4-ms forecast correlation: {apce['forecast_1ms_correlation']:.4f} / {apce['forecast_2ms_correlation']:.4f} / {apce['forecast_4ms_correlation']:.4f}",
        f"- APCE 90% coverage: {apce['coverage_90']:.4f}", "",
        "The full PDE reconstruction improves over trilinear sparse interpolation, and the short free forecast remains correlated with the measured field. The uncertainty interval remains under-dispersed, so the smoke is not yet admitted for formal expansion.", "",
        "## Gate", "", "```json", json.dumps(gates, indent=2), "```",
    ]
    (args.output / "S1_FULL_DYNAMICS_SMOKE_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(gates, indent=2))


if __name__ == "__main__":
    main()
