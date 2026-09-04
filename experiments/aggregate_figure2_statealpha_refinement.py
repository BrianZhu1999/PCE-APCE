"""Aggregate the independent Figure 2 state--alpha smoke/formal results."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments import aggregate_figure2_corrected_formal as base


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def read_v58_rows(root: Path, seeds: set[int]) -> list[dict[str, Any]]:
    rows=[]
    for path in sorted(root.rglob("seed_*.json")):
        try: row=json.loads(path.read_text(encoding="utf-8"))
        except Exception: continue
        if base.valid_row(row) and int(row.get("seed", -1)) in seeds:
            row["_path"]=str(path); rows.append(row)
    return rows


def trace_index(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"case":row["case"],"method":row["method"],"seed":row["seed"],"trace_path":row.get("trace_path",""),"trace_sha256":row.get("trace_sha256","")} for row in rows]


def cross_version_pairs(new_rows: list[dict[str, Any]], old_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    old={(str(r["case"]),str(r["method"]),int(r["seed"])):r for r in old_rows}
    output=[]
    for row in new_rows:
        if row["method"] not in {"pce","apce"}: continue
        reference=old.get((str(row["case"]),str(row["method"]),int(row["seed"])))
        if reference is None: continue
        for metric in ("nrmse","crps","alpha_absolute_error","coverage_90","interval_width_90","runtime_seconds","peak_gpu_memory_mb"):
            new, prior=base.numeric(row.get(metric)),base.numeric(reference.get(metric))
            if not (math.isfinite(new) and math.isfinite(prior)): continue
            output.append({"case":row["case"],"method":row["method"],"seed":row["seed"],"metric":metric,"new_value":new,"v58_value":prior,"new_minus_v58":new-prior,"new_better":(new-prior)<0 if metric!="coverage_90" else abs(new-0.90)<abs(prior-0.90)})
    return output


def decision_gate(summary: list[dict[str, Any]], cross: list[dict[str, Any]], audit: dict[str, Any]) -> dict[str, Any]:
    improvements={}
    coverage_worse={}
    for case in base.CASES:
        improvements[case]=False; coverage_worse[case]=False
        for method in ("pce","apce"):
            lookup={(r["case"],r["method"],r["metric"]):r for r in cross}
            nrmse=[r for r in cross if r["case"]==case and r["method"]==method and r["metric"]=="nrmse"]
            crps=[r for r in cross if r["case"]==case and r["method"]==method and r["metric"]=="crps"]
            coverage=[r for r in cross if r["case"]==case and r["method"]==method and r["metric"]=="coverage_90"]
            if nrmse and crps:
                improvements[case] |= float(np.mean([r["new_minus_v58"] for r in nrmse])) < 0 and float(np.mean([r["new_minus_v58"] for r in crps])) < 0
            if coverage:
                coverage_worse[case] |= not bool(np.mean([r["new_better"] for r in coverage]) >= 0.5)
    improved_cases=sum(improvements.values()); worsened_coverage=sum(coverage_worse.values())
    pass_gate=not audit["missing"] and not audit["failed_or_invalid"] and improved_cases>=2 and worsened_coverage<=1
    return {"formal_gate_passed":pass_gate,"improved_cases":improved_cases,"case_improves_nrmse_and_crps":improvements,"coverage_error_worse_cases":worsened_coverage,"coverage_error_worse_by_case":coverage_worse,"reason":"eligible for 50 paired-seed formal" if pass_gate else "retain as smoke/supplement or reject; do not replace v58"}


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,required=True); parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--v58-root",type=Path,required=True); parser.add_argument("--seed-base",type=int,default=2026080700); parser.add_argument("--seed-count",type=int,default=5)
    parser.add_argument("--bootstrap-resamples",type=int,default=10000); parser.add_argument("--seed",type=int,default=2026081317)
    args=parser.parse_args(); seeds=set(range(args.seed_base,args.seed_base+args.seed_count))
    raw_rows=base.read_rows(args.root)
    # A failed first pass may coexist with a later valid retry.  Keep precisely
    # one valid record per paired task so statistics cannot double-count it.
    unique: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in raw_rows:
        key=(str(row["case"]),str(row["method"]),int(row["seed"]))
        existing=unique.get(key)
        if existing is None or "retry" in str(row.get("_path", "")):
            unique[key]=row
    rows=list(unique.values()); summary=base.summarize(rows,args.bootstrap_resamples,args.seed); paired=base.paired(rows,args.bootstrap_resamples,args.seed)
    audit=base.audit(rows,args.root) if args.seed_count==50 else _smoke_audit(rows,args.root,seeds)
    old=read_v58_rows(args.v58_root,seeds); cross=cross_version_pairs(rows,old); decision=decision_gate(summary,cross,audit)
    args.output.mkdir(parents=True,exist_ok=True)
    write_csv(args.output/"figure2_statealpha_run_source_data.csv",rows); write_csv(args.output/"figure2_statealpha_method_summary.csv",summary)
    write_csv(args.output/"figure2_statealpha_paired_comparisons.csv",paired); write_csv(args.output/"figure2_statealpha_vs_v58_paired_deltas.csv",cross); write_csv(args.output/"figure2_statealpha_trace_index.csv",trace_index(rows))
    (args.output/"figure2_statealpha_aggregate.json").write_text(json.dumps({"root":str(args.root),"v58_root":str(args.v58_root),"audit":audit,"decision_gate":decision,"valid_rows":len(rows),"v58_rows":len(old)},indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps({"valid_rows":len(rows),"v58_rows":len(old),"missing":len(audit["missing"]),"gate":decision["formal_gate_passed"]},ensure_ascii=False))


def _smoke_audit(rows: list[dict[str, Any]], root: Path, seeds: set[int]) -> dict[str, Any]:
    expected={(case,method,seed) for case in base.CASES for method in base.METHODS for seed in seeds}
    present={(str(r["case"]),str(r["method"]),int(r["seed"])) for r in rows}
    failed=[]
    for path in root.rglob("seed_*.json"):
        try: row=json.loads(path.read_text(encoding="utf-8"))
        except Exception: continue
        key = (str(row.get("case")), str(row.get("method")), int(row.get("seed", -1)))
        # A later retry in a separate worker directory is the authoritative run.
        if key not in present and (row.get("status") != "completed" or not row.get("valid", False)):
            failed.append({"path":str(path),"case":row.get("case"),"method":row.get("method"),"seed":row.get("seed"),"error":row.get("error")})
    return {"expected_rows":len(expected),"valid_rows":len(rows),"missing":[{"case":c,"method":m,"seed":s} for c,m,s in sorted(expected-present)],"failed_or_invalid":failed}


if __name__ == "__main__": main()
