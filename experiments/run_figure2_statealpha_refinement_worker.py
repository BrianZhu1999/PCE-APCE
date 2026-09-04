"""Resumable Figure 2 state--alpha alternative worker; never writes v58 paths."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments import figure2_statealpha_refinement as statealpha
from experiments import run_figure2_corrected_formal_worker as frozen
from experiments import run_figure2_reviewer_gate as reviewer

PROTOCOL = "figure2-latest-statealpha-refinement-smoke-5paired-seeds-20260813"


def parse_tasks(path: Path) -> list[tuple[str, str, int]]:
    tasks=[]
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"): continue
        case, method, seed = [item.strip() for item in line.split(",")]
        if case not in reviewer.CASES or method not in frozen.METHODS: raise ValueError(line)
        tasks.append((case, method, int(seed)))
    return tasks


def run_method(case: str, method: str, seed: int, device: torch.device, profile: dict[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if method in {"pce", "apce"}:
        row = statealpha.run_statealpha_refined(case, method, seed, device, profile)
        trace = row.pop("_statealpha_trace")
        row.update(case=case, method=method, label=frozen.LABELS[method], seed=seed, implementation_method="statealpha_refinement")
    else:
        with frozen.record_traces(method) as recorder:
            row=frozen.run_method(case, method, seed, device)
        trace=recorder.arrays()
    row.update(protocol=PROTOCOL, source_hash=reviewer.source_hash(reviewer.PROJECT_ROOT), torch_version=torch.__version__, cuda_available=bool(torch.cuda.is_available()), device_name=torch.cuda.get_device_name(device) if device.type=="cuda" else "cpu", worker_pid=os.getpid(), valid=reviewer.finite_metrics(row), status="completed")
    return row, trace


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--task-file",type=Path,required=True); parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--artifact-root",type=Path,required=True); parser.add_argument("--tuning-matrix",type=Path,required=True)
    parser.add_argument("--profile",default=statealpha.PROFILE_NAME); parser.add_argument("--device",default="cpu")
    args=parser.parse_args(); device=torch.device(args.device); profile=statealpha.load_v52_profile(args.tuning_matrix,args.profile)
    torch.set_num_threads(max(1,int(os.environ.get("FIG2_TORCH_THREADS","2"))))
    completed=failed=0
    for case,method,seed in parse_tasks(args.task_file):
        path=args.output/case/method/f"seed_{seed}.json"; trace_path=args.artifact_root/"method_traces"/case/method/f"seed_{seed}.npz"
        if path.exists() and trace_path.exists():
            try:
                existing=json.loads(path.read_text(encoding="utf-8"))
                if existing.get("status")=="completed" and existing.get("valid"): completed+=1; continue
            except Exception: pass
        started=time.perf_counter()
        try:
            metadata=frozen.save_common_assets(case,seed,device,args.artifact_root/"common_assets")
            row,trace=run_method(case,method,seed,device,profile)
            frozen._atomic_npz(trace_path,trace)
            row.update(common_asset_path=metadata["asset_path"],common_asset_sha256=metadata["arrays_sha256"],trace_path=str(trace_path),trace_sha256=frozen._sha256(trace_path),worker_device=str(device),worker_elapsed_seconds=float(time.perf_counter()-started))
            frozen._atomic_json(path,row); completed+=1
        except Exception as exc:
            failed+=1; frozen._atomic_json(path,{"case":case,"method":method,"seed":seed,"protocol":PROTOCOL,"status":"failed","valid":False,"error_type":type(exc).__name__,"error":str(exc),"worker_device":str(device)})
    frozen._atomic_json(args.output/"worker_status.json",{"status":"completed" if not failed else "completed_with_failures","completed":completed,"failed":failed,"device":str(device),"protocol":PROTOCOL})


if __name__ == "__main__": main()
