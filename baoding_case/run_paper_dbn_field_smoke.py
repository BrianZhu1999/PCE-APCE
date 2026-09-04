#!/usr/bin/env python3
"""Smoke-adapt the public Zhang--Bao DBN tracker to Baoding field data.

The public repository is a synthetic demo. This adapter only tests whether its
DBN state-update machinery can consume the paper-matched eight-node Baoding
raw streams. It is not a reproduction claim until the full field protocol,
frequency grid, initialization, and all convergence diagnostics are frozen.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
import types
import types as _types
from copy import deepcopy
from pathlib import Path

import numpy as np


PAPER_NODES = (1, 3, 5, 6, 7, 8, 11, 13)
NODE_TO_IP = {1: 47, 2: 48, 3: 40, 5: 54, 6: 43, 7: 49, 8: 61, 11: 5, 13: 46}
CHANNEL_GROUPS = {"z": [19, 18, 17, 13, 14, 15, 16], "x": [9, 8, 7, 1, 2, 3], "y": [12, 11, 10, 4, 5, 6]}
XY_CHANNELS = [channel - 1 for channel in CHANNEL_GROUPS["x"] + CHANNEL_GROUPS["y"]]
PAPER_XYV = ((38614853.4, 4337388.27, 25.85033, 23.00194), (38615012.2, 4336467.20, -41.09208, 6.67753), (38615647.2, 4337215.10, 3.20862, -41.49795))
PAPER_TARGET_Z = (222.2, 227.017, 250.2)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def parse_nod(path: Path) -> dict[int, tuple[float, float, float]]:
    output = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = line.split()
        if len(fields) >= 6:
            output[int(fields[2])] = (float(fields[3]), float(fields[4]), float(fields[5]))
    return output


def array_positions(node_xy: tuple[float, float], spacing: float = 0.5) -> np.ndarray:
    x0, y0 = node_xy
    positions = np.zeros((19, 2), dtype=float)
    xy_offsets = (-3.0, -2.0, -1.0, 1.0, 2.0, 3.0)
    for channel, offset in zip(CHANNEL_GROUPS["x"], xy_offsets):
        positions[channel - 1] = (x0 + spacing * offset, y0)
    for channel, offset in zip(CHANNEL_GROUPS["y"], xy_offsets):
        positions[channel - 1] = (x0, y0 + spacing * offset)
    for channel in CHANNEL_GROUPS["z"]:
        index = CHANNEL_GROUPS["z"].index(channel) - 3.0
        positions[channel - 1] = (x0, y0)
    return positions


def hms_seconds(value: int | float) -> float:
    text = str(int(float(value))).zfill(6)
    return int(text[:2]) * 3600 + int(text[2:4]) * 60 + int(text[4:])


def gps_at(path: Path, time_seconds: float) -> tuple[float, float, float]:
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = line.split()
        if len(fields) >= 8:
            try: rows.append((abs(hms_seconds(fields[7]) - time_seconds), float(fields[4]), float(fields[5]), float(fields[6])))
            except ValueError: pass
    _, x, y, z = min(rows)
    return x, y, z


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--remote-root", type=Path, required=True)
    parser.add_argument("--segment-name", default="sanyuan_tongxinyuan_6")
    parser.add_argument("--nod", type=Path, required=True)
    parser.add_argument("--start-hhmmss", type=int, default=132754)
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--freq-step", type=int, default=30)
    parser.add_argument("--freq-start", type=int, default=3)
    parser.add_argument("--freq-stop", type=int, default=1497)
    parser.add_argument("--particles", type=int, default=100)
    parser.add_argument("--snapshot-len", type=int, default=640)
    parser.add_argument("--height-mode", choices=("paper230", "target_specific"), default="paper230")
    parser.add_argument("--xy-only", action="store_true")
    parser.add_argument("--max-step-m", type=float, default=0.0)
    parser.add_argument("--state-damping", type=float, default=1.0)
    parser.add_argument("--cov-floor", type=float, default=1e-8)
    parser.add_argument("--sound-speed", type=float, default=340.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base = load_module(args.code_root / "shuangyuan_dual_frontend.py", "baoding_base")
    # The public demo imports matplotlib only for optional plotting; the field
    # smoke uses no plotting path and Super-Server CPU environment need not
    # carry the plotting dependency.
    sys.modules.setdefault("matplotlib", types.ModuleType("matplotlib"))
    sys.modules.setdefault("matplotlib.pyplot", types.ModuleType("matplotlib.pyplot"))
    public_root = args.code_root / "MTT_WB_DBN"
    if str(public_root) not in sys.path:
        sys.path.insert(0, str(public_root))
    tracking = load_module(public_root / "Tracking.py", "public_tracking")
    settings = type("PaperFieldSettings", (), {})()
    nodes = parse_nod(args.nod)
    center = np.mean(np.asarray([nodes[node][:2] for node in PAPER_NODES]), axis=0)
    settings.C = float(args.sound_speed); settings.fs = 3050; settings.SnapshotLen = args.snapshot_len; settings.dt = settings.SnapshotLen / settings.fs
    settings.fr = np.arange(args.freq_start, args.freq_stop + 1, args.freq_step, dtype=float); settings.P = (12 if args.xy_only else 19) * len(PAPER_NODES); settings.N = len(PAPER_NODES); settings.M = 3
    settings.NodePosition = np.asarray([np.asarray(nodes[node][:2]) - center for node in PAPER_NODES])
    full_array = np.vstack([array_positions(tuple(settings.NodePosition[index])) for index in range(len(PAPER_NODES))])
    settings.ArrayShape = full_array.reshape(len(PAPER_NODES), 19, 2)[:, XY_CHANNELS, :].reshape(-1, 2) if args.xy_only else full_array
    settings.x0 = []
    for x, y, vx, vy in PAPER_XYV:
        settings.x0.append(np.asarray([[x - center[0]], [y - center[1]], [vx], [vy]], dtype=float))
    # Keep the public DBN state in the horizontal plane, but use the actual
    # target altitudes and node elevations in the near-field propagation
    # distance. This replaces the old flat 2-D range with a target-aware 3-D
    # spherical spreading and phase-delay model.
    settings.target_z = np.full(3, 230.0, dtype=float) if args.height_mode == "paper230" else np.asarray(PAPER_TARGET_Z, dtype=float)
    settings.node_z = np.asarray([nodes[node][2] for node in PAPER_NODES], dtype=float)
    settings.sensor_z = np.repeat(settings.node_z, 12 if args.xy_only else 19)
    settings.lambda_0 = [100.0 for _ in settings.fr]
    settings.Amp = [[1.0, 1.0] for _ in settings.fr]
    tracking.DbnTracking.SysSet = settings
    tracker = tracking.DbnTracking(sigma=10)
    tracker.SysSet = settings; tracker.ParticalNum = args.particles; tracker.Niter = 4
    tracker.Fk = np.eye(4) + tracker.F * settings.dt
    tracker._active_target = 0

    def _g3(self, x):
        xy = np.asarray(x, dtype=float).reshape(-1, 1, 2)
        dxy = xy - self.SysSet.ArrayShape
        dz = float(self.SysSet.target_z[self._active_target]) - self.SysSet.sensor_z.reshape(1, -1)
        d = np.sqrt(np.sum(dxy * dxy, axis=-1) + dz * dz)
        return 1.0 / np.maximum(d, 1e-6), d

    def _a3(self, f, x):
        g, d = self.g(x)
        tau = d / self.SysSet.C
        B = np.exp(-1j * 2 * np.pi * f * tau)
        return g * B, B, g

    tracker.g = _types.MethodType(_g3, tracker)
    tracker.A = _types.MethodType(_a3, tracker)

    def _ex_ax3(self, f):
        A = []
        for m in range(self.SysSet.M):
            self._active_target = m
            mean = self.xk[m][0:2, 0]
            cov = self.Rxk[m][0:2, 0:2]
            cov = (cov + cov.T) * 0.5 + np.eye(2) * 1e-8
            samples = np.random.multivariate_normal(mean, cov, self.ParticalNum)
            At, _, _ = self.A(f, samples)
            A.append(At)
        A = np.asarray(A)
        ex_A = A.mean(axis=1).T
        ex_AHA = np.zeros((self.SysSet.M, self.SysSet.M))
        ex_AHA[np.diag_indices(self.SysSet.M)] = (np.linalg.norm(A, axis=-1) ** 2).mean(axis=-1)
        return ex_AHA, ex_A

    original_grad_hess = tracker.GradHess_xk_f
    def _grad_hess3(self, xk, m, fi, f):
        self._active_target = m
        return original_grad_hess(xk, m, fi, f)

    tracker.EX_Ax = _types.MethodType(_ex_ax3, tracker)
    tracker.GradHess_xk_f = _types.MethodType(_grad_hess3, tracker)
    segment = args.remote_root / "20171107保定实验/project/20171107baoding" / args.segment_name
    sample_start = int((base.hms_seconds(args.start_hhmmss) - base.hms_seconds(132614)) * base.FS)
    streams = {}
    for node in PAPER_NODES:
        path = segment / f"20171107baoding_132614_{NODE_TO_IP[node]}_19.wavfm"
        data, metadata = base.decode_wavfm(path)
        streams[node] = data[XY_CHANNELS if args.xy_only else slice(0, 19), sample_start:sample_start + args.frames * settings.SnapshotLen]
    robust_scale = float(np.median([np.median(np.std(values, axis=1)) for values in streams.values()]))
    if not math.isfinite(robust_scale) or robust_scale <= 0:
        raise RuntimeError(f"invalid raw robust scale: {robust_scale}")
    streams = {node: values / robust_scale for node, values in streams.items()}

    def stabilize_covariances() -> None:
        for index, covariance in enumerate(tracker.Rxk):
            covariance = (np.asarray(covariance, dtype=float) + np.asarray(covariance, dtype=float).T) * 0.5
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            tracker.Rxk[index] = eigenvectors.dot(np.diag(np.maximum(eigenvalues, args.cov_floor))).dot(eigenvectors.T)

    def trust_region_update() -> None:
        if args.max_step_m <= 0 or not hasattr(tracker, "predictedxk"):
            return
        for index, state in enumerate(tracker.xk):
            predicted = np.asarray(tracker.predictedxk[index], dtype=float)
            updated = np.asarray(state, dtype=float)
            delta = updated[:2, 0] - predicted[:2, 0]
            norm = float(np.linalg.norm(delta))
            if norm > args.max_step_m:
                updated[:2, 0] = predicted[:2, 0] + delta * (args.max_step_m / norm)
                tracker.xk[index] = updated
    rows = []
    for frame in range(args.frames):
        block = np.vstack([streams[node][:, frame * settings.SnapshotLen:(frame + 1) * settings.SnapshotLen] for node in PAPER_NODES])
        stabilize_covariances(); tracker.input(block); tracker.Predictxk()
        predicted_states = [np.asarray(state, dtype=float).copy() for state in tracker.predictedxk]
        errors = []
        for _ in range(5):
            tracker.UpdateLambda_0(); tracker.UpdateLambda(); tracker.UpdateMu(); tracker.UpdateSk(); tracker.Updatexk()
            if args.state_damping < 1.0:
                for index, state in enumerate(tracker.xk):
                    tracker.xk[index] = predicted_states[index] + args.state_damping * (np.asarray(state, dtype=float) - predicted_states[index])
            trust_region_update(); stabilize_covariances()
        frame_time_seconds = base.hms_seconds(args.start_hhmmss) + frame * settings.dt
        time_code = base.seconds_hhmmss(frame_time_seconds)
        for target, state in enumerate(tracker.xk, 1):
            truth_file = {1: "GPS1_plane1.gps", 2: "GPS3_plane2.gps", 3: "GPS4_plane2to3.gps"}[target]
            truth = gps_at(args.remote_root / "20171107保定实验/GPS_data" / truth_file, frame_time_seconds)
            estimated = (float(state[0, 0] + center[0]), float(state[1, 0] + center[1]))
            errors.append({"target": target, "estimated_x": estimated[0], "estimated_y": estimated[1], "truth_x": truth[0], "truth_y": truth[1], "position_error_m": math.hypot(estimated[0] - truth[0], estimated[1] - truth[1]), "covariance_diag": [float(value) for value in np.diag(np.asarray(tracker.Rxk[target - 1], dtype=float))]})
        rows.append({"frame_index": frame, "time_hhmmss": time_code, "targets": errors})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {"claim_status": "paper_inspired_dbn_field_long_window", "public_code": "https://github.com/zhangwenqiong2017/MTT_WB_DBN", "paper_nodes": PAPER_NODES, "excluded_node": 2, "segment": str(segment), "start_hhmmss": args.start_hhmmss, "frames": args.frames, "freq_start_hz": args.freq_start, "freq_stop_hz": args.freq_stop, "freq_step_hz": args.freq_step, "frequency_count": len(settings.fr), "frequency_grid": f"{args.freq_start},{args.freq_start + args.freq_step},...,{args.freq_stop} Hz", "particles": args.particles, "xy_only": args.xy_only, "max_step_m": args.max_step_m, "state_damping": args.state_damping, "covariance_floor": args.cov_floor, "snapshot_len": settings.SnapshotLen, "sample_rate_hz": settings.fs, "sound_speed_mps": settings.C, "target_heights_m": settings.target_z.tolist(), "node_heights_m": settings.node_z.tolist(), "propagation_model": "3-D near-field spherical spreading: d=sqrt(dx^2+dy^2+dz^2), amplitude 1/d, phase delay d/C", "height_mode": args.height_mode, "height_source": "paper processing plane z=230 m" if args.height_mode == "paper230" else "paper-matched GPS initial heights at 13:27:54; fixed target-specific z prior", "raw_robust_scale": robust_scale, "covariance_projection": "eigenvalue floor cov_floor after each DBN state update", "rows": rows, "warning": "Paper-inspired approximate field baseline; not the authors private implementation."}
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
