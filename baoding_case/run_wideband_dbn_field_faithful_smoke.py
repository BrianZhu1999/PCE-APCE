#!/usr/bin/env python3
"""Faithful first field smoke for the public wideband DBN-LA-NM code.

This runner keeps the public tracker update order and uses paper-aligned raw
data after packet repair/resampling. It intentionally has no GPS correction,
trust region, trajectory alignment, or covariance rescue. A numerical failure
is therefore an admission failure, not silently repaired output.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np


PAPER_NODES = (1, 3, 5, 6, 7, 8, 11, 13)
NODE_TO_IP = {1: 47, 3: 40, 5: 54, 6: 43, 7: 49, 8: 61, 11: 5, 13: 46}
CHANNEL_GROUPS = {"z": [19, 18, 17, 13, 14, 15, 16], "x": [9, 8, 7, 1, 2, 3], "y": [12, 11, 10, 4, 5, 6]}
PAPER_XYV = ((38614853.4, 4337388.27, 25.85033, 23.00194), (38615012.2, 4336467.20, -41.09208, 6.67753), (38615647.2, 4337215.10, 3.20862, -41.49795))
TARGET_Z = (222.2, 227.017, 250.2)
FS = 3000.0
SNAPSHOT = 2048
PAPER_PROCESSING_HEIGHT = 230.0


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_nod(path: Path) -> dict[int, tuple[float, float, float]]:
    out = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = line.split()
        if len(fields) >= 6:
            out[int(fields[2])] = (float(fields[3]), float(fields[4]), float(fields[5]))
    return out


def array_geometry(node_xyz: tuple[float, float, float]) -> np.ndarray:
    x0, y0, z0 = node_xyz
    positions = np.zeros((19, 3), dtype=float)
    offsets = (-3.0, -2.0, -1.0, 1.0, 2.0, 3.0)
    for channel, offset in zip(CHANNEL_GROUPS["x"], offsets):
        positions[channel - 1] = (x0 + 0.5 * offset, y0, z0)
    for channel, offset in zip(CHANNEL_GROUPS["y"], offsets):
        positions[channel - 1] = (x0, y0 + 0.5 * offset, z0)
    for channel, offset in zip(CHANNEL_GROUPS["z"], (-3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 0.0)):
        positions[channel - 1] = (x0, y0, z0 + 0.5 * offset)
    return positions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--sync-root", type=Path, required=True)
    parser.add_argument("--nod", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--particles", type=int, default=64)
    parser.add_argument("--outer-iterations", type=int, default=5)
    parser.add_argument("--newton-iterations", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--init-mode", choices=("public", "eq44", "eq44eq45"), default="eq44eq45")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    np.random.seed(args.seed)
    if str(args.code_root.resolve()) not in sys.path:
        sys.path.insert(0, str(args.code_root.resolve()))
    tracking = load_module(args.code_root / "Tracking.py", "upstream_tracking_field")
    nodes = parse_nod(args.nod)
    center = np.mean(np.asarray([nodes[node] for node in PAPER_NODES]), axis=0)
    settings = type("PaperFieldSettings", (), {})()
    settings.C = 340.0
    settings.fs = FS
    settings.SnapshotLen = SNAPSHOT
    settings.dt = SNAPSHOT / FS
    settings.fr = np.arange(3.0, 1497.0 + 0.1, 3.0)
    settings.P = 19 * len(PAPER_NODES)
    settings.N = len(PAPER_NODES)
    settings.M = 3
    settings.NodePosition = np.asarray([np.asarray(nodes[node]) - center for node in PAPER_NODES])
    full_array = np.vstack([array_geometry(tuple(nodes[node])) for node in PAPER_NODES])
    settings.ArrayShape = full_array[:, :2] - center[:2]
    settings.sensor_z = full_array[:, 2] - center[2]
    # The field section of the paper explicitly replaces the 3-D range by a
    # fixed processing plane at z=230 m. Node elevations and target-specific
    # GPS heights are retained only as provenance, not in the steering range.
    settings.target_z = np.full(3, PAPER_PROCESSING_HEIGHT, dtype=float)
    settings.x0 = [np.asarray([[x - center[0]], [y - center[1]], [vx], [vy]], dtype=float) for x, y, vx, vy in PAPER_XYV]
    settings.lambda_0 = [100.0 for _ in settings.fr]
    settings.Amp = [[1.0, 1.0] for _ in settings.fr]
    tracking.DbnTracking.SysSet = settings
    tracker = tracking.DbnTracking(sigma=10)
    tracker.SysSet = settings
    tracker.ParticalNum = args.particles
    tracker.Niter = args.newton_iterations
    tracker.Fk = np.eye(4) + tracker.F * settings.dt
    tracker._active_target = 0

    def g3(self, x):
        xy = np.asarray(x, dtype=float).reshape(-1, 1, 2)
        dxy = xy - self.SysSet.ArrayShape.reshape(1, -1, 2)
        distance = np.sqrt(np.sum(dxy * dxy, axis=-1) + PAPER_PROCESSING_HEIGHT**2)
        return 1.0 / np.maximum(distance, 1e-9), distance

    def a3(self, f, x):
        gain, distance = self.g(x)
        phase = np.exp(-1j * 2 * np.pi * f * distance / self.SysSet.C)
        return gain * phase, phase, gain

    original_ex_ax = tracker.EX_Ax

    def ex_ax(self, f):
        # Keep the upstream Monte-Carlo approximation, but activate the target
        # height used by the 3-D steering model for each target in turn.
        values = []
        for target in range(self.SysSet.M):
            self._active_target = target
            values.append(original_ex_ax(f))
        ex_aha = np.zeros((self.SysSet.M, self.SysSet.M))
        ex_a = np.zeros((self.SysSet.P, self.SysSet.M), dtype=complex)
        for target, (aha, a) in enumerate(values):
            ex_aha[target, target] = aha[target, target]
            ex_a[:, target] = a[:, target]
        return ex_aha, ex_a

    original_grad_hess = tracker.GradHess_xk_f

    def grad_hess(self, xk, m, fi, f):
        self._active_target = m
        return original_grad_hess(xk, m, fi, f)

    tracker.g = g3.__get__(tracker)
    tracker.A = a3.__get__(tracker)
    tracker.EX_Ax = ex_ax.__get__(tracker)
    tracker.GradHess_xk_f = grad_hess.__get__(tracker)

    streams = {}
    for node in PAPER_NODES:
        path = args.sync_root / f"node{node}_ip{NODE_TO_IP[node]}_3khz.npy"
        streams[node] = np.load(path, mmap_mode="r")
        if streams[node].shape[0] != 19 or streams[node].shape[1] < args.frames * SNAPSHOT:
            raise RuntimeError(f"invalid synchronized stream: {path} {streams[node].shape}")

    records = []
    lambda_initialization = None
    source_initialization = None
    for frame in range(args.frames):
        block = np.vstack([streams[node][:, frame * SNAPSHOT:(frame + 1) * SNAPSHOT] for node in PAPER_NODES])
        tracker.input(block)
        tracker.Predictxk()
        if not np.isfinite(np.asarray(tracker.predictedxk, dtype=float)).all():
            raise FloatingPointError(f"non-finite prediction at frame {frame}")
        if frame == 0 and args.init_mode in ("eq44", "eq44eq45"):
            # Paper Eq. (44): initialize the subband noise precision from the
            # orthogonal residual of the initial steering model. The public
            # tracker stores this quantity as lambda_0.
            values = []
            source_norms = []
            for fi, frequency in enumerate(settings.fr):
                initial_xy = np.asarray([state[:2, 0] for state in settings.x0], dtype=float)
                tracker._active_target = 0
                initial_a, _, _ = tracker.A(float(frequency), initial_xy)
                initial_a = np.asarray(initial_a, dtype=complex).T
                z = tracker.zk[fi].reshape(-1, 1)
                gram = initial_a.conj().T.dot(initial_a)
                source = np.linalg.solve(gram + np.eye(settings.M) * 1e-10, initial_a.conj().T.dot(z)).reshape(settings.M)
                if args.init_mode == "eq44eq45":
                    for target in range(settings.M):
                        tracker.sk[fi][target].m.g = source[target]
                    source_norms.append(float(np.linalg.norm(source)))
                    residual = z - initial_a.dot(source.reshape(-1, 1))
                else:
                    residual = z - initial_a.dot(np.ones((settings.M, 1), dtype=complex))
                precision = float(max(1e-8, (settings.P - settings.M) / max(1e-12, np.vdot(residual, residual).real)))
                tracker.lambda_0[fi].a = max(1e-8, precision)
                tracker.lambda_0[fi].b = 1.0
                values.append(precision)
            lambda_initialization = {"method": "paper_eq44_initial_source_residual", "min": min(values), "median": float(np.median(values)), "max": max(values)}
            if args.init_mode == "eq44eq45":
                source_initialization = {"method": "paper_eq45_initial_least_squares_source", "norm_min": min(source_norms), "norm_median": float(np.median(source_norms)), "norm_max": max(source_norms)}
        for iteration in range(args.outer_iterations):
            tracker.UpdateLambda_0(); tracker.UpdateLambda(); tracker.UpdateMu(); tracker.UpdateSk(); tracker.Updatexk()
            if not np.isfinite(np.asarray(tracker.xk, dtype=float)).all():
                raise FloatingPointError(f"non-finite state at frame {frame}, iteration {iteration + 1}")
        for target, state in enumerate(tracker.xk, 1):
            records.append({"frame_index": frame, "time_s": frame * settings.dt, "target": target, "estimated_x": float(state[0, 0] + center[0]), "estimated_y": float(state[1, 0] + center[1]), "estimated_vx": float(state[2, 0]), "estimated_vy": float(state[3, 0])})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    manifest = {"claim_status": "paper_dbn_field_faithful_smoke", "public_code": "https://github.com/zhangwenqiong2017/MTT_WB_DBN", "paper_nodes": list(PAPER_NODES), "node_to_ip": NODE_TO_IP, "sync_root": str(args.sync_root), "frames": args.frames, "sample_rate_hz": FS, "snapshot_len": SNAPSHOT, "frequency_count": len(settings.fr), "frequency_grid_hz": "3,6,...,1497", "particles": args.particles, "outer_iterations": args.outer_iterations, "newton_iterations": args.newton_iterations, "seed": args.seed, "init_mode": args.init_mode, "geometry": "19-channel cross array, 0.5 m spacing; paper fixed processing plane", "processing_plane_height_m": PAPER_PROCESSING_HEIGHT, "lambda_initialization": lambda_initialization, "source_initialization": source_initialization, "gps_runtime_observation": False, "trust_region": False, "covariance_rescue": False, "output_csv": str(csv_path), "warning": "First paper-protocol field smoke; no admission claim until negative controls and full-interval stability pass."}
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
