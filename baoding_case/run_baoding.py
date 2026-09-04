#!/usr/bin/env python3
"""Remote 2017 Baoding MUSIC frontend and PCE/APCE trajectory runner.

The script intentionally uses only the Python standard library and PyTorch so
the Super-Server torch312env does not need a new scientific Python install.
It is an independent real-data transfer experiment, not a Figure 2/3 runner.
"""
from __future__ import annotations

import argparse
import array
import csv
import hashlib
import json
import math
import os
import re
import time
import wave
from pathlib import Path
from typing import Iterable

import torch

try:
    from . import ALPHA_GRID, CHANNEL_GROUPS, NODE_IDS
except ImportError:
    from __init__ import ALPHA_GRID, CHANNEL_GROUPS, NODE_IDS

IP_TO_NODE = {61: 8, 5: 11, 47: 1, 43: 6, 48: 2, 54: 5, 49: 7, 46: 13, 40: 3}
NODE_TO_IP = {v: k for k, v in IP_TO_NODE.items()}
CHANNELS = tuple(range(1, 20))
STAGE_DIR = "20171107保定实验/2017保定实验/20171107保定实验/project/20171107baoding"
GPS_DIR = "20171107保定实验/2017保定实验/20171107保定实验/GPS_data"


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def hms_seconds(value: str) -> int:
    value = value.replace(":", "").strip()
    value = value.zfill(6)
    return int(value[:2]) * 3600 + int(value[2:4]) * 60 + int(value[4:])


def angle_residual(pred: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
    return torch.remainder(pred - obs + math.pi, 2.0 * math.pi) - math.pi


def circular_deg(value: float) -> float:
    return float(value % 360.0)


def signed_deg(value: float) -> float:
    return float((value + 180.0) % 360.0 - 180.0)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_nod(path: Path) -> dict[int, dict[str, float]]:
    nodes: dict[int, dict[str, float]] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = line.split()
        if len(fields) < 10:
            continue
        try:
            ip = int(fields[0].split(".")[-1])
            network, node_id = int(fields[1]), int(fields[2])
            nodes[node_id] = {
                "ip": ip, "node_id": node_id, "network_id": network,
                "x": float(fields[3]), "y": float(fields[4]), "z": float(fields[5]),
                "node_type": int(fields[6]), "azimuth_offset": float(fields[7]),
                "elevation_offset": float(fields[8]), "azimuth_positive": float(fields[9]),
                "elevation_positive": float(fields[10]) if len(fields) > 10 else 0.0,
            }
        except (ValueError, IndexError):
            continue
    return nodes


def parse_gps(path: Path) -> list[tuple[int, float, float, float]]:
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = line.split()
        if len(fields) < 8:
            continue
        try:
            rows.append((hms_seconds(fields[7]), float(fields[4]), float(fields[5]), float(fields[6])))
        except ValueError:
            continue
    return rows


def discover_channel_groups(segment_dir: Path) -> dict[int, list[Path]]:
    groups: dict[tuple[int, str], dict[int, Path]] = {}
    pattern = re.compile(r"_(\d+)_19_(?:Ch)?(\d+)\.wav$", re.IGNORECASE)
    for path in segment_dir.rglob("*.wav"):
        match = pattern.search(path.name)
        if not match:
            continue
        ip, channel = int(match.group(1)), int(match.group(2))
        groups.setdefault((ip, path.name[: match.start() + len(match.group(1)) + 4]), {})[channel] = path
    selected: dict[int, list[Path]] = {}
    for (ip, _stem), files in groups.items():
        if ip not in IP_TO_NODE or not all(channel in files for channel in CHANNELS):
            continue
        selected[IP_TO_NODE[ip]] = [files[channel] for channel in CHANNELS]
    return selected


def wav_meta(path: Path) -> tuple[int, int, int, int]:
    with wave.open(str(path), "rb") as stream:
        return stream.getframerate(), stream.getnchannels(), stream.getsampwidth(), stream.getnframes()


def read_block(streams: list[wave.Wave_read], start: int, length: int) -> torch.Tensor:
    rows = []
    for stream in streams:
        stream.setpos(min(start, stream.getnframes()))
        raw = stream.readframes(length)
        sample_width = stream.getsampwidth()
        if sample_width == 2:
            values = array.array("h"); values.frombytes(raw)
        elif sample_width == 3:
            values = [int.from_bytes(raw[index:index + 3], "little", signed=True) for index in range(0, len(raw) - 2, 3)]
        elif sample_width == 4:
            values = array.array("i"); values.frombytes(raw)
        else:
            raise RuntimeError(f"unsupported PCM sample width: {sample_width}")
        if len(values) < length:
            values.extend([0] * (length - len(values)))
        rows.append(torch.tensor(values[:length], dtype=torch.float32))
    return torch.stack(rows)


def steering(positions: torch.Tensor, frequencies: torch.Tensor, angles: torch.Tensor, sound_speed: float) -> torch.Tensor:
    phase = 2.0 * math.pi * frequencies[:, None, None] / sound_speed
    phase = phase * positions[None, None, :] * angles[None, :, None]
    return torch.exp(1j * phase).permute(0, 2, 1)


def music_1d(data: torch.Tensor, positions: list[float], fs: int, cfg: dict, axis: str) -> tuple[float, float]:
    nfft, snapshots = int(cfg["nfft"]), int(cfg["snapshots"])
    band_low = float(cfg["fc_hz"]) - float(cfg["bandwidth_hz"]) / 2.0
    band_high = float(cfg["fc_hz"]) + float(cfg["bandwidth_hz"]) / 2.0
    frames = data[:, : nfft * snapshots].reshape(data.shape[0], snapshots, nfft)
    # Historical freq_decomp.m uses the unwindowed FFT; its Hanning line is
    # explicitly commented out. Preserve that executed convention here.
    spectrum = torch.fft.rfft(frames, dim=-1) / math.sqrt(nfft)
    bins = [k for k in range(spectrum.shape[-1]) if band_low <= k * fs / nfft <= band_high]
    if not bins:
        bins = [max(1, int(round(float(cfg["fc_hz"]) * nfft / fs)))]
    angle_step = float(cfg["angle_step_deg"])
    angle_max = 360.0 if axis == "azimuth" else 90.0
    grid = torch.arange(0.0, angle_max, angle_step, dtype=torch.float32) * math.pi / 180.0
    pos = torch.tensor(positions, dtype=torch.float32)
    denominator_sum = torch.zeros(grid.shape[0], dtype=torch.float64)
    # The archived MATLAB implementation accumulates every selected DFT bin.
    # Keep the historical accelerated stride as the default for old caches, but
    # expose it in the configuration so an exact all-bin replay can be audited.
    frequency_stride = max(1, int(cfg.get("frequency_stride", 2)))
    for index in bins[::frequency_stride]:
        snapshot = spectrum[:, :, index]
        covariance = snapshot @ snapshot.conj().transpose(0, 1) / snapshots
        _, vectors = torch.linalg.eigh(covariance)
        noise = vectors[:, : max(1, vectors.shape[1] - 1)]
        if axis == "azimuth":
            phase = 2.0 * math.pi * (index * fs / nfft) / float(cfg["sound_speed_mps"])
            # The historical implementation uses two six-element horizontal arms.
            a1 = torch.exp(1j * phase * pos[:, None] * torch.cos(grid)[None, :])
            a2 = torch.exp(1j * phase * pos[:, None] * torch.sin(grid)[None, :])
            steering_matrix = torch.cat((a1, a2), dim=0)
        else:
            phase = 2.0 * math.pi * (index * fs / nfft) / float(cfg["sound_speed_mps"])
            steering_matrix = torch.exp(1j * phase * pos[:, None] * torch.cos(grid)[None, :])
        denominator = (steering_matrix.conj().transpose(0, 1) @ noise).abs().square().sum(dim=1)
        # wb_MUSIC_xy/z sum a^H U_n U_n^H a across frequencies and invert once.
        denominator_sum += denominator.double()
    score = 1.0 / denominator_sum.clamp_min(1e-12)
    winner = int(torch.argmax(score))
    concentration = float(score[winner] / score.sum().clamp_min(1e-12))
    return float(grid[winner] * 180.0 / math.pi), concentration


def music_observation(block: torch.Tensor, fs: int, cfg: dict) -> tuple[float, float, float]:
    spacing = float(cfg["geometry_spacing_m"])
    horizontal = [-3.0 * spacing, -2.0 * spacing, -spacing, spacing, 2.0 * spacing, 3.0 * spacing]
    geometry_profile = cfg.get("geometry_profile", "legacy_nonuniform_z")
    if geometry_profile == "paper_uniform_cross":
        vertical = [-3.0 * spacing, -2.0 * spacing, -1.0 * spacing, 0.0, 1.0 * spacing, 2.0 * spacing, 3.0 * spacing]
    elif geometry_profile == "legacy_nonuniform_z":
        vertical = [-2.13 * spacing, -1.53 * spacing, -0.93 * spacing, 0.0, 1.0 * spacing, 2.0 * spacing, 3.0 * spacing]
    else:
        raise ValueError(f"unknown geometry_profile={geometry_profile!r}")
    x_idx = [channel - 1 for channel in CHANNEL_GROUPS["x"]]
    y_idx = [channel - 1 for channel in CHANNEL_GROUPS["y"]]
    z_idx = [channel - 1 for channel in CHANNEL_GROUPS["z"]]
    az, az_conc = music_1d(torch.cat((block[x_idx], block[y_idx])), horizontal, fs, cfg, "azimuth")
    el_zenith, el_conc = music_1d(block[z_idx], vertical, fs, cfg, "elevation")
    # Historical wb_MUSIC_z scans the angle from the array z-axis. Convert its
    # zenith convention to the ENU elevation used by the tracking observation.
    el = 90.0 - el_zenith
    return az, el, min(az_conc, el_conc)


def run_frontend(cfg: dict, output: Path, max_seconds: int | None = None) -> None:
    remote_root = Path(cfg["remote_root"])
    nod_path = remote_root / "20171107保定实验/GPS_data/20171107baoding.nod"
    nodes = parse_nod(nod_path)
    frontend_nodes = tuple(int(node) for node in cfg.get("frontend_nodes", NODE_IDS))
    if not set(frontend_nodes).issubset(nodes):
        raise RuntimeError(f"frontend nodes {frontend_nodes} are not present in nod file: {sorted(nodes)}")
    gps_path = remote_root / "20171107保定实验/GPS_data/GPS1_plane1.gps"
    gps = parse_gps(gps_path)
    if not gps:
        raise RuntimeError("GPS1_plane1.gps has no rows")
    gps_times = [row[0] for row in gps]
    gps_xyz = [[row[1] - sum(n["x"] for n in nodes.values()) / len(nodes), row[2] - sum(n["y"] for n in nodes.values()) / len(nodes), row[3] - sum(n["z"] for n in nodes.values()) / len(nodes)] for row in gps]
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"nodes": nodes, "gps_source": str(gps_path), "gps_source_sha256": sha256(gps_path), "segments": [], "config": cfg}
    all_rows = []
    hop_seconds = float(cfg.get("frontend_hop_seconds", 1.0))
    if hop_seconds <= 0.0:
        raise ValueError("frontend_hop_seconds must be positive")
    stage_root = remote_root / "20171107保定实验/project/20171107baoding"
    for segment in cfg["segments"]:
        segment_dir = stage_root / segment["name"]
        groups = discover_channel_groups(segment_dir)
        missing_nodes = sorted(set(frontend_nodes) - set(groups))
        if len(groups) < 3:
            raise RuntimeError(f"{segment['name']} has fewer than three usable nodes: {sorted(groups)}")
        metadata = {node: wav_meta(groups[node][0]) for node in groups}
        fs_values = {item[0] for item in metadata.values()}
        if len(fs_values) != 1:
            raise RuntimeError(f"inconsistent sample rates in {segment['name']}: {fs_values}")
        fs = fs_values.pop()
        if fs != int(cfg["sample_rate_expected"]):
            raise RuntimeError(f"unexpected sample rate {fs} in {segment['name']}")
        n_seconds = min(metadata[node][3] for node in groups) // fs
        if max_seconds is not None:
            n_seconds = min(n_seconds, max_seconds)
        analysis_samples = int(cfg["nfft"]) * int(cfg["snapshots"])
        hop_samples = max(1, int(round(hop_seconds * fs)))
        total_samples = min(metadata[node][3] for node in groups)
        n_frames = max(0, 1 + (min(total_samples, n_seconds * fs) - analysis_samples) // hop_samples)
        streams = {node: [wave.open(str(path), "rb") for path in groups[node]] for node in frontend_nodes if node in groups}
        try:
            segment_rows = []
            for frame_index in range(n_frames):
                start_sample = frame_index * hop_samples
                for node in sorted(groups):
                    block = read_block(streams[node], start_sample, analysis_samples)
                    start_time = time.monotonic()
                    az, el, concentration = music_observation(block, fs, cfg)
                    segment_rows.append({"segment": segment["name"], "time_s": float(hms_seconds(segment["start"]) + frame_index * hop_seconds), "node_id": node, "azimuth_deg": az, "elevation_deg": el, "concentration": concentration, "valid": True, "frontend_runtime_s": time.monotonic() - start_time})
        finally:
            for node_streams in streams.values():
                for stream in node_streams:
                    stream.close()
        manifest["segments"].append({"name": segment["name"], "directory": str(segment_dir), "n_seconds": n_seconds, "n_frames": n_frames, "frame_dt_s": hop_seconds, "analysis_samples": analysis_samples, "sample_rate": fs, "available_nodes": sorted(groups), "missing_nodes": missing_nodes, "groups": {str(node): [str(path) for path in groups[node]] for node in groups}})
        all_rows.extend(segment_rows)
    write_json(output / "frontend_manifest.json", manifest)
    # Calibration is deliberately restricted to the first configured window.
    # GPS is used here only to fix array orientation and the known cross-modal
    # clock offset; the resulting corrected DOA stream is the sole assimilation
    # observation for every method.
    truth_by_second = {row[0]: (row[1], row[2], row[3]) for row in gps}
    calibration_end = min(float(hms_seconds(cfg["segments"][0]["start"]) + cfg["initial_calibration_s"]), max(truth_by_second))
    node_center = torch.tensor([sum(value["x"] for value in nodes.values()) / len(nodes), sum(value["y"] for value in nodes.values()) / len(nodes), sum(value["z"] for value in nodes.values()) / len(nodes)], dtype=torch.float64)
    node_xyz = {node: torch.tensor([value["x"], value["y"], value["z"]], dtype=torch.float64) - node_center for node, value in nodes.items()}
    calibration = {"geometry_spacing_m": cfg["geometry_spacing_m"], "candidate_delay_s": list(cfg["gps_delay_candidates_s"]), "nodes": {}}
    best_delays = []
    for delay in cfg["gps_delay_candidates_s"]:
        errors = []
        for row in all_rows:
            t = int(row["time_s"])
            if t >= calibration_end or t - int(delay) not in truth_by_second:
                continue
            xyz = torch.tensor(truth_by_second[t - int(delay)], dtype=torch.float64) - node_center
            node = int(row["node_id"]); delta = xyz - node_xyz[node]
            truth_az = math.degrees(math.atan2(float(delta[1]), float(delta[0]))) % 360.0
            truth_el = math.degrees(math.atan2(float(delta[2]), math.hypot(float(delta[0]), float(delta[1]))))
            errors.append(abs(signed_deg(truth_az - float(row["azimuth_deg"]))) + abs(truth_el - float(row["elevation_deg"])))
        best_delays.append((sum(errors) / len(errors) if errors else float("inf"), int(delay)))
    selected_delay = min(best_delays)[1] if best_delays else 2
    calibration["selected_delay_s"] = selected_delay
    for node in sorted(nodes):
        node_rows = [row for row in all_rows if int(row["node_id"]) == node and int(row["time_s"]) < calibration_end and int(row["time_s"]) - selected_delay in truth_by_second]
        candidates = []
        for sign in (1.0, -1.0):
            az_diffs, el_diffs = [], []
            for row in node_rows:
                xyz = torch.tensor(truth_by_second[int(row["time_s"]) - selected_delay], dtype=torch.float64) - node_center
                delta = xyz - node_xyz[node]
                az = math.degrees(math.atan2(float(delta[1]), float(delta[0]))) % 360.0
                el = math.degrees(math.atan2(float(delta[2]), math.hypot(float(delta[0]), float(delta[1]))))
                az_diffs.append(signed_deg(az - sign * float(row["azimuth_deg"])))
                el_diffs.append(el - float(row["elevation_deg"]))
            az_offset = math.degrees(math.atan2(sum(math.sin(math.radians(value)) for value in az_diffs), sum(math.cos(math.radians(value)) for value in az_diffs))) if az_diffs else 0.0
            el_offset = sum(el_diffs) / len(el_diffs) if el_diffs else 0.0
            score = sum(abs(signed_deg(sign * float(row["azimuth_deg"]) + az_offset - math.degrees(math.atan2(float((torch.tensor(truth_by_second[int(row["time_s"]) - selected_delay], dtype=torch.float64) - node_center - node_xyz[node])[1]), float((torch.tensor(truth_by_second[int(row["time_s"]) - selected_delay], dtype=torch.float64) - node_center - node_xyz[node])[0]))) % 360.0)) + abs((float(row["elevation_deg"]) + el_offset) - math.degrees(math.atan2(float((torch.tensor(truth_by_second[int(row["time_s"]) - selected_delay], dtype=torch.float64) - node_center - node_xyz[node])[2]), math.hypot(float((torch.tensor(truth_by_second[int(row["time_s"]) - selected_delay], dtype=torch.float64) - node_center - node_xyz[node])[0]), float((torch.tensor(truth_by_second[int(row["time_s"]) - selected_delay], dtype=torch.float64) - node_center - node_xyz[node])[1]))))) for row in node_rows)
            candidates.append((score / max(len(node_rows), 1), sign, az_offset, el_offset))
        _, sign, az_offset, el_offset = min(candidates) if candidates else (0.0, 1.0, 0.0, 0.0)
        calibration["nodes"][str(node)] = {"azimuth_sign": sign, "azimuth_offset_deg": az_offset, "elevation_offset_deg": el_offset, "n_calibration_rows": len(node_rows)}
        for row in all_rows:
            if int(row["node_id"]) == node:
                row["raw_azimuth_deg"] = row["azimuth_deg"]; row["raw_elevation_deg"] = row["elevation_deg"]
                row["azimuth_deg"] = circular_deg(sign * float(row["azimuth_deg"]) + az_offset)
                row["elevation_deg"] = float(row["elevation_deg"]) + el_offset
    write_json(output / "frontend_calibration.json", calibration)
    truth_rows = []
    for row in all_rows:
        target = int(row["time_s"]) - selected_delay
        nearest = min(range(len(gps_times)), key=lambda index: abs(gps_times[index] - target))
        if abs(gps_times[nearest] - target) <= 2:
            truth_rows.append({"time_s": row["time_s"], "px": gps_xyz[nearest][0], "py": gps_xyz[nearest][1], "pz": gps_xyz[nearest][2]})
    with (output / "observations.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
        writer.writeheader(); writer.writerows(all_rows)
    with (output / "gps_truth.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["time_s", "px", "py", "pz"]); writer.writeheader(); writer.writerows(truth_rows)


def load_observations(path: Path) -> dict[float, dict[int, tuple[float, float, float]]]:
    output: dict[float, dict[int, tuple[float, float, float]]] = {}
    with path.open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if not row.get("valid", "False").lower() == "true":
                continue
            output.setdefault(float(row["time_s"]), {})[int(row["node_id"])] = (float(row["azimuth_deg"]), float(row["elevation_deg"]), float(row["concentration"]))
    return output


def load_truth(path: Path) -> dict[float, torch.Tensor]:
    truth = {}
    with path.open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            truth[float(row["time_s"])] = torch.tensor([float(row["px"]), float(row["py"]), float(row["pz"])])
    return truth


def interpolate_truth(truth: dict[float, torch.Tensor], times: list[float]) -> list[torch.Tensor | None]:
    if not truth:
        return [None] * len(times)
    keys = sorted(truth)
    out = []
    for t in times:
        key = min(keys, key=lambda value: abs(value - t))
        out.append(truth[key] if abs(key - t) <= 2.0 else None)
    return out


def estimate_alpha_star(truth: dict[float, torch.Tensor], q_min: float, q_max: float) -> float | None:
    keys = sorted(truth)
    if len(keys) < 4:
        return None
    accelerations = []
    for left, middle, right in zip(keys, keys[1:], keys[2:]):
        dt_left = max(middle - left, 1.0); dt_right = max(right - middle, 1.0)
        velocity_left = (truth[middle] - truth[left]) / dt_left
        velocity_right = (truth[right] - truth[middle]) / dt_right
        accelerations.append(float(torch.linalg.vector_norm(velocity_right - velocity_left) / max((dt_left + dt_right) * 0.5, 1.0)))
    q = sorted(accelerations)[max(0, int(0.10 * len(accelerations))): max(1, int(0.90 * len(accelerations)))]
    observed = sum(q) / len(q) if q else q_min
    return float(max(0.0, min(1.0, math.log(max(observed, q_min) / q_min) / math.log(q_max / q_min))))


def ray_direction(az: float, el: float) -> torch.Tensor:
    azr, elr = math.radians(az), math.radians(el)
    return torch.tensor([math.cos(elr) * math.cos(azr), math.cos(elr) * math.sin(azr), math.sin(elr)], dtype=torch.float64)


def triangulate(observation: dict[int, tuple[float, float, float]], node_xyz: dict[int, torch.Tensor], condition_limit: float) -> torch.Tensor | None:
    matrices, rhs = [], []
    for node, (az, el, concentration) in observation.items():
        if concentration <= 0.001 or node not in node_xyz:
            continue
        direction = ray_direction(az, el).to(node_xyz[node]); projection = torch.eye(3, dtype=torch.float64, device=node_xyz[node].device) - direction[:, None] @ direction[None, :]
        matrices.append(projection); rhs.append(projection @ node_xyz[node])
    if len(matrices) < 3:
        return None
    matrix, target = torch.stack(matrices).sum(dim=0), torch.stack(rhs).sum(dim=0)
    if float(torch.linalg.cond(matrix)) > condition_limit:
        return None
    return torch.linalg.solve(matrix, target)


def predict_angles(states: torch.Tensor, node_xyz: dict[int, torch.Tensor], node_ids: list[int]) -> torch.Tensor:
    # states: [ensemble, 6] or [branches, ensemble, 6]
    position = states[..., :3]
    outputs = []
    for node in node_ids:
        delta = position - node_xyz[node].to(position)
        az = torch.atan2(delta[..., 1], delta[..., 0])
        el = torch.atan2(delta[..., 2], torch.linalg.vector_norm(delta[..., :2], dim=-1))
        outputs.extend((az, el))
    return torch.stack(outputs, dim=-1)


def observation_tensor(observation: dict[int, tuple[float, float, float]], node_ids: list[int], device: torch.device) -> torch.Tensor:
    values = []
    for node in node_ids:
        az, el, _ = observation[node]
        values.extend((math.radians(az), math.radians(el)))
    return torch.tensor(values, dtype=torch.float64, device=device)


def enkf_update(states: torch.Tensor, observation: torch.Tensor, node_xyz: dict[int, torch.Tensor], node_ids: list[int], obs_std: float) -> torch.Tensor:
    predicted = predict_angles(states, node_xyz, node_ids)
    mean_state, mean_obs = states.mean(dim=-2), predicted.mean(dim=-2)
    state_anomaly, obs_anomaly = states - mean_state.unsqueeze(-2), predicted - mean_obs.unsqueeze(-2)
    covariance = state_anomaly.transpose(-2, -1) @ obs_anomaly / max(states.shape[-2] - 1, 1)
    innovation = angle_residual(observation.unsqueeze(-2), predicted)
    obs_cov = obs_anomaly.transpose(-2, -1) @ obs_anomaly / max(states.shape[-2] - 1, 1)
    obs_cov = obs_cov + torch.eye(obs_cov.shape[-1], dtype=torch.float64, device=states.device) * math.radians(obs_std) ** 2
    gain = torch.linalg.solve(obs_cov.transpose(-2, -1), covariance.transpose(-2, -1)).transpose(-2, -1)
    return states + torch.einsum("...ij,...nj->...ni", gain, innovation)


def propagate(states: torch.Tensor, alpha: torch.Tensor, dt: float, noise: torch.Tensor, q_min: float, q_max: float) -> torch.Tensor:
    q = q_min * torch.pow(torch.tensor(q_max / q_min, dtype=torch.float64, device=states.device), alpha)
    out = states.clone()
    out[..., :3] = out[..., :3] + out[..., 3:] * dt + noise[..., :3] * dt
    out[..., 3:] = out[..., 3:] + noise[..., 3:] * q.unsqueeze(-1) * math.sqrt(max(dt, 1e-6))
    return out


def entropy_project(weights: torch.Tensor, floor: float) -> torch.Tensor:
    uniform = torch.ones_like(weights) / weights.shape[-1]
    mixed = weights
    for _ in range(24):
        entropy = -(mixed.clamp_min(1e-12) * mixed.clamp_min(1e-12).log()).sum()
        if float(entropy) >= floor:
            break
        mixed = 0.5 * (mixed + uniform)
    return mixed / mixed.sum()


def score_weights(method: str, logits: torch.Tensor, predicted: torch.Tensor, observation: torch.Tensor, obs_std: float) -> tuple[torch.Tensor, torch.Tensor]:
    residual = angle_residual(predicted, observation.unsqueeze(0))
    variance = math.radians(obs_std) ** 2
    if method == "apce":
        between = predicted.mean(dim=1).var(dim=0, unbiased=False)
        dimension_weights = 0.35 + 0.65 * between / between.mean().clamp_min(1e-12)
        dimension_weights = dimension_weights / dimension_weights.mean()
    else:
        dimension_weights = torch.ones(predicted.shape[-1], dtype=torch.float64, device=predicted.device)
    raw = -0.5 * (dimension_weights * residual.square() / variance).mean(dim=(1, 2))
    if method == "apce":
        logits = 0.975 * logits + raw / 0.58
        weights = entropy_project(torch.softmax(logits - logits.max(), dim=0), 0.12)
    else:
        logits = logits + raw / 0.66
        weights = torch.softmax(logits - logits.max(), dim=0)
    return logits, weights


def weighted_quantile(values: torch.Tensor, weights: torch.Tensor, quantile: float) -> torch.Tensor:
    order = torch.argsort(values)
    values, weights = values[order], weights[order]
    cumulative = torch.cumsum(weights, dim=0)
    index = int(torch.searchsorted(cumulative, torch.tensor(quantile, dtype=weights.dtype, device=weights.device)).clamp(max=len(values) - 1))
    return values[index]


def crps_empirical(samples: torch.Tensor, weights: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    first = (weights[:, None] * (samples - truth).abs()).sum(dim=0)
    pair = (weights[:, None, None] * weights[None, :, None] * (samples[:, None] - samples[None, :]).abs()).sum(dim=(0, 1))
    return first - 0.5 * pair


def run_track(cfg: dict, frontend: Path, output: Path, method: str, seed: int, device_name: str) -> dict:
    observations = load_observations(frontend / "observations.csv")
    truth = load_truth(frontend / "gps_truth.csv")
    alpha_star = estimate_alpha_star(truth, float(cfg["q_min_accel_mps2"]), float(cfg["q_max_accel_mps2"]))
    times = sorted(observations)
    node_manifest = load_json(frontend / "frontend_manifest.json")["nodes"]
    track_node_ids = tuple(int(node) for node in cfg.get("track_nodes", sorted(int(key) for key in node_manifest)))
    node_xyz = {int(node): torch.tensor([node_manifest[str(node)]["x"], node_manifest[str(node)]["y"], node_manifest[str(node)]["z"]], dtype=torch.float64) for node in track_node_ids}
    center = torch.stack(list(node_xyz.values())).mean(dim=0)
    node_xyz = {node: value - center for node, value in node_xyz.items()}
    device = torch.device(device_name if torch.cuda.is_available() and device_name.startswith("cuda") else "cpu")
    generator = torch.Generator(device=device).manual_seed(seed)
    alpha = torch.tensor(ALPHA_GRID, dtype=torch.float64, device=device)
    initial_points = []
    for t in times[:10]:
        row = {node: value for node, value in observations[t].items() if node in node_xyz}
        point = triangulate(row, {node: value.to(device) for node, value in node_xyz.items()}, float(cfg["max_condition_number"]))
        if point is not None:
            initial_points.append(point.to(device))
    if len(initial_points) < 3:
        raise RuntimeError("acoustic initialization has fewer than three valid triangulations")
    initial_position = torch.stack(initial_points).mean(dim=0)
    initial_velocity = torch.zeros(3, dtype=torch.float64, device=device)
    if len(initial_points) >= 2:
        initial_velocity = (initial_points[-1] - initial_points[0]) / max(float(times[min(9, len(times) - 1)] - times[0]), 1.0)
    n = int(cfg["ensemble_size"]); init = torch.cat((initial_position, initial_velocity)).repeat(n, 1)
    init = init + torch.cat((torch.randn((n, 3), generator=generator, device=device, dtype=torch.float64) * cfg["position_init_std_m"], torch.randn((n, 3), generator=generator, device=device, dtype=torch.float64) * cfg["velocity_init_std_mps"]), dim=1)
    branches = init.unsqueeze(0).repeat(len(ALPHA_GRID), 1, 1)
    shadows = branches.clone(); logits = torch.zeros(len(ALPHA_GRID), dtype=torch.float64, device=device)
    single = init.clone(); aug = torch.cat((init, torch.full((n, 1), 0.50, dtype=torch.float64, device=device)), dim=1)
    records = []; previous_time = times[0]
    for time_s in times:
        dt = max(0.1, min(3.0, time_s - previous_time))
        base_noise = torch.randn((n, 6), generator=generator, device=device, dtype=torch.float64)
        if time_s != times[0]:
            single = propagate(single, torch.full((n,), 0.50, dtype=torch.float64, device=device), dt, base_noise, cfg["q_min_accel_mps2"], cfg["q_max_accel_mps2"])
            aug_state, aug_alpha = aug[:, :6], aug[:, 6].clamp(0.0, 1.0)
            aug[:, :6] = propagate(aug_state, aug_alpha, dt, base_noise, cfg["q_min_accel_mps2"], cfg["q_max_accel_mps2"])
            aug[:, 6] = (aug_alpha + 0.015 * torch.randn((n,), generator=generator, device=device, dtype=torch.float64)).clamp(0.0, 1.0)
            for index in range(len(ALPHA_GRID)):
                branches[index] = propagate(branches[index], alpha[index].expand(n), dt, base_noise, cfg["q_min_accel_mps2"], cfg["q_max_accel_mps2"])
                shadows[index] = propagate(shadows[index], alpha[index].expand(n), dt, base_noise, cfg["q_min_accel_mps2"], cfg["q_max_accel_mps2"])
        row = observations[time_s]; valid_nodes = [node for node in track_node_ids if node in row]
        if len(valid_nodes) < 3:
            previous_time = time_s; continue
        obs = observation_tensor(row, valid_nodes, device)
        node_sub = {node: node_xyz[node].to(device) for node in valid_nodes}
        if method == "denkf":
            single = enkf_update(single, obs, node_sub, valid_nodes, cfg["observation_angle_std_deg"])
            samples, weights, alpha_est = single, torch.ones(n, dtype=torch.float64, device=device) / n, torch.tensor(0.50, device=device)
        elif method == "aug_enkf":
            aug[:, :6] = enkf_update(aug[:, :6], obs, node_sub, valid_nodes, cfg["observation_angle_std_deg"]); aug[:, 6] = aug[:, 6].clamp(0.0, 1.0)
            samples, weights, alpha_est = aug[:, :6], torch.ones(n, dtype=torch.float64, device=device) / n, aug[:, 6].mean()
        else:
            evidence_source = predict_angles(shadows if method in ("pce", "apce") else branches, node_sub, valid_nodes)
            logits, branch_weights = score_weights(method, logits, evidence_source, obs, cfg["observation_angle_std_deg"])
            for index in range(len(ALPHA_GRID)):
                branches[index] = enkf_update(branches[index], obs, node_sub, valid_nodes, cfg["observation_angle_std_deg"])
            samples = branches.reshape(-1, 6); weights = branch_weights.repeat_interleave(n) / n; alpha_est = (branch_weights * alpha).sum()
        truth_key = min((key for key in truth if abs(key - time_s) <= 2.0), key=lambda key: abs(key - time_s), default=None)
        truth_point = truth[truth_key] if truth_key is not None else None
        mean = (samples * weights[:, None]).sum(dim=0)
        lower = torch.stack([weighted_quantile(samples[:, dim], weights, 0.05) for dim in range(6)])
        upper = torch.stack([weighted_quantile(samples[:, dim], weights, 0.95) for dim in range(6)])
        row_out = {"time_s": time_s, "method": method, "seed": seed, "valid_nodes": len(valid_nodes), "px": float(mean[0]), "py": float(mean[1]), "pz": float(mean[2]), "vx": float(mean[3]), "vy": float(mean[4]), "vz": float(mean[5]), "alpha_estimate": float(alpha_est), "evidence_entropy": float(-(weights.reshape(len(ALPHA_GRID), n).sum(dim=1).clamp_min(1e-12) * weights.reshape(len(ALPHA_GRID), n).sum(dim=1).clamp_min(1e-12).log()).sum()) if method in ("pce", "apce", "bma") else 0.0, "runtime_s": 0.0}
        for dim, name in enumerate(("px", "py", "pz", "vx", "vy", "vz")):
            row_out[f"{name}_lo"] = float(lower[dim]); row_out[f"{name}_hi"] = float(upper[dim])
        row_out["alpha_star"] = alpha_star
        row_out["alpha_error"] = abs(float(alpha_est) - alpha_star) if alpha_star is not None else None
        if truth_point is not None:
            err = mean[:3] - truth_point.to(device); row_out["position_error_m"] = float(torch.linalg.vector_norm(err)); row_out["crps_position_m"] = float(crps_empirical(samples[:, :3], weights, truth_point.to(device)).mean())
            row_out["alpha_error"] = None
            row_out["coverage_90"] = float(((truth_point.to(device) >= lower[:3]) & (truth_point.to(device) <= upper[:3])).double().mean()); row_out["interval_width_m"] = float((upper[:3] - lower[:3]).mean())
        else:
            row_out.update({"position_error_m": None, "crps_position_m": None, "coverage_90": None, "interval_width_m": None})
        records.append(row_out); previous_time = time_s
    output.mkdir(parents=True, exist_ok=True)
    run_path = output / f"{method}_seed_{seed}.json"
    payload = {"status": "valid", "method": method, "seed": seed, "source_frontend": str(frontend), "frontend_manifest_sha256": sha256(frontend / "frontend_manifest.json"), "frontend_calibration_sha256": sha256(frontend / "frontend_calibration.json"), "runner_sha256": sha256(Path(__file__)), "track_nodes": list(track_node_ids), "input_provenance": load_json(frontend / "frontend_manifest.json").get("input_provenance"), "alpha_grid": list(ALPHA_GRID), "records": records}
    write_json(run_path, payload)
    return payload


def aggregate(output: Path) -> None:
    rows = []
    for path in sorted(output.glob("*_seed_*.json")):
        payload = load_json(path)
        records = [row for row in payload.get("records", []) if row.get("position_error_m") is not None]
        if not records:
            continue
        def mean(name: str) -> float:
            values = [float(row[name]) for row in records if row.get(name) is not None]
            return sum(values) / len(values) if values else float("nan")
        errors = [float(row["position_error_m"]) for row in records]
        rows.append({"method": payload["method"], "seed": payload["seed"], "n_records": len(records), "position_rmse_m": math.sqrt(sum(value * value for value in errors) / len(errors)), "position_mae_m": mean("position_error_m"), "crps_position_m": mean("crps_position_m"), "coverage_90": mean("coverage_90"), "interval_width_m": mean("interval_width_m"), "alpha_mae": mean("alpha_error") if any(row.get("alpha_error") is not None for row in records) else None})
    with (output / "method_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["method"]); writer.writeheader(); writer.writerows(rows)
    write_json(output / "aggregate_manifest.json", {"valid_runs": len(rows), "methods": sorted({row["method"] for row in rows}), "source": str(output)})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--stage", choices=("manifest", "frontend", "track", "aggregate"), required=True)
    parser.add_argument("--method", choices=("denkf", "aug_enkf", "bma", "pce", "apce"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--frontend", type=Path)
    parser.add_argument("--max-seconds", type=int)
    parser.add_argument("--frontend-hop-seconds", type=float)
    parser.add_argument("--geometry-profile", choices=("legacy_nonuniform_z", "paper_uniform_cross"))
    args = parser.parse_args(); cfg = load_json(args.config)
    if args.frontend_hop_seconds is not None:
        cfg["frontend_hop_seconds"] = args.frontend_hop_seconds
    if args.geometry_profile is not None:
        cfg["geometry_profile"] = args.geometry_profile
    output = args.output or Path(cfg["remote_result_root"])
    if args.stage == "manifest":
        root = Path(cfg["remote_root"]); nod = root / "20171107保定实验/GPS_data/20171107baoding.nod"; write_json(output / "data_manifest.json", {"root": cfg["remote_root"], "nod": str(nod), "nod_sha256": sha256(nod), "nodes": parse_nod(nod), "segments": cfg["segments"], "channels": CHANNELS, "channel_groups": CHANNEL_GROUPS, "alpha_grid": list(ALPHA_GRID)})
    elif args.stage == "frontend":
        run_frontend(cfg, output / "frontend", args.max_seconds)
    elif args.stage == "track":
        if args.method is None or args.seed is None: raise SystemExit("--method and --seed are required for track")
        frontend = args.frontend or (output / "frontend")
        run_track(cfg, frontend, output / "runs", args.method, args.seed, args.device)
    elif args.stage == "aggregate":
        aggregate(output / "runs")


if __name__ == "__main__":
    main()
