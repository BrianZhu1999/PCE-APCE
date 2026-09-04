from __future__ import annotations

import hashlib
import io
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat


INNER_ZIP = "F16GVT_Files.zip"
FULLMSINE_TEMPLATE = "F16GVT_Files/BenchmarkData/F16Data_FullMSine_Level{level}{suffix}.mat"


@dataclass(frozen=True)
class FullMSineLevel:
    level: int
    force: np.ndarray
    voltage: np.ndarray
    acceleration: np.ndarray
    sample_rate_hz: float
    source_member: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inner_archive(outer_path: Path) -> zipfile.ZipFile:
    outer = zipfile.ZipFile(outer_path, "r")
    payload = outer.read(INNER_ZIP)
    outer.close()
    return zipfile.ZipFile(io.BytesIO(payload), "r")


def member_for_level(level: int) -> str:
    suffix = "_Validation" if level in (2, 4, 6) else ""
    return FULLMSINE_TEMPLATE.format(level=level, suffix=suffix)


def load_fullmsine_level(outer_path: Path, level: int) -> FullMSineLevel:
    member = member_for_level(level)
    with inner_archive(outer_path) as archive:
        payload = loadmat(io.BytesIO(archive.read(member)))
    force = np.asarray(payload["Force"], dtype=np.float64).reshape(-1)
    voltage = np.asarray(payload["Voltage"], dtype=np.float64).reshape(-1)
    acceleration = np.asarray(payload["Acceleration"], dtype=np.float64)
    sample_rate = float(np.asarray(payload["Fs"]).reshape(-1)[0])
    if acceleration.shape[0] != 3:
        raise ValueError(f"Level {level}: expected 3 acceleration outputs, found {acceleration.shape}")
    if not (force.size == voltage.size == acceleration.shape[1]):
        raise ValueError(f"Level {level}: channel length mismatch")
    if not (np.isfinite(force).all() and np.isfinite(voltage).all() and np.isfinite(acceleration).all()):
        raise ValueError(f"Level {level}: non-finite data")
    return FullMSineLevel(level, force, voltage, acceleration.T.copy(), sample_rate, member)


def audit_archive(outer_path: Path) -> dict:
    with zipfile.ZipFile(outer_path, "r") as outer:
        outer_names = outer.namelist()
    with inner_archive(outer_path) as inner:
        inner_names = inner.namelist()
        members = {name: inner.getinfo(name).file_size for name in inner_names}
    required = [member_for_level(level) for level in range(1, 8)]
    missing = [name for name in required if name not in members]
    return {
        "outer_path": str(outer_path),
        "outer_sha256": sha256_file(outer_path),
        "outer_members": outer_names,
        "inner_member_count": len(inner_names),
        "required_fullmsine_members": required,
        "missing_required_members": missing,
        "required_member_sizes": {name: members.get(name) for name in required},
    }
