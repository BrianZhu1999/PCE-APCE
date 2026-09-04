"""Read-only access to the stored members of the VIV-PIV NPZ archives."""
from __future__ import annotations

import hashlib
import io
import pathlib
import re
import struct
import zipfile
from dataclasses import dataclass
from typing import Iterator

import numpy as np


def parse_case_id(value: str | pathlib.Path) -> str:
    """Return a four-digit case id, tolerating comma-formatted filenames."""
    text = pathlib.Path(value).name if isinstance(value, pathlib.Path) else str(value)
    text = text.replace(",", "")
    matches = re.findall(r"(?<!\d)(\d{4})(?!\d)", text)
    if matches:
        return matches[-1]
    digits = re.findall(r"\d+", text)
    if not digits:
        raise ValueError(f"Cannot parse VIV case id from {value!r}")
    token = digits[-1].zfill(4)
    return token[-4:]


def locate_case(root: pathlib.Path, case_id: str) -> pathlib.Path:
    target = str(case_id).zfill(4)
    candidates = sorted(root.glob("reduced_velocity_*.npz"))
    for path in candidates:
        if parse_case_id(path) == target:
            return path
    raise FileNotFoundError(f"No NPZ for case {target} under {root}")


def sha256_file(path: pathlib.Path, block_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _npy_header(path: pathlib.Path, member: str) -> tuple[tuple[int, ...], np.dtype, bool, int]:
    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo(member + ".npy")
        if info.compress_type != zipfile.ZIP_STORED:
            raise ValueError(f"{path.name}:{member} is compressed; zero-copy reader requires stored NPZ members")
        header_offset = info.header_offset
    with path.open("rb") as handle:
        handle.seek(header_offset)
        local = handle.read(30)
        signature, _version, _flags, compression, _mtime, _mdate, _crc, _csize, _usize, name_len, extra_len = struct.unpack(
            "<IHHHHHIIIHH", local
        )
        if signature != 0x04034B50 or compression != zipfile.ZIP_STORED:
            raise ValueError(f"Invalid stored NPZ local header for {path}")
        handle.seek(name_len + extra_len, 1)
        version = np.lib.format.read_magic(handle)
        reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
        shape, fortran, dtype = reader(handle)
        return tuple(int(x) for x in shape), np.dtype(dtype), bool(fortran), int(handle.tell())


def stored_memmap(path: pathlib.Path, member: str) -> np.memmap:
    shape, dtype, fortran, offset = _npy_header(path, member)
    return np.memmap(path, mode="r", dtype=dtype, offset=offset, shape=shape, order="F" if fortran else "C")


def load_small(path: pathlib.Path, member: str) -> np.ndarray:
    with zipfile.ZipFile(path) as archive:
        with archive.open(member + ".npy") as handle:
            return np.load(io.BytesIO(handle.read()), allow_pickle=False)


@dataclass
class VIVCase:
    path: pathlib.Path
    case_id: str
    label: str
    x_mm: np.ndarray
    y_mm: np.ndarray
    time_s: np.ndarray
    cyl_displ_m: np.ndarray
    norm_values: np.ndarray

    @classmethod
    def open(cls, path: pathlib.Path) -> "VIVCase":
        label_array = load_small(path, "label")
        label = str(np.asarray(label_array).item())
        return cls(
            path=path,
            case_id=parse_case_id(path),
            label=label,
            x_mm=np.asarray(load_small(path, "x"), dtype=np.float64),
            y_mm=np.asarray(load_small(path, "y"), dtype=np.float64),
            time_s=np.asarray(load_small(path, "time"), dtype=np.float64),
            cyl_displ_m=np.asarray(load_small(path, "cyl_displ"), dtype=np.float64),
            norm_values=np.asarray(load_small(path, "norm_values"), dtype=np.float64),
        )

    @property
    def velocities(self) -> np.memmap:
        return stored_memmap(self.path, "velocities")

    @property
    def mask(self) -> np.memmap:
        return stored_memmap(self.path, "mask")

    @property
    def reduced_velocity(self) -> float:
        return int(self.case_id) / 100.0

    @property
    def dt_s(self) -> float:
        return float(np.median(np.diff(self.time_s)))

    def physical_frames(self, start: int = 0, stop: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        stop = self.time_s.size if stop is None else int(stop)
        values = np.asarray(self.velocities[start:stop], dtype=np.float32)
        valid = np.asarray(self.mask[start:stop] > 0.5, dtype=bool)
        low = self.norm_values[0].astype(np.float32)
        high = self.norm_values[1].astype(np.float32)
        values = values * (high - low)[None, None, None, :] + low[None, None, None, :]
        return values, valid

    def physical_flat(self, start: int = 0, stop: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        values, valid = self.physical_frames(start, stop)
        return values.reshape(values.shape[0], -1), np.repeat(valid[..., None], 2, axis=-1).reshape(values.shape[0], -1)

    def iter_physical(self, block: int = 32) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
        for start in range(0, self.time_s.size, block):
            stop = min(start + block, self.time_s.size)
            values, valid = self.physical_flat(start, stop)
            yield start, values, valid


def list_cases(root: pathlib.Path) -> dict[str, pathlib.Path]:
    output: dict[str, pathlib.Path] = {}
    for path in sorted(root.glob("reduced_velocity_*.npz")):
        case_id = parse_case_id(path)
        if case_id in output:
            raise ValueError(f"Duplicate parsed case id {case_id}: {output[case_id]} and {path}")
        output[case_id] = path
    return output


def nearest_grid_indices(x_mm: np.ndarray, y_mm: np.ndarray, x_over_d: list[float], y_over_d: list[float], diameter_m: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    targets_x = np.asarray(x_over_d, dtype=float) * diameter_m * 1000.0
    targets_y = np.asarray(y_over_d, dtype=float) * diameter_m * 1000.0
    x_idx = np.asarray([int(np.argmin(np.abs(x_mm - target))) for target in targets_x], dtype=np.int64)
    y_idx = np.asarray([int(np.argmin(np.abs(y_mm - target))) for target in targets_y], dtype=np.int64)
    coords = np.asarray([(float(x_mm[ix]), float(y_mm[iy])) for ix in x_idx for iy in y_idx])
    return x_idx, y_idx, coords
