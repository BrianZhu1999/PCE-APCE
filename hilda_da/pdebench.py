from __future__ import annotations

import csv
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import h5py
import numpy as np
import torch

from .observations import SparseObservation

PDEBenchSchemaKind = Literal["tensor", "cfd_fields", "grouped_data", "ns_incom"]
_CFD_FIELDS = ("density", "pressure", "Vx", "Vy", "Vz")
_COORDINATE_KEYS = ("x-coordinate", "y-coordinate", "z-coordinate")


@dataclass(frozen=True)
class PDEBenchManifestRecord:
    pde: str
    filename: str
    url: str
    relative_path: str
    expected_md5: str


@dataclass(frozen=True)
class PDEBenchSchema:
    kind: PDEBenchSchemaKind
    trajectory_count: int
    spatial_shape: tuple[int, ...]
    channel_names: tuple[str, ...]
    dataset_paths: tuple[str, ...]
    source_shapes: tuple[tuple[int, ...], ...]
    coordinate_keys: tuple[str, ...]


@dataclass(frozen=True)
class TrajectorySlice:
    trajectory_index: int = 0
    time_start: int | None = None
    time_stop: int | None = None
    time_step: int = 1
    spatial_stride: int | tuple[int, ...] = 1
    channel_indices: tuple[int, ...] | None = None

    @property
    def time_slice(self) -> slice:
        if self.time_step <= 0:
            raise ValueError("time_step must be positive")
        return slice(self.time_start, self.time_stop, self.time_step)


@dataclass(frozen=True)
class PDEBenchProvenance:
    source_path: str
    file_size_bytes: int
    file_mtime_ns: int
    schema: PDEBenchSchemaKind
    dataset_paths: tuple[str, ...]
    source_shapes: tuple[tuple[int, ...], ...]
    trajectory_index: int
    trajectory_name: str
    time_slice: tuple[int | None, int | None, int]
    spatial_stride: tuple[int, ...]
    channel_indices: tuple[int, ...]
    channel_names: tuple[str, ...]
    root_attributes: dict[str, Any]
    manifest_record: PDEBenchManifestRecord | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PDEBenchTrajectory:
    states: torch.Tensor
    spatial_shape: tuple[int, ...]
    channel_names: tuple[str, ...]
    times: torch.Tensor | None
    coordinates: tuple[torch.Tensor, ...]
    provenance: PDEBenchProvenance

    @property
    def state_dim(self) -> int:
        return self.states.shape[-1]

    def sparse_observation(
        self,
        sensor_mask: torch.Tensor | np.ndarray,
        *,
        channel_indices: tuple[int, ...] | None = None,
        transform: str = "linear",
    ) -> SparseObservation:
        return sparse_observation_from_sensor_mask(
            sensor_mask,
            self.spatial_shape,
            len(self.channel_names),
            channel_indices=channel_indices,
            transform=transform,
        )


def load_manifest_record(
    manifest_path: str | Path,
    filename: str,
) -> PDEBenchManifestRecord:
    with Path(manifest_path).open("r", encoding="utf-8-sig", newline="") as handle:
        matches = [
            row
            for row in csv.DictReader(handle)
            if row.get("Filename") == filename
        ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one manifest row for {filename!r}, found {len(matches)}"
        )
    row = matches[0]
    required = ("PDE", "Filename", "URL", "Path", "MD5")
    missing = [key for key in required if not row.get(key)]
    if missing:
        raise ValueError(f"PDEBench manifest row is missing fields: {missing}")
    return PDEBenchManifestRecord(
        pde=row["PDE"],
        filename=row["Filename"],
        url=row["URL"],
        relative_path=row["Path"],
        expected_md5=row["MD5"],
    )


def file_md5(path: str | Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.md5(usedforsecurity=False)
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest_checksum(
    path: str | Path,
    record: PDEBenchManifestRecord,
    *,
    chunk_size: int = 8 * 1024 * 1024,
) -> str:
    actual = file_md5(path, chunk_size=chunk_size)
    if actual.lower() != record.expected_md5.lower():
        raise ValueError(
            f"PDEBench MD5 mismatch for {Path(path).name}: "
            f"expected {record.expected_md5}, got {actual}"
        )
    return actual


def sparse_observation_from_sensor_mask(
    sensor_mask: torch.Tensor | np.ndarray,
    spatial_shape: tuple[int, ...],
    channel_count: int,
    *,
    channel_indices: tuple[int, ...] | None = None,
    transform: str = "linear",
) -> SparseObservation:
    mask = torch.as_tensor(sensor_mask, dtype=torch.bool, device="cpu")
    if tuple(mask.shape) != spatial_shape:
        raise ValueError(
            f"Sensor mask shape {tuple(mask.shape)} does not match spatial shape {spatial_shape}"
        )
    if channel_count < 1:
        raise ValueError("channel_count must be positive")
    channels = (
        tuple(range(channel_count)) if channel_indices is None else channel_indices
    )
    if not channels or len(set(channels)) != len(channels):
        raise ValueError("channel_indices must be non-empty and unique")
    if min(channels) < 0 or max(channels) >= channel_count:
        raise IndexError("channel index is outside the loaded trajectory")
    spatial_indices = torch.nonzero(mask.reshape(-1), as_tuple=False).flatten()
    if spatial_indices.numel() == 0:
        raise ValueError("Sensor mask must select at least one spatial point")
    channel_tensor = torch.tensor(channels, dtype=torch.int64)
    state_indices = (
        spatial_indices[:, None] * channel_count + channel_tensor[None, :]
    ).reshape(-1)
    return SparseObservation(state_indices, transform=transform)


def _json_attribute(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _normalize_stride(
    stride: int | tuple[int, ...],
    spatial_rank: int,
) -> tuple[int, ...]:
    values = (stride,) * spatial_rank if isinstance(stride, int) else tuple(stride)
    if len(values) != spatial_rank or any(value <= 0 for value in values):
        raise ValueError(
            "spatial_stride must contain one positive value per spatial dimension"
        )
    return values


class PDEBenchHDF5Adapter:
    """Read PDEBench trajectories without materializing unrelated samples."""

    def __init__(
        self,
        path: str | Path,
        *,
        manifest_path: str | Path | None = None,
        cfd_fields: tuple[str, ...] | None = None,
    ) -> None:
        self.path = Path(path).resolve()
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self._requested_cfd_fields = cfd_fields
        self.manifest_record = (
            load_manifest_record(manifest_path, self.path.name)
            if manifest_path is not None
            else None
        )
        self.schema = self._inspect_schema()

    def _inspect_schema(self) -> PDEBenchSchema:
        with h5py.File(self.path, "r") as handle:
            root_keys = set(handle.keys())
            coordinate_keys = tuple(key for key in _COORDINATE_KEYS if key in handle)
            if "velocity" in handle and "t" in handle:
                velocity_shape = tuple(handle["velocity"].shape)
                time_shape = tuple(handle["t"].shape)
                if (
                    len(velocity_shape) == 5
                    and velocity_shape[-1] == 2
                    and time_shape == velocity_shape[:2]
                ):
                    return PDEBenchSchema(
                        kind="ns_incom",
                        trajectory_count=velocity_shape[0],
                        spatial_shape=velocity_shape[2:-1],
                        channel_names=("velocity_x", "velocity_y"),
                        dataset_paths=("velocity", "t"),
                        source_shapes=(velocity_shape, time_shape),
                        coordinate_keys=coordinate_keys,
                    )
            if "tensor" in handle:
                shape = tuple(handle["tensor"].shape)
                if len(shape) < 3:
                    raise ValueError("PDEBench tensor must have [batch, time, spatial...] axes")
                return PDEBenchSchema(
                    kind="tensor",
                    trajectory_count=shape[0],
                    spatial_shape=shape[2:],
                    channel_names=("tensor",),
                    dataset_paths=("tensor",),
                    source_shapes=(shape,),
                    coordinate_keys=coordinate_keys,
                )

            available_fields = tuple(field for field in _CFD_FIELDS if field in root_keys)
            if available_fields:
                fields = self._requested_cfd_fields or available_fields
                if (
                    not fields
                    or len(set(fields)) != len(fields)
                    or any(field not in available_fields for field in fields)
                ):
                    raise ValueError(
                        f"Requested CFD fields must be selected from {available_fields}"
                    )
                shapes = tuple(tuple(handle[field].shape) for field in fields)
                if len(shapes[0]) < 3 or any(shape != shapes[0] for shape in shapes[1:]):
                    raise ValueError("PDEBench CFD fields must share [batch, time, spatial...] shape")
                return PDEBenchSchema(
                    kind="cfd_fields",
                    trajectory_count=shapes[0][0],
                    spatial_shape=shapes[0][2:],
                    channel_names=fields,
                    dataset_paths=fields,
                    source_shapes=shapes,
                    coordinate_keys=coordinate_keys,
                )

            group_names = sorted(
                key
                for key in handle.keys()
                if isinstance(handle[key], h5py.Group) and "data" in handle[key]
            )
            if not group_names:
                raise ValueError("Unrecognized PDEBench HDF5 schema")
            self._group_names = tuple(group_names)
            first_shape = tuple(handle[f"{group_names[0]}/data"].shape)
            if len(first_shape) < 3:
                raise ValueError("Grouped PDEBench data must have [time, spatial..., channel] axes")
            channel_count = first_shape[-1]
            channel_names = tuple(f"channel_{index}" for index in range(channel_count))
            return PDEBenchSchema(
                kind="grouped_data",
                trajectory_count=len(group_names),
                spatial_shape=first_shape[1:-1],
                channel_names=channel_names,
                dataset_paths=("<trajectory>/data",),
                source_shapes=(first_shape,),
                coordinate_keys=coordinate_keys,
            )

    def load_trajectory(
        self,
        selection: TrajectorySlice | None = None,
        *,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ) -> PDEBenchTrajectory:
        selection = selection or TrajectorySlice()
        if not 0 <= selection.trajectory_index < self.schema.trajectory_count:
            raise IndexError("trajectory_index is outside the PDEBench file")
        spatial_stride = _normalize_stride(
            selection.spatial_stride,
            len(self.schema.spatial_shape),
        )
        spatial_slices = tuple(slice(None, None, step) for step in spatial_stride)
        time_slice = selection.time_slice

        with h5py.File(self.path, "r") as handle:
            if self.schema.kind == "ns_incom":
                values = np.asarray(
                    handle["velocity"][
                        (selection.trajectory_index, time_slice, *spatial_slices, slice(None))
                    ]
                )
                dataset_paths = ("velocity", "t")
                source_shapes = self.schema.source_shapes
                trajectory_name = str(selection.trajectory_index)
            elif self.schema.kind == "tensor":
                raw = handle["tensor"][
                    (selection.trajectory_index, time_slice, *spatial_slices)
                ]
                values = np.asarray(raw)[..., None]
                dataset_paths = ("tensor",)
                source_shapes = self.schema.source_shapes
                trajectory_name = str(selection.trajectory_index)
            elif self.schema.kind == "cfd_fields":
                arrays = [
                    np.asarray(
                        handle[field][
                            (selection.trajectory_index, time_slice, *spatial_slices)
                        ]
                    )
                    for field in self.schema.channel_names
                ]
                values = np.stack(arrays, axis=-1)
                dataset_paths = self.schema.dataset_paths
                source_shapes = self.schema.source_shapes
                trajectory_name = str(selection.trajectory_index)
            else:
                trajectory_name = self._group_names[selection.trajectory_index]
                dataset_path = f"{trajectory_name}/data"
                dataset = handle[dataset_path]
                source_shape = tuple(dataset.shape)
                if source_shape[1:] != self.schema.source_shapes[0][1:]:
                    raise ValueError(
                        "Selected grouped trajectory does not match the inspected spatial/channel schema"
                    )
                raw = dataset[(time_slice, *spatial_slices, slice(None))]
                values = np.asarray(raw)
                dataset_paths = (dataset_path,)
                source_shapes = (source_shape,)
            if values.shape[0] == 0:
                raise ValueError("Trajectory slice selects no time steps")

            available_channel_count = values.shape[-1]
            channels = (
                tuple(range(available_channel_count))
                if selection.channel_indices is None
                else selection.channel_indices
            )
            if not channels or len(set(channels)) != len(channels):
                raise ValueError("channel_indices must be non-empty and unique")
            if min(channels) < 0 or max(channels) >= available_channel_count:
                raise IndexError("channel index is outside the PDEBench trajectory")
            values = np.take(values, channels, axis=-1)
            channel_names = tuple(self.schema.channel_names[index] for index in channels)

            times = None
            if self.schema.kind == "ns_incom":
                times = torch.as_tensor(
                    np.asarray(handle["t"][selection.trajectory_index, time_slice]),
                    dtype=dtype,
                    device=device,
                )
            elif "t-coordinate" in handle:
                times = torch.as_tensor(
                    np.asarray(handle["t-coordinate"][time_slice]),
                    dtype=dtype,
                    device=device,
                )
            coordinates = tuple(
                torch.as_tensor(
                    np.asarray(handle[key][:: spatial_stride[index]]),
                    dtype=dtype,
                    device=device,
                )
                for index, key in enumerate(self.schema.coordinate_keys)
            )
            root_attributes = {
                str(key): _json_attribute(value) for key, value in handle.attrs.items()
            }

        states = torch.as_tensor(values, dtype=dtype, device=device).reshape(values.shape[0], -1)
        spatial_shape = tuple(values.shape[1:-1])
        stat = self.path.stat()
        provenance = PDEBenchProvenance(
            source_path=str(self.path),
            file_size_bytes=stat.st_size,
            file_mtime_ns=stat.st_mtime_ns,
            schema=self.schema.kind,
            dataset_paths=dataset_paths,
            source_shapes=source_shapes,
            trajectory_index=selection.trajectory_index,
            trajectory_name=trajectory_name,
            time_slice=(selection.time_start, selection.time_stop, selection.time_step),
            spatial_stride=spatial_stride,
            channel_indices=channels,
            channel_names=channel_names,
            root_attributes=root_attributes,
            manifest_record=self.manifest_record,
        )
        return PDEBenchTrajectory(
            states=states,
            spatial_shape=spatial_shape,
            channel_names=channel_names,
            times=times,
            coordinates=coordinates,
            provenance=provenance,
        )
