"""Unit tests for the GPS-free A2 dual-DOA input boundary."""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from baoding_case.shuangyuan_a2_input import (
    FrontendRow,
    candidate_quality,
    infer_frozen_identity_prior,
    load_frontend_bundle,
)


FIELDNAMES = [
    "node_id",
    "time_s",
    "time_hhmmss",
    "frame_start_sample",
    "azimuth_1_deg",
    "azimuth_2_deg",
    "zenith_1_deg",
    "zenith_2_deg",
    "azimuth_strength_1",
    "azimuth_strength_2",
    "zenith_strength_1",
    "zenith_strength_2",
]


def write_frontend(path: Path, node: int, times: list[float]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDNAMES)
        writer.writeheader()
        for index, time_s in enumerate(times):
            writer.writerow({
                "node_id": node,
                "time_s": time_s,
                "time_hhmmss": 125540,
                "frame_start_sample": index * 640,
                "azimuth_1_deg": 10.0,
                "azimuth_2_deg": 200.0,
                "zenith_1_deg": 40.0,
                "zenith_2_deg": 0.0,
                "azimuth_strength_1": 4.0,
                "azimuth_strength_2": 1.0,
                "zenith_strength_1": 9.0,
                "zenith_strength_2": 1.0,
            })


def test_candidates_preserve_rank_pair_and_expose_boundary_flag() -> None:
    row = FrontendRow(
        node_id=1,
        frame_index=0,
        time_s=1.0,
        time_hhmmss=125540,
        frame_start_sample=0,
        azimuth_deg=(10.0, 200.0),
        zenith_deg=(40.0, 0.0),
        azimuth_strength=(4.0, 1.0),
        zenith_strength=(9.0, 1.0),
    )
    first, second = row.candidates({
        "az_sign": -1.0,
        "az_offset_deg": 15.0,
        "zenith_sign": 1.0,
        "zenith_offset_deg": -2.0,
    })
    assert (first.azimuth_deg, first.zenith_deg) == (5.0, 38.0)
    assert (second.azimuth_deg, second.zenith_deg) == (175.0, -2.0)
    assert not first.boundary
    assert second.zenith_boundary
    strength, ratio = candidate_quality(first, second)
    assert strength > 0.0
    assert ratio > 0.0


def test_bundle_fails_closed_on_timestamp_misalignment(tmp_path: Path) -> None:
    write_frontend(tmp_path / "dual_doa_node_1_125540_125900.csv", 1, [0.0, 0.2])
    write_frontend(tmp_path / "dual_doa_node_2_125540_125900.csv", 2, [0.0, 0.21])
    with pytest.raises(ValueError, match="timestamp mismatch"):
        load_frontend_bundle(tmp_path, [1, 2])


def test_frozen_association_is_recovered_as_prior_not_hard_observation() -> None:
    row = FrontendRow(
        node_id=1,
        frame_index=0,
        time_s=0.0,
        time_hhmmss=125540,
        frame_start_sample=0,
        azimuth_deg=(10.0, 200.0),
        zenith_deg=(40.0, 50.0),
        azimuth_strength=(1.0, 1.0),
        zenith_strength=(1.0, 1.0),
    )
    candidates = row.candidates()
    swapped = infer_frozen_identity_prior(candidates, (200.0, 50.0), (10.0, 40.0))
    assert (swapped.target1_peak_index, swapped.target2_peak_index) == (1, 0)
    assert swapped.matched_label_cost_deg == 0.0
    assert swapped.alternative_label_cost_deg > 0.0
