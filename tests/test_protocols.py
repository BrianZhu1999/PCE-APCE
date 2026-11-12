import json
from pathlib import Path

import numpy as np
import torch

from acoustic_field_reconstruction.boundary_candidates import (
    SHAPE_ZYX,
    causal_boundary_candidate_series_from_sparse,
    full_grid,
    sample_192,
)
from benchmarks.applied_odes import CASES, config_for_case
from benchmarks.kolmogorov_velocity import KOL64VelocityConfig
from benchmarks.kuramoto_sivashinsky import KSEConfig
from benchmarks.lorenz96 import L96ScalingConfig
from benchmarks.wave_protocol import METHOD_SEED_OFFSETS
from pce_assimilation.systems.applied_odes import applied_ode_case_spec


ROOT = Path(__file__).resolve().parents[1]


def load_json(relative_path: str) -> dict:
    return json.loads((ROOT / relative_path).read_text(encoding="utf-8"))


def test_five_applied_ode_cases_are_complete() -> None:
    expected = ("chemical", "pk_infusion", "pendulum", "fhn", "robertson")
    assert CASES == expected
    for name in expected:
        spec = applied_ode_case_spec(name)
        config = config_for_case(name, seed=7)
        assert spec.state_dim == spec.initial_state_factory(torch.device("cpu")).numel()
        assert spec.observation_dim == spec.observation_indices_factory(torch.device("cpu")).numel()
        assert len(config.alpha_grid) == 7


def test_high_dimensional_protocol_defaults() -> None:
    lorenz = L96ScalingConfig(seed=7)
    assert (lorenz.state_dim, lorenz.observed_points, lorenz.obs_interval) == (1024, 128, 8)
    assert len(lorenz.coarse_alpha_grid) == 7

    ks = KSEConfig(seed=7, sample_index=0)
    assert (ks.state_dim, ks.observed_points, ks.obs_interval) == (1024, 32, 2)
    assert len(ks.coarse_mu_grid) == 7

    kolmogorov = KOL64VelocityConfig(seed=7, sensor_grid=16, window_start=10)
    assert (kolmogorov.state_dim, kolmogorov.observed_points) == (8192, 512)
    assert len(kolmogorov.coarse_alpha_grid) == 7


def test_wave_comparison_seed_offsets_are_frozen() -> None:
    assert METHOD_SEED_OFFSETS == {
        "denkf": 1_679_088_713,
        "letkf": 3_337_870_379,
        "ensf_lr_ridge": 203_002_971,
    }


def test_viv_piv_split_and_sensor_layout() -> None:
    protocol = load_json("viv_piv/protocol.json")
    train = set(protocol["train_cases"])
    test = set(protocol["test_cases"])
    layout = protocol["sensor_layouts"]["adaptive_fullfield_valid_x40y20"]

    assert (len(train), len(test), train & test) == (12, 5, set())
    assert protocol["rank"] == 256
    assert protocol["ensemble_size"] == 64
    assert layout["x_points"] * layout["y_points"] == layout["target_points"] == 800


def test_acoustic_field_sampling_and_candidates() -> None:
    protocol = load_json("acoustic_field_reconstruction/protocol.json")
    sampling = sample_192()
    sparse = np.union1d(sampling["boundary_flat"], sampling["interior_flat"])
    assert protocol["grid"]["boundary_observed_count"] == 128
    assert protocol["grid"]["interior_observed_count"] == 64
    assert sparse.size == 192

    positions, _ = full_grid()
    truth = np.zeros((2,) + SHAPE_ZYX, dtype=np.float32)
    candidates, labels = causal_boundary_candidate_series_from_sparse(
        truth,
        sampling["boundary_flat"],
        positions,
        np.array([-0.8, 0.0, 0.0]),
        sample_rate=16_000.0,
        sound_speed=protocol["physics"]["nominal_speed_m_s"],
    )
    assert candidates.shape == (2, 8) + SHAPE_ZYX
    assert len(labels) == 8


def test_acoustic_tracking_protocols() -> None:
    protocol = load_json("acoustic_array_tracking/protocol.json")
    single = protocol["single_source"]
    dual = protocol["dual_source"]

    assert single["segment"] == "single_source_evaluation"
    assert (single["ensemble_size"], single["q_min_accel_mps2"], single["q_max_accel_mps2"]) == (48, 1.0, 8.0)
    assert (single["position_init_std_m"], single["velocity_init_std_mps"], single["turn_rate_radps"]) == (30.0, 5.0, -0.1)
    assert (dual["ensemble_size"], dual["q_min_accel_mps2"], dual["q_max_accel_mps2"]) == (48, 2.0, 12.0)
    assert (dual["position_init_std_m"], dual["velocity_init_std_mps"], dual["turn_rate_radps"]) == (50.0, 10.0, 0.2)
    assert dual["innovation_chi2_threshold"] == 16.26623619623813
