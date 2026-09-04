from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest
import torch

from viv_piv_case import assimilation
from viv_piv_case.assimilation import CandidateLibrary, Scenario
from viv_piv_case.io import VIVCase, parse_case_id, stored_memmap
from viv_piv_case.rom import DMDCCandidate, evaluation_indices, sensor_flat_indices
from viv_piv_case.run_case import _evaluation_keep_mask, _sensor_scalar_indices


def _candidate(case_id: str, coordinate: float, matrix: np.ndarray) -> DMDCCandidate:
    rank = matrix.shape[0]
    return DMDCCandidate(
        case_id=case_id,
        reduced_velocity=coordinate,
        a=matrix,
        b=np.zeros((rank, 3), dtype=np.float64),
        q_diag=np.full(rank, 1e-12, dtype=np.float64),
        residual_rms=0.0,
        spectral_radius=float(np.max(np.abs(np.linalg.eigvals(matrix)))),
    )


def test_case_id_parsing_handles_comma_names() -> None:
    assert parse_case_id("reduced_velocity_01,112.npz") == "1112"
    assert parse_case_id("reduced_velocity_01,359.npz") == "1359"
    assert parse_case_id("reduced_velocity_01,482.npz") == "1482"
    assert parse_case_id("reduced_velocity_00,463.npz") == "0463"


def test_stored_npz_member_is_read_only_memmap(tmp_path: pathlib.Path) -> None:
    expected = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    archive = tmp_path / "stored.npz"
    np.savez(archive, velocities=expected)
    mapped = stored_memmap(archive, "velocities")
    assert isinstance(mapped, np.memmap)
    assert mapped.mode == "r"
    np.testing.assert_array_equal(mapped, expected)


def test_train_test_and_sensor_evaluation_sets_are_disjoint() -> None:
    config_path = pathlib.Path(__file__).parents[1] / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert set(config["train_cases"]).isdisjoint(config["test_cases"])
    case = VIVCase(
        path=pathlib.Path("synthetic.npz"),
        case_id="0000",
        label="synthetic",
        x_mm=np.linspace(-100.0, 450.0, 416),
        y_mm=np.linspace(-150.0, 150.0, 201),
        time_s=np.arange(3, dtype=float) * 0.1,
        cyl_displ_m=np.zeros(3),
        norm_values=np.asarray([[0.0, 0.0], [1.0, 1.0]]),
    )
    sensors = set(sensor_flat_indices(case, config).tolist())
    evaluation = set(evaluation_indices(case, config, 2048).tolist())
    assert len(sensors) == 80
    assert sensors.isdisjoint(evaluation)


@pytest.mark.parametrize("density", [10, 20, 40])
def test_sensor_density_keeps_paired_components(density: int) -> None:
    selected = _sensor_scalar_indices(80, density)
    assert selected.size == 2 * density
    np.testing.assert_array_equal(selected[0::2] + 1, selected[1::2])
    assert np.unique(selected // 2).size == density


def test_layout_observations_are_removed_from_evaluation_dimensions() -> None:
    class PodStub:
        evaluation_flat_indices = np.asarray([1, 3, 5, 7, 9], dtype=np.int64)

    archive = {"sensor_flat_indices": np.asarray([2, 3, 9, 11], dtype=np.int64)}
    keep = _evaluation_keep_mask(PodStub(), archive)
    np.testing.assert_array_equal(keep, [True, False, True, True, False])


@pytest.mark.parametrize("method", ["pce", "apce"])
def test_full_covariance_evidence_accepts_uneven_final_patch(method: str) -> None:
    generator = torch.Generator().manual_seed(20260816)
    branch_observations = torch.randn((4, 12, 30), dtype=torch.float64, generator=generator)
    observation = torch.randn(30, dtype=torch.float64, generator=generator)
    covariance = 0.2 * torch.eye(30, dtype=torch.float64)
    config = {
        "evidence_patch_points": 4,
        "diagonal_evidence_above_dimensions": 0,
        "_full_observation_covariance": True,
        "evidence_shrinkage": 0.18,
        "dimension_weight_floor": 0.35,
        "dimension_weight_gain": 0.65,
    }
    patches = assimilation.observation_patches(observation.numel(), config)
    assert {int(patch.numel()) for patch in patches} == {6, 8}
    scores = assimilation.evidence_scores(
        branch_observations, observation, covariance, config, method
    )
    assert scores.shape == (4,)
    assert torch.isfinite(scores).all()


def test_entropy_projection_enforces_floor_and_normalization() -> None:
    weights = torch.tensor([0.999, 0.0005, 0.0005], dtype=torch.float64)
    projected = assimilation.entropy_project(weights, target=0.28)
    assert torch.isfinite(projected).all()
    assert float(projected.sum()) == pytest.approx(1.0, abs=1e-12)
    assert float(assimilation.entropy(projected)) >= 0.28 - 1e-10
    assert int(torch.argmax(projected)) == 0


def test_state_inflation_preserves_mean_and_scales_anomalies() -> None:
    ensemble = torch.tensor([[1.0, 4.0], [3.0, 8.0], [5.0, 12.0]], dtype=torch.float64)
    inflated = assimilation.inflate_ensemble(ensemble, 1.5)
    torch.testing.assert_close(inflated.mean(dim=0), ensemble.mean(dim=0))
    torch.testing.assert_close(
        inflated - inflated.mean(dim=0), 1.5 * (ensemble - ensemble.mean(dim=0))
    )


def test_state_inflation_rejects_nonpositive_factor() -> None:
    with pytest.raises(ValueError, match="positive"):
        assimilation.inflate_ensemble(torch.ones((2, 3), dtype=torch.float64), 0.0)


def test_ensemble_space_denkf_matches_observation_space_update() -> None:
    generator = torch.Generator().manual_seed(20260814)
    ensemble = torch.randn((12, 7), dtype=torch.float64, generator=generator)
    matrix = torch.randn((19, 7), dtype=torch.float64, generator=generator)
    observation = torch.randn(19, dtype=torch.float64, generator=generator)
    variances = 0.05 + torch.rand(19, dtype=torch.float64, generator=generator)
    covariance = torch.diag(variances)
    operator = lambda values: values @ matrix.mT
    expected = assimilation.denkf_analysis(ensemble, observation, operator, covariance)
    actual = assimilation.denkf_analysis_diagonal_ensemble_space(
        ensemble, observation, operator, covariance
    )
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)


def test_batched_general_covariance_denkf_matches_direct_update() -> None:
    generator = torch.Generator().manual_seed(14082026)
    branches = torch.randn((3, 12, 7), dtype=torch.float64, generator=generator)
    matrix = torch.randn((19, 7), dtype=torch.float64, generator=generator)
    observation = torch.randn(19, dtype=torch.float64, generator=generator)
    root = torch.randn((19, 19), dtype=torch.float64, generator=generator)
    covariance = root @ root.mT + 0.5 * torch.eye(19, dtype=torch.float64)
    covariance_factor = torch.linalg.cholesky(covariance)
    operator = lambda values: values @ matrix.mT
    actual = assimilation.denkf_analysis_general_ensemble_space_batch(
        branches, observation, operator, covariance_factor
    )
    expected = []
    for ensemble in branches:
        predicted = operator(ensemble)
        state_mean = ensemble.mean(dim=0)
        observation_mean = predicted.mean(dim=0)
        state_anomalies = ensemble - state_mean
        observation_anomalies = predicted - observation_mean
        cross = state_anomalies.mT @ observation_anomalies / (ensemble.shape[0] - 1)
        innovation = observation_anomalies.mT @ observation_anomalies / (ensemble.shape[0] - 1) + covariance
        gain = torch.linalg.solve(innovation, cross.mT).mT
        updated_mean = state_mean + gain @ (observation - observation_mean)
        expected.append(updated_mean[None, :] + state_anomalies - 0.5 * observation_anomalies @ gain.mT)
    expected_tensor = torch.stack(expected)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected_tensor, rtol=1e-10, atol=1e-10)


def test_local_candidate_interpolation_rejects_unstable_matrix() -> None:
    library = CandidateLibrary(
        [
            _candidate("0400", 4.0, np.eye(2) * 0.95),
            _candidate("0500", 5.0, np.eye(2) * 1.10),
        ]
    )
    with pytest.raises(ValueError, match="Unstable local candidate"):
        library.parameters(np.asarray([4.8]), torch.device("cpu"), torch.float64)


def test_vectorized_torch_interpolation_matches_audited_numpy_path() -> None:
    first = np.asarray([[0.80, 0.02], [0.01, 0.75]])
    second = np.asarray([[0.90, -0.01], [0.03, 0.70]])
    library = CandidateLibrary(
        [
            _candidate("0400", 4.0, first),
            _candidate("0600", 6.0, second),
        ]
    )
    coordinates = np.asarray([4.0, 4.5, 5.25, 6.0], dtype=np.float64)
    expected = library.parameters(coordinates, torch.device("cpu"), torch.float64)
    actual = library.parameters_torch(
        torch.as_tensor(coordinates, dtype=torch.float64), torch.device("cpu"), torch.float64
    )
    for expected_tensor, actual_tensor in zip(expected, actual):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=1e-12, atol=1e-12)


def test_aug_enkf_uses_one_total_ensemble_and_tracks_member_coordinates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    library = CandidateLibrary(
        [
            _candidate("0400", 4.0, np.eye(2)),
            _candidate("0500", 5.0, np.eye(2)),
            _candidate("0600", 6.0, np.eye(2)),
        ]
    )
    scenario = Scenario(
        case_id="test",
        time_s=np.arange(5, dtype=np.float64) * 0.1,
        control=np.zeros((5, 3), dtype=np.float64),
        observations=np.zeros((5, 2), dtype=np.float64),
        sensor_mean=np.zeros(2, dtype=np.float64),
        sensor_basis=np.eye(2, dtype=np.float64),
        evaluation_truth=np.zeros((5, 2), dtype=np.float64),
        evaluation_mean=np.zeros(2, dtype=np.float64),
        evaluation_basis=np.eye(2, dtype=np.float64),
        observation_noise=np.full(2, 0.1, dtype=np.float64),
        initial_std=np.full(2, 0.1, dtype=np.float64),
    )
    analysed_shapes: list[tuple[int, ...]] = []

    def identity_augmented_analysis(ensemble, *args, **kwargs):
        analysed_shapes.append(tuple(ensemble.shape))
        return ensemble

    monkeypatch.setattr(assimilation, "scalable_denkf_analysis", identity_augmented_analysis)
    config = {
        "ensemble_size": 8,
        "warmup_seconds": 0.0,
        "time_step_s": 0.1,
        "probabilistic_metric_stride": 1,
        "pce_temperature": 0.62,
        "apce_temperature": 0.50,
        "apce_min_temperature": 0.16,
        "apce_forgetting": 0.975,
        "apce_entropy_floor": 0.28,
        "reference_evidence_interval_s": 0.5,
    }
    result = assimilation.run_pass(
        scenario,
        library,
        config,
        "aug_enkf",
        seed=12,
        device=torch.device("cpu"),
        record_trace=True,
        blackout_origins={1},
    )
    assert analysed_shapes == [(8, 3)] * (scenario.steps - 1)
    assert result.final_weights.shape == (1,)
    np.testing.assert_allclose(result.final_weights, [1.0])
    assert result.grid.shape == (1,)
    snapshot = result.blackout_states[1]
    assert snapshot["branches"].shape == (1, 8, 2)
    assert snapshot["member_coordinates"].shape == (8,)
    assert result.trace["aug_coordinate_mean"].shape == (scenario.steps,)
    assert result.trace["aug_coordinate_std"].shape == (scenario.steps,)


def test_pce_evidence_uses_unanalysed_shadow(monkeypatch: pytest.MonkeyPatch) -> None:
    library = CandidateLibrary(
        [
            _candidate("0400", 4.0, np.eye(2)),
            _candidate("0500", 5.0, np.eye(2)),
        ]
    )
    scenario = Scenario(
        case_id="test",
        time_s=np.arange(5, dtype=np.float64) * 0.1,
        control=np.zeros((5, 3), dtype=np.float64),
        observations=np.zeros((5, 2), dtype=np.float64),
        sensor_mean=np.zeros(2, dtype=np.float64),
        sensor_basis=np.eye(2, dtype=np.float64),
        evaluation_truth=np.zeros((5, 2), dtype=np.float64),
        evaluation_mean=np.zeros(2, dtype=np.float64),
        evaluation_basis=np.eye(2, dtype=np.float64),
        observation_noise=np.full(2, 0.1, dtype=np.float64),
        initial_std=np.full(2, 0.1, dtype=np.float64),
    )
    captured_maxima: list[float] = []

    def fake_scores(branch_observations, observation, covariance, config, method):
        captured_maxima.append(float(branch_observations.abs().max()))
        return torch.zeros(branch_observations.shape[0], dtype=branch_observations.dtype)

    def large_analysis_shift(ensemble, observation, operator, covariance):
        return ensemble + 50.0

    monkeypatch.setattr(assimilation, "evidence_scores", fake_scores)
    monkeypatch.setattr(assimilation, "denkf_analysis", large_analysis_shift)
    config = {
        "ensemble_size": 8,
        "warmup_seconds": 0.0,
        "time_step_s": 0.1,
        "probabilistic_metric_stride": 1,
        "pce_temperature": 0.62,
        "apce_temperature": 0.50,
        "apce_min_temperature": 0.16,
        "apce_forgetting": 0.975,
        "apce_entropy_floor": 0.28,
        "reference_evidence_interval_s": 0.5,
    }
    result = assimilation.run_pass(
        scenario,
        library,
        config,
        "pce",
        seed=7,
        device=torch.device("cpu"),
        record_trace=True,
    )
    assert result.trace["weights"].shape[0] == scenario.steps - 1
    assert max(captured_maxima) < 1.0
    np.testing.assert_allclose(result.final_weights.sum(), 1.0, atol=1e-12)


def test_evidence_window_is_a_causal_trailing_average(monkeypatch: pytest.MonkeyPatch) -> None:
    library = CandidateLibrary(
        [
            _candidate("0400", 4.0, np.eye(2)),
            _candidate("0500", 5.0, np.eye(2)),
        ]
    )
    scenario = Scenario(
        case_id="test",
        time_s=np.arange(5, dtype=np.float64) * 0.1,
        control=np.zeros((5, 3), dtype=np.float64),
        observations=np.zeros((5, 2), dtype=np.float64),
        sensor_mean=np.zeros(2, dtype=np.float64),
        sensor_basis=np.eye(2, dtype=np.float64),
        evaluation_truth=np.zeros((5, 2), dtype=np.float64),
        evaluation_mean=np.zeros(2, dtype=np.float64),
        evaluation_basis=np.eye(2, dtype=np.float64),
        observation_noise=np.full(2, 0.1, dtype=np.float64),
        initial_std=np.full(2, 0.1, dtype=np.float64),
    )
    calls = 0

    def ordered_scores(branch_observations, observation, covariance, config, method):
        nonlocal calls
        calls += 1
        return torch.as_tensor([float(calls), 0.0], dtype=branch_observations.dtype)

    monkeypatch.setattr(assimilation, "evidence_scores", ordered_scores)
    config = {
        "ensemble_size": 8,
        "warmup_seconds": 0.0,
        "time_step_s": 0.1,
        "probabilistic_metric_stride": 1,
        "pce_temperature": 0.62,
        "apce_temperature": 0.50,
        "apce_min_temperature": 0.16,
        "apce_forgetting": 0.975,
        "apce_entropy_floor": 0.28,
        "reference_evidence_interval_s": 0.5,
        "evidence_window_frames": 3,
    }
    result = assimilation.run_pass(
        scenario,
        library,
        config,
        "pce",
        seed=9,
        device=torch.device("cpu"),
        record_trace=True,
    )
    np.testing.assert_allclose(
        result.trace["scores"],
        np.asarray([[0.5, -0.5], [0.75, -0.75], [1.0, -1.0], [1.5, -1.5]]),
        atol=1e-12,
    )
