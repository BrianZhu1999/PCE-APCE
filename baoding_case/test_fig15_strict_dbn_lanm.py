import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

from baoding_case.run_fig15_strict_dbn_lanm import (
    eq44_eq45_initialization,
    expected_residual_power,
    expected_steering_stats,
    log_posterior,
    sensor_geometry,
    steering,
    update_hidden,
    update_author_hierarchical_source,
    update_noise_precision,
)


def test_nod_azimuth_convention_reflects_y_axis_only_at_zero_offset() -> None:
    nodes = {}
    for node in (1, 3, 5, 6, 7, 8, 11, 13):
        nodes[node] = {
            "x": float(node * 10), "y": float(node * 20), "z": 0.0,
            "azimuth_offset_deg": 0.0, "elevation_offset_deg": 0.0,
            "horizontal_reverse": 1, "vertical_reverse": 0,
        }
    geometry = sensor_geometry(nodes, "field_nonuniform_z", "nod_azimuth_convention")
    first = geometry[:19]
    # Channel 9 is the -1.5 m endpoint of the x arm and remains on global x.
    np.testing.assert_allclose(first[8, :2], [8.5, 20.0])
    # Channel 12 is the -1.5 m endpoint in array-y; clockwise azimuth reverses y.
    np.testing.assert_allclose(first[11, :2], [10.0, 21.5])


def _device() -> torch.device:
    if not torch.cuda.is_available() or torch.cuda.device_count() <= 2:
        raise RuntimeError("Fig. 15 tests require physical GPU 2 or 3")
    return torch.device("cuda:2")


def test_eq44_uses_physical_snapshot_average() -> None:
    device = _device()
    torch.manual_seed(11)
    dtype = torch.complex128
    a = torch.randn((5, 12, 3), device=device, dtype=dtype)
    z = torch.randn((5, 12), device=device, dtype=dtype)
    lambda_1, source_1, diagnostics_1 = eq44_eq45_initialization(a, z)
    z_repeat = z[:, None, :].repeat(1, 4, 1)
    lambda_4, source_4, diagnostics_4 = eq44_eq45_initialization(a, z_repeat)
    torch.testing.assert_close(lambda_4, lambda_1, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(source_4, source_1, rtol=1e-12, atol=1e-12)
    assert diagnostics_1["eq44_snapshot_count_l"] == 1
    assert diagnostics_4["eq44_snapshot_count_l"] == 4


def test_eq44_does_not_treat_snapshot_source_variation_as_noise() -> None:
    device = _device()
    torch.manual_seed(13)
    dtype = torch.complex128
    a = torch.randn((4, 9, 2), device=device, dtype=dtype)
    sources = torch.randn((4, 5, 2), device=device, dtype=dtype)
    z = torch.einsum("fpm,flm->flp", a, sources)
    precision, _, diagnostics = eq44_eq45_initialization(a, z)
    assert diagnostics["eq44_snapshot_count_l"] == 5
    assert float(torch.min(precision).item()) > 1e15


def test_eq21_averages_residual_over_physical_snapshots() -> None:
    device = _device()
    dtype = torch.complex128
    a = torch.ones((1, 3, 1), device=device, dtype=dtype)
    z = torch.full((1, 3), 2.0 + 0.0j, device=device, dtype=dtype)
    aha = torch.einsum("fpm,fpn->fmn", torch.conj(a), a)
    mean = torch.zeros((1, 1), device=device, dtype=dtype)
    covariance = torch.full((1, 1, 1), 0.5, device=device, dtype=dtype)
    alpha = torch.ones((1,), device=device, dtype=torch.float64)
    beta = torch.ones((1,), device=device, dtype=torch.float64)
    _, beta_1, _, _ = update_noise_precision(z, a, aha, alpha, beta, mean, covariance)
    z_repeat = z[:, None, :].repeat(1, 4, 1)
    _, beta_4, _, _ = update_noise_precision(z_repeat, a, aha, alpha, beta, mean, covariance)
    # Mean residual is 12 and covariance trace is 1.5; beta increment is
    # 0.5*(12 + 1.5) for both one and four identical snapshots.
    torch.testing.assert_close(
        beta_1, torch.tensor([7.75], device=device, dtype=torch.float64), rtol=1e-12, atol=1e-12
    )
    torch.testing.assert_close(
        beta_4, beta_1, rtol=1e-12, atol=1e-12
    )


def test_eq24_source_update_uses_snapshot_mean() -> None:
    device = _device()
    dtype = torch.complex128
    a = torch.ones((1, 3, 1), device=device, dtype=dtype)
    z = torch.tensor([[[1.0 + 0.0j, 2.0 + 0.0j, 3.0 + 0.0j]]], device=device, dtype=dtype)
    mean = torch.zeros((1, 1), device=device, dtype=dtype)
    covariance = torch.full((1, 1, 1), 0.5, device=device, dtype=dtype)
    alpha = torch.ones((1,), device=device, dtype=torch.float64)
    beta = torch.ones((1,), device=device, dtype=torch.float64)
    _, _, source_one, _, _ = update_hidden(
        z[:, 0, :], a, torch.einsum("fpm,fpn->fmn", torch.conj(a), a),
        alpha, beta, mean, covariance, mean, covariance,
    )
    _, _, source_multi, _, _ = update_hidden(
        z, a, torch.einsum("fpm,fpn->fmn", torch.conj(a), a),
        alpha, beta, mean, covariance, mean, covariance,
    )
    torch.testing.assert_close(source_multi, source_one, rtol=1e-12, atol=1e-12)


def test_eq26_literal_and_eq21_consistent_snapshot_scaling() -> None:
    device = _device()
    dtype = torch.float64
    complex_dtype = torch.complex128
    states = torch.tensor([[2.0, 3.0, 0.0, 0.0]], device=device, dtype=dtype)
    sensors = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], device=device, dtype=dtype)
    frequencies = torch.tensor([300.0], device=device, dtype=dtype)
    source_mean = torch.tensor([[0.4 + 0.2j]], device=device, dtype=complex_dtype)
    source_cov = torch.tensor([[[0.6 + 0.0j]]], device=device, dtype=complex_dtype)
    expected_precision = torch.tensor([2.5], device=device, dtype=dtype)
    snapshots = torch.tensor(
        [[[1.0 + 0.0j, 0.2 + 0.3j], [0.7 - 0.1j, -0.2 + 0.4j],
          [0.1 + 0.5j, 0.8 - 0.3j], [-0.4 + 0.2j, 0.3 + 0.1j]]],
        device=device, dtype=complex_dtype,
    )
    predicted_covariance = torch.eye(4, device=device, dtype=dtype)[None, :, :]

    a = steering(states[:, [0, 2]], sensors, frequencies, "sensor_z")
    residual = snapshots - torch.einsum("fpm,fm->fp", a, source_mean)[:, None, :]
    mean_residual = torch.mean(torch.sum(torch.abs(residual) ** 2, dim=2), dim=1).real
    gram = torch.einsum("fpm,fpn->fmn", torch.conj(a), a)
    trace_term = torch.einsum("fmn,fnm->f", gram, source_cov).real
    expected_consistent = -0.5 * torch.mean(expected_precision * (mean_residual + trace_term))
    expected_literal = -0.5 * torch.mean(
        expected_precision * (mean_residual + trace_term / snapshots.shape[1])
    )

    actual_consistent = log_posterior(
        states, states, predicted_covariance, snapshots, source_mean, source_cov,
        expected_precision, sensors, frequencies, "sensor_z",
        "paper_eq26_eq21_consistent",
    )
    actual_literal = log_posterior(
        states, states, predicted_covariance, snapshots, source_mean, source_cov,
        expected_precision, sensors, frequencies, "sensor_z", "paper_eq26_literal",
    )
    torch.testing.assert_close(actual_consistent, expected_consistent, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(actual_literal, expected_literal, rtol=1e-12, atol=1e-12)
    assert float(actual_literal.item()) > float(actual_consistent.item())


def test_planar_steering_matches_horizontal_distance() -> None:
    device = _device()
    dtype = torch.float64
    sensors = torch.tensor([[0.0, 0.0, 100.0]], device=device, dtype=dtype)
    xy = torch.tensor([[3.0, 4.0]], device=device, dtype=dtype)
    frequencies = torch.tensor([0.0], device=device, dtype=dtype)
    planar = steering(xy, sensors, frequencies, "planar")
    expected = torch.tensor([[[0.2 + 0.0j]]], device=device, dtype=torch.complex128)
    torch.testing.assert_close(planar, expected, rtol=1e-12, atol=1e-12)


def test_monte_carlo_steering_samples_interleaved_uv_coordinates() -> None:
    device = _device()
    dtype = torch.float64
    mean = torch.tensor([[2.0, 99.0, 3.0, 88.0]], device=device, dtype=dtype)
    covariance = torch.eye(4, device=device, dtype=dtype)[None, :, :] * 1e-20
    sensors = torch.tensor([[0.0, 0.0, 0.0]], device=device, dtype=dtype)
    frequencies = torch.tensor([0.0], device=device, dtype=dtype)
    generator = torch.Generator(device=device).manual_seed(7)
    expected_a, _ = expected_steering_stats(
        mean, covariance, sensors, frequencies, 1, "planar", generator, 1, "paper_full"
    )
    expected = torch.tensor([[[1.0 / (13.0**0.5) + 0.0j]]], device=device, dtype=torch.complex128)
    torch.testing.assert_close(expected_a, expected, rtol=1e-8, atol=1e-8)


def test_expected_residual_reduces_to_deterministic_residual() -> None:
    device = _device()
    torch.manual_seed(17)
    dtype = torch.complex128
    a = torch.randn((3, 7, 2), device=device, dtype=dtype)
    z = torch.randn((3, 7), device=device, dtype=dtype)
    mean = torch.randn((3, 2), device=device, dtype=dtype)
    aha = torch.einsum("fpm,fpn->fmn", torch.conj(a), a)
    covariance = torch.zeros((3, 2, 2), device=device, dtype=dtype)
    actual = expected_residual_power(z, a, aha, mean, covariance)
    expected = torch.sum(torch.abs(z - torch.einsum("fpm,fm->fp", a, mean)) ** 2, dim=1).real
    torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-11)


def test_expected_residual_matches_direct_monte_carlo_average() -> None:
    device = _device()
    torch.manual_seed(23)
    dtype = torch.complex128
    draws, sensors, targets = 120000, 4, 2
    z = torch.randn((1, sensors), device=device, dtype=dtype)
    a_draws = torch.randn((draws, sensors, targets), device=device, dtype=dtype)
    source_mean = torch.tensor([[0.7 + 0.2j, -0.3 + 0.4j]], device=device, dtype=dtype)
    source_cov = torch.diag_embed(
        torch.tensor([[0.15, 0.08]], device=device, dtype=torch.float64)
    ).to(dtype)
    source_noise = torch.randn((draws, targets), device=device, dtype=dtype)
    source_draws = source_mean + source_noise * torch.sqrt(
        torch.tensor([0.15, 0.08], device=device, dtype=torch.float64)
    )
    expected_a = torch.mean(a_draws, dim=0, keepdim=True)
    expected_aha = torch.mean(
        torch.einsum("jpm,jpn->jmn", torch.conj(a_draws), a_draws), dim=0, keepdim=True
    )
    actual = expected_residual_power(z, expected_a, expected_aha, source_mean, source_cov)[0]
    residual = z[0][None, :] - torch.einsum("jpm,jm->jp", a_draws, source_draws)
    direct = torch.mean(torch.sum(torch.abs(residual) ** 2, dim=1))
    torch.testing.assert_close(actual, direct, rtol=8e-3, atol=8e-3)


def test_author_hierarchy_matches_delivered_tracking_update() -> None:
    device = _device()
    author_root = Path("<HILDA_RESULTS_ROOT>/code/baoding_case/MTT_WB_DBN")
    sys.path.insert(0, str(author_root))
    spec = importlib.util.spec_from_file_location("author_tracking", author_root / "Tracking.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    rng = np.random.RandomState(31)
    tracker = module.DbnTracking(sigma=10)
    frequencies = list(tracker.SysSet.fr)
    frequency_count = len(frequencies)
    sensor_count = tracker.SysSet.P
    target_count = tracker.SysSet.M
    z_np = rng.normal(size=(frequency_count, sensor_count)) + 1j * rng.normal(
        size=(frequency_count, sensor_count)
    )
    a_np = rng.normal(size=(frequency_count, sensor_count, target_count)) + 1j * rng.normal(
        size=(frequency_count, sensor_count, target_count)
    )
    diagonal = rng.uniform(0.2, 1.2, size=(frequency_count, target_count))
    aha_np = np.zeros((frequency_count, target_count, target_count), dtype=np.complex128)
    for fi in range(frequency_count):
        np.fill_diagonal(aha_np[fi], diagonal[fi])
    tracker.zk = z_np.copy()
    frequency_to_index = {float(frequency): index for index, frequency in enumerate(frequencies)}
    tracker.EX_Ax = lambda frequency: (
        aha_np[frequency_to_index[float(frequency)]],
        a_np[frequency_to_index[float(frequency)]],
    )

    z = torch.as_tensor(z_np, device=device, dtype=torch.complex128)
    expected_a = torch.as_tensor(a_np, device=device, dtype=torch.complex128)
    expected_aha = torch.as_tensor(aha_np, device=device, dtype=torch.complex128)
    source_mean = torch.ones((frequency_count, target_count), device=device, dtype=torch.complex128)
    source_cov = torch.diag_embed(
        torch.full((frequency_count, target_count), 0.1, device=device, dtype=torch.float64)
    ).to(torch.complex128)
    alpha = torch.full((frequency_count,), 100.0, device=device, dtype=torch.float64)
    beta = torch.ones((frequency_count,), device=device, dtype=torch.float64)
    alpha_new, beta_new, noise_precision, _ = update_noise_precision(
        z, expected_a, expected_aha, alpha, beta, source_mean, source_cov
    )
    result = update_author_hierarchical_source(
        z, expected_a, expected_aha, noise_precision, source_mean, source_cov,
        source_mean.clone(),
        torch.full((frequency_count, target_count), 0.1, device=device, dtype=torch.complex128),
        torch.full((frequency_count, target_count), 10.0, device=device, dtype=torch.complex128),
        torch.ones((frequency_count, target_count), device=device, dtype=torch.complex128),
        "author_complex_square",
    )
    source_new, source_cov_new, hyper_mean_new, hyper_variance_new, shape_new, rate_new, _ = result

    tracker.UpdateLambda_0()
    tracker.UpdateLambda()
    tracker.UpdateMu()
    tracker.UpdateSk()
    author_alpha = np.asarray([item.a for item in tracker.lambda_0])
    author_beta = np.asarray([item.b for item in tracker.lambda_0])
    author_source = np.asarray([[tracker.sk[fi][m].m.g for m in range(target_count)] for fi in range(frequency_count)])
    author_source_variance = np.asarray(
        [[1.0 / tracker.sk[fi][m].s.lam for m in range(target_count)] for fi in range(frequency_count)]
    )
    author_hyper_mean = np.asarray(
        [[tracker.sk[fi][m].m.m for m in range(target_count)] for fi in range(frequency_count)]
    )
    author_hyper_variance = np.asarray(
        [[tracker.sk[fi][m].m.s for m in range(target_count)] for fi in range(frequency_count)]
    )
    author_shape = np.asarray(
        [[tracker.sk[fi][m].s.a for m in range(target_count)] for fi in range(frequency_count)]
    )
    author_rate = np.asarray(
        [[tracker.sk[fi][m].s.b for m in range(target_count)] for fi in range(frequency_count)]
    )

    np.testing.assert_allclose(alpha_new.cpu().numpy(), author_alpha, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(beta_new.cpu().numpy(), author_beta, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(source_new.cpu().numpy(), author_source, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(
        torch.diagonal(source_cov_new, dim1=-2, dim2=-1).cpu().numpy(),
        author_source_variance, rtol=1e-10, atol=1e-10,
    )
    np.testing.assert_allclose(hyper_mean_new.cpu().numpy(), author_hyper_mean, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(
        hyper_variance_new.cpu().numpy(), author_hyper_variance, rtol=1e-10, atol=1e-10
    )
    np.testing.assert_allclose(shape_new.cpu().numpy(), author_shape, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(rate_new.cpu().numpy(), author_rate, rtol=1e-10, atol=1e-10)


def test_delivered_state_derivatives_match_their_implicit_objective() -> None:
    """Check whether Tracking.py's GradHess_xk_f is internally consistent.

    The delivered state derivative omits the noise precision and cross-target
    terms from Eq. (26).  Its remaining terms imply the scalar objective used
    below.  This test separates an analytic-derivative bug from a deliberate
    code-versus-paper objective change.
    """
    device = _device()
    author_root = Path("<HILDA_RESULTS_ROOT>/code/baoding_case/MTT_WB_DBN")
    sys.path.insert(0, str(author_root))
    spec = importlib.util.spec_from_file_location("author_tracking_derivative", author_root / "Tracking.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    rng = np.random.RandomState(43)
    tracker = module.DbnTracking(sigma=10)
    target = 0
    frequency_index = 2
    frequency = float(tracker.SysSet.fr[frequency_index])
    x_np = np.array([[4.0], [11.0], [0.7], [-0.2]], dtype=np.float64)
    z_np = rng.normal(size=tracker.SysSet.P) + 1j * rng.normal(size=tracker.SysSet.P)
    source_mean = np.complex128(0.8 - 0.35j)
    source_precision = 3.7
    tracker.zk = np.zeros((len(tracker.SysSet.fr), tracker.SysSet.P), dtype=np.complex128)
    tracker.zk[frequency_index] = z_np
    tracker.sk[frequency_index][target].m.g = source_mean
    tracker.sk[frequency_index][target].s.lam = source_precision
    author_gradient, author_hessian = tracker.GradHess_xk_f(
        x_np, target, frequency_index, frequency
    )

    variable = torch.as_tensor(x_np[:, 0], device=device, dtype=torch.float64).requires_grad_(True)
    sensors = torch.as_tensor(tracker.SysSet.ArrayShape, device=device, dtype=torch.float64)
    z = torch.as_tensor(z_np, device=device, dtype=torch.complex128)
    mu = torch.as_tensor(source_mean, device=device, dtype=torch.complex128)

    def implicit_objective(value: torch.Tensor) -> torch.Tensor:
        distance = torch.linalg.vector_norm(value[:2][None, :] - sensors, dim=1)
        a = torch.exp(-1j * 2.0 * torch.pi * frequency * distance / tracker.SysSet.C) / distance
        source_power = torch.abs(mu) ** 2 + 1.0 / source_precision
        matched = torch.real(torch.conj(mu) * torch.sum(torch.conj(a) * z))
        return matched - 0.5 * source_power * torch.sum(torch.abs(a) ** 2)

    value = implicit_objective(variable)
    automatic_gradient = torch.autograd.grad(value, variable, create_graph=True)[0]
    automatic_hessian = torch.autograd.functional.hessian(implicit_objective, variable, vectorize=True)
    np.testing.assert_allclose(
        author_gradient[:, 0], automatic_gradient.detach().cpu().numpy(), rtol=2e-10, atol=2e-10
    )
    np.testing.assert_allclose(
        author_hessian, automatic_hessian.detach().cpu().numpy(), rtol=2e-9, atol=2e-9
    )
