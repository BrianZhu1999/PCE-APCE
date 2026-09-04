from __future__ import annotations

import unittest

import torch

from hilda_da.systems import (
    AllenCahn1D,
    AllenCahnConfig,
    Burgers1D,
    BurgersConfig,
    EnFFNavierStokes2D,
    EnFFNavierStokesConfig,
    Heat1D,
    HeatConfig,
    NavierStokes2D,
    NavierStokesConfig,
    SpringOscillator,
    Wave1D,
    WaveConfig,
)
from hilda_da.systems.base import HybridSystem


class _PureNoise(HybridSystem):
    state_dim = 4

    def drift(self, state, time, alpha_quantile):
        return torch.zeros_like(state)

    def diffusion(self, state, time):
        return torch.ones_like(state)


class SystemTests(unittest.TestCase):
    def _assert_finite_step(self, system, state, dt) -> torch.Tensor:
        generator = torch.Generator().manual_seed(20260804)
        result = system.step(state, 0.0, dt, 0.65, generator)
        self.assertEqual(result.shape, state.shape)
        self.assertTrue(torch.isfinite(result).all())
        return result

    def test_stochastic_increment_uses_square_root_dt(self) -> None:
        system = _PureNoise()
        initial = torch.zeros(3, 4, dtype=torch.float64)
        small = system.step(initial, 0.0, 0.01, 0.5, torch.Generator().manual_seed(4))
        large = system.step(initial, 0.0, 0.04, 0.5, torch.Generator().manual_seed(4))
        self.assertTrue(torch.allclose(large, 2.0 * small))

    def test_spring(self) -> None:
        self._assert_finite_step(
            SpringOscillator(),
            torch.zeros(5, 2, dtype=torch.float64),
            0.01,
        )

    def test_heat_dirichlet_boundaries(self) -> None:
        system = Heat1D(HeatConfig(nx=32, diffusivity=0.02))
        state = torch.rand(4, system.state_dim, dtype=torch.float64)
        result = self._assert_finite_step(system, state, 1e-4)
        self.assertTrue(torch.all(result[..., 0] == 0))
        self.assertTrue(torch.all(result[..., -1] == 0))

    def test_wave_dirichlet_boundaries(self) -> None:
        system = Wave1D(WaveConfig(nx=32, wave_speed=0.5))
        state = torch.rand(4, system.state_dim, dtype=torch.float64)
        result = self._assert_finite_step(system, state, 5e-4)
        nx = system.config.nx
        self.assertTrue(torch.all(result[..., 0] == 0))
        self.assertTrue(torch.all(result[..., nx - 1] == 0))
        self.assertTrue(torch.all(result[..., nx] == 0))
        self.assertTrue(torch.all(result[..., -1] == 0))

    def test_burgers_spectral_step(self) -> None:
        system = Burgers1D(BurgersConfig(nx=32, viscosity=0.04))
        grid = torch.linspace(0, 2 * torch.pi, 33, dtype=torch.float64)[:-1]
        state = torch.sin(grid).repeat(3, 1)
        self._assert_finite_step(system, state, 5e-4)

    def test_allen_cahn_spectral_step(self) -> None:
        system = AllenCahn1D(AllenCahnConfig(nx=32))
        state = 0.2 * torch.randn(3, 32, dtype=torch.float64)
        self._assert_finite_step(system, state, 1e-3)

    def test_navier_stokes_spectral_step(self) -> None:
        system = NavierStokes2D(NavierStokesConfig(nx=16, ny=16, viscosity=2e-3))
        state = 0.1 * torch.randn(2, system.state_dim, dtype=torch.float64)
        self._assert_finite_step(system, state, 2e-4)

    def test_enff_navier_stokes_projection_and_alpha_forcing(self) -> None:
        system = EnFFNavierStokes2D(
            EnFFNavierStokesConfig(
                nx=8,
                ny=8,
                pressure_iterations=40,
                stochastic_scale=0.0,
            )
        )
        state = system.taylor_green_state(dtype=torch.float64, device=torch.device("cpu"))
        perturbed = state + 0.01 * torch.randn(3, system.state_dim, dtype=torch.float64)
        projected = system.project_state(perturbed, 1e-3)
        fields = system._reshape(projected)
        divergence = system._divergence(fields[..., 1, :, :], fields[..., 2, :, :])
        self.assertLess(float(divergence.square().mean().sqrt()), 0.2)
        low = system.step(state, 0.0, 1e-3, 0.2)
        high = system.step(state, 0.0, 1e-3, 0.8)
        self.assertEqual(low.shape, (system.state_dim,))
        self.assertTrue(torch.isfinite(low).all())
        self.assertFalse(torch.equal(low, high))


if __name__ == "__main__":
    unittest.main()
