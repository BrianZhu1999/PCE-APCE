from __future__ import annotations

import unittest

import torch

import paper_experiments.run_figure3_applied_ode as figure3_runner
from hilda_da.alpha_refinement import (
    apce_calibration_parameters,
    torch_local_alpha_grid,
    torch_refined_alpha_map,
    torch_regrid_paths,
)
from hilda_da.systems.applied_odes import final_figure3_case_names, final_figure3_case_spec


class Figure3FinalRegistryTest(unittest.TestCase):
    def test_final_registry_has_exactly_fourteen_cases(self) -> None:
        names = final_figure3_case_names()
        self.assertEqual(len(names), 14)
        self.assertEqual(
            names,
            (
                "chemical",
                "pk_infusion",
                "sir",
                "sis",
                "seiar",
                "logistic",
                "gordon_schaefer",
                "rl_circuit",
                "van_der_pol",
                "duffing",
                "lotka_volterra",
                "fhn",
                "robertson",
                "pendulum",
            ),
        )

    def test_registry_entries_are_accessible(self) -> None:
        seiar = final_figure3_case_spec("seiar")
        robertson = final_figure3_case_spec("robertson")
        pendulum = final_figure3_case_spec("pendulum")
        self.assertEqual(seiar.tier, "A")
        self.assertEqual(robertson.tier, "B")
        self.assertEqual(pendulum.tier, "B")
        self.assertEqual(seiar.observation_dim, 3)
        self.assertEqual(robertson.state_dim, 3)
        self.assertEqual(pendulum.state_dim, 2)

    def test_legacy_case_names_are_rejected_by_final_registry(self) -> None:
        for legacy in ("pk_bolus", "sir_epidemic", "seir", "rlc_circuit"):
            with self.assertRaises(KeyError):
                final_figure3_case_spec(legacy)

    def test_runner_config_uses_registry_defaults(self) -> None:
        config = figure3_runner.config_for_case("seiar", 2026081200)
        self.assertEqual(config.steps, final_figure3_case_spec("seiar").default_steps)
        self.assertEqual(config.dt, final_figure3_case_spec("seiar").default_dt)
        self.assertEqual(config.obs_interval, final_figure3_case_spec("seiar").default_obs_interval)
        self.assertEqual(config.ensemble_size, final_figure3_case_spec("seiar").default_ensemble_size)
        self.assertEqual(config.obs_noise, final_figure3_case_spec("seiar").default_obs_noise)
        self.assertEqual(config.alpha_true, 0.12)

    def test_observation_geometry_matches_registry(self) -> None:
        device = torch.device("cpu")
        for name in final_figure3_case_names():
            spec = final_figure3_case_spec(name)
            indices = spec.observation_indices_factory(device)
            self.assertEqual(int(indices.numel()), spec.observation_dim, name)
            self.assertTrue(bool(torch.all(indices >= 0)), name)
            self.assertTrue(bool(torch.all(indices < spec.state_dim)), name)
            self.assertEqual(int(indices.unique().numel()), int(indices.numel()), name)

    def test_figure3_uses_shared_refinement_core(self) -> None:
        self.assertIs(figure3_runner.apce_calibration_parameters, apce_calibration_parameters)
        self.assertIs(figure3_runner.torch_local_alpha_grid, torch_local_alpha_grid)
        self.assertIs(figure3_runner.torch_refined_alpha_map, torch_refined_alpha_map)
        self.assertIs(figure3_runner.torch_regrid_paths, torch_regrid_paths)

    def test_method_specific_case_overrides_are_separated(self) -> None:
        config = figure3_runner.config_for_case("logistic", 2026081200)
        profile = {
            "pce_apce_overrides": {},
            "method_overrides": {},
            "case_overrides": {
                "logistic": {
                    "state_weight_power": 2.0,
                    "pce": {"state_weight_power": 7.0},
                    "apce": {"state_weight_power": 3.0, "global_analysis_strength": 0.5},
                }
            },
        }
        pce = figure3_runner.apply_tuning_profile(config, profile, "logistic", "pce")
        apce = figure3_runner.apply_tuning_profile(config, profile, "logistic", "apce")
        self.assertEqual(pce.state_weight_power, 7.0)
        self.assertEqual(apce.state_weight_power, 3.0)
        self.assertEqual(pce.global_analysis_strength, 0.0)
        self.assertEqual(apce.global_analysis_strength, 0.5)

    def test_global_augmented_alpha_tuning_field_is_supported(self) -> None:
        config = figure3_runner.config_for_case("duffing", 2026081200)
        profile = {
            "pce_apce_overrides": {"global_augmented_alpha_analysis_strength": 0.25},
            "method_overrides": {"apce": {"global_augmented_alpha_analysis_strength": 0.4}},
            "case_overrides": {
                "duffing": {
                    "pce": {"global_augmented_alpha_analysis_strength": 0.6},
                    "apce": {"global_augmented_alpha_analysis_strength": 0.8},
                }
            },
        }
        pce = figure3_runner.apply_tuning_profile(config, profile, "duffing", "pce")
        apce = figure3_runner.apply_tuning_profile(config, profile, "duffing", "apce")
        self.assertEqual(pce.global_augmented_alpha_analysis_strength, 0.6)
        self.assertEqual(apce.global_augmented_alpha_analysis_strength, 0.8)


if __name__ == "__main__":
    unittest.main()
