from __future__ import annotations

import unittest

import torch

from experiments.figure2_statealpha_refinement import StateAlphaHooks, _member_alpha_cloud


class Figure2StateAlphaRefinementTest(unittest.TestCase):
    def test_v52_hook_subset_is_read_from_method_profile(self) -> None:
        profile={"method_overrides":{"pce":{"branch_member_alpha_jitter":0.012,"branch_augmented_alpha_analysis_strength":0.6,"global_augmented_alpha_analysis_strength":0.25,"global_analysis_strength":1.0}}}
        hooks=StateAlphaHooks.from_profile(profile,"pce")
        self.assertEqual(hooks.global_analysis_strength,1.0)
        self.assertEqual(hooks.branch_augmented_alpha_analysis_strength,0.6)

    def test_unknown_profile_field_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            StateAlphaHooks.from_profile({"method_overrides":{"pce":{"not_a_figure2_field":1}}},"pce")

    def test_zero_jitter_returns_path_centres(self) -> None:
        grid=torch.tensor([0.1,0.4,0.8],dtype=torch.float64)
        cloud=_member_alpha_cloud(grid,4,StateAlphaHooks(),torch.zeros(3,dtype=torch.float64))
        self.assertTrue(torch.allclose(cloud,grid[:,None].expand(-1,4)))

    def test_jitter_is_bounded_and_has_expected_shape(self) -> None:
        grid=torch.tensor([0.1,0.4,0.8],dtype=torch.float64)
        cloud=_member_alpha_cloud(grid,5,StateAlphaHooks(branch_member_alpha_jitter=0.012),torch.zeros(3,dtype=torch.float64))
        self.assertEqual(tuple(cloud.shape),(3,5))
        self.assertTrue(bool(torch.all(cloud>=0.1) and torch.all(cloud<=0.8)))


if __name__ == "__main__":
    unittest.main()
