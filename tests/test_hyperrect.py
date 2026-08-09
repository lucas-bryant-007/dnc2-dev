import itertools
import unittest

import torch

from analysis.hyperrect import (
    crossfit_probe_geometry,
    predicted_box_corners,
    split_balanced_whitening_and_probe_folds,
)


def _corners_by_combo(corners):
    return {tuple(entry["combo"]): entry["center"] for entry in corners}


class PredictedBoxCornersTest(unittest.TestCase):
    def test_theorem_coordinates_for_nonorthogonal_axes(self):
        basis = torch.tensor(
            [
                [1.0, 0.6, 0.0],
                [0.0, 0.8, 0.6],
                [0.0, 0.0, 0.8],
            ]
        )
        # Every column is a unit task axis, but the axes are not orthogonal.
        self.assertTrue(torch.allclose(basis.norm(dim=0), torch.ones(3)))
        self.assertFalse(torch.allclose(basis.T @ basis, torch.eye(3)))

        w_cols = basis * torch.tensor([0.2, 0.3, 0.4])
        capture = [0.25, 0.36, 0.49]
        corners = _corners_by_combo(
            predicted_box_corners(w_cols, basis, capture=capture)
        )

        half_sides = torch.tensor(capture).sqrt()
        for combo in itertools.product((0, 1), repeat=3):
            signs = torch.tensor([2 * bit - 1 for bit in combo])
            self.assertTrue(
                torch.allclose(torch.tensor(corners[combo]), signs * half_sides)
            )

    def test_uses_probe_norms_without_capture_override(self):
        basis = torch.eye(3)
        w_cols = basis * torch.tensor([0.25, 0.5, 0.75])
        corners = _corners_by_combo(predicted_box_corners(w_cols, basis))

        self.assertEqual(corners[(0, 1, 0)], [-0.25, 0.5, -0.75])
        self.assertEqual(corners[(1, 0, 1)], [0.25, -0.5, 0.75])

    def test_rejects_mismatched_coordinate_basis(self):
        with self.assertRaisesRegex(ValueError, "basis must match"):
            predicted_box_corners(torch.eye(3), torch.eye(4, 3))


class BalancedWhiteningFoldTest(unittest.TestCase):
    def test_whitening_and_probe_folds_are_disjoint_within_every_cell(self):
        samples_per_cell = 9
        selected = torch.arange(8 * samples_per_cell)
        whitening, first, second = split_balanced_whitening_and_probe_folds(
            selected,
            samples_per_cell,
        )

        self.assertEqual(whitening.numel(), 8 * 3)
        self.assertEqual(first.numel(), 8 * 3)
        self.assertEqual(second.numel(), 8 * 3)
        all_indices = torch.cat((whitening, first, second))
        self.assertEqual(torch.unique(all_indices).numel(), selected.numel())
        self.assertEqual(set(all_indices.tolist()), set(selected.tolist()))

    def test_three_folds_require_three_samples_per_cell(self):
        with self.assertRaisesRegex(ValueError, "at least three"):
            split_balanced_whitening_and_probe_folds(torch.arange(16), 2)


class CrossfitInterpretationTest(unittest.TestCase):
    def setUp(self):
        self.features = torch.tensor(
            [
                [-1.0, -1.0, -1.0],
                [-1.0, 1.0, 1.0],
                [1.0, -1.0, 1.0],
                [1.0, 1.0, -1.0],
            ]
        )
        self.attrs = (self.features > 0).long()

    def test_training_crossfit_is_not_labeled_post_selection_unbiased(self):
        geometry = crossfit_probe_geometry(
            self.features,
            self.attrs,
            self.features,
            self.attrs,
            ["a", "b", "c"],
            task_selection_status="selected_using_same_probe_observations",
        )
        interpretation = geometry["statistical_interpretation"]

        self.assertTrue(
            interpretation[
                "fixed_prespecified_task_unbiased_under_iid_sampling"
            ]
        )
        self.assertFalse(interpretation["post_selection_unbiasedness_claimed"])
        self.assertEqual(
            interpretation["reported_role"],
            "selection_conditioned_training_fit_not_unbiased_inference",
        )

    def test_heldout_crossfit_records_independent_task_freezing(self):
        geometry = crossfit_probe_geometry(
            self.features,
            self.attrs,
            self.features,
            self.attrs,
            ["a", "b", "c"],
            task_selection_status="frozen_from_independent_training_split",
        )
        interpretation = geometry["statistical_interpretation"]

        self.assertFalse(interpretation["post_selection_unbiasedness_claimed"])
        self.assertTrue(
            interpretation[
                "conditionally_unbiased_for_frozen_task_and_representation_"
                "under_iid_sampling"
            ]
        )
        self.assertEqual(
            interpretation["reported_role"],
            "conditionally_unbiased_evaluation_of_frozen_task_and_"
            "train_fitted_representation_under_iid_heldout_sampling",
        )
