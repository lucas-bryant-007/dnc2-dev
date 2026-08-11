import unittest

import torch

from analysis.bounds import (
    _build_instance_sampling_layout,
    _sample_grouped_binary_trial,
    cdnv_from_B,
    combined_fewshot_curves,
    directional_cdnv_from_B,
    directional_nccc_bound,
    empirical_nccc_error,
    hyperrectangle_half_side_lengths,
    hyperrectangle_side_lengths,
    luthra2025_aggregates_from_features,
    luthra2025_fixed_a16_from_aggregates,
    luthra2025_official_optimized_bound,
    luthra2025_official_optimized_details,
    nccc_error_bound,
    nccc_error_bound_from_tilde_v,
    nccc_pair_c2,
    nccc_pair_thm41,
)
from analysis.cdnv_conventions import (
    ORIGINAL_HALF_SYMMETRIC,
    UNHALVED_SYMMETRIC,
    convert_symmetric_cdnv,
)
from analysis.spectral import estimate_B_r_corrected


def _two_way_pairwise():
    return {
        (0, 1): {
            "Vtilde_ij": 0.10,
            "Vij": 999.0,
            "Theta_ij": 0.20,
            "vi": 0.20,
            "vj": 0.30,
            "d2": 1.0,
        },
        (1, 0): {
            "Vtilde_ij": 0.20,
            "Vij": 999.0,
            "Theta_ij": 0.20,
            "vi": 0.30,
            "vj": 0.20,
            "d2": 1.0,
        },
    }


class PublishedFormulaTest(unittest.TestCase):
    def test_cdnv_normalization_conversions_and_B_identities(self):
        self.assertEqual(
            convert_symmetric_cdnv(
                3.0,
                source=ORIGINAL_HALF_SYMMETRIC,
                target=UNHALVED_SYMMETRIC,
            ),
            6.0,
        )
        self.assertEqual(
            convert_symmetric_cdnv(
                6.0,
                source=UNHALVED_SYMMETRIC,
                target=ORIGINAL_HALF_SYMMETRIC,
            ),
            3.0,
        )
        self.assertAlmostEqual(directional_cdnv_from_B(0.2), 2.0)
        self.assertAlmostEqual(
            directional_cdnv_from_B(
                0.2,
                normalization=ORIGINAL_HALF_SYMMETRIC,
            ),
            1.0,
        )
        self.assertAlmostEqual(cdnv_from_B(0.2, 3), 7.0)
        self.assertAlmostEqual(
            cdnv_from_B(0.2, 3, normalization=ORIGINAL_HALF_SYMMETRIC),
            3.5,
        )
        with self.assertRaisesRegex(ValueError, "integer"):
            cdnv_from_B(0.2, 1.5)
        with self.assertRaisesRegex(ValueError, "unsupported CDNV"):
            directional_cdnv_from_B(0.0, normalization="ambiguous")

    def test_luthra_fixed_a16_golden_coefficients(self):
        self.assertAlmostEqual(
            luthra2025_fixed_a16_from_aggregates(0.1, 0.25, 0.5, 10, 2),
            2.7973665961010274,
        )
        self.assertAlmostEqual(
            luthra2025_fixed_a16_from_aggregates(0.1, 0.25, 0.5, 100, 2),
            1.4100000000000001,
        )

    def test_official_optimized_a_matches_reference_commit(self):
        expected = {
            10: (2.5357338875451765, 11.526335664232267),
            100: (1.3233244738990941, 14.875033254353715),
        }
        for m, (value, a_opt) in expected.items():
            details = luthra2025_official_optimized_details(
                0.1,
                0.25,
                m,
                2,
            )
            self.assertAlmostEqual(details["value"], value)
            self.assertAlmostEqual(details["a_opt"], a_opt)

    def test_2025_adapter_ignores_ambiguous_legacy_Vij(self):
        pairwise = _two_way_pairwise()
        observed = luthra2025_official_optimized_bound(pairwise, 10)
        # Ordered aggregates are alpha=.15 and beta=.25. Vij=999 must not leak
        # into the provenance-exact 2025 interface.
        expected = luthra2025_official_optimized_details(0.15, 0.25, 10, 2)[
            "value"
        ]
        self.assertAlmostEqual(observed, expected)

    def test_official_feature_aggregates_use_population_variance(self):
        features = torch.tensor([[-2.0], [0.0], [0.0], [2.0]])
        labels = torch.tensor([0, 0, 1, 1])
        n_classes, directional, cdnv, sqrt_cdnv = (
            luthra2025_aggregates_from_features(features, labels)
        )
        self.assertEqual(n_classes, 2)
        self.assertAlmostEqual(directional, 0.25)
        self.assertAlmostEqual(cdnv, 0.25)
        self.assertAlmostEqual(sqrt_cdnv, 0.5)

    def test_theorem_validity_domains_are_enforced(self):
        with self.assertRaisesRegex(ValueError, "m >= 10"):
            nccc_pair_thm41(0.1, 0.2, 0.3, 0.2, 0.3, 1.0, 9)
        self.assertGreater(
            nccc_pair_c2(0.1, 0.2, 0.3, 0.2, 0.3, 1.0, 1),
            0.0,
        )
        with self.assertRaisesRegex(ValueError, "positive expected margin"):
            nccc_pair_c2(0.1, 0.2, 0.3, 10.0, 1.0, 1.0, 1)
        with self.assertRaisesRegex(ValueError, "m >= 10"):
            luthra2025_official_optimized_bound(_two_way_pairwise(), 5)
        with self.assertRaisesRegex(ValueError, "dij2 must be finite and positive"):
            pairwise = _two_way_pairwise()
            pairwise[(0, 1)]["d2"] = 0.0
            directional_nccc_bound(
                pairwise,
                10,
                "thm41",
            )
        with self.assertRaisesRegex(ValueError, "every ordered class pair"):
            directional_nccc_bound({(0, 1): _two_way_pairwise()[(0, 1)]}, 10)

    def test_edge_lengths_and_B_zero_bound_are_literal(self):
        self.assertEqual(hyperrectangle_half_side_lengths([0.25]), [0.5])
        self.assertEqual(hyperrectangle_side_lengths([0.25]), [1.0])
        for B in (0.1, 0.25, 0.75, 1.0):
            tilde_v = directional_cdnv_from_B(B)
            self.assertAlmostEqual(
                nccc_error_bound(B, 3, 10, clamp=False),
                nccc_error_bound_from_tilde_v(
                    tilde_v,
                    3,
                    10,
                    clamp=False,
                ),
            )
        self.assertAlmostEqual(nccc_error_bound(0.0, 3, 2, clamp=False), 3.5)
        self.assertAlmostEqual(
            nccc_error_bound_from_tilde_v(float("inf"), 3, 2, clamp=False),
            3.5,
        )
        with self.assertRaisesRegex(ValueError, "tilde_v"):
            nccc_error_bound_from_tilde_v(float("nan"), 3, 2)
        with self.assertRaisesRegex(ValueError, "integer"):
            nccc_error_bound(0.5, 1.5, 2)

    def test_legacy_corrected_estimator_honors_center_labels(self):
        psi = torch.tensor([[1.0], [2.0], [3.0]])
        labels = torch.tensor([1.0, 1.0, -1.0])
        uncentered = estimate_B_r_corrected(
            psi,
            labels,
            [1],
            center_labels=False,
            ridge=0.0,
        )[1]
        centered = estimate_B_r_corrected(
            psi,
            labels,
            [1],
            center_labels=True,
            ridge=0.0,
        )[1]
        self.assertAlmostEqual(uncentered, 0.0)
        self.assertGreater(centered, 0.0)


class InstanceAwareNccTest(unittest.TestCase):
    def setUp(self):
        # Eight latent instances, two identical sibling views per instance.
        self.instance_ids = torch.arange(8).repeat(2)
        instance_labels = torch.tensor([17] * 4 + [45] * 4)
        self.labels = instance_labels.repeat(2)
        instance_features = torch.arange(8, dtype=torch.float32)[:, None]
        self.features = instance_features.repeat(2, 1)

    def test_sibling_views_never_cross_support_and_query(self):
        layout = _build_instance_sampling_layout(
            self.labels,
            self.instance_ids,
        )
        generator = torch.Generator(device="cpu").manual_seed(23)
        for _ in range(20):
            trial = _sample_grouped_binary_trial(
                layout,
                m=1,
                generator=generator,
                max_query=2,
            )
            support_ids = set(trial["support_instance_ids"].tolist())
            query_ids = set(trial["query_instance_ids"].tolist())
            self.assertTrue(support_ids.isdisjoint(query_ids))

            selected_rows = torch.cat(
                (
                    trial["support_positive_rows"],
                    trial["support_negative_rows"],
                    trial["query_positive_rows"],
                    trial["query_negative_rows"],
                )
            )
            # One row is chosen per selected instance even though each has two.
            selected_ids = self.instance_ids[selected_rows]
            self.assertEqual(
                torch.unique(selected_ids).numel(),
                selected_ids.numel(),
            )
            self.assertTrue(
                torch.equal(self.features[selected_rows, 0], selected_ids.float())
            )

    def test_public_empirical_ncc_and_metadata_are_instance_aware(self):
        error = empirical_nccc_error(
            self.features,
            self.labels,
            m=1,
            n_trials=5,
            seed=11,
            max_query=2,
            instance_ids=self.instance_ids,
        )
        pair = {
            (0, 1): {
                "Vtilde_ij": 0.1,
                "Vij": 0.2,
                "Theta_ij": 0.1,
                "vi": 0.0,
                "vj": 0.0,
                "d2": 1.0,
            },
            (1, 0): {
                "Vtilde_ij": 0.1,
                "Vij": 0.2,
                "Theta_ij": 0.1,
                "vi": 0.0,
                "vj": 0.0,
                "d2": 1.0,
            },
        }
        curves = combined_fewshot_curves(
            self.features,
            self.labels,
            B_by_r={1: 0.5},
            r_values=[1],
            m_values=[1],
            pairwise_by_r={1: pair},
            instance_ids=self.instance_ids,
            n_trials=3,
            max_query=2,
        )

        self.assertGreaterEqual(error, 0.0)
        self.assertLessEqual(error, 1.0)
        sampling = curves[1]["empirical_sampling"]
        self.assertEqual(sampling["n_rows"], 16)
        self.assertEqual(sampling["n_instances"], 8)
        self.assertEqual(sampling["class_instance_counts"], {"-1": 4, "+1": 4})
        self.assertEqual(sampling["min_rows_per_instance"], 2)
        self.assertEqual(sampling["max_rows_per_instance"], 2)
        self.assertEqual(
            sampling["view_selection"],
            "one_uniform_random_view_per_selected_instance",
        )
        self.assertEqual(sampling["seed"], 0)
        self.assertEqual(sampling["n_trials"], 3)
        self.assertEqual(sampling["max_query_instances_per_class"], 2)
        self.assertEqual(
            curves[1]["curves"][1]["empirical_group_counts"],
            {
                "support_instances_per_class": 1,
                "query_instances": {"-1": 2, "+1": 2},
            },
        )
        point = curves[1]["curves"][1]
        self.assertIsNone(point["thm41_dir"])
        self.assertFalse(point["validity"]["thm41_dir"]["valid"])
        self.assertIn("m >= 10", point["validity"]["thm41_dir"]["reason"])
        self.assertIsNotNone(point["validity"]["lim"]["value"])

    def test_conflicting_sibling_labels_are_rejected(self):
        labels = self.labels.clone()
        labels[8] = 45
        with self.assertRaisesRegex(ValueError, "exactly one class label"):
            empirical_nccc_error(
                self.features,
                labels,
                m=1,
                instance_ids=self.instance_ids,
            )

    def test_combined_theorem_curve_requires_instance_ids(self):
        with self.assertRaisesRegex(ValueError, "requires instance_ids"):
            combined_fewshot_curves(
                self.features,
                self.labels,
                B_by_r={1: 0.5},
                r_values=[1],
                m_values=[1],
                pairwise_by_r={1: {}},
            )

    def test_theorem_wrapper_rejects_imbalanced_instance_counts(self):
        keep = self.instance_ids != 3
        with self.assertRaisesRegex(ValueError, "equal class counts"):
            combined_fewshot_curves(
                self.features[keep],
                self.labels[keep],
                B_by_r={1: 0.5},
                r_values=[1],
                m_values=[1],
                pairwise_by_r={1: {}},
                instance_ids=self.instance_ids[keep],
            )

    def test_single_view_mode_must_be_explicit(self):
        single_features = self.features[:8]
        single_labels = self.labels[:8]
        with self.assertRaisesRegex(ValueError, "explicitly set"):
            empirical_nccc_error(single_features, single_labels, m=1)

        error = empirical_nccc_error(
            single_features,
            single_labels,
            m=1,
            n_trials=2,
            assume_single_view=True,
        )
        self.assertGreaterEqual(error, 0.0)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            empirical_nccc_error(
                single_features,
                single_labels,
                m=1,
                instance_ids=torch.arange(8),
                assume_single_view=True,
            )


if __name__ == "__main__":
    unittest.main()
