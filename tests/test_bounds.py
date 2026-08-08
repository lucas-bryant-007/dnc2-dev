import unittest

import torch

from analysis.bounds import (
    _build_instance_sampling_layout,
    _sample_grouped_binary_trial,
    combined_fewshot_curves,
    empirical_nccc_error,
)


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
            }
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
