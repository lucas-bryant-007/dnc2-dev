import unittest

import torch

from analysis.br.ssl_subspace import fit_ssl_subspace
from analysis.br.whitening import (
    ABSOLUTE_WHITENING_ELIGIBILITY_POLICY,
    absolute_whitening_eligibility,
    canonicalize_balanced_binary_labels,
    fit_exact_whitener,
    require_balanced_binary_labels,
    split_balanced_paired_fit_eval,
    whitening_diagnostics,
)
from analysis.br.br_estimators import estimate_B_r_raw
from analysis.br.geometric_estimators import estimate_tilde_V
from analysis.hyperrect import apply_rewhitener, fit_rewhitener, rewhiten
from analysis.metrics_io import build_csv_rows


def _empirical_covariance(features):
    centered = features - features.mean(dim=0, keepdim=True)
    return (centered.T @ centered) / centered.shape[0]


class ExactWhiteningTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)

    def assert_identity_covariance(self, features, atol=2e-5):
        covariance = _empirical_covariance(features)
        identity = torch.eye(
            covariance.shape[0],
            dtype=covariance.dtype,
            device=covariance.device,
        )
        self.assertTrue(torch.allclose(covariance, identity, atol=atol, rtol=0.0))

    def test_exact_whitener_produces_identity_covariance(self):
        features = torch.randn(600, 4) @ torch.diag(
            torch.tensor([4.0, 2.0, 0.5, 0.1])
        )
        fit = fit_exact_whitener(features, rel_eig_threshold=1e-5)
        whitened = (features - fit.mean) @ fit.whitener

        self.assertEqual(fit.output_dim, 4)
        self.assert_identity_covariance(whitened)

    def test_rank_deficient_population_is_truncated_then_exactly_whitened(self):
        latent = torch.randn(500, 2)
        features = torch.column_stack(
            (
                latent[:, 0],
                latent[:, 1],
                latent[:, 0] + 2.0 * latent[:, 1],
                torch.zeros(latent.shape[0]),
            )
        )
        transform = fit_rewhitener(features, rel_eig_threshold=1e-5)
        whitened = apply_rewhitener(features, transform)

        self.assertEqual(transform.input_dim, 4)
        self.assertEqual(transform.output_dim, 2)
        self.assertTrue(torch.isfinite(whitened).all())
        self.assert_identity_covariance(whitened)

    def test_high_dimensional_rank_deficiency_respects_numerical_floor(self):
        latent = torch.randn(800, 5)
        features = latent @ torch.randn(5, 30)
        fit = fit_exact_whitener(features, rel_eig_threshold=1e-8)
        whitened = (features - fit.mean) @ fit.whitener

        self.assertEqual(fit.output_dim, 5)
        self.assertGreater(fit.effective_rel_eig_threshold, 1e-8)
        self.assert_identity_covariance(whitened)

    def test_float64_is_preserved_through_ssl_fit_and_transform(self):
        z1 = torch.randn(200, 4, dtype=torch.float64)
        z2 = z1 + 0.05 * torch.randn_like(z1)
        estimator = fit_ssl_subspace(z1, z2, rel_eig_threshold=1e-8)
        transformed = estimator.transform(z1)

        self.assertEqual(estimator.mean_.dtype, torch.float64)
        self.assertEqual(estimator.whiten_.dtype, torch.float64)
        self.assertEqual(transformed.dtype, torch.float64)
        self.assertGreaterEqual(estimator.effective_rel_eig_threshold, 1e-8)
        self.assertAlmostEqual(
            estimator.covariance_cutoff,
            estimator.effective_rel_eig_threshold * estimator.lam_max,
        )

    def test_convenience_rewhitening_is_exact_on_its_fit_population(self):
        features = torch.randn(300, 5)
        whitened, transform = rewhiten(
            features,
            rel_eig_threshold=1e-4,
            return_transform=True,
        )

        self.assertLessEqual(whitened.shape[1], features.shape[1])
        self.assert_identity_covariance(whitened)
        metadata = transform.metadata()
        self.assertEqual(metadata["input_dim"], features.shape[1])
        self.assertEqual(metadata["output_dim"], whitened.shape[1])
        self.assertEqual(metadata["requested_rel_eig_threshold"], 1e-4)
        self.assertIn("effective_rel_eig_threshold", metadata)
        self.assertIn("lambda_min_retained", metadata)

    def test_diagnostics_distinguish_fit_and_shifted_populations(self):
        fit_population = torch.randn(500, 4)
        transform = fit_rewhitener(fit_population, rel_eig_threshold=1e-5)
        fit_coordinates = apply_rewhitener(fit_population, transform)
        shifted_population = 3.0 * torch.randn(500, 4) + 4.0
        shifted_coordinates = apply_rewhitener(shifted_population, transform)

        fit_diagnostic = whitening_diagnostics(fit_coordinates)
        shifted_diagnostic = whitening_diagnostics(shifted_coordinates)
        self.assertTrue(
            fit_diagnostic.satisfies_absolute_cutoffs(1e-4, 1e-4)
        )
        self.assertFalse(
            shifted_diagnostic.satisfies_absolute_cutoffs(0.05, 0.10)
        )

    def test_spectral_gate_rejects_small_entrywise_large_operator_error(self):
        dimension = 20
        target_second_moment = (
            0.91 * torch.eye(dimension, dtype=torch.float64)
            + 0.09 * torch.ones((dimension, dimension), dtype=torch.float64)
        )
        factor = torch.linalg.cholesky(target_second_moment)
        half = (dimension ** 0.5) * factor.T
        features = torch.cat((half, -half), dim=0)
        diagnostic = whitening_diagnostics(
            features,
            n_independent_samples=dimension,
        )

        self.assertAlmostEqual(
            diagnostic.second_moment_max_abs_error,
            0.09,
            places=12,
        )
        self.assertAlmostEqual(
            diagnostic.second_moment_operator_error,
            1.71,
            places=12,
        )
        self.assertAlmostEqual(
            diagnostic.second_moment_min_eigenvalue,
            0.91,
            places=12,
        )
        self.assertAlmostEqual(
            diagnostic.second_moment_max_eigenvalue,
            2.71,
            places=12,
        )
        self.assertAlmostEqual(diagnostic.mean_l2_norm, 0.0, places=12)
        self.assertEqual(diagnostic.n_samples, 40)
        self.assertEqual(diagnostic.n_independent_samples, 20)
        self.assertEqual(diagnostic.rows_per_independent_sample, 2.0)
        self.assertFalse(
            diagnostic.satisfies_absolute_cutoffs(
                max_mean_l2_error=0.05,
                max_operator_error=0.10,
            )
        )
        decision = absolute_whitening_eligibility(diagnostic, 0.05, 0.10)
        self.assertFalse(decision["eligible"])
        self.assertIn(
            "second_moment_operator_error_exceeds_absolute_cutoff",
            decision["ineligibility_reasons"],
        )

    def test_absolute_policy_does_not_mislabel_sampling_fluctuation(self):
        torch.manual_seed(23)
        features = torch.randn(4000, 128)
        diagnostic = whitening_diagnostics(
            features,
            n_independent_samples=2000,
        )
        decision = absolute_whitening_eligibility(diagnostic, 0.05, 0.10)

        self.assertFalse(decision["eligible"])
        self.assertEqual(
            decision["policy_meaning"],
            "strict_absolute_empirical_geometry_not_population_compatibility_test",
        )
        self.assertEqual(
            decision["sampling_normalization_role"],
            "diagnostic_only_not_decision_rule",
        )
        ratio = 128 / 2000
        self.assertAlmostEqual(
            diagnostic.mean_sampling_reference_scale,
            ratio ** 0.5,
        )
        self.assertAlmostEqual(
            diagnostic.operator_sampling_reference_scale,
            2.0 * ratio ** 0.5 + ratio,
        )
        self.assertLess(diagnostic.mean_l2_sampling_normalized_error, 1.0)
        self.assertLess(
            diagnostic.second_moment_operator_sampling_normalized_error,
            1.0,
        )

    def test_ineligible_rank_is_retained_in_summary_rows(self):
        diagnostic = whitening_diagnostics(
            torch.randn(4000, 128),
            n_independent_samples=2000,
        )
        eligibility = absolute_whitening_eligibility(diagnostic, 0.05, 0.10)
        whitening_row = {**diagnostic.as_dict(), **eligibility}
        rows = build_csv_rows(
            method="test",
            attribute="binary",
            tag="",
            epoch=1,
            per_cap_results={
                "adaptive": {
                    "k_eff": 128,
                    "requested_r_values": [128],
                    "r_values": [],
                    "B_r": {},
                    "whitening_diagnostics": {128: whitening_row},
                }
            },
            orig_cdnv=None,
            orig_directional_cdnv=None,
            b_metric="orth",
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["r"], 128)
        self.assertFalse(rows[0]["theorem_eligible"])
        self.assertIsNone(rows[0]["B_r"])
        self.assertEqual(
            rows[0]["whitening_eligibility_policy"],
            ABSOLUTE_WHITENING_ELIGIBILITY_POLICY,
        )

    def test_paired_population_split_balances_and_keeps_views_together(self):
        instance_ids = torch.arange(12, dtype=torch.float32)
        z1 = instance_ids[:, None]
        z2 = z1 + 100.0
        labels = torch.tensor([17] * 8 + [45] * 4)
        split = split_balanced_paired_fit_eval(
            z1,
            z2,
            labels,
            fit_fraction=0.5,
            seed=3,
            instance_ids=instance_ids.long(),
        )

        self.assertTrue(torch.equal(split.fit_view2, split.fit_view1 + 100.0))
        self.assertTrue(torch.equal(split.eval_view2, split.eval_view1 + 100.0))
        self.assertTrue(
            set(split.fit_instance_ids.tolist()).isdisjoint(
                set(split.eval_instance_ids.tolist())
            )
        )
        self.assertEqual(
            torch.bincount((split.fit_labels > 0).long()).tolist(),
            [2, 2],
        )
        self.assertEqual(
            torch.bincount((split.evaluation_labels > 0).long()).tolist(),
            [4, 4],
        )
        self.assertEqual(split.original_class_counts, (8, 4))
        self.assertEqual(split.label_mapping, ((17, -1), (45, 1)))
        self.assertEqual(
            split.metadata()["instance_id_source"],
            "caller_supplied",
        )
        self.assertEqual(split.balanced_instances_per_class, 4)
        self.assertTrue(
            torch.equal(
                split.evaluation_instance_ids[: split.eval_instance_ids.numel()],
                split.eval_instance_ids,
            )
        )

    def test_balanced_label_contract_rejects_centered_but_imbalanced_labels(self):
        labels = torch.tensor([-1, -1, -1, 1])
        centered = labels.float() - labels.float().mean()
        with self.assertRaisesRegex(ValueError, "equally represented"):
            require_balanced_binary_labels(centered, "test population")

    def test_arbitrary_binary_values_are_canonicalized_before_capture(self):
        arbitrary = torch.tensor([17, 17, 45, 45])
        expected = torch.tensor([-1.0, -1.0, 1.0, 1.0])
        canonical, diagnostics = canonicalize_balanced_binary_labels(
            arbitrary,
            "test population",
        )
        features = torch.tensor([[-1.0], [-0.5], [0.5], [1.0]])

        capture = estimate_B_r_raw(
            features,
            canonical,
            [1],
            center_labels=True,
        )[1]
        expected_capture = estimate_B_r_raw(
            features,
            expected,
            [1],
            center_labels=True,
        )[1]
        self.assertTrue(torch.equal(canonical, expected))
        self.assertEqual(
            diagnostics["canonical_mapping"],
            [
                {"observed": 17, "canonical": -1},
                {"observed": 45, "canonical": 1},
            ],
        )
        self.assertEqual(capture, expected_capture)

    def test_ssl_coordinates_are_white_on_the_fitted_view_marginal(self):
        z1 = torch.randn(400, 5) @ torch.diag(
            torch.tensor([3.0, 2.0, 1.0, 0.5, 0.2])
        )
        z2 = z1 + 0.1 * torch.randn_like(z1)
        estimator = fit_ssl_subspace(z1, z2, rel_eig_threshold=1e-5)
        psi = estimator.transform(torch.cat((z1, z2), dim=0))

        self.assertEqual(psi.shape[1], estimator.k_eff)
        self.assert_identity_covariance(psi)

    def test_first_stage_whitener_provenance_identifies_theorem_features(self):
        z1 = torch.randn(120, 6) @ torch.diag(
            torch.tensor([4.0, 2.0, 1.0, 0.5, 0.2, 0.05])
        )
        z2 = z1 + 0.05 * torch.randn_like(z1)
        estimator = fit_ssl_subspace(
            z1,
            z2,
            k_cap=4,
            rel_eig_threshold=1e-4,
        )
        provenance = estimator.first_stage_whitener_provenance(
            fit_split="train",
            fit_population="paired_training_instances",
            view_marginal="equal_weight_two_augmented_views",
            frozen_for_test=True,
        )

        self.assertEqual(
            provenance["stage"],
            "first_stage_ssl_marginal_whitener",
        )
        self.assertEqual(provenance["n_fit_latent_instances"], 120)
        self.assertEqual(provenance["n_fit_view_rows"], 240)
        self.assertEqual(provenance["input_dimension"], 6)
        self.assertEqual(provenance["requested_rank_cap"], 4)
        self.assertEqual(provenance["retained_rank"], estimator.k_eff)
        self.assertEqual(
            provenance["relative_rank_cutoff"]["effective"],
            estimator.effective_rel_eig_threshold,
        )
        self.assertEqual(
            provenance["retained_covariance_eigenvalue_range"]["max"],
            estimator.lam_max,
        )
        self.assertTrue(provenance["transform_frozen_after_fit"])
        self.assertTrue(provenance["frozen_for_test"])

    def test_exact_whitening_restores_directional_cdnv_identity(self):
        labels = torch.cat((torch.ones(250), -torch.ones(250)))
        raw = torch.randn(500, 4)
        raw[:, 0] += 0.8 * labels
        features = rewhiten(raw, rel_eig_threshold=1e-5)
        capture = ((labels[:, None] * features).mean(dim=0) ** 2).sum()

        observed = estimate_tilde_V(
            features,
            labels,
            r=features.shape[1],
        )
        predicted = float(((1.0 - capture) / (2.0 * capture)).item())
        self.assertAlmostEqual(observed, predicted, places=5)

    def test_zero_variance_population_is_rejected(self):
        features = torch.zeros(20, 3)
        with self.assertRaisesRegex(ValueError, "zero-variance"):
            fit_exact_whitener(features)
        with self.assertRaisesRegex(ValueError, "zero-variance"):
            fit_ssl_subspace(features, features)


if __name__ == "__main__":
    unittest.main()
