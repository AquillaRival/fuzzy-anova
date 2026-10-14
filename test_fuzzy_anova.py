"""
Automated Test Suite for Fuzzy ANOVA
====================================
Tests benchmark results, identical groups, widely separated groups,
unequal sample sizes, overlapping spreads, and comprehensive input validation.
"""

import unittest
from fuzzy_anova import (
    fuzzy_anova,
    get_alpha_cut,
    alpha_cut_anova,
    FuzzyANOVAResult,
    AlphaCutANOVAResult,
    QuadraticCoefficients,
)


class TestFuzzyANOVA(unittest.TestCase):

    def test_wu_benchmark_example(self):
        """Test with benchmark data matching the 3-treatment fuzzy ANOVA setup."""
        sample_data = {
            "Group_1": [(10, 12, 14), (11, 13, 15), (9, 11, 13)],
            "Group_2": [(20, 22, 24), (19, 21, 23), (21, 23, 25)],
            "Group_3": [(15, 17, 19), (14, 16, 18), (16, 18, 20)],
        }
        res = fuzzy_anova(sample_data, alpha_sig=0.05)

        self.assertIsInstance(res, FuzzyANOVAResult)
        self.assertEqual(res.decision, "H1")
        self.assertTrue(res.is_null_rejected)
        self.assertAlmostEqual(res.sstr, 150.0, places=4)
        self.assertAlmostEqual(res.sse, 6.0, places=4)
        self.assertEqual(res.df_between, 2)
        self.assertEqual(res.df_within, 6)
        self.assertAlmostEqual(res.f_crit, 5.14325, places=4)
        self.assertAlmostEqual(res.optimistic_degree, 1.0, places=4)
        self.assertAlmostEqual(res.pessimistic_degree, 0.0, places=4)
        self.assertAlmostEqual(res.confidence_degree, 1.0, places=4)
        self.assertLess(res.p_value, 0.001)

        # Test coefficient unpacking
        al, bl, cl = res.lower_coeffs
        self.assertAlmostEqual(al, res.AL)
        self.assertAlmostEqual(bl, res.BL)
        self.assertAlmostEqual(cl, res.CL)

    def test_list_of_lists_input_format(self):
        """Test passing a list of groups instead of a dict."""
        group_list = [
            [(10, 12, 14), (11, 13, 15), (9, 11, 13)],
            [(20, 22, 24), (19, 21, 23), (21, 23, 25)],
            [(15, 17, 19), (14, 16, 18), (16, 18, 20)],
        ]
        res = fuzzy_anova(group_list, alpha_sig=0.05)
        self.assertEqual(res.decision, "H1")
        self.assertAlmostEqual(res.sstr, 150.0, places=4)

    def test_identical_groups_accepts_null_hypothesis(self):
        """Identical groups must yield SSTR=0 and accept H0."""
        identical_data = {
            "Group_A": [(10, 12, 14), (11, 13, 15), (9, 11, 13)],
            "Group_B": [(10, 12, 14), (11, 13, 15), (9, 11, 13)],
        }
        res = fuzzy_anova(identical_data, alpha_sig=0.05)

        self.assertEqual(res.decision, "H0")
        self.assertFalse(res.is_null_rejected)
        self.assertAlmostEqual(res.sstr, 0.0, places=6)
        self.assertGreater(res.sse, 0.0)
        self.assertAlmostEqual(res.optimistic_degree, 1.0, places=4)
        self.assertAlmostEqual(res.pessimistic_degree, 0.0, places=4)
        self.assertAlmostEqual(res.confidence_degree, 1.0, places=4)
        self.assertAlmostEqual(res.p_value, 1.0, places=4)

    def test_widely_separated_groups_rejects_null_hypothesis(self):
        """Groups with huge difference in means must reject H0 with high confidence."""
        separated_data = {
            "Group_Low": [(1, 2, 3), (2, 3, 4), (1, 2, 3)],
            "Group_High": [(100, 102, 104), (101, 103, 105), (99, 101, 103)],
        }
        res = fuzzy_anova(separated_data, alpha_sig=0.01)

        self.assertEqual(res.decision, "H1")
        self.assertTrue(res.is_null_rejected)
        self.assertAlmostEqual(res.confidence_degree, 1.0, places=4)

    def test_unequal_sample_sizes(self):
        """ANOVA must support groups with unequal number of observations."""
        unbalanced_data = {
            "G1": [(10, 12, 14), (11, 13, 15)],
            "G2": [(20, 22, 24), (19, 21, 23), (21, 23, 25), (20, 22, 24)],
            "G3": [(15, 17, 19), (14, 16, 18), (16, 18, 20)],
        }
        res = fuzzy_anova(unbalanced_data, alpha_sig=0.05)

        self.assertEqual(res.df_between, 2)
        self.assertEqual(res.df_within, 9 - 3)
        self.assertIn(res.decision, ["H0", "H1"])
        self.assertTrue(0.0 <= res.confidence_degree <= 1.0)
        self.assertTrue(0.0 <= res.optimistic_degree <= 1.0)
        self.assertTrue(0.0 <= res.pessimistic_degree <= 1.0)

    def test_asymmetric_and_overlapping_spreads(self):
        """Test asymmetric fuzzy numbers where (Core - L) != (U - Core)."""
        asymmetric_data = {
            "Group_X": [(2, 8, 15), (3, 9, 14), (1, 7, 16)],
            "Group_Y": [(8, 10, 12), (7, 9, 13), (9, 11, 14)],
            "Group_Z": [(6, 12, 20), (5, 11, 19), (7, 13, 21)],
        }
        res = fuzzy_anova(asymmetric_data, alpha_sig=0.05)

        self.assertIsInstance(res.lower_coeffs, QuadraticCoefficients)
        self.assertIsInstance(res.upper_coeffs, QuadraticCoefficients)
        self.assertTrue(0.0 <= res.confidence_degree <= 1.0)

    def test_result_to_dict_and_summary(self):
        """Test dictionary serialization and summary formatting."""
        data = {
            "A": [(1, 2, 3), (2, 3, 4)],
            "B": [(5, 6, 7), (6, 7, 8)],
        }
        res = fuzzy_anova(data, alpha_sig=0.05)

        d = res.to_dict()
        self.assertIn("decision", d)
        self.assertIn("confidence_degree", d)
        self.assertIn("p_value", d)
        self.assertIn("lower_coefficients", d)
        self.assertIn("upper_coefficients", d)
        self.assertIn("A", d["lower_coefficients"])

        summary = res.summary()
        self.assertIn("ONE-WAY FUZZY ANOVA SUMMARY", summary)
        self.assertIn("Confidence Degree", summary)


class TestFuzzyANOVAInputValidation(unittest.TestCase):

    def test_empty_groups_raises_error(self):
        with self.assertRaises(ValueError):
            fuzzy_anova({})

    def test_invalid_type_groups_raises_type_error(self):
        with self.assertRaises(TypeError):
            fuzzy_anova(12345)  # type: ignore

    def test_single_group_raises_error(self):
        with self.assertRaises(ValueError):
            fuzzy_anova({"Group1": [(1, 2, 3), (2, 3, 4)]})

    def test_group_with_zero_observations_raises_error(self):
        with self.assertRaises(ValueError):
            fuzzy_anova({
                "Group1": [(1, 2, 3), (2, 3, 4)],
                "Group2": []
            })

    def test_total_observations_less_than_groups_raises_error(self):
        # 2 groups with 1 observation each -> total N = 2, k = 2 -> df_within = 0
        with self.assertRaises(ValueError):
            fuzzy_anova({
                "Group1": [(1, 2, 3)],
                "Group2": [(4, 5, 6)]
            })

    def test_invalid_alpha_raises_error(self):
        valid_data = {
            "G1": [(1, 2, 3), (2, 3, 4)],
            "G2": [(5, 6, 7), (6, 7, 8)]
        }
        with self.assertRaises(ValueError):
            fuzzy_anova(valid_data, alpha_sig=0.0)
        with self.assertRaises(ValueError):
            fuzzy_anova(valid_data, alpha_sig=1.0)
        with self.assertRaises(ValueError):
            fuzzy_anova(valid_data, alpha_sig=-0.05)
        with self.assertRaises(ValueError):
            fuzzy_anova(valid_data, alpha_sig=1.5)

    def test_non_numeric_alpha_raises_type_error(self):
        valid_data = {
            "G1": [(1, 2, 3), (2, 3, 4)],
            "G2": [(5, 6, 7), (6, 7, 8)]
        }
        with self.assertRaises(TypeError):
            fuzzy_anova(valid_data, alpha_sig="0.05")  # type: ignore

    def test_invalid_observation_shape_raises_error(self):
        invalid_shape_data = {
            "G1": [(1, 2), (2, 3, 4)],  # only 2 values
            "G2": [(5, 6, 7), (6, 7, 8)]
        }
        with self.assertRaises(ValueError):
            fuzzy_anova(invalid_shape_data)

    def test_violating_fuzzy_condition_raises_error(self):
        # L > Core
        invalid_fuzzy_data = {
            "G1": [(5, 3, 7), (2, 3, 4)],
            "G2": [(5, 6, 7), (6, 7, 8)]
        }
        with self.assertRaises(ValueError):
            fuzzy_anova(invalid_fuzzy_data)

        # Core > U
        invalid_fuzzy_data2 = {
            "G1": [(1, 8, 7), (2, 3, 4)],
            "G2": [(5, 6, 7), (6, 7, 8)]
        }
        with self.assertRaises(ValueError):
            fuzzy_anova(invalid_fuzzy_data2)

    def test_nan_or_inf_in_observations_raises_error(self):
        nan_data = {
            "G1": [(1, float("nan"), 4), (2, 3, 4)],
            "G2": [(5, 6, 7), (6, 7, 8)]
        }
        with self.assertRaises(ValueError):
            fuzzy_anova(nan_data)

        inf_data = {
            "G1": [(1, 2, float("inf")), (2, 3, 4)],
            "G2": [(5, 6, 7), (6, 7, 8)]
        }
        with self.assertRaises(ValueError):
            fuzzy_anova(inf_data)

    def test_invalid_alpha_cut_level_raises_error(self):
        valid_data = {
            "G1": [(1, 2, 3), (2, 3, 4)],
            "G2": [(5, 6, 7), (6, 7, 8)]
        }
        with self.assertRaises(ValueError):
            get_alpha_cut(valid_data, h=-0.1)
        with self.assertRaises(ValueError):
            get_alpha_cut(valid_data, h=1.1)
        with self.assertRaises(ValueError):
            get_alpha_cut(valid_data, h=float("nan"))
        with self.assertRaises(ValueError):
            alpha_cut_anova(valid_data, h=-0.5)
        with self.assertRaises(ValueError):
            alpha_cut_anova(valid_data, h=1.5)


class TestWu2007PaperBenchmark(unittest.TestCase):
    """
    Direct validation against the exact numerical values published in:
    Hsien-Chung Wu (2007), Section 5 "A numerical example" (pages 16-17).
    """

    def setUp(self):
        self.wu_data = {
            "Design_1": [(9, 11, 13), (14, 16, 18)],
            "Design_2": [(11, 15, 19), (10, 15, 21), (11, 13, 15)],
            "Design_3": [(15, 18, 21), (14, 17, 20), (17, 20, 23)],
            "Design_4": [(15, 19, 23), (21, 24, 27)],
        }

    def test_wu_paper_with_textbook_f_crit(self):
        """Matches Wu (2007) Page 17 textbook results to 4 decimal places with F_crit=4.76."""
        res = fuzzy_anova(self.wu_data, alpha_sig=0.05, f_crit=4.76)

        # Core sums of squares & degrees of freedom
        self.assertEqual(res.decision, "H1")
        self.assertTrue(res.is_null_rejected)
        self.assertAlmostEqual(res.sstr, 91.2667, places=3)
        self.assertAlmostEqual(res.sse, 32.3333, places=3)
        self.assertEqual(res.df_between, 3)
        self.assertEqual(res.df_within, 6)
        self.assertAlmostEqual(res.rho, 2.38, places=2)

        # Quadratic boundary coefficients (AL, BL, CL)
        self.assertAlmostEqual(res.AL, -8.5633, places=3)
        self.assertAlmostEqual(res.BL, 25.8933, places=3)
        self.assertAlmostEqual(res.CL, -3.0167, places=3)

        # Quadratic boundary coefficients (AU, BU, CU)
        self.assertAlmostEqual(res.AU, -15.1300, places=3)
        self.assertAlmostEqual(res.BU, 29.6000, places=3)
        self.assertAlmostEqual(res.CU, -0.1567, places=3)

        # Roots of quadratic equations
        roots_l = res.lower_coeffs.roots()
        roots_u = res.upper_coeffs.roots()
        self.assertAlmostEqual(roots_l[0], 0.1214, places=3)
        self.assertAlmostEqual(roots_l[1], 2.9024, places=3)
        self.assertAlmostEqual(roots_u[0], 0.0053, places=3)
        self.assertAlmostEqual(roots_u[1], 1.9511, places=3)

        # Optimistic, Pessimistic, and Confidence degrees
        self.assertAlmostEqual(res.optimistic_degree, 0.8786, places=3)
        self.assertAlmostEqual(res.pessimistic_degree, 0.0053, places=3)
        self.assertAlmostEqual(res.confidence_degree, 0.8786, places=3)

        # Summary output wording
        summary_text = res.summary()
        self.assertIn("accept the alternative hypothesis H1", summary_text)
        self.assertIn("optimistic degree 0.8786", summary_text)
        self.assertIn("pessimistic degree 0.0053", summary_text)
        self.assertIn("ANOVA Table: Lower Bound Case", summary_text)
        self.assertIn("ANOVA Table: Upper Bound Case", summary_text)

    def test_wu_paper_with_automatic_f_crit(self):
        """Validates result with exact theoretical F_crit (4.7571)."""
        res = fuzzy_anova(self.wu_data, alpha_sig=0.05)
        self.assertEqual(res.decision, "H1")
        self.assertAlmostEqual(res.sstr, 91.2667, places=3)
        self.assertAlmostEqual(res.sse, 32.3333, places=3)
        self.assertAlmostEqual(res.f_stat, 5.6454, places=3)
        self.assertAlmostEqual(res.optimistic_degree, 0.8808, places=3)
        self.assertAlmostEqual(res.pessimistic_degree, 0.0031, places=3)

    def test_alpha_cut_evaluation(self):
        """Tests explicit alpha cuts at h=0.0, 0.5, 1.0 (Eq. 11)."""
        # For Observation 1 of Design 1: (9, 11, 13)
        cut_h0 = get_alpha_cut(self.wu_data, h=0.0)
        self.assertEqual(cut_h0["Design_1"][0], (9.0, 13.0))

        cut_h5 = get_alpha_cut(self.wu_data, h=0.5)
        self.assertEqual(cut_h5["Design_1"][0], (10.0, 12.0))

        cut_h1 = get_alpha_cut(self.wu_data, h=1.0)
        self.assertEqual(cut_h1["Design_1"][0], (11.0, 11.0))

    def test_alpha_cut_anova_tables(self):
        """Tests Table 2 and Table 3 calculations at h=0 and h=1."""
        res_tables_h0 = alpha_cut_anova(self.wu_data, h=0.0, alpha_sig=0.05)
        self.assertIsInstance(res_tables_h0, AlphaCutANOVAResult)
        # Lower bound at h=0: SSTR=82.2667, SSE=35.8333, SSTO=118.1
        self.assertAlmostEqual(res_tables_h0.lower.sstr, 82.2667, places=3)
        self.assertAlmostEqual(res_tables_h0.lower.sse, 35.8333, places=3)
        self.assertAlmostEqual(res_tables_h0.lower.ssto, 118.1000, places=3)
        # Upper bound at h=0: SSTR=104.1667, SSE=43.8333, SSTO=148.0
        self.assertAlmostEqual(res_tables_h0.upper.sstr, 104.1667, places=3)
        self.assertAlmostEqual(res_tables_h0.upper.sse, 43.8333, places=3)
        self.assertAlmostEqual(res_tables_h0.upper.ssto, 148.0000, places=3)

        # At h=1.0, lower and upper bounds must converge to core ANOVA
        res_tables_h1 = alpha_cut_anova(self.wu_data, h=1.0, alpha_sig=0.05)
        self.assertAlmostEqual(res_tables_h1.lower.sstr, 91.2667, places=3)
        self.assertAlmostEqual(res_tables_h1.upper.sstr, 91.2667, places=3)
        self.assertAlmostEqual(res_tables_h1.lower.sse, 32.3333, places=3)
        self.assertAlmostEqual(res_tables_h1.upper.sse, 32.3333, places=3)

    def test_degenerate_crisp_numbers(self):
        """Fuzzy ANOVA with crisp observations (L = Core = U) collapses to classical ANOVA."""
        crisp_data = {
            g: [(c, c, c) for _, c, _ in obs_list]
            for g, obs_list in self.wu_data.items()
        }
        res = fuzzy_anova(crisp_data, alpha_sig=0.05)
        self.assertEqual(res.decision, "H1")
        self.assertAlmostEqual(res.f_stat, 5.6454, places=3)
        # For crisp numbers, AL=0, BL=0, AU=0, BU=0, and bounds are flat
        self.assertAlmostEqual(res.AL, 0.0, places=4)
        self.assertAlmostEqual(res.AU, 0.0, places=4)


if __name__ == "__main__":
    unittest.main()

