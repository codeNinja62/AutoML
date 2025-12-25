import unittest

import pandas as pd

from modules.Module1 import (
    get_cardinality_buckets,
    get_all_missing_columns,
    get_constant_columns,
    get_dataset_profile,
    get_dataset_schema,
    get_duplicate_row_count,
    get_missing_cell_count,
    get_shape,
    get_unique_counts,
    infer_target_candidates,
    validate_target_column,
)


class TestModule1(unittest.TestCase):
    def test_basic_counts(self):
        df = pd.DataFrame(
            {
                "a": [1, 2, 2, None],
                "b": ["x", "y", "y", "y"],
            }
        )
        self.assertEqual(get_shape(df), (4, 2))
        self.assertEqual(get_missing_cell_count(df), 1)

        # Duplicate rows: row 1 and row 2 are identical
        df2 = pd.DataFrame({"a": [1, 2, 2], "b": ["x", "y", "y"]})
        self.assertEqual(get_duplicate_row_count(df2), 1)

    def test_unique_and_cardinality(self):
        df = pd.DataFrame({"id": [1, 2, 3, 4], "cat": ["a", "a", "b", "b"], "num": [10, 11, 12, 13]})
        uniq = get_unique_counts(df)
        self.assertEqual(int(uniq["id"]), 4)
        self.assertEqual(int(uniq["cat"]), 2)

        buckets = get_cardinality_buckets(df, low_max=2, medium_max=3, high_max=4)
        self.assertIn("bucket", buckets.columns)

    def test_schema(self):
        df = pd.DataFrame({"a": [1, None, 3], "b": ["x", "y", None]})
        schema = get_dataset_schema(df, sample_values=2)
        self.assertEqual(set(schema.columns), {"column", "dtype", "non_null", "missing_pct", "unique", "samples"})
        self.assertEqual(len(schema), 2)

    def test_target_validation(self):
        df = pd.DataFrame({"x": [1, 2, 3, 4], "y": [0, 0, 1, 1]})
        tv = validate_target_column(df, "y")
        self.assertTrue(tv.ok)
        self.assertEqual(tv.n_classes, 2)

        df_bad = pd.DataFrame({"x": [1, 2, 3], "y": [1, 1, 1]})
        tv2 = validate_target_column(df_bad, "y")
        self.assertFalse(tv2.ok)
        self.assertTrue(any("at least 2" in e.lower() for e in tv2.errors))

    def test_target_candidate_inference(self):
        df = pd.DataFrame({"feature": [1, 2, 3, 4], "label": ["a", "b", "a", "b"]})
        cands = infer_target_candidates(df)
        self.assertIn("label", cands)

    def test_profile(self):
        df = pd.DataFrame({"x": [1, 2, 3, 4], "target": ["a", "b", "a", "b"]})
        prof = get_dataset_profile(df, target_col="target")
        self.assertIn("rows", prof)
        self.assertIn("target", prof)
        self.assertTrue(prof["target"]["ok"])

    def test_constant_columns(self):
        df = pd.DataFrame({"a": [1, 1, 1], "b": [1, 2, 3], "c": [None, None, None]})
        const = get_constant_columns(df)
        self.assertIn("a", const)
        self.assertIn("c", const)
        self.assertNotIn("b", const)

        const2 = get_constant_columns(df, exclude=["a"])
        self.assertNotIn("a", const2)

    def test_all_missing_columns(self):
        df = pd.DataFrame({"a": [None, None], "b": [1, None], "c": ["x", "y"]})
        all_missing = get_all_missing_columns(df)
        self.assertIn("a", all_missing)
        self.assertNotIn("b", all_missing)
        self.assertNotIn("c", all_missing)

        all_missing2 = get_all_missing_columns(df, exclude=["a"])
        self.assertNotIn("a", all_missing2)


if __name__ == "__main__":
    unittest.main()
