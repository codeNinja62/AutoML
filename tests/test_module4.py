import unittest

import pandas as pd

from modules.Module4 import apply_outlier_handling, build_preprocessor, split_train_test_stratified


class TestModule4SplitTrainTestStratified(unittest.TestCase):
    def test_raises_when_target_missing(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        with self.assertRaises(ValueError):
            split_train_test_stratified(df, target_column="y")

    def test_raises_when_no_feature_columns(self):
        df = pd.DataFrame({"y": [0, 1, 0, 1]})
        with self.assertRaises(ValueError):
            split_train_test_stratified(df, target_column="y")

    def test_raises_when_class_has_single_sample(self):
        df = pd.DataFrame({"x": [1, 2, 3], "y": [0, 0, 1]})
        with self.assertRaises(ValueError):
            split_train_test_stratified(df, target_column="y", test_size=0.33, random_state=0)

    def test_raises_when_test_split_too_small_for_classes(self):
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5, 6], "y": ["a", "a", "b", "b", "c", "c"]})
        # With 3 classes and 6 rows, test_size=0.2 => ceil(1.2)=2 test samples, which can't contain all 3 classes.
        with self.assertRaises(ValueError):
            split_train_test_stratified(df, target_column="y", test_size=0.2, random_state=0)


class TestModule4Preprocessor(unittest.TestCase):
    def test_build_preprocessor_fit_transform_onehot(self):
        X = pd.DataFrame({
            "num": [1.0, 2.0, None, 4.0],
            "cat": ["a", "b", "a", "c"],
        })
        pre = build_preprocessor(
            X,
            numeric_impute='median',
            categorical_impute='most_frequent',
            scaling='standard',
            encoding='onehot',
        )
        Xt = pre.fit_transform(X)
        # 1 numeric + 3 one-hot categories
        self.assertEqual(Xt.shape, (4, 4))

    def test_build_preprocessor_fit_transform_ordinal(self):
        X = pd.DataFrame({
            "num": [1.0, 2.0, None, 4.0],
            "cat": ["a", "b", "a", "c"],
        })
        pre = build_preprocessor(
            X,
            numeric_impute='median',
            categorical_impute='most_frequent',
            scaling='none',
            encoding='ordinal',
        )
        Xt = pre.fit_transform(X)
        # 1 numeric + 1 ordinal categorical
        self.assertEqual(Xt.shape, (4, 2))


class TestModule4OutlierHandling(unittest.TestCase):
    def test_cap_iqr_caps_extremes(self):
        df = pd.DataFrame({
            "x": [0.0, 0.0, 0.0, 0.0, 100.0],
            "y": ["a", "a", "a", "a", "a"],
        })
        out, summary = apply_outlier_handling(df, action='cap_iqr', method='iqr')
        self.assertEqual(int(summary['values_capped']), 1)
        self.assertTrue((out["x"] == 0.0).all())

    def test_remove_rows_iqr_removes_outlier_row(self):
        df = pd.DataFrame({
            "x": [0.0, 0.0, 0.0, 0.0, 100.0],
            "target": [0, 0, 0, 0, 1],
        })
        out, summary = apply_outlier_handling(df, action='remove_rows', method='iqr', exclude_columns=['target'])
        self.assertEqual(int(summary['rows_removed']), 1)
        self.assertEqual(out.shape[0], 4)

    def test_remove_rows_zscore_removes_outlier_row_with_low_threshold(self):
        df = pd.DataFrame({
            "x": [0.0, 0.0, 0.0, 0.0, 100.0],
            "target": [0, 0, 0, 0, 1],
        })
        out, summary = apply_outlier_handling(
            df,
            action='remove_rows',
            method='zscore',
            exclude_columns=['target'],
            zscore_threshold=1.0,
        )
        self.assertEqual(int(summary['rows_removed']), 1)
        self.assertEqual(out.shape[0], 4)


if __name__ == "__main__":
    unittest.main()
