import unittest

import pandas as pd

from project.modules.Module4 import split_train_test_stratified


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


if __name__ == "__main__":
    unittest.main()
