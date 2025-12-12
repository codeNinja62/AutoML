import unittest

import numpy as np
import pandas as pd

from project.modules.Module4 import build_preprocessor
from project.modules.Module5 import evaluate_models, train_and_optimize_models


class TestModule5Training(unittest.TestCase):
    def test_reduces_cv_when_min_class_small(self):
        # min class count = 2, so requested cv=5 should reduce to 2
        X = pd.DataFrame({"x": [0, 1, 2, 3]})
        y = pd.Series([0, 0, 1, 1])
        trained = train_and_optimize_models(
            X,
            y,
            search_type="grid",
            cv=5,
            scoring="accuracy",
            include_models=["Rule-based (Most Frequent)"],
            preprocessor=None,
            n_jobs=1,
            cache=False,
        )
        self.assertIn("Rule-based (Most Frequent)", trained)
        self.assertEqual(int(trained["Rule-based (Most Frequent)"]["cv_folds"]), 2)
        self.assertIsNotNone(trained["Rule-based (Most Frequent)"]["model"])

    def test_training_with_preprocessor_pipeline(self):
        X = pd.DataFrame({"num": [1.0, 2.0, 3.0, 4.0], "cat": ["a", "b", "a", "b"]})
        y = pd.Series([0, 0, 1, 1])
        pre = build_preprocessor(X, scaling="standard", encoding="onehot")

        trained = train_and_optimize_models(
            X,
            y,
            search_type="grid",
            cv=2,
            scoring="accuracy",
            include_models=["Logistic Regression"],
            preprocessor=pre,
            n_jobs=1,
            cache=False,
        )
        model = trained["Logistic Regression"]["model"]
        self.assertIsNotNone(model)
        # When a preprocessor is provided, we expect a sklearn Pipeline
        self.assertTrue(hasattr(model, "predict"))


class TestModule5Evaluation(unittest.TestCase):
    def test_evaluate_models_binary_has_confusion_matrix(self):
        X_train = pd.DataFrame({"x": [0, 1, 2, 3, 4, 5]})
        y_train = pd.Series([0, 0, 0, 1, 1, 1])
        X_test = pd.DataFrame({"x": [10, 11, 12, 13]})
        y_test = pd.Series([0, 0, 1, 1])

        trained = train_and_optimize_models(
            X_train,
            y_train,
            search_type="grid",
            cv=2,
            scoring="accuracy",
            include_models=["Logistic Regression"],
            preprocessor=None,
            n_jobs=1,
            cache=False,
        )
        evals = evaluate_models(trained, X_test, y_test)
        self.assertIn("Logistic Regression", evals)
        cm = evals["Logistic Regression"]["confusion_matrix"]
        self.assertEqual(np.asarray(cm).shape, (2, 2))

    def test_evaluate_models_multiclass_roc_auc(self):
        # Create a simple multiclass dataset with 3 classes.
        X_train = pd.DataFrame({"x": [0, 1, 2, 3, 4, 5, 6, 7, 8]})
        y_train = pd.Series(["a", "a", "a", "b", "b", "b", "c", "c", "c"])
        X_test = pd.DataFrame({"x": [9, 10, 11, 12, 13, 14]})
        y_test = pd.Series(["a", "a", "b", "b", "c", "c"])

        trained = train_and_optimize_models(
            X_train,
            y_train,
            search_type="grid",
            cv=3,
            scoring="accuracy",
            include_models=["Logistic Regression"],
            preprocessor=None,
            n_jobs=1,
            cache=False,
        )
        evals = evaluate_models(trained, X_test, y_test)
        roc_auc = evals["Logistic Regression"].get("roc_auc")
        self.assertIsNotNone(roc_auc)
        self.assertTrue(0.0 <= float(roc_auc) <= 1.0)


if __name__ == "__main__":
    unittest.main()
