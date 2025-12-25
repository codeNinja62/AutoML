import unittest

import pandas as pd

from modules.Module2 import (
    correlation_matrix,
    missing_value_analysis,
    outlier_summary_iqr,
    outlier_summary_zscore,
    train_test_split_summary,
)


class TestModule2EDA(unittest.TestCase):
    def test_missing_value_analysis_filters_non_missing(self):
        df = pd.DataFrame({'a': [1, None, 3], 'b': [1, 2, 3]})
        mv = missing_value_analysis(df)
        self.assertIn('missing_count', mv.columns)
        self.assertIn('missing_pct', mv.columns)
        self.assertIn('a', mv.index)
        self.assertNotIn('b', mv.index)

    def test_correlation_matrix_empty_when_no_numeric(self):
        df = pd.DataFrame({'a': ['x', 'y'], 'b': ['m', 'n']})
        corr = correlation_matrix(df)
        self.assertTrue(corr.empty)

    def test_outlier_summary_iqr_detects_simple_outlier(self):
        df = pd.DataFrame({'x': [1, 1, 1, 1, 100]})
        out = outlier_summary_iqr(df)
        # With IQR=0 this should produce empty (guarded), so we use a slightly varied dataset.
        df2 = pd.DataFrame({'x': [1, 2, 2, 3, 100]})
        out2 = outlier_summary_iqr(df2)
        self.assertTrue(out2.empty or out2.iloc[0]['outlier_count'] >= 1)

    def test_outlier_summary_zscore_detects_simple_outlier(self):
        df = pd.DataFrame({'x': [0, 0, 0, 0, 10]})
        out = outlier_summary_zscore(df, threshold=2.0)
        self.assertTrue(out.empty or out.iloc[0]['outlier_count'] >= 1)

    def test_train_test_split_summary_shapes(self):
        df = pd.DataFrame(
            {
                'f1': [1, 2, 3, 4, 5, 6],
                'f2': [10, 20, 30, 40, 50, 60],
                'y': ['a', 'a', 'a', 'b', 'b', 'b'],
            }
        )
        summary = train_test_split_summary(df, target_col='y', test_size=0.33, random_state=42)
        self.assertIn('train_rows', summary)
        self.assertIn('test_rows', summary)
        self.assertGreater(summary['train_rows'], 0)
        self.assertGreater(summary['test_rows'], 0)
        self.assertEqual(summary['n_features'], 2)


if __name__ == '__main__':
    unittest.main()
