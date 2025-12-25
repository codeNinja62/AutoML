import unittest

import pandas as pd

from modules.Module3 import detect_issues


class TestModule3IssueDetection(unittest.TestCase):
    def test_detect_issues_missing_values(self):
        df = pd.DataFrame({'a': [1, None, 3], 'y': [0, 1, 0]})
        issues = detect_issues(df, target_column='y')
        self.assertIn('missing_values', issues)
        self.assertTrue(isinstance(issues.get('findings', []), list))

    def test_detect_issues_high_cardinality(self):
        df = pd.DataFrame({'cat': [str(i) for i in range(100)], 'y': [0, 1] * 50})
        issues = detect_issues(df, target_column='y', high_cardinality_ratio_threshold=0.5)
        self.assertIn('high_cardinality', issues)
        self.assertIn('cat', [str(c) for c in issues['high_cardinality']])

    def test_detect_issues_class_imbalance(self):
        df = pd.DataFrame({'x': list(range(10)), 'y': [1] * 9 + [0]})
        issues = detect_issues(df, target_column='y', class_imbalance_threshold=0.7)
        self.assertIn('class_imbalance', issues)
        self.assertTrue(bool(issues['class_imbalance']))

    def test_detect_issues_constant_features(self):
        df = pd.DataFrame({'x': [1, 1, 1, 1], 'y': [0, 1, 0, 1]})
        issues = detect_issues(df, target_column='y', constant_threshold=0.75)
        self.assertIn('constant_features', issues)
        self.assertIn('x', [str(c) for c in issues['constant_features']])


if __name__ == '__main__':
    unittest.main()
