import os
import tempfile
import unittest

import pandas as pd

from modules.Module6 import plot_metric_bars, rank_algorithms, results_to_csv_bytes, save_results_to_csv, show_comparison_table


class TestModule6(unittest.TestCase):
    def test_show_comparison_table_sets_index_when_model_present(self):
        df = show_comparison_table([
            {"model": "A", "accuracy": 0.5},
            {"model": "B", "accuracy": 0.7},
        ])
        self.assertIn("model", df.columns)
        self.assertEqual(list(df.index), ["A", "B"])

    def test_rank_algorithms_sorts_desc_and_nans_last(self):
        comparison_df = pd.DataFrame([
            {"model": "A", "f1_score": 0.8},
            {"model": "B", "f1_score": None},
            {"model": "C", "f1_score": 0.9},
        ])
        ranked = rank_algorithms(comparison_df, "f1_score")
        self.assertEqual(list(ranked["model"]), ["C", "A", "B"])

    def test_results_to_csv_bytes(self):
        comparison_df = pd.DataFrame([{"model": "A", "accuracy": 1.0}])
        b = results_to_csv_bytes(comparison_df)
        self.assertTrue(isinstance(b, (bytes, bytearray)))
        self.assertIn(b"model", b)

    def test_save_results_to_csv_writes_file(self):
        comparison_df = pd.DataFrame([{"model": "A", "accuracy": 1.0}])
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "out.csv")
            save_results_to_csv(comparison_df, path)
            self.assertTrue(os.path.exists(path))
            with open(path, "rb") as f:
                content = f.read()
            self.assertIn(b"model", content)

    def test_plot_metric_bars_returns_figure(self):
        comparison_df = pd.DataFrame([
            {"model": "A", "accuracy": 0.5},
            {"model": "B", "accuracy": 0.7},
        ])
        fig = plot_metric_bars(comparison_df, "accuracy")
        self.assertTrue(hasattr(fig, "savefig"))


if __name__ == "__main__":
    unittest.main()
