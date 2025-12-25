import unittest

import pandas as pd

from modules.Module7 import build_markdown_report, export_report_as_markdown_bytes


class TestModule7Markdown(unittest.TestCase):
    def test_build_markdown_report_includes_sections(self):
        sections = {
            "Dataset Overview": {"rows": 3, "cols": 2},
            "Model Comparison": pd.DataFrame([
                {"model": "A", "accuracy": 0.9},
                {"model": "B", "accuracy": 0.8},
            ]),
        }
        md = build_markdown_report(sections, title="My Report")
        self.assertIn("# My Report", md)
        self.assertIn("## Dataset Overview", md)
        self.assertIn("## Model Comparison", md)
        self.assertIn("model", md)

    def test_export_report_as_markdown_bytes(self):
        b = export_report_as_markdown_bytes({"A": {"k": 1}}, title="T")
        self.assertTrue(isinstance(b, (bytes, bytearray)))
        self.assertIn(b"# T", b)


class TestModule7Pdf(unittest.TestCase):
    def test_export_report_as_pdf_bytes_if_available(self):
        try:
            from modules.Module7 import export_report_as_pdf_bytes
        except Exception:
            self.skipTest("Module7 missing PDF export")

        try:
            pdf_bytes = export_report_as_pdf_bytes({"A": {"k": 1}}, title="T")
        except RuntimeError:
            self.skipTest("fpdf2 not installed")

        self.assertTrue(isinstance(pdf_bytes, (bytes, bytearray)))
        self.assertTrue(pdf_bytes.startswith(b"%PDF"))


if __name__ == "__main__":
    unittest.main()
