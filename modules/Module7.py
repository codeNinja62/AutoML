# Module 7: Auto-Generated Final Report
# Dataset overview
# EDA results
# Issues detected
# Preprocessing decisions
# Model settings and hyperparameters
# Comparison tables
# Best model explanation
# Export as PDF/Markdown


"""Module 7: Auto-Generated Final Report.

Minimum-spec implementation focuses on generating a downloadable Markdown report.
"""

from __future__ import annotations

import io
from typing import Any

import pandas as pd

# 1. Generate dataset overview section
def generate_dataset_overview(df):
    overview = {
        "Number of Rows": df.shape[0],
        "Number of Columns": df.shape[1],
        "Column Types": df.dtypes.to_dict(),
        "Summary Statistics": df.describe().to_dict()
    }
    return overview

# 2. Compile EDA results section
def compile_eda_results(eda_results):
    return eda_results

# 3. Document issues detected section
def document_issues_detected(issues):
    return issues

# 4. Summarize preprocessing decisions section
def summarize_preprocessing_decisions(preprocessing_steps):
    return preprocessing_steps

# 5. Detail model settings and hyperparameters section
def detail_model_settings(model_info):
    return model_info

# 6. Create comparison tables section
def create_comparison_tables(comparison_df):
    return comparison_df.to_dict()

# 7. Explain best model section
def explain_best_model(best_model_info):
    return best_model_info


def _to_markdown_block(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, pd.DataFrame):
        return value.to_markdown(index=False)
    if isinstance(value, dict):
        lines = [f"- **{k}**: {v}" for k, v in value.items()]
        return "\n".join(lines)
    return str(value)


def build_markdown_report(report_sections: dict[str, Any], title: str = "AutoML Classification Report") -> str:
    parts: list[str] = [f"# {title}", ""]
    for section_title, content in report_sections.items():
        parts.append(f"## {section_title}")
        parts.append(_to_markdown_block(content))
        parts.append("")
    return "\n".join(parts).strip() + "\n"


def export_report_as_markdown_bytes(report_sections: dict[str, Any], title: str = "AutoML Classification Report") -> bytes:
    md = build_markdown_report(report_sections, title=title)
    buffer = io.BytesIO()
    buffer.write(md.encode("utf-8"))
    return buffer.getvalue()