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

Implements Markdown and PDF exports.
"""

from __future__ import annotations

import io
import json
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
    # For nested structures, a JSON code block is the most robust.
    if isinstance(value, (dict, list, tuple)):
        try:
            return "```json\n" + json.dumps(value, indent=2, default=str) + "\n```"
        except Exception:
            return "```\n" + str(value) + "\n```"
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


def export_report_as_pdf_bytes(report_sections: dict[str, Any], title: str = "AutoML Classification Report") -> bytes:
    """Export report as PDF bytes.

    Uses fpdf2 (pure Python). If unavailable, raises RuntimeError with install hint.
    """

    try:
        from fpdf import FPDF  # type: ignore
    except Exception as e:
        raise RuntimeError('PDF export requires fpdf2. Install it via `pip install fpdf2`.') from e

    md = build_markdown_report(report_sections, title=title)

    pdf = FPDF(orientation='P', unit='mm', format='A4')
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    # Built-in font; avoids external font dependencies.
    pdf.set_font('Helvetica', size=10)

    effective_width = float(pdf.w - pdf.l_margin - pdf.r_margin)

    def _write_line(text: str) -> None:
        # fpdf2 can throw if it cannot render even a single character in the remaining width.
        pdf.set_x(pdf.l_margin)
        if text == "":
            pdf.ln(5)
            return
        try:
            pdf.multi_cell(effective_width, 5, text)
        except Exception:
            # Fallback: chunk long unbreakable lines.
            chunk_size = 80
            for i in range(0, len(text), chunk_size):
                pdf.set_x(pdf.l_margin)
                pdf.multi_cell(effective_width, 5, text[i:i + chunk_size])

    # Render markdown as plain text. (Minimum spec is exportability, not rich PDF formatting.)
    for line in md.splitlines():
        # fpdf core fonts are latin-1; replace unsupported chars.
        safe_line = line.encode('latin-1', errors='replace').decode('latin-1')
        _write_line(safe_line)

    out = pdf.output()
    if isinstance(out, (bytes, bytearray)):
        return bytes(out)
    return str(out).encode('latin-1')