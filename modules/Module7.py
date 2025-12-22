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
from datetime import datetime
from typing import Any

import pandas as pd

# 1. Generate dataset overview section
def generate_dataset_overview(df):
    # Keep complex structures as DataFrames for better table rendering in Markdown/PDF
    overview = {
        "Number of Rows": df.shape[0],
        "Number of Columns": df.shape[1],
        "Column Types": df.dtypes.astype(str).to_frame(name='dtype').reset_index().rename(columns={'index': 'column'}),
        "Summary Statistics": df.describe().round(3).reset_index().rename(columns={'index': 'statistic'})
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
    """Recursively convert a value to a markdown string representation."""
    
    def _recruit(v: Any, level: int) -> str:
        # Indent for lists
        indent = "  " * level
        
        if v is None:
            return ""
        
        if isinstance(v, pd.DataFrame):
            # Render dataframe as table.
            # Markdown tables don't support indentation well, so we print them on a new line.
            return "\n" + v.to_markdown(index=False) + "\n"
        
        if isinstance(v, dict):
            if not v:
                return "_(empty)_"
            lines = []
            for k, val in v.items():
                # If the value is a complex type (dict/list/df), label it on its own line
                if isinstance(val, (dict, list, pd.DataFrame)):
                    lines.append(f"{indent}- **{k}**:")
                    lines.append(_recruit(val, level + 1))
                else:
                    lines.append(f"{indent}- **{k}**: {val}")
            return "\n".join(lines)
            
        if isinstance(v, (list, tuple)):
            if not v:
                return "_(empty)_"
            lines = []
            for item in v:
                if isinstance(item, (dict, list, pd.DataFrame)):
                    lines.append(f"{indent}-") 
                    lines.append(_recruit(item, level + 1))
                else:
                    lines.append(f"{indent}- {item}")
            return "\n".join(lines)
            
        return str(v)

    return _recruit(value, 0)


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
    """Export report as a professionally formatted PDF.

    Uses fpdf2 (pure Python). If unavailable, raises RuntimeError with install hint.
    Features:
    - Styled title page
    - Section headings with colors
    - Properly formatted tables
    - Key-value pair formatting for dictionaries
    - UTF-8 support with fallback
    """

    try:
        from fpdf import FPDF  # type: ignore
    except Exception as e:
        raise RuntimeError('PDF export requires fpdf2. Install it via `pip install fpdf2`.') from e

    # Custom PDF class with header/footer
    class AutoMLPDF(FPDF):
        def __init__(self):
            super().__init__(orientation='P', unit='mm', format='A4')
            self.set_auto_page_break(auto=True, margin=15)
            
        def header(self):
            if self.page_no() > 1:
                self.set_font('Helvetica', 'I', 8)
                self.set_text_color(128, 128, 128)
                self.cell(0, 10, title, align='C')
                self.ln(5)
                self.set_draw_color(200, 200, 200)
                self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
                self.ln(5)

        def footer(self):
            self.set_y(-15)
            self.set_font('Helvetica', 'I', 8)
            self.set_text_color(128, 128, 128)
            self.cell(0, 10, f'Page {self.page_no()}', align='C')

    pdf = AutoMLPDF()
    effective_width = float(pdf.w - pdf.l_margin - pdf.r_margin)

    # Colors
    PRIMARY_COLOR = (41, 128, 185)  # Blue
    SECONDARY_COLOR = (44, 62, 80)  # Dark gray
    ACCENT_COLOR = (39, 174, 96)    # Green
    TABLE_HEADER_BG = (52, 73, 94)  # Dark blue-gray
    TABLE_ALT_ROW = (245, 245, 245) # Light gray

    def safe_text(text: Any) -> str:
        """Convert to string and handle encoding."""
        s = str(text) if text is not None else ""
        # Replace problematic characters
        return s.encode('latin-1', errors='replace').decode('latin-1')

    def add_title_page():
        """Create a styled title page."""
        pdf.add_page()
        pdf.ln(60)
        
        # Title
        pdf.set_font('Helvetica', 'B', 28)
        pdf.set_text_color(*PRIMARY_COLOR)
        pdf.multi_cell(effective_width, 12, safe_text(title), align='C')
        
        pdf.ln(10)
        
        # Subtitle/date
        pdf.set_font('Helvetica', '', 12)
        pdf.set_text_color(128, 128, 128)
        pdf.cell(0, 10, f"Generated on {datetime.now().strftime('%B %d, %Y at %H:%M')}", align='C')
        
        pdf.ln(30)
        
        # Decorative line
        pdf.set_draw_color(*PRIMARY_COLOR)
        pdf.set_line_width(0.5)
        center = pdf.w / 2
        pdf.line(center - 40, pdf.get_y(), center + 40, pdf.get_y())

    def add_section_header(section_title: str):
        """Add a styled section header."""
        pdf.ln(8)
        
        # Section header background
        pdf.set_fill_color(*PRIMARY_COLOR)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font('Helvetica', 'B', 14)
        
        pdf.set_x(pdf.l_margin)
        pdf.cell(effective_width, 10, f"  {safe_text(section_title)}", fill=True)
        pdf.ln(12)
        
        # Reset text color
        pdf.set_text_color(0, 0, 0)

    def add_key_value(key: str, value: Any, indent: int = 0):
        """Add a key-value pair with proper formatting."""
        pdf.set_font('Helvetica', 'B', 10)
        pdf.set_text_color(*SECONDARY_COLOR)
        
        x_offset = pdf.l_margin + (indent * 5)
        pdf.set_x(x_offset)
        
        key_text = safe_text(f"{key}: ")
        pdf.cell(pdf.get_string_width(key_text) + 2, 6, key_text)
        
        pdf.set_font('Helvetica', '', 10)
        pdf.set_text_color(0, 0, 0)
        
        value_text = safe_text(str(value))
        # Truncate very long values
        if len(value_text) > 100:
            value_text = value_text[:97] + "..."
        pdf.multi_cell(0, 6, value_text)

    def add_table(df: pd.DataFrame):
        """Add a properly formatted table."""
        if df.empty:
            pdf.set_font('Helvetica', 'I', 10)
            pdf.cell(0, 8, "(No data)", ln=True)
            return

        # Limit columns and rows for PDF readability
        display_df = df.head(20)
        cols = list(display_df.columns)[:8]  # Max 8 columns
        display_df = display_df[cols]
        
        # Calculate column widths
        n_cols = len(cols)
        col_width = min(effective_width / n_cols, 35)
        row_height = 7
        
        # Check if we need a new page
        if pdf.get_y() + 50 > pdf.h - 20:
            pdf.add_page()

        # Table header
        pdf.set_fill_color(*TABLE_HEADER_BG)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font('Helvetica', 'B', 8)
        
        for col in cols:
            col_text = safe_text(str(col))[:12]  # Truncate long column names
            pdf.cell(col_width, row_height, col_text, border=1, fill=True, align='C')
        pdf.ln()
        
        # Table rows
        pdf.set_text_color(0, 0, 0)
        pdf.set_font('Helvetica', '', 8)
        
        for i, (_, row) in enumerate(display_df.iterrows()):
            # Alternate row colors
            if i % 2 == 1:
                pdf.set_fill_color(*TABLE_ALT_ROW)
                fill = True
            else:
                pdf.set_fill_color(255, 255, 255)
                fill = True
            
            for col in cols:
                val = row[col]
                # Format numbers nicely
                if isinstance(val, float):
                    cell_text = f"{val:.4f}" if abs(val) < 1000 else f"{val:.2e}"
                else:
                    cell_text = safe_text(str(val))[:15]
                pdf.cell(col_width, row_height, cell_text, border=1, fill=fill, align='C')
            pdf.ln()
        
        # Show truncation notice if needed
        if len(df) > 20 or len(df.columns) > 8:
            pdf.set_font('Helvetica', 'I', 8)
            pdf.set_text_color(128, 128, 128)
            pdf.cell(0, 6, f"(Showing {min(len(df), 20)} of {len(df)} rows, {len(cols)} of {len(df.columns)} columns)", ln=True)
            pdf.set_text_color(0, 0, 0)
        
        pdf.ln(3)

    def add_dict_content(d: dict, level: int = 0):
        """Recursively add dictionary content."""
        for key, value in d.items():
            if isinstance(value, dict):
                # Nested dict: show as sub-section
                pdf.set_font('Helvetica', 'B', 10)
                pdf.set_text_color(*SECONDARY_COLOR)
                pdf.set_x(pdf.l_margin + (level * 5))
                pdf.cell(0, 7, safe_text(f"{key}:"), ln=True)
                add_dict_content(value, level + 1)
            elif isinstance(value, pd.DataFrame):
                pdf.set_font('Helvetica', 'B', 10)
                pdf.set_text_color(*SECONDARY_COLOR)
                pdf.cell(0, 7, safe_text(f"{key}:"), ln=True)
                add_table(value)
            elif isinstance(value, (list, tuple)):
                if len(value) == 0:
                    add_key_value(key, "(empty)", indent=level)
                elif len(value) <= 5 and all(not isinstance(v, (dict, list)) for v in value):
                    add_key_value(key, ", ".join(str(v) for v in value), indent=level)
                else:
                    pdf.set_font('Helvetica', 'B', 10)
                    pdf.set_text_color(*SECONDARY_COLOR)
                    pdf.set_x(pdf.l_margin + (level * 5))
                    pdf.cell(0, 7, safe_text(f"{key}: ({len(value)} items)"), ln=True)
                    pdf.set_font('Helvetica', '', 9)
                    pdf.set_text_color(0, 0, 0)
                    for i, item in enumerate(value[:10]):
                        pdf.set_x(pdf.l_margin + ((level + 1) * 5))
                        pdf.multi_cell(0, 5, safe_text(f"• {item}"))
                    if len(value) > 10:
                        pdf.set_x(pdf.l_margin + ((level + 1) * 5))
                        pdf.set_font('Helvetica', 'I', 8)
                        pdf.cell(0, 5, f"... and {len(value) - 10} more", ln=True)
            else:
                add_key_value(key, value, indent=level)

    def add_section_content(content: Any):
        """Add content based on its type."""
        pdf.set_font('Helvetica', '', 10)
        pdf.set_text_color(0, 0, 0)
        
        if content is None:
            pdf.set_font('Helvetica', 'I', 10)
            pdf.cell(0, 8, "(No data available)", ln=True)
        elif isinstance(content, pd.DataFrame):
            add_table(content)
        elif isinstance(content, dict):
            add_dict_content(content)
        elif isinstance(content, (list, tuple)):
            for item in content[:20]:
                pdf.set_x(pdf.l_margin + 5)
                pdf.multi_cell(0, 6, safe_text(f"• {item}"))
            if len(content) > 20:
                pdf.set_font('Helvetica', 'I', 8)
                pdf.cell(0, 5, f"... and {len(content) - 20} more items", ln=True)
        else:
            # Plain text
            text = safe_text(str(content))
            for line in text.split('\n'):
                if line.strip():
                    pdf.multi_cell(effective_width, 6, line)
                else:
                    pdf.ln(3)

    # Build the PDF
    add_title_page()
    
    for section_title, content in report_sections.items():
        pdf.add_page()
        add_section_header(section_title)
        add_section_content(content)

    # Output
    out = pdf.output()
    if isinstance(out, (bytes, bytearray)):
        return bytes(out)
    return str(out).encode('latin-1')