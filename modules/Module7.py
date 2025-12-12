# Module 7: Auto-Generated Final Report
# Dataset overview
# EDA results
# Issues detected
# Preprocessing decisions
# Model settings and hyperparameters
# Comparison tables
# Best model explanation
# Export as PDF/Markdown


# Imports
import pandas as pd
from fpdf import FPDF
import matplotlib.pyplot as plt

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

# 8. Export report as PDF
def export_report_as_pdf(report_sections, file_path):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    for section_title, content in report_sections.items():
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, section_title, ln=True)
        pdf.set_font("Arial", size=12)

        if isinstance(content, dict):
            for key, value in content.items():
                pdf.multi_cell(0, 10, f"{key}: {value}")
        elif isinstance(content, pd.DataFrame):
            pdf.multi_cell(0, 10, content.to_string())
        else:
            pdf.multi_cell(0, 10, str(content))

        pdf.ln(10)

    pdf.output(file_path)