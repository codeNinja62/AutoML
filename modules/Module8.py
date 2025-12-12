# Module 8: Streamlit Deployment Module
# Streamlit UI integration
# Proper requirements.txt
# Public hosting on Streamlit Cloud
# README with instructions and app link


# Imports
import streamlit as st
import pandas as pd
import joblib
import pdfplumber
from project.modules.Module7 import (generate_dataset_overview,
                                      compile_eda_results, document_issues_detected,
                                      summarize_preprocessing_decisions, detail_model_settings, create_comparison_tables, explain_best_model)

# 1. Load model and data
@st.cache_data
def load_model_and_data(model_path, data_path):
    model = joblib.load(model_path)
    data = pd.read_csv(data_path)
    return model, data

# 2. Main Streamlit app function
def main():
    st.title("Machine Learning Model Report")

    # Load model and data
    model, data = load_model_and_data('model.pkl', 'data.csv')

    # Generate report sections
    dataset_overview = generate_dataset_overview(data)
    eda_results = compile_eda_results("EDA results placeholder")
    issues_detected = document_issues_detected("Issues detected placeholder")
    preprocessing_decisions = summarize_preprocessing_decisions("Preprocessing decisions placeholder")
    model_settings = detail_model_settings("Model settings placeholder")
    comparison_tables = create_comparison_tables(pd.DataFrame({"Model": ["A", "B"], "Accuracy": [0.9, 0.85]}))
    best_model_explanation = explain_best_model("Best model explanation placeholder")

    # Display report sections in Streamlit
    st.header("Dataset Overview")
    st.json(dataset_overview)

    st.header("EDA Results")
    st.text(eda_results)

    st.header("Issues Detected")
    st.text(issues_detected)

    st.header("Preprocessing Decisions")
    st.text(preprocessing_decisions)

    st.header("Model Settings and Hyperparameters")
    st.text(model_settings)

    st.header("Comparison Tables")
    st.json(comparison_tables)

    st.header("Best Model Explanation")
    st.text(best_model_explanation)