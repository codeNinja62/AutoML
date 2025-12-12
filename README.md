# CS-245 AutoML System for Classification (Streamlit)

A minimum-spec AutoML pipeline for supervised **classification** tasks:
- Upload a CSV dataset
- Automated EDA (missing values, correlations, distributions)
- Issue detection + user-approved preprocessing choices
- Train & tune multiple classifiers
- Compare models and download results
- Download a final Markdown report

This repo is organized as a set of Python modules (Modules 1–8) with a Streamlit UI on top.

## Features (Minimum Spec)
- Dataset upload + basic info (shape, dtypes, summary stats, class distribution)
- EDA: missing value chart, correlation heatmap, numeric/categorical distributions, outlier boxplot
- Issue detection: missing values, outliers, class imbalance, high-cardinality categoricals, constant features
- Preprocessing (user-controlled): imputation, scaling (Standard/MinMax/none), encoding (OneHot/Ordinal), outlier handling
- Training & tuning: Grid/Random search across LR, KNN, DT, NB, RF, SVM, and a baseline rule-based classifier
- Metrics: accuracy, precision, recall, F1, confusion matrix, ROC-AUC (binary + multiclass when supported)
- Comparison dashboard: ranking + bar chart + CSV download
- Report: downloadable Markdown + PDF

## Run Locally
### 1) Install
1. Create/activate a Python environment (Python 3.9+ recommended)
2. Install dependencies:
   - `pip install -r requirements.txt`

### 2) Run the app
- `python -m streamlit run streamlit_app.py`

Optional flags:
- Headless: `python -m streamlit run streamlit_app.py --server.headless true`
- Custom port: `python -m streamlit run streamlit_app.py --server.port 8509`

### 3) Run unit tests
- `python -m unittest -q`

## Deploy on Streamlit Cloud
- Set the app entrypoint to: `streamlit_app.py`
- Ensure `requirements.txt` is present at repo root
- Push to GitHub and deploy via Streamlit Cloud UI

App link (Streamlit Cloud): _add your deployed URL here_

## Project Structure
- `streamlit_app.py` — Streamlit Cloud entrypoint
- `project/modules/Module1.py` … `Module8.py` — modular pipeline implementation
- `project/tests/` — unittest suite for Modules 1–7
- `sample_dataset.csv` — small sample dataset for quick manual testing

## Outputs
From the app you can download:
- Model comparison results as CSV
- Final report as Markdown (and PDF if `fpdf2` is installed)
- Best trained model as a single `joblib` file (an sklearn Pipeline that includes preprocessing + estimator)

## Troubleshooting
- If Streamlit prints “Stopping…” and exits with a non-zero code, it’s often just the server shutting down (e.g., the process was interrupted). Re-run the command and keep the terminal open.
- If deployment fails on Streamlit Cloud, confirm the entrypoint is `streamlit_app.py` and that `requirements.txt` installs cleanly.

## Notes
- Uploaded data is processed in-memory for the current session.
- The downloaded best model is exported as a single sklearn Pipeline (preprocessor + estimator).
