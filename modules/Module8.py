"""Module 8: Streamlit Deployment Module.

Implements the minimum-spec end-to-end Streamlit AutoML app:
upload -> EDA -> issue detection + user-approved preprocessing -> model training/tuning -> comparison -> report.
"""

from __future__ import annotations

import io
import re
import time
from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.pipeline import Pipeline

from project.modules.Module1 import get_class_distribution, get_column_types, get_shape, get_summary_statistics
from project.modules.Module2 import (
    correlation_matrix,
    missing_value_analysis,
    plot_categorical_distribution,
    plot_correlation_heatmap,
    plot_missing_values,
    plot_numerical_distribution,
    plot_outlier_boxplot,
)
from project.modules.Module3 import (
    detect_class_imbalance,
    detect_constant_features,
    detect_high_cardinality,
    detect_missing_values,
    detect_outliers_iqr,
    detect_outliers_zscore,
)
from project.modules.Module4 import build_preprocessor, handle_outliers_capping, split_train_test_stratified
from project.modules.Module5 import evaluate_models, train_and_optimize_models
from project.modules.Module6 import plot_metric_bars, rank_algorithms, show_comparison_table
from project.modules.Module7 import generate_dataset_overview, export_report_as_markdown_bytes


@dataclass
class PreprocessChoices:
    numeric_impute: str
    categorical_impute: str
    scaling: str
    encoding: str
    outlier_action: str
    outlier_method: str


def _sanitize_column_name(name: str) -> str:
    # NFR-S2: sanitize user-provided text used as a column name
    cleaned = re.sub(r"[^0-9a-zA-Z_\- ]+", "", str(name)).strip()
    return cleaned or "col"


@st.cache_data(show_spinner=False)
def _read_csv_from_upload(uploaded_file) -> pd.DataFrame:
    return pd.read_csv(uploaded_file)


def _health_check(df: pd.DataFrame) -> dict[str, Any]:
    unnamed = [c for c in df.columns if str(c).lower().startswith('unnamed')]
    whitespace_cols = [c for c in df.columns if str(c) != str(c).strip()]
    non_ascii_cols = [c for c in df.columns if any(ord(ch) > 127 for ch in str(c))]
    return {
        'rows': int(df.shape[0]),
        'cols': int(df.shape[1]),
        'unnamed_columns': unnamed,
        'whitespace_columns': whitespace_cols,
        'non_ascii_columns': non_ascii_cols,
    }


def _render_basic_info(df: pd.DataFrame, target_col: str | None):
    rows, cols = get_shape(df)
    st.subheader('Dataset Upload & Basic Info')
    c1, c2, c3 = st.columns(3)
    c1.metric('Rows', rows)
    c2.metric('Columns', cols)
    c3.metric('Missing Cells', int(df.isna().sum().sum()))

    st.markdown('**Column Types**')
    st.dataframe(get_column_types(df).astype(str), use_container_width=True)

    st.markdown('**Summary Statistics (Numeric)**')
    try:
        st.dataframe(get_summary_statistics(df), use_container_width=True)
    except Exception:
        st.info('No numeric columns available for summary statistics.')

    if target_col:
        st.markdown('**Class Distribution**')
        st.dataframe(get_class_distribution(df, target_col).rename('count'), use_container_width=True)


def _render_preview_tools(df: pd.DataFrame) -> pd.DataFrame:
    st.subheader('Preview & Cleanup (before analysis)')
    st.caption('These operations affect only the current session (in-memory).')

    st.markdown('**Preview (first 100 rows)**')
    st.dataframe(df.head(100), use_container_width=True)

    if st.button('Shuffle preview (random 100 rows)'):
        sample_n = min(100, len(df))
        st.dataframe(df.sample(n=sample_n, random_state=int(time.time()) % 10_000), use_container_width=True)

    drop_cols = st.multiselect('Remove columns (e.g., IDs)', options=list(df.columns))
    if drop_cols:
        df = df.drop(columns=drop_cols)

    st.markdown('**Rename columns**')
    rename_df = pd.DataFrame({'current': list(df.columns), 'new': list(df.columns)})
    edited = st.data_editor(rename_df, num_rows='fixed', use_container_width=True)
    if st.button('Apply column renames'):
        mapping = {}
        for _, row in edited.iterrows():
            cur = row['current']
            new = _sanitize_column_name(row['new'])
            if cur != new:
                mapping[cur] = new
        if mapping:
            df = df.rename(columns=mapping)
            st.success('Renamed columns applied.')
        else:
            st.info('No column renames detected.')

    return df


def _render_eda(df: pd.DataFrame):
    st.subheader('Automated EDA')

    missing_df = missing_value_analysis(df)
    st.markdown('**Missing Value Analysis**')
    st.pyplot(plot_missing_values(missing_df), clear_figure=True)
    if not missing_df.empty:
        st.dataframe(missing_df, use_container_width=True)

    corr_df = correlation_matrix(df)
    st.markdown('**Correlation Matrix**')
    st.pyplot(plot_correlation_heatmap(corr_df), clear_figure=True)

    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    cat_cols = [c for c in df.columns if c not in numeric_cols]

    st.markdown('**Distribution Plots**')
    with st.expander('Numeric feature distribution'):
        if numeric_cols:
            num_choice = st.selectbox('Select numeric feature', options=numeric_cols)
            st.pyplot(plot_numerical_distribution(df, num_choice), clear_figure=True)
        else:
            st.info('No numeric features found.')

    with st.expander('Categorical feature bar plot'):
        if cat_cols:
            cat_choice = st.selectbox('Select categorical feature', options=cat_cols)
            st.pyplot(plot_categorical_distribution(df, cat_choice), clear_figure=True)
        else:
            st.info('No categorical features found.')

    with st.expander('Outlier visualization (boxplot)'):
        if numeric_cols:
            out_choice = st.selectbox('Select numeric feature for boxplot', options=numeric_cols)
            st.pyplot(plot_outlier_boxplot(df, out_choice), clear_figure=True)
        else:
            st.info('No numeric features found.')


def _detect_issues(df: pd.DataFrame, target_col: str) -> dict[str, Any]:
    warnings: dict[str, Any] = {}

    mv = detect_missing_values(df)
    if not mv.empty:
        warnings['missing_values'] = mv.to_dict()

    outliers_z = detect_outliers_zscore(df)
    if outliers_z:
        warnings['outliers_zscore'] = {k: len(v) for k, v in outliers_z.items()}

    outliers_iqr = detect_outliers_iqr(df)
    if outliers_iqr:
        warnings['outliers_iqr'] = outliers_iqr

    imbalance = detect_class_imbalance(df, target_col)
    if not imbalance.empty:
        warnings['class_imbalance'] = imbalance.to_dict()

    high_card = detect_high_cardinality(df)
    if high_card:
        warnings['high_cardinality'] = high_card

    constant = detect_constant_features(df)
    if constant:
        warnings['constant_features'] = constant

    return warnings


def _render_issue_detection_and_choices(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, PreprocessChoices, dict[str, Any]]:
    st.subheader('Issue Detection & User Approval')
    issues = _detect_issues(df, target_col)

    if not issues:
        st.success('No major data quality issues detected with default rules.')
    else:
        for k, v in issues.items():
            st.warning(f"Detected issue: {k}")
            st.write(v)

        st.markdown('**Suggested fixes (examples)**')
        if 'missing_values' in issues:
            st.write('- Consider imputing missing numeric values with median and categorical with most frequent.')
        if 'outliers_iqr' in issues or 'outliers_zscore' in issues:
            st.write('- Consider capping outliers (IQR) or removing extreme rows if justified by domain knowledge.')
        if 'class_imbalance' in issues:
            st.write('- Consider using stratified split (enabled) and evaluating with F1-score; optionally use class weights in models later.')
        if 'high_cardinality' in issues:
            st.write('- Consider One-Hot encoding with rare-category grouping, or Ordinal encoding if categories are ordered.')
        if 'constant_features' in issues:
            st.write('- Consider dropping constant/near-constant features before training.')

    st.markdown('**Choose preprocessing options (nothing is applied until you confirm)**')
    c1, c2 = st.columns(2)

    with c1:
        numeric_impute = st.selectbox(
            'Missing values (numeric):',
            options=['mean', 'median', 'most_frequent'],
            index=1,
            help='How to impute missing values in numeric columns.',
        )
        scaling = st.selectbox(
            'Scaling:',
            options=['standard', 'minmax', 'none'],
            index=0,
            help='StandardScaler or MinMaxScaler (or no scaling).',
        )
        outlier_method = st.selectbox(
            'Outlier detection method:',
            options=['iqr', 'zscore'],
            index=0,
            help='Used to decide which rows/values are considered outliers.',
        )

    with c2:
        categorical_impute = st.selectbox(
            'Missing values (categorical):',
            options=['most_frequent', 'constant'],
            index=0,
            help='How to impute missing values in categorical columns.',
        )
        encoding = st.selectbox(
            'Categorical encoding:',
            options=['onehot', 'ordinal'],
            index=0,
            help='One-Hot Encoding or Ordinal Encoding.',
        )
        outlier_action = st.selectbox(
            'Outlier handling:',
            options=['no_action', 'cap_iqr', 'remove_rows'],
            index=0,
            help='Cap using IQR bounds, remove outlier rows, or take no action.',
        )

    choices = PreprocessChoices(
        numeric_impute='median' if numeric_impute == 'median' else ('mean' if numeric_impute == 'mean' else 'most_frequent'),
        categorical_impute='most_frequent' if categorical_impute == 'most_frequent' else 'constant',
        scaling='none' if scaling == 'none' else scaling,
        encoding=encoding,
        outlier_action=outlier_action,
        outlier_method=outlier_method,
    )

    apply_now = st.button('Apply preprocessing (with approval)')
    if not apply_now:
        return df, choices, issues

    before_df = df.copy()

    # Outlier handling (simple, dataset-level) BEFORE split.
    if choices.outlier_action == 'cap_iqr':
        for col in before_df.select_dtypes(include='number').columns:
            before_df = handle_outliers_capping(before_df, col)
    elif choices.outlier_action == 'remove_rows':
        numeric_cols = before_df.select_dtypes(include='number').columns
        if len(numeric_cols) > 0:
            mask = pd.Series(False, index=before_df.index)
            if choices.outlier_method == 'zscore':
                out = detect_outliers_zscore(before_df)
                for idxs in out.values():
                    mask.loc[idxs] = True
            else:
                # IQR row removal: any row outlying on any numeric column
                for col in numeric_cols:
                    q1 = before_df[col].quantile(0.25)
                    q3 = before_df[col].quantile(0.75)
                    iqr = q3 - q1
                    lb = q1 - 1.5 * iqr
                    ub = q3 + 1.5 * iqr
                    mask = mask | (before_df[col] < lb) | (before_df[col] > ub)
            before_df = before_df.loc[~mask].copy()

    st.markdown('**Before vs After (preview)**')
    st.write('Before:')
    st.dataframe(df.head(10), use_container_width=True)
    st.write('After:')
    st.dataframe(before_df.head(10), use_container_width=True)
    st.success('Preprocessing choices stored for training. (Transformations are applied via a sklearn Pipeline during training.)')

    return before_df, choices, issues


def _metric_to_sklearn_scoring(metric: str, is_binary: bool) -> str:
    if metric == 'accuracy':
        return 'accuracy'
    if metric == 'precision':
        return 'precision' if is_binary else 'precision_weighted'
    if metric == 'recall':
        return 'recall' if is_binary else 'recall_weighted'
    # default f1
    return 'f1' if is_binary else 'f1_weighted'


def main():
    st.set_page_config(page_title='AutoML Classification (CS-245)', layout='wide')
    st.title('AutoML System for Classification')
    st.caption('Minimum-spec implementation: upload → EDA → user-approved preprocessing → train/tune → compare → report')

    with st.sidebar:
        st.header('Configuration')
        primary_metric = st.selectbox('Primary ranking metric', options=['f1', 'accuracy', 'precision', 'recall'], index=0)
        search_type = st.selectbox('Hyperparameter search', options=['grid', 'random'], index=0)
        test_ratio = st.slider('Test split ratio', min_value=0.1, max_value=0.5, value=0.2, step=0.05)
        cv = st.slider('CV folds', min_value=3, max_value=5, value=3, step=1)

        model_options = [
            'Logistic Regression',
            'K-Nearest Neighbors',
            'Decision Tree',
            'Naive Bayes',
            'Random Forest',
            'Support Vector Machine',
            'Rule-based (Most Frequent)',
        ]
        selected_models = st.multiselect('Models to train', options=model_options, default=model_options)

        if st.button('Reset session (undo preprocessing/training)'):
            for k in ['raw_df', 'trained_models', 'evaluation_results', 'preprocess_choices', 'issues']:
                if k in st.session_state:
                    del st.session_state[k]
            st.rerun()

    uploaded = st.file_uploader('Upload a CSV file', type=['csv'])
    if uploaded is None:
        st.info('Upload a CSV to start.')
        return

    # FR-1 / NFR-P1: basic file-size validation
    try:
        if hasattr(uploaded, 'size') and uploaded.size is not None and uploaded.size > 50 * 1024 * 1024:
            st.error('File too large. Please upload a CSV under 50MB.')
            return
    except Exception:
        pass

    try:
        df = _read_csv_from_upload(uploaded)
    except Exception as e:
        st.error(f'File Read Error: {e}')
        return

    # Persist base dataset across reruns
    st.session_state['raw_df'] = df

    health = _health_check(df)
    if health['unnamed_columns']:
        st.warning(f"Unnamed columns detected: {health['unnamed_columns']}")
    if health['whitespace_columns']:
        st.warning(f"Columns with leading/trailing whitespace: {health['whitespace_columns']}")
    if health['non_ascii_columns']:
        st.warning(f"Columns with non-ASCII characters: {health['non_ascii_columns']}")

    df = _render_preview_tools(df)

    target_col = st.selectbox('Select target column (classification label)', options=list(df.columns))

    _render_basic_info(df, target_col)
    _render_eda(df)

    df_after, choices, issues = _render_issue_detection_and_choices(df, target_col)

    st.subheader('Train/Test Split Summary')
    try:
        X_train, X_test, y_train, y_test = split_train_test_stratified(df_after, target_col, test_size=float(test_ratio))
        st.write({
            'train_samples': int(len(X_train)),
            'test_samples': int(len(X_test)),
            'train_ratio': float(1 - test_ratio),
            'test_ratio': float(test_ratio),
        })
    except Exception as e:
        st.error(f'Unable to split dataset: {e}')
        return

    st.subheader('Model Training & Hyperparameter Optimization')
    run_training = st.button('Train models')
    if not run_training and 'evaluation_results' not in st.session_state:
        st.info('Click “Train models” to run training and evaluation.')
        return

    is_binary = y_train.nunique() == 2
    scoring = _metric_to_sklearn_scoring(primary_metric, is_binary=is_binary)

    preprocessor = build_preprocessor(
        X_train,
        numeric_impute=choices.numeric_impute,
        categorical_impute=choices.categorical_impute,
        scaling=choices.scaling,
        encoding=choices.encoding,
    )

    if run_training:
        with st.spinner('Training models (this may take a bit)...'):
            # Train all models on transformed data by wrapping each estimator in the same preprocessor pipeline.
            X_train_t = preprocessor.fit_transform(X_train)
            X_test_t = preprocessor.transform(X_test)

            trained = train_and_optimize_models(
                X_train_t,
                y_train,
                search_type=search_type,
                cv=int(cv),
                scoring=scoring,
                include_models=selected_models,
            )
            evals = evaluate_models(trained, X_test_t, y_test)
            st.session_state['trained_models'] = trained
            st.session_state['evaluation_results'] = evals
            st.session_state['preprocess_choices'] = choices.__dict__
            st.session_state['issues'] = issues

    evals = st.session_state['evaluation_results']
    results_list = [
        {
            'model': k,
            'accuracy': v.get('accuracy'),
            'precision': v.get('precision'),
            'recall': v.get('recall'),
            'f1_score': v.get('f1_score'),
            'roc_auc': v.get('roc_auc'),
            'training_time': v.get('training_time'),
            'best_params': v.get('best_params'),
            'error': v.get('error'),
        }
        for k, v in evals.items()
    ]
    comparison_df = show_comparison_table(results_list)

    st.subheader('Model Comparison Dashboard')
    st.dataframe(comparison_df, use_container_width=True)

    metric_map = {'f1': 'f1_score', 'accuracy': 'accuracy', 'precision': 'precision', 'recall': 'recall'}
    sort_metric = metric_map[primary_metric]
    ranked = rank_algorithms(comparison_df.reset_index(drop=True), sort_metric)
    st.markdown(f'**Ranking by {sort_metric}**')
    st.dataframe(ranked, use_container_width=True)

    st.pyplot(plot_metric_bars(comparison_df.reset_index(drop=True), sort_metric), clear_figure=True)

    st.download_button(
        'Download results as CSV',
        data=comparison_df.reset_index(drop=True).to_csv(index=False).encode('utf-8'),
        file_name='model_comparison.csv',
        mime='text/csv',
    )

    # Show confusion matrices
    st.subheader('Confusion Matrices')
    for model_name, v in evals.items():
        if v.get('confusion_matrix') is None:
            continue
        st.markdown(f'**{model_name}**')
        cm = np.array(v['confusion_matrix'])
        fig = None
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)

    # Best model + downloads
    valid_ranked = ranked[ranked[sort_metric].notna()]
    best_name = valid_ranked.iloc[0]['model'] if not valid_ranked.empty else None

    st.subheader('Auto-Generated Final Report')
    if best_name:
        st.success(f'Best model by {sort_metric}: {best_name}')
    else:
        st.warning('No successful models to select as best model.')

    report_sections: dict[str, Any] = {
        'Dataset Overview': generate_dataset_overview(df_after),
        'EDA Findings': {
            'missing_value_columns': int((df_after.isna().sum() > 0).sum()),
            'total_missing_cells': int(df_after.isna().sum().sum()),
            'num_features': int(df_after.select_dtypes(include='number').shape[1]),
            'cat_features': int(df_after.select_dtypes(exclude='number').shape[1]),
        },
        'Detected Issues': st.session_state.get('issues', {}),
        'Preprocessing Decisions': st.session_state.get('preprocess_choices', {}),
        'Model Comparison': comparison_df.reset_index(drop=True),
        'Best Model Summary': {
            'best_model': best_name,
            'ranking_metric': sort_metric,
        },
    }

    st.download_button(
        'Download report (Markdown)',
        data=export_report_as_markdown_bytes(report_sections, title='CS-245 AutoML Report'),
        file_name='automl_report.md',
        mime='text/markdown',
    )

    if best_name and st.session_state.get('trained_models', {}).get(best_name, {}).get('model') is not None:
        # Export winning model pipeline as a single object (preprocessor + estimator)
        best_est = st.session_state['trained_models'][best_name]['model']
        export_pipeline = Pipeline(steps=[('preprocess', preprocessor), ('model', best_est)])
        buffer = io.BytesIO()
        joblib.dump(export_pipeline, buffer)
        st.download_button(
            'Download best model (pickle/joblib)',
            data=buffer.getvalue(),
            file_name='best_model.joblib',
            mime='application/octet-stream',
        )