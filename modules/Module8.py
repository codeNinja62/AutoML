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
from sklearn.metrics import roc_curve

from sklearn.pipeline import Pipeline

from project.modules.Module1 import (
    get_cardinality_buckets,
    get_class_distribution,
    get_column_types,
    get_dataset_schema,
    get_shape,
    get_summary_statistics,
    get_all_missing_columns,
    get_constant_columns,
    infer_target_candidates,
    validate_target_column,
)
from project.modules.Module2 import (
    correlation_matrix,
    missing_value_analysis,
    outlier_summary_iqr,
    outlier_summary_zscore,
    plot_categorical_distribution,
    plot_correlation_heatmap,
    plot_missing_values,
    plot_numerical_distribution,
    plot_outlier_counts,
    plot_outlier_boxplot,
    train_test_split_summary,
)
from project.modules.Module3 import (
    detect_issues,
    detect_class_imbalance,
    detect_constant_features,
    detect_high_cardinality,
    detect_missing_values,
    detect_outliers_iqr,
    detect_outliers_zscore,
)
from project.modules.Module4 import apply_outlier_handling, build_preprocessor, split_train_test_stratified
from project.modules.Module5 import evaluate_models, train_and_optimize_models
from project.modules.Module6 import plot_metric_bars, rank_algorithms, show_comparison_table
from project.modules.Module7 import generate_dataset_overview, export_report_as_markdown_bytes, export_report_as_pdf_bytes


@dataclass
class PreprocessChoices:
    numeric_impute: str
    categorical_impute: str
    numeric_fill_value: float | None
    categorical_fill_value: str | None
    scaling: str
    encoding: str
    outlier_action: str
    outlier_method: str


def _format_ts(ts: float | None) -> str:
    if not ts:
        return '—'
    try:
        return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(float(ts)))
    except Exception:
        return '—'


def _estimate_workload_text(search_type: str, cv: int, n_models: int) -> str:
    # Simple, user-friendly estimate (not exact) to set expectations.
    if n_models <= 0:
        return 'Select at least one model.'
    if cv <= 0:
        return 'Invalid CV configuration.'
    base = f"{n_models} model(s) × {cv} fold(s)"
    if search_type in {'random', 'halving_random'}:
        return base + ' × ~10 candidates (per model)'
    if search_type in {'halving_grid'}:
        return base + ' × (adaptive candidates; faster than full grid)'
    if search_type == 'grid':
        return base + ' × (grid size varies by model)'
    return base


def _split_feasibility_hint(df: pd.DataFrame, target_col: str, test_ratio: float) -> dict[str, Any]:
    """Return simple diagnostics + a recommended test ratio when stratified split fails."""

    hint: dict[str, Any] = {
        'n_samples': int(len(df)),
        'n_classes': None,
        'min_class_count': None,
        'recommended_test_ratio': None,
    }
    try:
        y = pd.Series(df[target_col]).dropna()
        n_samples = int(len(y))
        n_classes = int(y.nunique())
        hint['n_samples'] = n_samples
        hint['n_classes'] = n_classes
        if n_classes >= 2:
            class_counts = y.value_counts()
            hint['min_class_count'] = int(class_counts.min()) if not class_counts.empty else None

        # For stratification, each split should contain at least one sample per class.
        # Our split helper uses ceil(test_ratio * n_samples) and enforces test_n >= n_classes and train_n >= n_classes.
        if n_samples > 0 and n_classes and n_classes > 0:
            min_test_n = n_classes
            max_test_n = max(n_classes, n_samples - n_classes)
            # Choose the smallest feasible test_n, then convert to ratio.
            feasible_test_n = min_test_n if min_test_n <= (n_samples - n_classes) else None
            if feasible_test_n is not None and n_samples:
                hint['recommended_test_ratio'] = float(np.clip(feasible_test_n / n_samples, 0.1, 0.5))
    except Exception:
        pass
    return hint


def _section(step: int, title: str, caption: str | None = None) -> None:
    st.header(f"Step {step} — {title}")
    if caption:
        st.caption(caption)


def _sanitize_column_name(name: str) -> str:
    # NFR-S2: sanitize user-provided text used as a column name
    cleaned = re.sub(r"[^0-9a-zA-Z_\- ]+", "", str(name)).strip()
    return cleaned or "col"


@st.cache_data(show_spinner=False)
def _read_csv_from_upload(uploaded_file) -> pd.DataFrame:
    return pd.read_csv(uploaded_file)


@st.cache_data(show_spinner=False)
def _read_csv_from_bytes(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))


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
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Rows', rows)
    c2.metric('Columns', cols)
    c3.metric('Missing cells', int(df.isna().sum().sum()))
    c4.metric('Duplicates', int(df.duplicated().sum()))

    with st.expander('Column types', expanded=False):
        st.dataframe(get_column_types(df).astype(str), use_container_width=True)

    with st.expander('Summary statistics (numeric)', expanded=False):
        try:
            st.dataframe(get_summary_statistics(df), use_container_width=True)
        except Exception:
            st.info('No numeric columns available for summary statistics.')

    if target_col:
        with st.expander('Class distribution', expanded=True):
            st.dataframe(get_class_distribution(df, target_col).rename('count'), use_container_width=True)


def _render_preview_tools(df: pd.DataFrame) -> pd.DataFrame:
    st.subheader('Preview & cleanup')
    st.caption('Changes are applied in-memory for this session. Use “Reset session” to undo.')

    with st.expander('Preview (first 100 rows)', expanded=True):
        st.dataframe(df.head(100), use_container_width=True)

    c1, c2 = st.columns([1, 3])
    with c1:
        if st.button('Shuffle preview'):
            sample_n = min(100, len(df))
            st.dataframe(df.sample(n=sample_n, random_state=int(time.time()) % 10_000), use_container_width=True)
    with c2:
        st.caption('Shows a random sample (does not change the dataset).')

    with st.expander('Cleanup actions', expanded=True):
        st.caption('To avoid accidental changes, actions are applied only when you click the button.')

        with st.form('drop_cols_form', clear_on_submit=False):
            drop_cols = st.multiselect('Remove columns (e.g., IDs)', options=list(df.columns))
            drop_apply = st.form_submit_button('Apply column removals')
        if drop_apply and drop_cols:
            df = df.drop(columns=drop_cols)
            st.success(f'Removed {len(drop_cols)} column(s).')
            # Dataset changed; clear any previously trained artifacts to avoid stale results.
            for k in ['trained_models', 'evaluation_results', 'preprocess_choices', 'issues']:
                if k in st.session_state:
                    del st.session_state[k]

        st.divider()
        st.markdown('**Feature creation (combine two numeric columns)**')
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        if len(numeric_cols) >= 2:
            with st.form('feature_create_form', clear_on_submit=False):
                fc1, fc2, fc3 = st.columns([3, 2, 3])
                with fc1:
                    col_a = st.selectbox('Column A', options=numeric_cols, key='cleanup_feat_col_a')
                with fc2:
                    op = st.selectbox('Operation', options=['+', '-', '*', '/'], key='cleanup_feat_op')
                with fc3:
                    col_b = st.selectbox('Column B', options=numeric_cols, key='cleanup_feat_col_b')

                default_name = _sanitize_column_name(f"{col_a}{op}{col_b}")
                new_name = st.text_input('New feature name', value=default_name, key='cleanup_feat_name')
                add_feature = st.form_submit_button('Add feature')
            if add_feature:
                new_name = _sanitize_column_name(new_name)
                if new_name in df.columns:
                    st.error('Feature name already exists. Choose a different name.')
                else:
                    if op == '+':
                        df[new_name] = df[col_a] + df[col_b]
                    elif op == '-':
                        df[new_name] = df[col_a] - df[col_b]
                    elif op == '*':
                        df[new_name] = df[col_a] * df[col_b]
                    else:
                        df[new_name] = df[col_a] / df[col_b].replace({0: np.nan})
                    st.success(f'Added new feature: {new_name}')
                    # Dataset changed; clear any previously trained artifacts to avoid stale results.
                    for k in ['trained_models', 'evaluation_results', 'preprocess_choices', 'issues']:
                        if k in st.session_state:
                            del st.session_state[k]
        else:
            st.info('Need at least two numeric columns to create a combined feature.')

        st.divider()
        st.markdown('**Rename columns**')
        rename_df = pd.DataFrame({'current': list(df.columns), 'new': list(df.columns)})
        edited = st.data_editor(rename_df, num_rows='fixed', use_container_width=True)
        with st.form('rename_cols_form', clear_on_submit=False):
            rename_apply = st.form_submit_button('Apply column renames')
        if rename_apply:
            mapping = {}
            for _, row in edited.iterrows():
                cur = row['current']
                new = _sanitize_column_name(row['new'])
                if cur != new:
                    mapping[cur] = new
            if mapping:
                df = df.rename(columns=mapping)
                st.success('Renamed columns applied.')
                # Dataset changed; clear any previously trained artifacts to avoid stale results.
                for k in ['trained_models', 'evaluation_results', 'preprocess_choices', 'issues']:
                    if k in st.session_state:
                        del st.session_state[k]
            else:
                st.info('No column renames detected.')

    # Note: cleanup actions are intentionally only available inside the expander/forms above
    # to prevent accidental dataset mutations during Streamlit reruns.

    return df


def _render_eda_summary(df: pd.DataFrame, target_col: str):
    """FR-16: One-page summary of main problems."""
    issues = _detect_issues(df, target_col)
    missing_cols = int((df.isna().sum() > 0).sum())
    total_missing = int(df.isna().sum().sum())
    out_iqr = issues.get('outliers_iqr', {})
    out_z = issues.get('outliers_zscore', {})
    st.markdown('**EDA summary (main problems)**')
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Columns w/ missing', missing_cols)
    c2.metric('Total missing cells', total_missing)
    c3.metric('Outlier cols (IQR)', len(out_iqr) if isinstance(out_iqr, dict) else 0)
    c4.metric('High-cardinality', len(issues.get('high_cardinality', [])) if isinstance(issues.get('high_cardinality', []), list) else 0)

    c5, c6, c7 = st.columns(3)
    c5.metric('Outlier cols (Z-score)', len(out_z) if isinstance(out_z, dict) else 0)
    c6.metric('Constant/near-constant', len(issues.get('constant_features', [])) if isinstance(issues.get('constant_features', []), list) else 0)
    c7.metric('Class imbalance', 'Yes' if 'class_imbalance' in issues else 'No')


def _render_eda(df: pd.DataFrame, *, target_col: Any, test_ratio: float):
    st.subheader('Automated EDA')

    st.caption('Charts are generated automatically; use expanders to focus on what you need.')

    missing_df = missing_value_analysis(df)
    with st.expander('Missing value analysis', expanded=True):
        st.pyplot(plot_missing_values(missing_df), clear_figure=True)
        if not missing_df.empty:
            st.dataframe(missing_df, use_container_width=True)

    corr_df = correlation_matrix(df)
    with st.expander('Correlation matrix (numeric)', expanded=False):
        st.pyplot(plot_correlation_heatmap(corr_df), clear_figure=True)

    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    cat_cols = [c for c in df.columns if c not in numeric_cols]

    st.markdown('**Distributions**')
    with st.expander('Numeric feature distribution', expanded=False):
        if numeric_cols:
            num_choice = st.selectbox('Select numeric feature', options=numeric_cols)
            st.pyplot(plot_numerical_distribution(df, num_choice), clear_figure=True)
        else:
            st.info('No numeric features found.')

    with st.expander('Categorical feature bar plot', expanded=False):
        if cat_cols:
            cat_choice = st.selectbox('Select categorical feature', options=cat_cols)
            st.pyplot(plot_categorical_distribution(df, cat_choice), clear_figure=True)
        else:
            st.info('No categorical features found.')

    with st.expander('Outlier visualization (boxplot)', expanded=False):
        if numeric_cols:
            out_choice = st.selectbox('Select numeric feature for boxplot', options=numeric_cols)
            st.pyplot(plot_outlier_boxplot(df, out_choice), clear_figure=True)
        else:
            st.info('No numeric features found.')

    with st.expander('Outlier detection summary (IQR / Z-score)', expanded=False):
        if numeric_cols:
            iqr_df = outlier_summary_iqr(df)
            z_df = outlier_summary_zscore(df, threshold=3.0)

            st.markdown('**IQR rule (1.5×IQR)**')
            st.pyplot(plot_outlier_counts(iqr_df, title='Outliers (%) by feature — IQR'), clear_figure=True)
            if not iqr_df.empty:
                st.dataframe(iqr_df[['column', 'outlier_count', 'outlier_pct']], use_container_width=True)
            else:
                st.info('No IQR outliers detected in numeric features.')

            st.markdown('**Z-score (|z| > 3.0)**')
            st.pyplot(plot_outlier_counts(z_df, title='Outliers (%) by feature — Z-score'), clear_figure=True)
            if not z_df.empty:
                st.dataframe(z_df[['column', 'outlier_count', 'outlier_pct']], use_container_width=True)
            else:
                st.info('No Z-score outliers detected in numeric features.')
        else:
            st.info('No numeric features found.')

    with st.expander('Train/test split summary', expanded=False):
        try:
            summary = train_test_split_summary(df, target_col=target_col, test_size=float(test_ratio))
            c1, c2, c3 = st.columns(3)
            c1.metric('Train rows', int(summary['train_rows']))
            c2.metric('Test rows', int(summary['test_rows']))
            c3.metric('Features (X)', int(summary['n_features']))

            st.caption('Stratified split summary (same split logic used for training).')
            st.dataframe(
                pd.DataFrame(
                    {
                        'train': pd.Series(summary['train_class_counts']),
                        'test': pd.Series(summary['test_class_counts']),
                    }
                ).fillna(0).astype(int),
                use_container_width=True,
            )
        except Exception as e:
            st.warning(f'Unable to compute a stratified split summary: {e}')


def _detect_issues(df: pd.DataFrame, target_col: str) -> dict[str, Any]:
    # Prefer the structured API from Module3. (Keeps the summary keys stable for other code paths.)
    try:
        return detect_issues(df, target_column=target_col)
    except Exception:
        # Fallback to legacy behavior if anything goes wrong.
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
    st.subheader('Issue detection & approval')
    issues = _detect_issues(df, target_col)

    findings = issues.get('findings', []) if isinstance(issues, dict) else []

    if not issues:
        st.success('No major data quality issues detected with default rules.')
    else:
        if findings:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            'issue': f.get('key'),
                            'severity': f.get('severity'),
                            'affected_cols': len(f.get('affected_columns', []) or []),
                            'title': f.get('title'),
                        }
                        for f in findings
                    ]
                ),
                use_container_width=True,
            )
        else:
            # Backwards-compatible display
            issue_rows = []
            for k, v in issues.items():
                if k == 'findings':
                    continue
                if isinstance(v, dict):
                    issue_rows.append({'issue': k, 'count': len(v), 'notes': 'dictionary details'})
                elif isinstance(v, list):
                    issue_rows.append({'issue': k, 'count': len(v), 'notes': 'feature list'})
                else:
                    issue_rows.append({'issue': k, 'count': None, 'notes': str(v)})
            st.dataframe(pd.DataFrame(issue_rows), use_container_width=True)

        with st.expander('Issue details', expanded=False):
            if findings:
                for f in findings:
                    sev = (f.get('severity') or 'warning').lower()
                    title = f.get('title') or f.get('key')
                    cols = f.get('affected_columns', []) or []
                    desc = f.get('description') or ''
                    fixes = f.get('suggested_fixes', []) or []

                    if sev == 'critical':
                        st.error(title)
                    elif sev == 'info':
                        st.info(title)
                    else:
                        st.warning(title)

                    if desc:
                        st.write(desc)
                    if cols:
                        st.caption('Affected columns')
                        st.write(', '.join([str(c) for c in cols[:25]]) + ('' if len(cols) <= 25 else f' (+{len(cols) - 25} more)'))
                    if fixes:
                        st.caption('Suggested fixes')
                        for fx in fixes:
                            st.write(f"- {fx}")
            else:
                for k, v in issues.items():
                    if k == 'findings':
                        continue
                    st.warning(f"Detected: {k}")
                    st.write(v)

    # Recognition over recall: recommend sensible defaults derived from detected issues.
    stored_choices: dict[str, Any] = st.session_state.get('preprocess_choices') or {}

    def _safe_index(options: list[str], value: Any, fallback: int = 0) -> int:
        try:
            return options.index(value)
        except Exception:
            return int(fallback)

    recommended: dict[str, Any] = {
        'numeric_impute_ui': 'median',
        'categorical_impute_ui': 'most_frequent',
        'scaling_ui': 'standard',
        'encoding_ui': 'onehot',
        'outlier_action_ui': 'no_action',
        'outlier_method_ui': 'iqr',
    }

    try:
        issue_keys = set(str(f.get('key')) for f in findings if f.get('key'))
        if 'missing_values' in issue_keys:
            recommended['numeric_impute_ui'] = 'median'
            recommended['categorical_impute_ui'] = 'most_frequent'
        if 'outliers_iqr' in issue_keys or 'outliers_zscore' in issue_keys:
            recommended['outlier_action_ui'] = 'cap_iqr'
            recommended['outlier_method_ui'] = 'iqr'
        if 'high_cardinality' in issue_keys:
            recommended['encoding_ui'] = 'auto'
    except Exception:
        pass

    with st.expander('Recommended preprocessing (based on detected issues)', expanded=bool(findings)):
        st.write('- Missing values → numeric: ' + str(recommended['numeric_impute_ui']) + ', categorical: ' + str(recommended['categorical_impute_ui']))
        st.write('- Encoding → ' + str(recommended['encoding_ui']))
        st.write('- Scaling → ' + str(recommended['scaling_ui']))
        st.write('- Outliers → ' + str(recommended['outlier_action_ui']) + ' (method: ' + str(recommended['outlier_method_ui']) + ')')

        c1, c2 = st.columns(2)
        with c1:
            if st.button('Use recommended defaults'):
                st.session_state['pp_numeric_impute_ui'] = recommended['numeric_impute_ui']
                st.session_state['pp_categorical_impute_ui'] = recommended['categorical_impute_ui']
                st.session_state['pp_scaling_ui'] = recommended['scaling_ui']
                st.session_state['pp_encoding_ui'] = recommended['encoding_ui']
                st.session_state['pp_outlier_action_ui'] = recommended['outlier_action_ui']
                st.session_state['pp_outlier_method_ui'] = recommended['outlier_method_ui']
                st.rerun()
        with c2:
            if stored_choices and st.button('Use last applied settings'):
                st.session_state['pp_numeric_impute_ui'] = stored_choices.get('numeric_impute', 'median')
                st.session_state['pp_categorical_impute_ui'] = stored_choices.get('categorical_impute', 'most_frequent')
                st.session_state['pp_scaling_ui'] = stored_choices.get('scaling', 'standard')
                st.session_state['pp_encoding_ui'] = stored_choices.get('encoding', 'onehot')
                st.session_state['pp_outlier_action_ui'] = stored_choices.get('outlier_action', 'no_action')
                st.session_state['pp_outlier_method_ui'] = stored_choices.get('outlier_method', 'iqr')
                st.rerun()

    st.markdown('**Choose preprocessing options (applied only when you click “Apply preprocessing”)**')

    with st.form('preprocess_form', clear_on_submit=False):
        st.caption(
            'Effect summary: Outlier handling (cap/remove rows) and dropping all-missing feature columns happen immediately on Apply. '
            'Imputation, scaling, and encoding are applied inside the training Pipeline (leakage-safe).'
        )
        c1, c2 = st.columns(2)

        with c1:
            numeric_impute = st.selectbox(
                'Missing values (numeric):',
                options=['mean', 'median', 'mode', 'constant'],
                index=_safe_index(
                    ['mean', 'median', 'mode', 'constant'],
                    st.session_state.get('pp_numeric_impute_ui', stored_choices.get('numeric_impute', recommended['numeric_impute_ui'])),
                    fallback=1,
                ),
                help='How to impute missing values in numeric columns.',
                key='pp_numeric_impute_ui',
            )
            numeric_fill_value = None
            if numeric_impute == 'constant':
                numeric_fill_value = st.number_input('Numeric constant value', value=0.0, step=1.0)

            scaling = st.selectbox(
                'Scaling:',
                options=['standard', 'minmax', 'none'],
                index=_safe_index(
                    ['standard', 'minmax', 'none'],
                    st.session_state.get('pp_scaling_ui', stored_choices.get('scaling', recommended['scaling_ui'])),
                    fallback=0,
                ),
                help='StandardScaler or MinMaxScaler (or no scaling).',
                key='pp_scaling_ui',
            )
            outlier_method = st.selectbox(
                'Outlier detection method:',
                options=['iqr', 'zscore'],
                index=_safe_index(
                    ['iqr', 'zscore'],
                    st.session_state.get('pp_outlier_method_ui', stored_choices.get('outlier_method', recommended['outlier_method_ui'])),
                    fallback=0,
                ),
                help='Used to decide which rows/values are considered outliers.',
                key='pp_outlier_method_ui',
            )

        with c2:
            categorical_impute = st.selectbox(
                'Missing values (categorical):',
                options=['most_frequent', 'constant'],
                index=_safe_index(
                    ['most_frequent', 'constant'],
                    st.session_state.get('pp_categorical_impute_ui', stored_choices.get('categorical_impute', recommended['categorical_impute_ui'])),
                    fallback=0,
                ),
                help='How to impute missing values in categorical columns.',
                key='pp_categorical_impute_ui',
            )
            categorical_fill_value = None
            if categorical_impute == 'constant':
                categorical_fill_value = st.text_input('Categorical constant value', value='missing')

            encoding = st.selectbox(
                'Categorical encoding:',
                options=['onehot', 'ordinal', 'auto'],
                index=_safe_index(
                    ['onehot', 'ordinal', 'auto'],
                    st.session_state.get('pp_encoding_ui', stored_choices.get('encoding', recommended['encoding_ui'])),
                    fallback=0,
                ),
                help='One-Hot Encoding or Ordinal Encoding.',
                key='pp_encoding_ui',
            )
            outlier_action = st.selectbox(
                'Outlier handling:',
                options=['no_action', 'cap_iqr', 'remove_rows'],
                index=_safe_index(
                    ['no_action', 'cap_iqr', 'remove_rows'],
                    st.session_state.get('pp_outlier_action_ui', stored_choices.get('outlier_action', recommended['outlier_action_ui'])),
                    fallback=0,
                ),
                help='Cap using IQR bounds, remove outlier rows, or take no action.',
                key='pp_outlier_action_ui',
            )

        st.divider()
        c_apply_1, c_apply_2 = st.columns([3, 2])
        with c_apply_1:
            st.warning('Applying preprocessing will clear any existing training results.', icon=None)
        with c_apply_2:
            apply_now = st.form_submit_button('Apply preprocessing (with approval)')

    numeric_impute_strategy = (
        'median' if numeric_impute == 'median'
        else ('mean' if numeric_impute == 'mean' else ('most_frequent' if numeric_impute == 'mode' else 'constant'))
    )

    choices = PreprocessChoices(
        numeric_impute=numeric_impute_strategy,
        categorical_impute='most_frequent' if categorical_impute == 'most_frequent' else 'constant',
        numeric_fill_value=float(numeric_fill_value) if numeric_fill_value is not None else None,
        categorical_fill_value=str(categorical_fill_value) if categorical_fill_value is not None else None,
        scaling='none' if scaling == 'none' else scaling,
        encoding=encoding,
        outlier_action=outlier_action,
        outlier_method=outlier_method,
    )
    if not apply_now:
        # Commit-based UX: training should rely on the last applied choices.
        if 'preprocess_choices' in st.session_state:
            try:
                stored = st.session_state.get('preprocess_choices') or {}
                choices = PreprocessChoices(**stored)
                st.session_state['preprocess_approved'] = True
                st.info('Using previously applied preprocessing choices. Change settings and click “Apply preprocessing” to update.')
            except Exception:
                st.session_state['preprocess_approved'] = False
        else:
            st.session_state['preprocess_approved'] = False
            st.info('Preprocessing has not been applied yet. Review settings and click “Apply preprocessing (with approval)” to continue to training.')
        return df, choices, issues

    # Preprocessing changes the dataset; clear any previously trained artifacts to avoid stale results.
    try:
        for k in ['trained_models', 'evaluation_results', 'preprocess_choices', 'best_model_name']:
            if k in st.session_state:
                del st.session_state[k]
        st.info('Cleared previous training results (dataset changed after preprocessing).')
    except Exception:
        pass

    before_df = df.copy()

    # Outlier handling (simple, dataset-level) BEFORE split.
    try:
        before_df, _outlier_summary = apply_outlier_handling(
            before_df,
            action=choices.outlier_action,
            method=choices.outlier_method,
            exclude_columns=[target_col],
        )
    except Exception:
        # Outlier handling should never block the user; any hard failures become no-ops.
        pass

    # Auto-clean: drop feature columns that are entirely missing.
    # (These break many sklearn preprocessors and carry no signal.)
    try:
        all_missing_cols = get_all_missing_columns(before_df, exclude=[target_col])
        if all_missing_cols:
            before_df = before_df.drop(columns=all_missing_cols)
            preview = ', '.join([str(c) for c in all_missing_cols[:8]])
            more = '' if len(all_missing_cols) <= 8 else f' (+{len(all_missing_cols) - 8} more)'
            st.info(f'Automatically dropped {len(all_missing_cols)} all-missing column(s): {preview}{more}.')
    except Exception:
        pass

    with st.expander('Before vs after (preview)', expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            st.write('Before')
            st.dataframe(df.head(10), use_container_width=True)
        with c2:
            st.write('After')
            st.dataframe(before_df.head(10), use_container_width=True)
    st.success('Preprocessing choices stored for training. (Imputation/encoding/scaling are applied via a sklearn Pipeline during training.)')

    # Persist the applied choices so progress/status is accurate and training can be safely gated.
    try:
        st.session_state['preprocess_choices'] = choices.__dict__
        st.session_state['issues'] = issues
        st.session_state['preprocess_approved'] = True
        st.session_state['last_preprocess_applied_at'] = time.time()
    except Exception:
        pass

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
    st.caption('Upload → understand → approve preprocessing → train/tune → compare → export')

    with st.sidebar:
        st.header('Configuration')
        st.caption('Tip: adjust settings → apply → then train.')

        default_cfg = {
            'primary_metric': 'f1',
            'search_type': 'grid',
            'test_ratio': 0.2,
            'cv': 3,
        }
        if 'train_cfg' not in st.session_state:
            st.session_state['train_cfg'] = default_cfg

        cfg = st.session_state.get('train_cfg', default_cfg)

        model_options = [
            'Logistic Regression',
            'K-Nearest Neighbors',
            'Decision Tree',
            'Naive Bayes',
            'Random Forest',
            'Support Vector Machine',
            'Rule-based (Most Frequent)',
        ]
        if 'selected_models' not in st.session_state:
            st.session_state['selected_models'] = model_options

        with st.form('training_config_form', clear_on_submit=False):
            primary_metric = st.selectbox(
                'Primary ranking metric',
                options=['f1', 'accuracy', 'precision', 'recall'],
                index=['f1', 'accuracy', 'precision', 'recall'].index(cfg.get('primary_metric', 'f1')),
                key='cfg_primary_metric',
            )
            search_type = st.selectbox(
                'Hyperparameter search',
                options=['grid', 'random', 'halving_grid', 'halving_random'],
                index=['grid', 'random', 'halving_grid', 'halving_random'].index(cfg.get('search_type', 'grid')),
                key='cfg_search_type',
            )
            test_ratio = st.slider(
                'Test split ratio',
                min_value=0.1,
                max_value=0.5,
                value=float(cfg.get('test_ratio', 0.2)),
                step=0.05,
                key='cfg_test_ratio',
            )
            cv = st.slider(
                'CV folds',
                min_value=3,
                max_value=5,
                value=int(cfg.get('cv', 3)),
                step=1,
                key='cfg_cv',
            )

            selected_models = st.multiselect(
                'Models to train',
                options=model_options,
                default=st.session_state.get('selected_models', model_options),
                key='cfg_selected_models',
            )

            apply_training_cfg = st.form_submit_button('Apply training settings')

        if apply_training_cfg:
            st.session_state['train_cfg'] = {
                'primary_metric': primary_metric,
                'search_type': search_type,
                'test_ratio': float(test_ratio),
                'cv': int(cv),
            }
            st.session_state['selected_models'] = selected_models
            st.success('Training settings applied.')

        cfg = st.session_state.get('train_cfg', default_cfg)
        primary_metric = cfg['primary_metric']
        search_type = cfg['search_type']
        test_ratio = float(cfg['test_ratio'])
        cv = int(cfg['cv'])
        selected_models = st.session_state.get('selected_models', model_options)

        st.divider()
        st.subheader('Progress')
        has_data = 'uploaded_bytes' in st.session_state
        has_training = 'evaluation_results' in st.session_state
        has_preprocess = bool(st.session_state.get('preprocess_choices'))
        st.write('1) Dataset:', 'Done' if has_data else 'Not yet')
        st.write('2) EDA:', 'Done' if has_data else 'Not yet')
        st.write('3) Preprocessing approval:', 'Done' if has_preprocess else 'Not yet')
        st.write('4) Training complete:', 'Done' if has_training else 'Not yet')

        with st.expander('Status snapshot', expanded=True):
            st.write('Dataset:', st.session_state.get('uploaded_name', '—'))
            st.write('Loaded at:', _format_ts(st.session_state.get('last_dataset_loaded_at')))
            st.write('Preprocessing applied at:', _format_ts(st.session_state.get('last_preprocess_applied_at')))
            st.write('Training run at:', _format_ts(st.session_state.get('last_training_ran_at')))
            st.write('Best model:', st.session_state.get('best_model_name', '—'))

        c1, c2 = st.columns(2)
        with c1:
            if st.button('Reset training results'):
                for k in ['trained_models', 'evaluation_results', 'best_model_name', 'last_training_ran_at']:
                    if k in st.session_state:
                        del st.session_state[k]
                st.rerun()
        with c2:
            if st.button('Reset preprocessing'):
                for k in ['preprocess_choices', 'preprocess_approved', 'last_preprocess_applied_at']:
                    if k in st.session_state:
                        del st.session_state[k]
                # Resetting preprocessing implies training results are stale.
                for k in ['trained_models', 'evaluation_results', 'best_model_name', 'last_training_ran_at']:
                    if k in st.session_state:
                        del st.session_state[k]
                st.rerun()

        if st.button('Reset session (start over)'):
            for k in [
                'uploaded_bytes', 'uploaded_name',
                'raw_df', 'working_df', 'working_df_sig',
                'trained_models', 'evaluation_results',
                'preprocess_choices', 'preprocess_approved',
                'issues', 'train_cfg', 'selected_models',
                'best_model_name',
                'last_dataset_loaded_at', 'last_preprocess_applied_at', 'last_training_ran_at',
            ]:
                if k in st.session_state:
                    del st.session_state[k]
            st.rerun()

    _section(1, 'Upload dataset', 'Supported format: CSV. Recommended: header row + clean column names.')
    uploaded = st.file_uploader('Upload a CSV file', type=['csv'])
    if uploaded is None and 'uploaded_bytes' not in st.session_state:
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
        if uploaded is not None:
            st.session_state['uploaded_bytes'] = uploaded.getvalue()
            st.session_state['uploaded_name'] = getattr(uploaded, 'name', 'uploaded.csv')
        df = _read_csv_from_bytes(st.session_state['uploaded_bytes'])
    except Exception as e:
        st.error(f'File Read Error: {e}')
        return

    # Persist dataset across reruns and keep user cleanup changes stable.
    sig = (st.session_state.get('uploaded_name'), len(st.session_state.get('uploaded_bytes', b'')))
    if st.session_state.get('working_df_sig') != sig:
        st.session_state['raw_df'] = df
        st.session_state['working_df'] = df
        st.session_state['working_df_sig'] = sig
        st.session_state['last_dataset_loaded_at'] = time.time()
        # New dataset => any prior approvals/results are stale.
        for k in ['trained_models', 'evaluation_results', 'preprocess_choices', 'preprocess_approved', 'issues', 'best_model_name', 'last_preprocess_applied_at', 'last_training_ran_at']:
            if k in st.session_state:
                del st.session_state[k]

    df = st.session_state.get('working_df', df)

    health = _health_check(df)
    if health['unnamed_columns']:
        st.warning(f"Unnamed columns detected: {health['unnamed_columns']}")
    if health['whitespace_columns']:
        st.warning(f"Columns with leading/trailing whitespace: {health['whitespace_columns']}")
    if health['non_ascii_columns']:
        st.warning(f"Columns with non-ASCII characters: {health['non_ascii_columns']}")

    _render_basic_info(df, target_col=None)

    st.divider()
    _section(2, 'Prepare dataset', 'Optional cleanup before analysis/training.')
    df = _render_preview_tools(df)
    st.session_state['working_df'] = df

    with st.expander('Dataset schema & cardinality', expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('**Schema**')
            st.dataframe(get_dataset_schema(df), use_container_width=True, height=260)
        with c2:
            st.markdown('**Cardinality buckets**')
            st.dataframe(get_cardinality_buckets(df), use_container_width=True, height=260)

    # Target selection: suggest likely candidates but keep user in control.
    cols = list(df.columns)
    col_by_str = {str(c): c for c in cols}
    suggested = infer_target_candidates(df)
    default_target = None
    for s in suggested:
        if s in col_by_str:
            default_target = col_by_str[s]
            break
    if default_target is None and cols:
        default_target = cols[-1]
    default_index = cols.index(default_target) if default_target in cols else 0

    target_col = st.selectbox(
        'Select target column (classification label)',
        options=cols,
        index=default_index,
        key='target_col',
        help='This is the label the models will learn to predict.',
    )

    tv = validate_target_column(df, target_col)
    if tv.warnings:
        for w in tv.warnings:
            st.warning(w)
    if not tv.ok:
        for e in tv.errors:
            st.error(e)
        st.info('Pick a different target column to continue.')
        return

    st.divider()
    _section(3, 'Understand dataset (EDA)', 'Automated charts and a one-page summary of key issues.')
    _render_eda_summary(df, target_col)
    _render_eda(df, target_col=target_col, test_ratio=float(test_ratio))

    st.divider()
    _section(4, 'Approve preprocessing', 'Review detected issues and confirm preprocessing decisions.')

    df_after, choices, issues = _render_issue_detection_and_choices(df, target_col)

    # Safety: preprocessing (especially outlier row removal) can change the target distribution.
    tv_after = validate_target_column(df_after, target_col)
    if tv_after.warnings:
        for w in tv_after.warnings:
            st.warning(w)
    if not tv_after.ok:
        for e in tv_after.errors:
            st.error(e)
        st.info('Preprocessing made the target invalid (e.g., only one class left). Adjust preprocessing or choose another target.')
        return

    # Safety: ensure at least one feature column remains.
    try:
        remaining_features = [c for c in df_after.columns if c != target_col]
        if len(remaining_features) == 0:
            st.error('No feature columns remain (dataset contains only the target). Add features or upload a dataset with predictors.')
            return
    except Exception:
        pass

    # Quality warning: constant feature columns add no predictive signal.
    try:
        const_cols = get_constant_columns(df_after, exclude=[target_col])
        if const_cols:
            preview = ', '.join([str(c) for c in const_cols[:8]])
            more = '' if len(const_cols) <= 8 else f' (+{len(const_cols) - 8} more)'
            st.warning(
                f'Found {len(const_cols)} constant feature(s): {preview}{more}. '
                "These won't help models and may indicate a data issue."
            )
    except Exception:
        pass

    # Guardrail: all-missing feature columns can break imputers/scalers.
    try:
        all_missing_cols = get_all_missing_columns(df_after, exclude=[target_col])
        if all_missing_cols:
            preview = ', '.join([str(c) for c in all_missing_cols[:8]])
            more = '' if len(all_missing_cols) <= 8 else f' (+{len(all_missing_cols) - 8} more)'
            st.error(
                f'Found {len(all_missing_cols)} feature column(s) with all values missing: {preview}{more}. '
                'Drop these columns or adjust preprocessing before training.'
            )
            return
    except Exception:
        pass

    # Quality warning: high-cardinality categoricals can explode feature space.
    try:
        feature_df = df_after.drop(columns=[target_col])
        high_card_after = detect_high_cardinality(feature_df)
        if high_card_after:
            preview = ', '.join([str(c) for c in high_card_after[:8]])
            more = '' if len(high_card_after) <= 8 else f' (+{len(high_card_after) - 8} more)'
            st.warning(
                f'High-cardinality categorical feature(s) detected after preprocessing: {preview}{more}. '
                'Consider ordinal encoding or dropping ID-like columns.'
            )
    except Exception:
        pass
    st.divider()

    _section(5, 'Train & compare models', 'Models are trained with your selected options from the sidebar.')
    st.subheader('Train/test split')
    try:
        X_train, X_test, y_train, y_test = split_train_test_stratified(df_after, target_col, test_size=float(test_ratio))
        c1, c2, c3, c4 = st.columns(4)
        c1.metric('Train samples', int(len(X_train)))
        c2.metric('Test samples', int(len(X_test)))
        c3.metric('Train ratio', float(1 - test_ratio))
        c4.metric('Test ratio', float(test_ratio))
    except Exception as e:
        st.error(f'Unable to split dataset (stratified): {e}')
        hint = _split_feasibility_hint(df_after, target_col, float(test_ratio))
        with st.expander('Fix this (recommended actions)', expanded=True):
            n_samples = hint.get('n_samples')
            n_classes = hint.get('n_classes')
            min_class = hint.get('min_class_count')
            st.write(f"Samples (non-null target): {n_samples}")
            st.write(f"Classes: {n_classes}")
            st.write(f"Smallest class size: {min_class}")

            rec = hint.get('recommended_test_ratio')
            if rec is not None:
                st.caption(f"Suggested test ratio (feasible minimum): {rec:.2f}")
                c1, c2 = st.columns(2)
                with c1:
                    if st.button(f'Set test ratio to {rec:.2f} and retry'):
                        cfg2 = dict(st.session_state.get('train_cfg', {}))
                        cfg2['test_ratio'] = float(rec)
                        st.session_state['train_cfg'] = cfg2
                        st.rerun()
                with c2:
                    if st.button('Set CV folds to 2 and retry'):
                        cfg2 = dict(st.session_state.get('train_cfg', {}))
                        cfg2['cv'] = 2
                        st.session_state['train_cfg'] = cfg2
                        st.rerun()
            else:
                st.info('Try increasing dataset size or reducing number of classes (merge rare classes) so stratification is possible.')

            if st.button('Train baseline only (Most Frequent)'):
                st.session_state['selected_models'] = ['Rule-based (Most Frequent)']
                st.rerun()
        return

    st.subheader('Training')
    st.caption('Tip: start with fewer models for faster iteration, then expand the list once the dataset looks good.')
    st.markdown('**Current settings (applied)**')
    applied_choices = st.session_state.get('preprocess_choices') or choices.__dict__
    st.write(
        f"Preprocessing: impute(num={applied_choices.get('numeric_impute')}, cat={applied_choices.get('categorical_impute')}), "
        f"scale={applied_choices.get('scaling')}, encode={applied_choices.get('encoding')}, "
        f"outliers={applied_choices.get('outlier_action')} ({applied_choices.get('outlier_method')})"
    )
    st.write(f"Training: metric={primary_metric}, search={search_type}, test_ratio={test_ratio}, cv={cv}, models={len(selected_models)}")
    st.caption('Estimated workload: ' + _estimate_workload_text(search_type, int(cv), int(len(selected_models))))

    disable_reasons: list[str] = []
    if not selected_models:
        disable_reasons.append('Select at least one model in the sidebar.')
    if not bool(st.session_state.get('preprocess_approved')):
        disable_reasons.append('Apply preprocessing (Step 4) before training.')

    run_training = st.button('Train models', type='primary', disabled=bool(disable_reasons))
    if disable_reasons:
        st.info('Training is disabled: ' + ' '.join(disable_reasons))
    if not run_training and 'evaluation_results' not in st.session_state:
        st.info('Click “Train models” to run training and evaluation.')
        return

    is_binary = y_train.nunique() == 2
    scoring = _metric_to_sklearn_scoring(primary_metric, is_binary=is_binary)

    preprocessor = build_preprocessor(
        X_train,
        numeric_impute=choices.numeric_impute,
        categorical_impute=choices.categorical_impute,
        numeric_fill_value=choices.numeric_fill_value,
        categorical_fill_value=choices.categorical_fill_value,
        scaling=choices.scaling,
        encoding=choices.encoding,
    )

    class_weight_auto = bool(st.session_state.get('issues', {}).get('class_imbalance'))

    if run_training:
        with st.spinner('Training models (this may take a bit)...'):
            trained = train_and_optimize_models(
                X_train,
                y_train,
                search_type=search_type,
                cv=int(cv),
                scoring=scoring,
                include_models=selected_models,
                preprocessor=preprocessor,
                n_jobs=-1,
                cache=True,
                class_weight_auto=class_weight_auto,
            )

            evals = evaluate_models(trained, X_test, y_test)
            st.session_state['trained_models'] = trained
            st.session_state['evaluation_results'] = evals
            st.session_state['preprocess_choices'] = choices.__dict__
            st.session_state['issues'] = issues
            st.session_state['last_training_ran_at'] = time.time()

    evals = st.session_state['evaluation_results']
    results_list = [
        {
            'model': k,
            'accuracy': v.get('accuracy'),
            'precision': v.get('precision'),
            'recall': v.get('recall'),
            'f1_score': v.get('f1_score'),
            'roc_auc': v.get('roc_auc'),
            'cv_mean': v.get('cv_mean'),
            'cv_std': v.get('cv_std'),
            'training_time': v.get('training_time'),
            'best_params': v.get('best_params'),
            'error': v.get('error'),
        }
        for k, v in evals.items()
    ]
    comparison_df = show_comparison_table(results_list)

    st.subheader('Model Comparison Dashboard')

    metric_map = {'f1': 'f1_score', 'accuracy': 'accuracy', 'precision': 'precision', 'recall': 'recall'}
    sort_metric = metric_map[primary_metric]
    ranked = rank_algorithms(comparison_df.reset_index(drop=True), sort_metric)

    valid_ranked = ranked[ranked[sort_metric].notna()]
    best_name = valid_ranked.iloc[0]['model'] if not valid_ranked.empty else None
    try:
        st.session_state['best_model_name'] = best_name
    except Exception:
        pass

    # Actionable diagnostics if training produced no successful models.
    if best_name is None:
        st.error('No models successfully trained. Review errors below and try a simpler configuration.')
        with st.expander('Fix this (recommended actions)', expanded=True):
            c1, c2, c3 = st.columns(3)
            with c1:
                if st.button('Use baseline only'):
                    st.session_state['selected_models'] = ['Rule-based (Most Frequent)']
                    # Clear results so the user is prompted to retrain.
                    for k in ['trained_models', 'evaluation_results', 'best_model_name', 'last_training_ran_at']:
                        if k in st.session_state:
                            del st.session_state[k]
                    st.rerun()
            with c2:
                if st.button('Switch search to grid'):
                    cfg2 = dict(st.session_state.get('train_cfg', {}))
                    cfg2['search_type'] = 'grid'
                    st.session_state['train_cfg'] = cfg2
                    for k in ['trained_models', 'evaluation_results', 'best_model_name', 'last_training_ran_at']:
                        if k in st.session_state:
                            del st.session_state[k]
                    st.rerun()
            with c3:
                if st.button('Set CV folds to 2'):
                    cfg2 = dict(st.session_state.get('train_cfg', {}))
                    cfg2['cv'] = 2
                    st.session_state['train_cfg'] = cfg2
                    for k in ['trained_models', 'evaluation_results', 'best_model_name', 'last_training_ran_at']:
                        if k in st.session_state:
                            del st.session_state[k]
                    st.rerun()

    if best_name is not None:
        top_row = valid_ranked.iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric('Best model', str(best_name))
        c2.metric(f'Best {primary_metric}', float(top_row[sort_metric]) if pd.notna(top_row[sort_metric]) else None)
        c3.metric('ROC-AUC', float(top_row.get('roc_auc')) if pd.notna(top_row.get('roc_auc')) else None)
        c4.metric('Train time (s)', float(top_row.get('training_time')) if pd.notna(top_row.get('training_time')) else None)

        display_df = comparison_df.reset_index(drop=True).copy()
        display_df['best'] = display_df['model'].apply(lambda m: '⭐' if m == best_name else '')
        st.dataframe(display_df, use_container_width=True, height=260)
    else:
        st.dataframe(comparison_df, use_container_width=True, height=260)
    with st.expander(f'Ranking by {sort_metric}', expanded=False):
        st.dataframe(ranked, use_container_width=True)

    st.pyplot(plot_metric_bars(comparison_df.reset_index(drop=True), sort_metric), clear_figure=True)

    st.download_button(
        'Download results as CSV',
        data=comparison_df.reset_index(drop=True).to_csv(index=False).encode('utf-8'),
        file_name='model_comparison.csv',
        mime='text/csv',
    )

    # Show confusion matrices
    st.subheader('Confusion matrices')
    classes_sorted = sorted(pd.unique(y_test))
    for model_name, v in evals.items():
        if v.get('confusion_matrix') is None:
            continue
        with st.expander(f'{model_name}', expanded=(model_name == best_name)):
            cm = np.array(v['confusion_matrix'])

            import matplotlib.pyplot as plt
            import seaborn as sns

            # Binary: show a simple 2x2 quadrant matrix with explicit labels
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                annot = np.array([
                    [f"TN\n{tn}", f"FP\n{fp}"],
                    [f"FN\n{fn}", f"TP\n{tp}"],
                ])
                fig, ax = plt.subplots(figsize=(4.5, 3.5))
                sns.heatmap(
                    cm,
                    annot=annot,
                    fmt='',
                    cmap='Blues',
                    cbar=False,
                    linewidths=1,
                    linecolor='white',
                    square=True,
                    ax=ax,
                )
                ax.set_title('Confusion Matrix (2×2)')
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                tick_labels = [str(classes_sorted[0]), str(classes_sorted[1])] if len(classes_sorted) == 2 else ['0', '1']
                ax.set_xticklabels(tick_labels)
                ax.set_yticklabels(tick_labels, rotation=0)
                fig.tight_layout()
                st.pyplot(fig, clear_figure=True)
            else:
                fig, ax = plt.subplots(figsize=(6, 4))
                n = cm.shape[0]
                if n <= 10 and len(classes_sorted) == n:
                    labels = [str(c) for c in classes_sorted]
                    sns.heatmap(
                        cm,
                        annot=True,
                        fmt='d',
                        cmap='Blues',
                        ax=ax,
                        xticklabels=labels,
                        yticklabels=labels,
                    )
                    ax.tick_params(axis='x', rotation=45)
                elif n <= 25 and len(classes_sorted) == n:
                    labels = [str(c) for c in classes_sorted]
                    sns.heatmap(
                        cm,
                        annot=False,
                        cmap='Blues',
                        ax=ax,
                        xticklabels=labels,
                        yticklabels=labels,
                    )
                    ax.tick_params(axis='x', rotation=45)
                else:
                    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', ax=ax)
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    st.caption(f'Confusion matrix has {n} classes; tick labels hidden for readability.')
                ax.set_title('Confusion Matrix')
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                fig.tight_layout()
                st.pyplot(fig, clear_figure=True)

    # Best model + downloads
    # ROC curve (binary only)
    if is_binary:
        st.subheader('ROC curves (binary classification)')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot([0, 1], [0, 1], linestyle='--', label='Chance')
        plotted_any = False
        for model_name, model_info in st.session_state.get('trained_models', {}).items():
            model = model_info.get('model')
            if model is None or not hasattr(model, 'predict_proba'):
                continue
            try:
                proba = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, proba)
                ax.plot(fpr, tpr, label=model_name)
                plotted_any = True
            except Exception:
                continue
        if plotted_any:
            ax.set_title('ROC Curves')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.legend(fontsize=8)
            fig.tight_layout()
            st.pyplot(fig, clear_figure=True)
        else:
            st.info('No ROC curves available (models may not support probabilities).')

    st.divider()
    _section(6, 'Export & report', 'Download results, the report, and the best trained pipeline.')
    st.subheader('Auto-generated final report')
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

    try:
        st.download_button(
            'Download report (PDF)',
            data=export_report_as_pdf_bytes(report_sections, title='CS-245 AutoML Report'),
            file_name='automl_report.pdf',
            mime='application/pdf',
        )
    except Exception:
        # If the PDF backend isn't installed, keep the UI working.
        st.info('PDF export is unavailable (install fpdf2 to enable it).')

    if best_name and st.session_state.get('trained_models', {}).get(best_name, {}).get('model') is not None:
        # Export winning model pipeline as a single object (already includes preprocessing)
        export_pipeline = st.session_state['trained_models'][best_name]['model']
        buffer = io.BytesIO()
        joblib.dump(export_pipeline, buffer)
        st.download_button(
            'Download best model (pickle/joblib)',
            data=buffer.getvalue(),
            file_name='best_model.joblib',
            mime='application/octet-stream',
        )