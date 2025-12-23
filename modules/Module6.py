# Module 6: Model Comparison Dashboard
# Show comparison table of metrics
# Rank algorithms by chosen metric
# Allow CSV download of results
# Visualize metrics using bar charts


"""Module 6: Model Comparison Dashboard.

This module is UI-framework agnostic: it returns DataFrames and matplotlib Figures
that can be rendered in Streamlit via st.dataframe / st.pyplot.
"""

from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# 1. Show comparison table of metrics
def show_comparison_table(results) -> pd.DataFrame:
    """
    Display a comparison table of model performance metrics.

    Parameters:
    results (list of dict): List containing dictionaries of model metrics.

    Returns:
    pd.DataFrame: DataFrame containing the comparison table.
    """
    comparison_df = pd.DataFrame(results or [])
    if not comparison_df.empty and 'model' in comparison_df.columns:
        comparison_df = comparison_df.set_index('model', drop=False)
    return comparison_df

# 2. Rank algorithms by chosen metric
def rank_algorithms(comparison_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    Rank algorithms based on a chosen performance metric.

    Parameters:
    comparison_df (pd.DataFrame): DataFrame containing model metrics.
    metric (str): The metric to rank by (e.g., 'accuracy', 'f1_score').

    Returns:
    pd.DataFrame: DataFrame sorted by the chosen metric in descending order.
    """
    if not isinstance(comparison_df, pd.DataFrame):
        raise TypeError('comparison_df must be a pandas DataFrame')
    if comparison_df.empty:
        return comparison_df.copy()
    if metric not in comparison_df.columns:
        raise ValueError(f"Metric column {metric!r} not found")

    ranked_df = comparison_df.sort_values(by=metric, ascending=False, na_position='last').reset_index(drop=True)
    return ranked_df

# 3. Allow CSV download of results
def save_results_to_csv(comparison_df: pd.DataFrame, file_path: str) -> None:
    """
    Save the comparison results to a CSV file.

    Parameters:
    comparison_df (pd.DataFrame): DataFrame containing model metrics.
    file_path (str): The file path to save the CSV.
    """
    comparison_df.to_csv(file_path, index=False)


def results_to_csv_bytes(comparison_df: pd.DataFrame) -> bytes:
    """Convert comparison results to CSV bytes (UTF-8), suitable for Streamlit downloads."""

    if not isinstance(comparison_df, pd.DataFrame):
        raise TypeError('comparison_df must be a pandas DataFrame')
    return comparison_df.to_csv(index=False).encode('utf-8')

# 4. Visualize metrics using bar charts
def plot_metric_bars(comparison_df: pd.DataFrame, metric: str):
    """
    Plot a bar chart of model performance metrics.

    Parameters:
    comparison_df (pd.DataFrame): DataFrame containing model metrics.
    metric (str): The metric to visualize (e.g., 'accuracy', 'f1_score').
    """

    if not isinstance(comparison_df, pd.DataFrame):
        raise TypeError('comparison_df must be a pandas DataFrame')

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_df = comparison_df.copy()
    if plot_df.empty:
        ax.set_title('Model Comparison')
        ax.set_xlabel('Model')
        ax.set_ylabel(metric.replace('_', ' ').title())
        fig.tight_layout()
        return fig

    if 'model' not in plot_df.columns:
        plot_df = plot_df.reset_index()
    if metric not in plot_df.columns:
        raise ValueError(f"Metric column {metric!r} not found")

    # Drop NaNs to avoid seaborn warnings and empty plots.
    plot_df = plot_df.dropna(subset=[metric])
    if plot_df.empty:
        ax.set_title(f'Model Comparison by {metric.replace("_", " ").title()}')
        ax.set_xlabel('Model')
        ax.set_ylabel(metric.replace('_', ' ').title())
        fig.tight_layout()
        return fig

    sns.barplot(x='model', y=metric, data=plot_df, ax=ax)
    ax.set_title(f'Model Comparison by {metric.replace("_", " ").title()}')
    ax.set_xlabel('Model')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.tick_params(axis='x', rotation=45)
    fig.tight_layout()
    return fig


def plot_decision_tree_viz(pipeline, feature_names=None, class_names=None, max_depth=3):
    """
    Plot the decision tree from a fitted pipeline (if the final step is a decision tree).
    
    Parameters:
    pipeline: Fitted sklearn Pipeline.
    feature_names (list): List of feature names (optional).
    class_names (list): List of class names (optional).
    max_depth (int): Maximum depth to display to keep the plot readable.
    """
    from sklearn.tree import plot_tree, DecisionTreeClassifier
    
    if not hasattr(pipeline, 'steps'):
        return None
        
    # Assume the last step is the classifier
    classifier = pipeline.steps[-1][1]
    
    if not isinstance(classifier, DecisionTreeClassifier):
        return None
        
    fig, ax = plt.subplots(figsize=(20, 10))
    plot_tree(
        classifier, 
        max_depth=max_depth, 
        feature_names=feature_names, 
        class_names=class_names, 
        filled=True, 
        rounded=True, 
        fontsize=10, 
        ax=ax
    )
    ax.set_title(f"Decision Tree Visualization (Max Depth: {max_depth})")
    fig.tight_layout()
    return fig


def plot_feature_importance(model, feature_names, top_n=20):
    """
    Plot feature importance for a given model.
    Support tree-based models (feature_importances_) and linear models (coef_).
    """
    import numpy as np
    
    # Handle pipeline: extract the final estimator
    if hasattr(model, 'steps'):
        model = model.steps[-1][1]

    if not hasattr(model, 'feature_importances_') and not hasattr(model, 'coef_'):
        return None

    importances = None
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        # For linear models, take absolute coefficient values (magnitude)
        # Handle LogisticRegression which might have coefs for multiple classes
        if model.coef_.ndim > 1:
            importances = np.abs(model.coef_).mean(axis=0)
        else:
            importances = np.abs(model.coef_)

    if importances is None:
        return None

    # Sort and select top N
    indices = np.argsort(importances)[::-1]
    if top_n:
        indices = indices[:top_n]

    # Map indices to names if provided
    if feature_names is not None and len(feature_names) == len(importances):
        names = [feature_names[i] for i in indices]
    else:
        names = [f"Feature {i}" for i in indices]

    values = importances[indices]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=values, y=names, ax=ax, palette='viridis')
    ax.set_title(f'Feature Importance (Top {len(values)})')
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    fig.tight_layout()
    return fig
