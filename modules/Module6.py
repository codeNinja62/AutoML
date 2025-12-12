# Module 6: Model Comparison Dashboard
# Show comparison table of metrics
# Rank algorithms by chosen metric
# Allow CSV download of results
# Visualize metrics using bar charts


# Imports
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# 1. Show comparison table of metrics
def show_comparison_table(results):
    """
    Display a comparison table of model performance metrics.

    Parameters:
    results (list of dict): List containing dictionaries of model metrics.

    Returns:
    pd.DataFrame: DataFrame containing the comparison table.
    """
    comparison_df = pd.DataFrame(results)
    return comparison_df

# 2. Rank algorithms by chosen metric
def rank_algorithms(comparison_df, metric):
    """
    Rank algorithms based on a chosen performance metric.

    Parameters:
    comparison_df (pd.DataFrame): DataFrame containing model metrics.
    metric (str): The metric to rank by (e.g., 'accuracy', 'f1_score').

    Returns:
    pd.DataFrame: DataFrame sorted by the chosen metric in descending order.
    """
    ranked_df = comparison_df.sort_values(by=metric, ascending=False).reset_index(drop=True)
    return ranked_df

# 3. Allow CSV download of results
def save_results_to_csv(comparison_df, file_path):
    """
    Save the comparison results to a CSV file.

    Parameters:
    comparison_df (pd.DataFrame): DataFrame containing model metrics.
    file_path (str): The file path to save the CSV.
    """
    comparison_df.to_csv(file_path, index=False)

# 4. Visualize metrics using bar charts
def plot_metric_bars(comparison_df, metric):
    """
    Plot a bar chart of model performance metrics.

    Parameters:
    comparison_df (pd.DataFrame): DataFrame containing model metrics.
    metric (str): The metric to visualize (e.g., 'accuracy', 'f1_score').
    """

    plt.figure(figsize=(10, 6))
    sns.barplot(x='model', y=metric, data=comparison_df)
    plt.title(f'Model Comparison by {metric.capitalize()}')
    plt.xlabel('Model')
    plt.ylabel(metric.capitalize())
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    high_cardinality = detect_high_cardinality(df)
    if high_cardinality:
        warnings['high_cardinality'] = high_cardinality
    constant_features = detect_constant_features(df)
    if constant_features:
        warnings['constant_features'] = constant_features
        return warnings
    return plt
