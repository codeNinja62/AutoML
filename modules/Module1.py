# Module 1: Dataset Upload and Basic Information
# Upload CSV file
# Show number of rows/columns
# Display column types
# Show summary statistics
# Display class distribution


# Imports
import pandas as pd


# 1. Upload CSV file
def upload_csv(file_path):
    df = pd.read_csv(file_path)
    return df

# 2. Show number of rows/columns
def get_shape(df):
    return df.shape

# 3. Display column types
def get_column_types(df):
    return df.dtypes

# 4. Show summary statistics
def get_summary_statistics(df):
    return df.describe()

# 5. Display class distribution
def get_class_distribution(df, target_column):
    return df[target_column].value_counts()