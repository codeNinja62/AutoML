# Module 2: Automated Exploratory Data Analysis (EDA)
# Missing value analysis
# Outlier detection (IQR / Z-score)
# Correlation matrix
# Numerical feature distributions
# Categorical feature bar plots
# Train/test split summary


# Imports
import matplotlib.pyplot as plt
import seaborn as sns


# 1. Missing value analysis
def missing_value_analysis(df):
    return df.isnull().sum()

# 2. Outlier detection using IQR
def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers

# 3. Correlation matrix
def correlation_matrix(df):
    return df.corr()

# 4. Numerical feature distributions
def plot_numerical_distributions(df, numerical_columns):
    for column in numerical_columns:
        plt.figure(figsize=(8, 4))
        sns.histplot(df[column], kde=True)
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.show()