# Project Modules Overview

## Module 1: Dataset Upload and Basic Information
- Upload CSV file
- Show number of rows/columns
- Display column types
- Show summary statistics
- Display class distribution

## Module 2: Automated Exploratory Data Analysis (EDA)
- Missing value analysis
- Outlier detection (IQR / Z-score)
- Correlation matrix
- Numerical feature distributions
- Categorical feature bar plots
- Train/test split summary

## Module 3: Issue Detection and User Approval
- Detect missing values
- Detect outliers
- Detect class imbalance
- Detect high-cardinality categorical features
- Detect constant/near-constant features
- Show warnings and suggest fixes
- Ask user confirmation before applying fixes

## Module 4: Preprocessing
- Missing value handling (mean/median/mode/constant)
- Outlier handling if needed
- Scaling
- Encoding categorical variables (One-Hot/Ordinal)
- Train-test split based on user ratio

## Module 5: Model Training and Hyperparameter Optimization
- Train multiple classifiers (LR, KNN, DT, NB, RF, SVM, Rule-based)
- Apply Grid Search or Randomized Search
- Evaluate models using accuracy, precision, recall, F1, ROC-AUC
- Generate confusion matrix
- Track training time

## Module 6: Model Comparison Dashboard
- Show comparison table of metrics
- Rank algorithms by chosen metric
- Allow CSV download of results
- Visualize metrics using bar charts

## Module 7: Auto-Generated Final Report
- Dataset overview
- EDA results
- Issues detected
- Preprocessing decisions
- Model settings and hyperparameters
- Comparison tables
- Best model explanation
- Export as PDF/Markdown

## Module 8: Streamlit Deployment Module
- Streamlit UI integration
- Proper requirements.txt
- Public hosting on Streamlit Cloud
- README with instructions and app link