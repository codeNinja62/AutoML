# CS-245 AutoML System for Classification (Streamlit)

A full-featured AutoML pipeline for supervised **classification** tasks with an intuitive Streamlit UI.

## Live Demo
[AutoML on Streamlit Cloud](https://automl-project.streamlit.app/)

## Features

### Core AutoML Pipeline
- **Dataset Upload**: CSV file support with automatic schema detection
- **Automated EDA**: Missing values, correlations, distributions, outliers
- **Issue Detection**: Missing values, outliers, class imbalance, high-cardinality categoricals, constant features
- **Preprocessing**: User-controlled imputation, scaling (Standard/MinMax), encoding (OneHot/Ordinal), outlier handling
- **Model Training**: Logistic Regression, KNN, Decision Tree, Naive Bayes, Random Forest, SVM, XGBoost, and baseline classifiers
- **Hyperparameter Tuning**: Grid/Random/Halving search with cross-validation
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix

### AI-Powered Chat Assistant
- **Gemini Integration**: Chat with your dataset using Google's Gemini AI
- **Context-Aware**: The chatbot understands your training results, best model, and data statistics
- **Streamlit Cloud Ready**: Supports `st.secrets` and `.env` for API key configuration

### User Experience
- **Pipeline Progress Bar**: Visual indicator showing current step (1-6) in the AutoML workflow
- **Data Quality Visualization**: Pie chart showing missing vs present data ratio
- **Auto-Collapsing Sections**: EDA and preprocessing sections collapse after training to reduce clutter
- **Tab-Based Results View**: Organized into Results Overview, Detailed Analysis, Chat, and Export tabs
- **Fragment-Based Chat**: Chat interface uses `@st.fragment` to prevent page resets when messaging
- **Responsive Tables**: Horizontal scroll support for wide data tables

### Export Options
- **Markdown Report**: Styled, professional markdown export
- **HTML Report**: Fully styled HTML with embedded CSS
- **PDF Report**: Professional PDF with colored headers, formatted tables, and title page
- **Model Export**: Download best model as joblib/pickle file

## Installation

### Prerequisites
- Python 3.9+
- pip

### Setup
```bash
# Clone the repository
git clone https://github.com/codeNinja62/AutoML.git
cd AutoML

# Install dependencies
pip install -r requirements.txt

# Set up API key for chat feature (optional)
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

### Run Locally
```bash
streamlit run streamlit_app.py
```

### Run Tests
```bash
python -m unittest -q
```
## Test Suite

The project includes 38 unit tests covering all core modules. Run with:
```bash
python -m unittest -q
```

### Test Coverage

| Test File | Module | Tests | Description |
|-----------|--------|-------|-------------|
| `test_module1.py` | Data Loading | 8 | Dataset shape, missing values, duplicates, unique counts, cardinality buckets, schema generation, target validation, and candidate inference |
| `test_module2.py` | EDA | 5 | Missing value analysis, correlation matrix, outlier detection (IQR and Z-score), train/test split summary |
| `test_module3.py` | Issue Detection | 4 | Detects missing values, high cardinality features, class imbalance, and constant features |
| `test_module4.py` | Preprocessing | 7 | Train/test split validation, preprocessor pipeline with OneHot/Ordinal encoding, outlier handling (cap/remove with IQR/Z-score) |
| `test_module5.py` | Model Training | 4 | CV fold reduction for small classes, pipeline integration, confusion matrix generation, multiclass ROC-AUC |
| `test_module6.py` | Comparison | 5 | Comparison table creation, algorithm ranking, CSV export (bytes and file), metric bar chart generation |
| `test_module7.py` | Report Export | 3 | Markdown report generation, Markdown bytes export, PDF generation (if fpdf2 installed) |

### Key Test Scenarios

- **Split Validation**: Tests edge cases like missing target columns, single-sample classes, and insufficient test sizes
- **Outlier Handling**: Verifies both IQR capping and row removal with configurable thresholds
- **Pipeline Integration**: Ensures preprocessor correctly chains with estimators
- **Multiclass Support**: Tests ROC-AUC calculation for 3+ class problems
- **Report Generation**: Validates Markdown structure and PDF byte output



## Deployment on Streamlit Cloud

1. Push to GitHub
2. Connect repo on [share.streamlit.io](https://share.streamlit.io)
3. Set entrypoint: `streamlit_app.py`
4. Add secrets (Settings > Secrets):
   ```toml
   GEMINI_API_KEY = "your-api-key-here"
   ```

## Project Structure
```
AutoML/
├── streamlit_app.py          # Main entry point
├── modules/
│   ├── Module1.py            # Data loading and schema detection
│   ├── Module2.py            # EDA visualizations
│   ├── Module3.py            # Issue detection
│   ├── Module4.py            # Preprocessing pipeline
│   ├── Module5.py            # Model training and optimization
│   ├── Module6.py            # Model comparison and visualization
│   ├── Module7.py            # Report generation (Markdown/PDF)
│   ├── Module8.py            # Streamlit UI orchestration
│   └── Module9.py            # Gemini chat integration
├── requirements.txt          # Python dependencies
├── .env.example              # Environment variable template
└── assets/                   # UI assets (logo, etc.)
```

## Dependencies
- streamlit
- pandas, numpy
- scikit-learn
- matplotlib, seaborn
- xgboost
- google-generativeai (Gemini)
- fpdf2 (PDF export)
- tabulate (Markdown tables)
- reportlab
- joblib
- python-dotenv

## Configuration

### Environment Variables
| Variable | Description | Required |
|----------|-------------|----------|
| `GEMINI_API_KEY` | Google Gemini API key for chat feature | No (chat disabled without it) |

### Streamlit Secrets (for Cloud deployment)
Add to `.streamlit/secrets.toml` or via Streamlit Cloud UI:
```toml
GEMINI_API_KEY = "your-api-key"
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| PDF export unavailable | Ensure `fpdf2` is installed: `pip install fpdf2` |
| Chat not working | Check GEMINI_API_KEY in .env or st.secrets |


