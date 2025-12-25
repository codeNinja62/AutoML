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


