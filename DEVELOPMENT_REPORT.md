# AutoML Development Report: Under the Hood

This report details how the AutoML application was built, focusing on the code structure, implementation decisions, and specific libraries used in each module.

## Development Philosophy

*   **Modular Design:** We avoided a "monolithic" script. Functionality is split into 8 distinct modules (`Module1.py` - `Module8.py`) in the `project/modules/` directory. Each module handles a specific stage of the machine learning pipeline (e.g., EDA, Training, Reporting).
*   **Separation of Concerns:** The Streamlit app (`streamlit_app.py` and `Module8.py`) is just a "Display Layer". All the hard logic (math, data processing) happens in Modules 1-7, which are "UI-agnostic" (they don't import `streamlit`). This means we can test them via command line or unit tests without needing a browser.
*   **Type Hinting:** We used Python 3.9+ type hints (e.g., `def func(df: pd.DataFrame) -> dict:`) everywhere to catch bugs early and make the code readable.

---

## Module-by-Module Technical Deep Dive

### Module 1: `Dataset Upload`
**File:** `modules/Module1.py`
*   **Core Logic:** Wraps `pandas.read_csv`.
*   **Key Implementations:**
    *   **`get_constant_columns`**: Iterates through columns and uses `.nunique()` to find features with only 1 unique value (which are useless for ML).
    *   **`TargetValidation` Class**: A Python `@dataclass` used to structurally return validation results (errors, warnings) instead of just printing them.
    *   **Heuristics**: Uses string matching (searching for "target", "label") and cardinality ratios to guess which column is the target variable.

### Module 2: `Automated EDA`
**File:** `modules/Module2.py`
*   **Libraries:** `matplotlib`, `seaborn` for static plotting.
*   **Optimization:** Contains a function `_sample_for_plots` that downsamples large datasets (e.g., max 5000 rows) before plotting. This prevents the app from crashing when visualization sets are huge.
*   **Plots:**
    *   **Missing Values**: Uses `df.isna().mean()` to calculate percentages and plots a bar chart.
    *   **Correlation**: Uses `df.corr()` (Pearson correlation) and `sns.heatmap` to visualize relationships.

### Module 3: `Issue Detection`
**File:** `modules/Module3.py`
*   **Concept:** Returns a list of `IssueFinding` objects (another @dataclass).
*   **Detection Logic:**
    *   **Outliers (Z-score)**: Calculates `(x - mean) / std`. If > 3.0, it's flagged.
    *   **High Cardinality**: Checks if a categorical column has too many unique values relative to the dataset size (e.g., >10% of rows are unique strings).
    *   **Class Imbalance**: Checks if any target class appears >70% of the time, suggesting a biased dataset.

### Module 4: `Preprocessing`
**File:** `modules/Module4.py`
*   **The "Engine" Room:** Uses `scikit-learn`.
*   **One-Step Pipeline:** We implement `build_preprocessor` which returns a `sklearn.compose.ColumnTransformer`.
    *   **Process:** It applies `StandardScaler` to numeric columns and `OneHotEncoder` (with `handle_unknown='ignore'`) to categorical columns simultaneously.
    *   **Why this matters:** By bundling these into a Pipeline, we ensure that when we predict on new data later, the *exact same* scaling and encoding rules are applied automatically.

### Module 5: `Model Training`
**File:** `modules/Module5.py`
*   **Training Loop:** Iterates through a dictionary of models (LogisticRegression, RandomForest, etc.).
*   **Hyperparameter Tuning:** Uses `GridSearchCV` or `RandomizedSearchCV` depending on the user's choice.
*   **Testing Logic:**
    *   Splits data into Train/Test sets.
    *   Trains on Train set with Cross-Validation (CV).
    *   Predicts on the held-out Test set for final evaluation.
*   **Metric Handling:** Automatically detects if the problem is Binary or Multiclass to prefer the correct ROC-AUC method (`"ovr"` - One-vs-Rest for multiclass).

### Module 6: `Comparison Dashboard`
**File:** `modules/Module6.py`
*   **Implementation:** Creates a Pandas DataFrame of all results.
*   **Visualization:** Uses `seaborn.barplot` to create a visual leaderboard.
*   **Ranking:** `rank_algorithms` sorts the DataFrame based on the user's chosen metric (e.g., highest F1 score).

### Module 7: `Reporting`
**File:** `modules/Module7.py`
*   **PDF Generation:** Uses `fpdf2` library.
*   **Trick:** PDF generation is hard because standard libraries struggle with layout. We implemented a custom `_write_line` function that manually handles text wrapping to ensure long sentences do not run off the page.
*   **Encoding:** Forces `latin-1` encoding with replacement for unsupported characters to ensure the PDF generator doesn't crash on special symbols.

---

## Streamlit Integration

*   **File:** `streamlit_app.py` is the entry point, but it imports `main()` from `modules/Module8.py`.
*   **State Management:** We use `st.session_state` heavily.
    *   When you upload a file, it's saved to `st.session_state['df']`.
    *   When you train models, results are saved to `st.session_state['results']`.
    *   This allows the user to click around (e.g., change tabs) without losing their training data.
*   **Caching:** We use `@st.cache_data` decorators on expensive functions (like loading data or training) so that if you change a minor UI setting, the app doesn't re-run the entire validatoin or training process from scratch.

---

## Testing Strategy

**Framework:** `unittest` (Standard Python library).
**Location:** `project/tests/`

*   **Test Isolation:** Each module has its own test file (e.g., `test_module5.py`).
*   **Toy Data:** Tests create small "toy" DataFrames (e.g., 5 rows, 2 columns) in memory to verify logic strings. They do *not* require external CSV files.
*   **Verification:**
    *   **Logic Checks:** Does `get_shape` return the correct tuple?
    *   **Coverage:** We test edge cases, like "What happens if the target column is missing?" or "What if the dataset is empty?".
    *   **Mocking:** We simulate user inputs to ensure the pipeline runs smoothly without human intervention.



## CA VIVA
## CA PROJECT
## DV PRESENTATION
## ITM PRESENTATION
## HCI PRESENTATION
## CA THOERY 
## OESTEO DATA