# Module 5: Model Training and Hyperparameter Optimization
# Train multiple classifiers (LR, KNN, DT, NB, RF, SVM, Rule-based)
# Apply Grid Search or Randomized Search
# Evaluate models using accuracy, precision, recall, F1, ROC-AUC
# Generate confusion matrix
# Track training time


"""Module 5: Model Training and Hyperparameter Optimization."""

from __future__ import annotations

import os
import time
import tempfile
from typing import Any

import numpy as np
import pandas as pd
from joblib import Memory
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


def _compute_roc_auc(model, X_test, y_test) -> float | None:
    """Compute ROC-AUC for binary or multiclass classification.

    - Binary: uses predict_proba[:, 1] if available, else decision_function.
    - Multiclass: uses OVR ROC-AUC with weighted average.
    Returns None if it cannot be computed safely.
    """

    try:
        y_true = pd.Series(y_test)
        y_true_non_null = y_true.dropna()
        if y_true_non_null.empty:
            return None

        classes = None
        if hasattr(model, 'classes_'):
            try:
                classes = list(model.classes_)
            except Exception:
                classes = None
        if classes is None:
            classes = sorted(pd.unique(y_true_non_null))

        n_classes = len(classes)
        if n_classes < 2:
            return None

        # Require all classes present in y_test for multiclass AUC.
        present = set(pd.unique(y_true_non_null))
        if n_classes > 2 and not set(classes).issubset(present):
            return None

        # Score extraction
        y_score = None
        if hasattr(model, 'predict_proba'):
            try:
                y_score = model.predict_proba(X_test)
            except Exception:
                y_score = None
        if y_score is None and hasattr(model, 'decision_function'):
            try:
                y_score = model.decision_function(X_test)
            except Exception:
                y_score = None

        if y_score is None:
            return None

        y_score_arr = np.asarray(y_score)
        if n_classes == 2:
            # Ensure we pass a continuous score for the positive class.
            pos_label = classes[1]
            y_true_bin = (y_true == pos_label).astype(int)
            if y_score_arr.ndim == 2 and y_score_arr.shape[1] >= 2:
                y_score_pos = y_score_arr[:, 1]
            else:
                y_score_pos = y_score_arr.reshape(-1)
            return float(roc_auc_score(y_true_bin, y_score_pos))

        # Multiclass (n_samples, n_classes)
        if y_score_arr.ndim != 2 or y_score_arr.shape[1] != n_classes:
            return None

        return float(
            roc_auc_score(
                y_true,
                y_score_arr,
                multi_class='ovr',
                average='weighted',
                labels=classes,
            )
        )
    except Exception:
        return None

# 1. Train multiple classifiers with hyperparameter optimization
def train_and_optimize_models(
    X_train,
    y_train,
    search_type='grid',
    cv=5,
    scoring: str | None = None,
    random_state: int = 42,
    include_models: list[str] | None = None,
    preprocessor=None,
    n_jobs: int | None = -1,
    cache: bool = True,
    class_weight_auto: bool = False,
    compute_cv_summary: bool = True,
):
    # Defensive validation: avoid confusing downstream sklearn errors.
    if X_train is None:
        raise ValueError('X_train is required')
    if y_train is None:
        raise ValueError('y_train is required')

    # Feature matrix must have at least one column.
    try:
        if hasattr(X_train, 'shape') and int(X_train.shape[1]) == 0:
            raise ValueError('No feature columns available for training')
    except Exception:
        # If shape inspection fails, let sklearn raise the detailed error.
        pass

    y_series = pd.Series(y_train)
    y_non_null = y_series.dropna()
    if int(y_non_null.nunique()) < 2:
        raise ValueError('Target must have at least 2 classes for classification training')

    # If the user requests more folds than the smallest class can support, reduce CV.
    try:
        class_counts = y_non_null.value_counts()
        min_class_count = int(class_counts.min()) if not class_counts.empty else 0
        requested_cv = int(cv)
        effective_cv = min(requested_cv, min_class_count)
        if effective_cv < 2:
            raise ValueError('Not enough samples per class for cross-validation (need at least 2 per class)')
        cv = effective_cv
    except Exception:
        cv = int(cv)

    cw = 'balanced' if class_weight_auto else None
    models: dict[str, tuple[Any, dict[str, Any]]] = {
        'Logistic Regression': (LogisticRegression(max_iter=2000, random_state=random_state, class_weight=cw), {
            'C': [0.01, 0.1, 1, 10],
            'solver': ['liblinear']
        }),
        'K-Nearest Neighbors': (KNeighborsClassifier(), {
            'n_neighbors': [3, 5, 7],
            'weights': ['uniform', 'distance']
        }),
        'Decision Tree': (DecisionTreeClassifier(random_state=random_state, class_weight=cw), {
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        }),
        'Naive Bayes': (GaussianNB(), {}),
        'Random Forest': (RandomForestClassifier(random_state=random_state, n_jobs=n_jobs, class_weight=cw), {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20]
        }),
        'Support Vector Machine': (SVC(probability=True, random_state=random_state, class_weight=cw), {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf']
        }),
        # Simple baseline often called "rule-based" in teaching contexts.
        'Rule-based (Most Frequent)': (DummyClassifier(strategy='most_frequent', random_state=random_state), {}),
    }
    
    best_models = {}
    
    cv_splitter = StratifiedKFold(n_splits=int(cv), shuffle=True, random_state=random_state)
    cache_dir = os.path.join(tempfile.gettempdir(), 'cs245_automl_cache')
    memory = Memory(location=cache_dir, verbose=0) if cache else None

    for model_name, (model, params) in models.items():
        if include_models is not None and model_name not in include_models:
            continue
        start_time = time.time()
        
        cv_mean: float | None = None
        cv_std: float | None = None

        try:
            # Best practice: run CV/search on the full pipeline so preprocessing happens inside each fold.
            if preprocessor is not None:
                estimator: Any = Pipeline(
                    steps=[('preprocess', preprocessor), ('model', model)],
                    memory=memory,
                )
                tuned_params = {f'model__{k}': v for k, v in (params or {}).items()}
            else:
                estimator = model
                tuned_params = params or {}

            if search_type == 'grid' and tuned_params:
                search = GridSearchCV(
                    estimator,
                    tuned_params,
                    cv=cv_splitter,
                    n_jobs=n_jobs,
                    scoring=scoring,
                )
            elif search_type == 'random' and tuned_params:
                search = RandomizedSearchCV(
                    estimator,
                    tuned_params,
                    cv=cv_splitter,
                    n_jobs=n_jobs,
                    n_iter=10,
                    scoring=scoring,
                    random_state=random_state,
                )
            elif search_type in {'halving_grid', 'halving_random'} and tuned_params:
                # Optional faster search that prunes weak configs early.
                try:
                    from sklearn.experimental import enable_halving_search_cv  # noqa: F401
                    from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV
                except Exception as ie:
                    raise RuntimeError('Halving search not available in this scikit-learn build.') from ie

                if search_type == 'halving_grid':
                    search = HalvingGridSearchCV(
                        estimator,
                        tuned_params,
                        cv=cv_splitter,
                        factor=3,
                        scoring=scoring,
                        n_jobs=n_jobs,
                    )
                else:
                    search = HalvingRandomSearchCV(
                        estimator,
                        tuned_params,
                        cv=cv_splitter,
                        factor=3,
                        scoring=scoring,
                        n_jobs=n_jobs,
                        n_candidates=12,
                        random_state=random_state,
                    )
            else:
                search = estimator

            search.fit(X_train, y_train)
            fitted = search
            best_estimator = fitted.best_estimator_ if tuned_params and hasattr(fitted, 'best_estimator_') else fitted
            best_params_raw = fitted.best_params_ if tuned_params and hasattr(fitted, 'best_params_') else {}
            # Strip pipeline prefix for display
            best_params = {
                (k.replace('model__', '', 1) if isinstance(k, str) else k): v
                for k, v in (best_params_raw or {}).items()
            }

            # CV summary for trustworthiness: mean ± std across folds.
            if compute_cv_summary:
                if hasattr(fitted, 'cv_results_') and hasattr(fitted, 'best_index_'):
                    try:
                        idx = int(fitted.best_index_)
                        cv_mean = float(fitted.cv_results_['mean_test_score'][idx])
                        cv_std = float(fitted.cv_results_['std_test_score'][idx])
                    except Exception:
                        cv_mean, cv_std = None, None
                elif hasattr(fitted, 'score'):
                    try:
                        scores = cross_val_score(fitted, X_train, y_train, cv=cv_splitter, n_jobs=n_jobs, scoring=scoring)
                        cv_mean = float(np.mean(scores))
                        cv_std = float(np.std(scores))
                    except Exception:
                        cv_mean, cv_std = None, None
            error = None
        except Exception as e:
            best_estimator = None
            best_params = {}
            cv_mean, cv_std = None, None
            error = str(e)
        
        end_time = time.time()
        training_time = end_time - start_time
        
        best_models[model_name] = {
            'model': best_estimator,
            'best_params': best_params,
            'training_time': training_time,
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'cv_folds': int(cv),
            'error': error,
        }
    
    return best_models

# 2. Evaluate models
def evaluate_models(models, X_test, y_test):
    evaluation_results = {}
    
    classes = np.unique(y_test)
    is_binary = len(classes) == 2
    average = 'binary' if is_binary else 'weighted'

    for model_name, model_info in models.items():
        model = model_info.get('model')
        if model is None:
            evaluation_results[model_name] = {
                'model': model_name,
                'accuracy': None,
                'precision': None,
                'recall': None,
                'f1_score': None,
                'roc_auc': None,
                'confusion_matrix': None,
                'training_time': model_info.get('training_time'),
                'cv_mean': model_info.get('cv_mean'),
                'cv_std': model_info.get('cv_std'),
                'cv_folds': model_info.get('cv_folds'),
                'best_params': model_info.get('best_params', {}),
                'error': model_info.get('error'),
            }
            continue

        try:
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average=average, zero_division=0)
            recall = recall_score(y_test, y_pred, average=average, zero_division=0)
            f1 = f1_score(y_test, y_pred, average=average, zero_division=0)
            conf_matrix = confusion_matrix(y_test, y_pred)

            roc_auc = _compute_roc_auc(model, X_test, y_test)

            evaluation_results[model_name] = {
                'model': model_name,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'confusion_matrix': conf_matrix,
                'training_time': model_info.get('training_time'),
                'cv_mean': model_info.get('cv_mean'),
                'cv_std': model_info.get('cv_std'),
                'cv_folds': model_info.get('cv_folds'),
                'best_params': model_info.get('best_params', {}),
                'error': None,
            }
        except Exception as e:
            evaluation_results[model_name] = {
                'model': model_name,
                'accuracy': None,
                'precision': None,
                'recall': None,
                'f1_score': None,
                'roc_auc': None,
                'confusion_matrix': None,
                'training_time': model_info.get('training_time'),
                'cv_mean': model_info.get('cv_mean'),
                'cv_std': model_info.get('cv_std'),
                'cv_folds': model_info.get('cv_folds'),
                'best_params': model_info.get('best_params', {}),
                'error': str(e),
            }
    
    return evaluation_results

