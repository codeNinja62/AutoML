# Module 5: Model Training and Hyperparameter Optimization
# Train multiple classifiers (LR, KNN, DT, NB, RF, SVM, Rule-based)
# Apply Grid Search or Randomized Search
# Evaluate models using accuracy, precision, recall, F1, ROC-AUC
# Generate confusion matrix
# Track training time


"""Module 5: Model Training and Hyperparameter Optimization."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
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

# 1. Train multiple classifiers with hyperparameter optimization
def train_and_optimize_models(
    X_train,
    y_train,
    search_type='grid',
    cv=5,
    scoring: str | None = None,
    random_state: int = 42,
    include_models: list[str] | None = None,
):
    models: dict[str, tuple[Any, dict[str, Any]]] = {
        'Logistic Regression': (LogisticRegression(max_iter=2000, random_state=random_state), {
            'C': [0.01, 0.1, 1, 10],
            'solver': ['liblinear']
        }),
        'K-Nearest Neighbors': (KNeighborsClassifier(), {
            'n_neighbors': [3, 5, 7],
            'weights': ['uniform', 'distance']
        }),
        'Decision Tree': (DecisionTreeClassifier(), {
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        }),
        'Naive Bayes': (GaussianNB(), {}),
        'Random Forest': (RandomForestClassifier(random_state=random_state), {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20]
        }),
        'Support Vector Machine': (SVC(probability=True, random_state=random_state), {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf']
        }),
        # Simple baseline often called "rule-based" in teaching contexts.
        'Rule-based (Most Frequent)': (DummyClassifier(strategy='most_frequent', random_state=random_state), {}),
    }
    
    best_models = {}
    
    for model_name, (model, params) in models.items():
        if include_models is not None and model_name not in include_models:
            continue
        start_time = time.time()
        
        try:
            if search_type == 'grid' and params:
                search = GridSearchCV(model, params, cv=cv, n_jobs=-1, scoring=scoring)
            elif search_type == 'random' and params:
                search = RandomizedSearchCV(model, params, cv=cv, n_jobs=-1, n_iter=10, scoring=scoring, random_state=random_state)
            else:
                search = model

            search.fit(X_train, y_train)
            fitted = search
            best_estimator = fitted.best_estimator_ if params and hasattr(fitted, 'best_estimator_') else fitted
            best_params = fitted.best_params_ if params and hasattr(fitted, 'best_params_') else {}
            error = None
        except Exception as e:
            best_estimator = None
            best_params = {}
            error = str(e)
        
        end_time = time.time()
        training_time = end_time - start_time
        
        best_models[model_name] = {
            'model': best_estimator,
            'best_params': best_params,
            'training_time': training_time,
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

            roc_auc = None
            if is_binary and hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)[:, 1]
                roc_auc = roc_auc_score(y_test, y_proba)

            evaluation_results[model_name] = {
                'model': model_name,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'confusion_matrix': conf_matrix,
                'training_time': model_info.get('training_time'),
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
                'best_params': model_info.get('best_params', {}),
                'error': str(e),
            }
    
    return evaluation_results

