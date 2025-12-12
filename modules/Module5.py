# Module 5: Model Training and Hyperparameter Optimization
# Train multiple classifiers (LR, KNN, DT, NB, RF, SVM, Rule-based)
# Apply Grid Search or Randomized Search
# Evaluate models using accuracy, precision, recall, F1, ROC-AUC
# Generate confusion matrix
# Track training time


# Imports
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import time

# 1. Train multiple classifiers with hyperparameter optimization
def train_and_optimize_models(X_train, y_train, search_type='grid', cv=5):
    models = {
        'Logistic Regression': (LogisticRegression(), {
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
        'Random Forest': (RandomForestClassifier(), {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20]
        }),
        'Support Vector Machine': (SVC(probability=True), {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf']
        })
    }
    
    best_models = {}
    
    for model_name, (model, params) in models.items():
        start_time = time.time()
        
        if search_type == 'grid' and params:
            search = GridSearchCV(model, params, cv=cv, n_jobs=-1)
        elif search_type == 'random' and params:
            search = RandomizedSearchCV(model, params, cv=cv, n_jobs=-1, n_iter=10)
        else:
            search = model
        
        search.fit(X_train, y_train)
        
        end_time = time.time()
        training_time = end_time - start_time
        
        best_models[model_name] = {
            'model': search.best_estimator_ if params else search,
            'training_time': training_time
        }
    
    return best_models

# 2. Evaluate models
def evaluate_models(models, X_test, y_test):
    evaluation_results = {}
    
    for model_name, model_info in models.items():
        model = model_info['model']
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        evaluation_results[model_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': conf_matrix,
            'training_time': model_info['training_time']
        }
    
    return evaluation_results

