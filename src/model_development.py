import os
# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             classification_report, roc_curve)

# Optional MLflow integration
try:
    import mlflow
    import mlflow.sklearn
    import mlflow.xgboost
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')

# Create necessary directories
for d in [MODELS_DIR, REPORTS_DIR]:
    if not os.path.exists(d):
        os.makedirs(d)

def train_and_evaluate_model(model_name, model, param_grid, X_train, y_train, X_test, y_test):
    """
    Step 4, 5 & 6: Train models, tune hyperparameters, log with MLflow, and evaluate performance.
    """
    print(f"\n--- Training and Tuning: {model_name} ---")
    
    # Grid search for hyperparameter tuning
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    
    # If MLflow is available, run inside a nested/parent run
    if MLFLOW_AVAILABLE:
        mlflow.set_experiment("SpaceX_Landing_Prediction")
        with mlflow.start_run(run_name=model_name):
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            
            # Predict and evaluate
            y_pred = best_model.predict(X_test)
            y_prob = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, "predict_proba") else None
            
            # Compute metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1_score': f1_score(y_test, y_pred, zero_division=0),
                'roc_auc': roc_auc_score(y_test, y_prob) if y_prob is not None else 0.0
            }
            
            # Log hyperparameters
            mlflow.log_params(grid_search.best_params_)
            mlflow.log_param("model_name", model_name)
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Create artifacts (Plots)
            create_artifacts(model_name, best_model, X_test, y_test, y_pred, y_prob)
            
            # Log artifacts folder
            mlflow.log_artifacts(os.path.join(REPORTS_DIR, model_name))
            
            # Log model
            mlflow.sklearn.log_model(best_model, f"model_{model_name}")
            
            print(f"MLflow logged successfully for {model_name}!")
            print(f"Best Params: {grid_search.best_params_}")
            print(f"Metrics: {metrics}")
            
            return best_model, metrics
    else:
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, "predict_proba") else None
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_prob) if y_prob is not None else 0.0
        }
        
        create_artifacts(model_name, best_model, X_test, y_test, y_pred, y_prob)
        print(f"Best Params: {grid_search.best_params_}")
        print(f"Metrics: {metrics}")
        
        return best_model, metrics

def create_artifacts(model_name, model, X_test, y_test, y_pred, y_prob):
    """
    Step 6 Artifacts: Confusion Matrix, ROC Curve, Feature Importance.
    """
    model_report_dir = os.path.join(REPORTS_DIR, model_name)
    if not os.path.exists(model_report_dir):
        os.makedirs(model_report_dir)
        
    # 1. Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fail', 'Success'], yticklabels=['Fail', 'Success'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(model_report_dir, 'confusion_matrix.png'))
    plt.close()
    
    # 2. ROC Curve Plot
    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.figure(figsize=(6, 4))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc_score(y_test, y_prob):.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(model_report_dir, 'roc_curve.png'))
        plt.close()

    # 3. Feature Importance Plot (if applicable)
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        plt.figure(figsize=(10, 6))
        plt.title(f"Feature Importance - {model_name}")
        sns.barplot(x=importances[indices][:10], y=[f"Feature {i}" for i in indices[:10]], palette="viridis")
        plt.tight_layout()
        plt.savefig(os.path.join(model_report_dir, 'feature_importance.png'))
        plt.close()
        
    # 4. Classification Report text file
    with open(os.path.join(model_report_dir, 'classification_report.txt'), 'w') as f:
        f.write(classification_report(y_test, y_pred))

def get_candidate_models():
    """Defines models and hyperparameter grids for tuning."""
    models_to_test = {
        'LogisticRegression': (
            LogisticRegression(max_iter=1000),
            {'C': [0.01, 0.1, 1, 10, 100], 'penalty': ['l2']}
        ),
        'DecisionTree': (
            DecisionTreeClassifier(random_state=42),
            {'criterion': ['gini', 'entropy'], 'max_depth': [4, 6, 8, 10, None], 'min_samples_split': [2, 5, 10]}
        ),
        'RandomForest': (
            RandomForestClassifier(random_state=42),
            {'n_estimators': [50, 100, 200], 'max_depth': [6, 8, 10], 'min_samples_split': [2, 5]}
        ),
        'SVM': (
            SVC(probability=True, random_state=42),
            {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
        ),
        'XGBoost': (
            XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
            {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 5, 7]}
        )
    }
    return models_to_test

if __name__ == '__main__':
    print("Candidate models and configurations ready.")
