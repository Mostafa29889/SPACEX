import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')

def compare_models(results_dict):
    """
    Step 7 & 8: Compare models using accuracy, precision, recall, f1, and ROC-AUC.
    Creates a comparison table and saves it.
    """
    print("\n--- Model Evaluation & Comparison ---")
    
    # Construct DataFrame
    records = []
    for model_name, metrics in results_dict.items():
        record = {
            'Model': model_name,
            'Accuracy': metrics.get('accuracy', 0.0),
            'Precision': metrics.get('precision', 0.0),
            'Recall': metrics.get('recall', 0.0),
            'F1-Score': metrics.get('f1_score', 0.0),
            'ROC-AUC': metrics.get('roc_auc', 0.0)
        }
        records.append(record)
        
    df_compare = pd.DataFrame(records)
    df_compare = df_compare.sort_values(by='Accuracy', ascending=False)
    
    # Save the table to reports/comparison_table.csv
    if not os.path.exists(REPORTS_DIR):
        os.makedirs(REPORTS_DIR)
        
    comparison_path = os.path.join(REPORTS_DIR, 'model_comparison_table.csv')
    df_compare.to_csv(comparison_path, index=False)
    print(df_compare.to_string(index=False))
    print(f"\nSaved model comparison table to: {comparison_path}")
    
    # Find the best model
    best_model_name = df_compare.iloc[0]['Model']
    best_acc = df_compare.iloc[0]['Accuracy']
    print(f"\nBest Model: {best_model_name} with Accuracy of {best_acc:.4f}")
    
    return df_compare

if __name__ == '__main__':
    # Test stub
    dummy_results = {
        'LogisticRegression': {'accuracy': 0.833, 'precision': 0.80, 'recall': 0.85, 'f1_score': 0.824, 'roc_auc': 0.90},
        'DecisionTree': {'accuracy': 0.778, 'precision': 0.75, 'recall': 0.80, 'f1_score': 0.774, 'roc_auc': 0.78},
        'RandomForest': {'accuracy': 0.889, 'precision': 0.88, 'recall': 0.90, 'f1_score': 0.890, 'roc_auc': 0.94},
        'SVM': {'accuracy': 0.833, 'precision': 0.80, 'recall': 0.85, 'f1_score': 0.824, 'roc_auc': 0.88},
        'XGBoost': {'accuracy': 0.889, 'precision': 0.88, 'recall': 0.90, 'f1_score': 0.890, 'roc_auc': 0.95}
    }
    compare_models(dummy_results)
