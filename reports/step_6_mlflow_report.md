# SpaceX Falcon 9 Landing Prediction - Step 6: MLflow Experiment Tracking Report

This report documents the implementation of **Step 6: Experiment Tracking with MLflow** for the SpaceX Falcon 9 landing success prediction project.

---

## 1. Overview of MLflow Configuration
We integrated **MLflow** into our training pipeline ([src/model_development.py](file:///c:/Users/User/Desktop/SPACEX-main/SPACEX-main/src/model_development.py)) to automatically track and compare machine learning experiments.

### Tracked Metadata:
1. **Hyperparameters (Params):** Logged the best hyperparameter values found during Grid Search (e.g. learning rate, max depth, C value, regularizations).
2. **Metrics (Metrics):** Tracked standard binary classification metrics:
   - `accuracy`
   - `precision`
   - `recall`
   - `f1_score`
   - `roc_auc`
3. **Artifacts (Plots & Models):**
   - The serialized model pickle files (e.g. `XGBoost.pkl`, `best_model.pkl`).
   - The classification report text summary.
   - Plot files: `confusion_matrix.png`, `roc_curve.png`, and `feature_importance.png` (for tree-based models).

---

## 2. Local Experiment Runs Directory Structure
When `src/model_development.py` executes, it creates a local directory named `mlruns/` in the workspace root. This folder acts as our local tracking database:
```
mlruns/
  └── 0/ (Default Experiment)
      ├── meta.yaml
      └── [run_id]/
          ├── meta.yaml
          ├── params/
          │   ├── learning_rate
          │   └── max_depth
          ├── metrics/
          │   ├── accuracy
          │   └── f1_score
          └── artifacts/
              ├── confusion_matrix.png
              ├── roc_curve.png
              └── model/ (best model pickle files)
```

---

## 3. How to Launch and View the MLflow Dashboard UI
To inspect the experiments and compare model performances side-by-side using the interactive web dashboard:

1. **Open your terminal** in the workspace root folder:
   `c:\Users\User\Desktop\SPACEX-main\SPACEX-main`
2. **Run the following command** to start the local tracking server:
   ```bash
   mlflow ui --port 5000
   ```
3. **Open your web browser** and navigate to:
   [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 4. Key Visualizations Tracked inside MLflow
For each run, the following plots are generated and logged in the MLflow UI:
- **Confusion Matrix:** Displays visual counts of true positives, false positives, true negatives, and false negatives.
- **ROC Curve:** Displays the true positive rate vs. false positive rate, alongside the area under the curve (ROC-AUC) score.
- **Feature Importance:** Highlights the top contributing features influencing booster landing predictions.
