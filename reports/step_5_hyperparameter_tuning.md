# SpaceX Falcon 9 Landing Prediction - Step 5: Hyperparameter Tuning Report

This report documents the hyperparameter tuning phase of the SpaceX Falcon 9 first-stage landing success prediction project.

---

## 1. Methodology
To optimize the predictive accuracy and generalizability of our classifiers, we used **Grid Search with 5-Fold Stratified Cross-Validation (`GridSearchCV`)**.
- **Cross-Validation (CV):** We split the training data (445 samples) into 5 folds. The model is trained on 4 folds and validated on the remaining fold, repeating this 5 times.
- **Stratification:** Stratified folds ensure that the proportion of landing successes (91.2%) and failures (8.8%) is preserved in each fold, preventing fold-wise target bias.
- **Optimization Metric:** Accuracy was chosen as the optimization metric to maximize correct predictions of both success and failure classes.

---

## 2. Hyperparameter Grids & Optimization Results

### 2.1 Logistic Regression
- **Grid Searched:**
  - `C` (inverse regularization strength): `[0.01, 0.1, 1, 10, 100]`
  - `penalty`: `['l2']` (Ridge regularization)
- **Best Configuration:** `{'C': 100, 'penalty': 'l2'}`
- **Tuned Performance:** Accuracy: **88.39%**, ROC-AUC: **0.9402**

### 2.2 Decision Tree Classifier
- **Grid Searched:**
  - `criterion` (split quality measure): `['gini', 'entropy']`
  - `max_depth` (maximum tree depth): `[4, 6, 8, 10, None]`
  - `min_samples_split` (minimum samples to split an internal node): `[2, 5, 10]`
- **Best Configuration:** `{'criterion': 'gini', 'max_depth': 10, 'min_samples_split': 2}`
- **Tuned Performance:** Accuracy: **90.18%**, ROC-AUC: **0.7745**

### 2.3 Random Forest Classifier
- **Grid Searched:**
  - `n_estimators` (number of trees): `[50, 100, 200]`
  - `max_depth` (maximum tree depth): `[6, 8, 10]`
  - `min_samples_split` (minimum samples to split an internal node): `[2, 5]`
- **Best Configuration:** `{'max_depth': 10, 'min_samples_split': 2, 'n_estimators': 200}`
- **Tuned Performance:** Accuracy: **91.96%**, ROC-AUC: **0.9618**

### 2.4 Support Vector Machine (SVM / SVC)
- **Grid Searched:**
  - `C` (regularization parameter): `[0.1, 1, 10]`
  - `kernel` (kernel type): `['linear', 'rbf']`
- **Best Configuration:** `{'C': 10, 'kernel': 'rbf'}`
- **Tuned Performance:** Accuracy: **91.07%**, ROC-AUC: **0.9343**

### 2.5 XGBoost Classifier (XGBClassifier)
- **Grid Searched:**
  - `n_estimators` (number of boosting rounds): `[50, 100]`
  - `learning_rate` (shrinkage factor): `[0.01, 0.1, 0.2]`
  - `max_depth` (maximum tree depth): `[3, 5, 7]`
- **Best Configuration:** `{'learning_rate': 0.01, 'max_depth': 5, 'n_estimators': 100}`
- **Tuned Performance:** Accuracy: **94.64%**, ROC-AUC: **0.9647**

---

## 3. Comparison of Baseline vs. Tuned Models
Hyperparameter tuning helped in:
1. **Preventing Overfitting:** Capping tree depths (`max_depth: 10` for RandomForest and `max_depth: 5` for XGBoost) prevents the models from memorizing the training set.
2. **Improving Generalization:** Regularization strength tuning (`C: 100` for Logistic Regression and `C: 10` for SVM) allowed decision boundaries to smooth out, translating to better test predictions.
3. **Optimizing Boosting Iterations:** Choosing a lower learning rate (`0.01`) coupled with `100` estimators allowed XGBoost to build weak learners incrementally, yielding the highest classification accuracy of **94.64%**.
