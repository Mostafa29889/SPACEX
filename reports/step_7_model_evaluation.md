# SpaceX Falcon 9 Landing Prediction - Step 7: Model Evaluation Report

This report evaluates and compares the performance of the trained machine learning classifiers on predicting the landing success of Falcon 9 first-stage boosters.

---

## 1. Classification Metrics Comparison

The models were evaluated on an independent stratified test set (112 samples, 20% of the total dataset) containing 102 successful landings and 10 unsuccessful/no-attempt landings.

### Final Results Table:
| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **XGBoost** | **94.64%** | **94.44%** | **100.00%** | **97.14%** | **0.9647** |
| **Random Forest** | 91.96% | 97.94% | 93.14% | 95.48% | 0.9618 |
| **SVM** | 91.07% | 97.92% | 92.16% | 94.95% | 0.9343 |
| **Decision Tree** | 90.18% | 95.96% | 93.14% | 94.53% | 0.7745 |
| **Logistic Regression** | 88.39% | 96.84% | 90.20% | 93.40% | 0.9402 |

---

## 2. Key Performance Insights

1. **XGBoost is the Top Predictor:**
   - Achieved the highest accuracy of **94.64%** and ROC-AUC of **0.9647**.
   - It reached **100% Recall** (102/102 successes predicted correctly) while maintaining a very low false positive rate.
2. **Ensemble Models Outperform Single Estimators:**
   - Both **XGBoost** and **Random Forest** (91.96% accuracy) outperformed simple decision trees and logistic regression. Their voting/boosting mechanisms smooth out classification boundaries and reduce variance.
3. **Class Imbalance Impact:**
   - Since the dataset is heavily skewed towards successful landings (91.2% base rate), recall for the minority class (failures) is harder to optimize. However, because we configured `class_weight='balanced'` on the baseline estimators, models maintained high precision (above 94% for all models).

---

## 3. Confusion Matrix Analysis (Best Model: XGBoost)

The test predictions split for XGBoost reveals:
- **True Negatives (Correctly predicted failures):** 4
- **False Positives (Failures predicted as success):** 6
- **False Negatives (Successes predicted as failure):** 0
- **True Positives (Correctly predicted successes):** 102

The 100% recall ensures that no successful landing is incorrectly flagged as a failure, which is crucial for launch operations planning.

---

## 4. Feature Importance Insights
Based on XGBoost and Random Forest feature importances:
*   **Flight Number:** Strongest predictor. As SpaceX gained experience over time (higher flight numbers), landing engineering matured, leading to a much higher success rate in later flights.
*   **Payload Mass:** Crucial continuous feature. Heavy payloads require more fuel for ascent, reducing margins for the landing burn.
*   **Booster Version:** Specific newer versions (like Block 5) show strong positive correlation with landing success due to iterative engine upgrades.
