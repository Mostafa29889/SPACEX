# SpaceX Falcon 9 Landing Prediction - Step 9: Conclusion Report

This report presents the final summary, conclusions, model limitations, and suggested next steps for the SpaceX Falcon 9 first-stage landing success prediction project.

---

## 1. Project Summary & Key Findings
We successfully built a machine learning pipeline to predict whether the first stage of a SpaceX Falcon 9 booster will land successfully after launch.
- **Dataset:** Scraped launch history from Wikipedia/Kaggle (557 launches).
- **Target Variable:** Derived landing outcome success (91.20% baseline success rate).
- **Tuned Model Comparison:**
  - **XGBoost (Best Model):** Test Accuracy of **94.64%**, ROC-AUC of **0.9647**, and Recall of **100%**.
  - **Random Forest:** Test Accuracy of **91.96%**, ROC-AUC of **0.9618**.
  - **SVM:** Test Accuracy of **91.07%**, ROC-AUC of **0.9343**.
  - **Decision Tree:** Test Accuracy of **90.18%**, ROC-AUC of **0.7745**.
  - **Logistic Regression:** Test Accuracy of **88.39%**, ROC-AUC of **0.9402**.

---

## 2. Model Performance Discussion
1. **Gradient Boosting Supremacy:** XGBoost outperformed all other classifiers. The combination of regularized gradient boosting and grid-tuned tree depth allowed it to capture non-linear relationships without overfitting, yielding perfect recall and extremely high overall accuracy.
2. **Feature Influence:** Flight Number was the most important feature across tree-based models. This aligns with history: early SpaceX flights had low landing success rates (experimental phase), whereas later flights (highly experienced team, block upgrades) have close to 100% success.
3. **Imbalance Handling:** Despite the heavy target skew (only 8.8% failures), applying `class_weight='balanced'` and regularization constraints helped maintain high precision across all classifiers, ensuring model decisions are reliable.

---

## 3. Model Limitations
1. **Feature Space Skew/Bias:** The dataset relies on flight metadata (e.g. site, orbit, payload mass). It does not contain critical telemetry variables such as landing site wind speeds, atmospheric pressure, temperature profiles, or thruster gimbal angles.
2. **Data Quantity:** While 557 samples are sufficient for basic classifiers, deep learning architectures or advanced boosting algorithms would benefit from larger datasets (e.g. individual stage telemetry logs).
3. **Typos/Wikipedia Scraping Quality:** Data scraped from Wikipedia contains minor typographical variance (such as cite brackets `[` or approximate indicators `~` on mass numbers) which require active text cleaning before ingestion.

---

## 4. Suggested Future Work
1. **Telemetry & Weather Integration:** Retrieve and merge actual local weather metrics at the drone ship / landing zone locations at launch hour (wind velocity, sea state/waves, visibility).
2. **Booster Recycle/Life Metrics:** Add features tracking individual booster history (e.g., number of thermal cycles, age, refit time between flights) to capture wear-and-tear degradation.
3. **Advanced Sampling:** Apply SMOTE (Synthetic Minority Over-sampling Technique) or ADASYN to upsample failure events, helping the model define a more robust failure decision boundary.
4. **Deep Learning:** Train a multi-layer neural network with Dropout regularization to see if complex feature interactions further push accuracy.
5. **Real-time Deployment:** Containerize the FastAPI/Streamlit application with Docker and deploy to AWS Elastic Beanstalk or Heroku for cloud predictions.
