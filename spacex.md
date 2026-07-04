# SpaceX Falcon 9 Landing Prediction Project

## Objective

Build a machine learning model to predict whether a Falcon 9 first-stage landing will be successful using the provided dataset.

---

# Project Workflow

## 1. Understand the Dataset

- Read the dataset description.
- Identify the target variable.
- Classify features into:
  - Numerical
  - Categorical
  - Date/Time
- Describe each feature in your own words.

---

## 2. Data Exploration (EDA)

Perform exploratory data analysis to understand the data.

Suggested analyses:

- Dataset shape
- Data types
- Missing values
- Duplicate records
- Class distribution
- Summary statistics
- Correlation matrix
- Distribution of numerical variables
- Frequency of categorical variables
- Landing success by:
  - Launch Site
  - Orbit
  - Booster Version
  - Launch Year

Suggested visualizations:

- Histogram
- Boxplot
- Countplot
- Bar chart
- Heatmap
- Scatter plot

---

## 3. Data Preparation

This phase is critical and should be well documented.

### 3.1 Handle Missing Values

- Identify missing values.
- Explain why a strategy was chosen.
- Possible methods:
  - Remove rows
  - Remove columns
  - Mean/Median imputation
  - Mode imputation
  - Constant values

---

### 3.2 Remove Duplicates

- Detect duplicate rows.
- Remove duplicates if appropriate.

---

### 3.3 Correct Data Types

Examples:

- Convert dates to datetime
- Convert numerical columns stored as strings
- Convert categorical columns to category type

---

### 3.4 Handle Outliers

Possible methods:

- IQR - boxplot



---

### 3.6 Feature Selection

Determine which features should be used.

Possible techniques:

- Feature importance


---

### 3.7 Encode Categorical Features

Choose appropriate encoding methods.

Examples:

- One-Hot Encoding
- Label Encoding

Explain why the chosen method is appropriate.

---

### 3.8 Scale Numerical Features

If required, apply:

- StandardScaler
- MinMaxScaler

Explain which models require scaling.

---

### 3.9 Handle Class Imbalance

If the dataset is imbalanced, consider:

- Class weights
- SMOTE upsample / downsample

Compare results before and after balancing.

---

### 3.10 Train/Test Split

Split the dataset into:

- Training
- Validation (optional)
- Testing

Suggested ratios:

- 80/20

---

## 4. Model Development

Train different machine learning models.

Suggested models:

- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine
- XGBoost

---

## 5. Hyperparameter Tuning

Improve model performance using:

- Grid Search
- Random Search

Document:

- Parameters tested
- Best parameters
- Performance improvements

---

## 6. Experiment Tracking with MLflow

Use **MLflow** to track all experiments.

Each run should log:

### Parameters

- Model name
- Hyperparameters
- Random seed
- Feature selection method
- Scaling method

### Metrics

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

### Artifacts

- Confusion Matrix
- ROC Curve
- Feature Importance Plot
- Classification Report
- Trained Model


---

## 7. Model Evaluation

Compare all trained models.

Evaluate using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

Present results in a comparison table.

---

## 8. Model Interpretation

Explain:

- Most important features
- Why the best model performed well
- Limitations of the model

---

## 9. Final Report

The report should include:

- Introduction
- Problem Statement
- Dataset Description
- Data Preparation
- Exploratory Data Analysis
- Feature Engineering
- Model Development
- Experiment Tracking (MLflow)
- Model Evaluation
- Conclusion
- Future Improvements

---

# Submission Requirements

Submit the following:

github repo with previous details

---

# Bonus (Optional)

- Streamlit Dashboard
- FastAPI Deployment

---
