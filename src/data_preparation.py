import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')

def load_dataset(filename='dataset_part_1.csv'):
    """Step 1 & 2: Load the dataset and perform initial exploration."""
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")
    df = pd.read_csv(filepath)
    print(f"Dataset loaded: {filename} with shape {df.shape}")
    return df

def handle_missing_values(df):
    """Step 3.1: Identify and handle missing values."""
    print("Checking for missing values...")
    missing = df.isnull().sum()
    print(missing[missing > 0])
    
    # Example: Impute numeric columns with median, categorical with mode
    # Implement strategy based on dataset analysis
    df_clean = df.copy()
    return df_clean

def remove_duplicates(df):
    """Step 3.2: Detect and remove duplicate rows."""
    duplicate_count = df.duplicated().sum()
    print(f"Number of duplicate rows: {duplicate_count}")
    if duplicate_count > 0:
        df = df.drop_duplicates()
        print("Duplicates removed.")
    return df

def correct_data_types(df):
    """Step 3.3: Correct columns stored with incorrect data types (e.g. Date to datetime)."""
    # Example: Convert Launch Date to datetime
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        print("Converted 'Date' column to datetime.")
    return df

def handle_outliers(df, columns):
    """Step 3.4: Handle outliers using the IQR method."""
    df_clean = df.copy()
    for col in columns:
        if col in df_clean.columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            # Option: cap outliers or remove them
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
            print(f"Handled outliers in '{col}' using IQR bounds [{lower_bound:.2f}, {upper_bound:.2f}]")
    return df_clean

def encode_categorical(df, columns_to_encode):
    """Step 3.7: Encode categorical features (One-Hot or Label Encoding)."""
    print(f"Encoding categorical features: {columns_to_encode}")
    # Example: One-Hot Encoding
    df_encoded = pd.get_dummies(df, columns=columns_to_encode, drop_first=True)
    return df_encoded

def scale_numerical(df, columns_to_scale, method='standard'):
    """Step 3.8: Scale numerical features (StandardScaler or MinMaxScaler)."""
    df_scaled = df.copy()
    if method == 'standard':
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()
    
    df_scaled[columns_to_scale] = scaler.fit_transform(df_scaled[columns_to_scale])
    print(f"Scaled numerical features using {method} scaler: {columns_to_scale}")
    return df_scaled, scaler

def handle_class_imbalance(X, y):
    """Step 3.9: Address class imbalance if it exists (e.g., using SMOTE or class weights)."""
    # Placeholder for SMOTE or other balancing technique
    print(f"Class distribution: {np.bincount(y)}")
    return X, y

def prepare_pipeline(filename='dataset_part_1.csv'):
    """Executes the full preprocessing pipeline."""
    df = load_dataset(filename)
    df = handle_missing_values(df)
    df = remove_duplicates(df)
    df = correct_data_types(df)
    
    # Feature Selection, Encoding, Scaling steps will be custom tailored
    # Split into features (X) and target (y)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    
    return df

if __name__ == '__main__':
    prepare_pipeline()
