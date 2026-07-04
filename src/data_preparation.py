import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')

def load_dataset(filename='spacex_web_scraped.csv'):
    """Step 1 & 2: Load the dataset and perform initial exploration."""
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")
    df = pd.read_csv(filepath)
    print(f"Dataset loaded: {filename} with shape {df.shape}")
    return df

def handle_missing_values(df):
    """Step 3.1: Identify and handle missing values."""
    print("\n--- Step 3.1: Handling Missing Values ---")
    df_clean = df.copy()
    
    # Check initial nulls
    nulls = df_clean.isnull().sum()
    print("Missing values before imputation:")
    print(nulls[nulls > 0])
    
    # Impute Orbit with mode
    if 'Orbit' in df_clean.columns:
        orbit_mode = df_clean['Orbit'].mode()[0] if not df_clean['Orbit'].mode().empty else 'LEO'
        df_clean['Orbit'] = df_clean['Orbit'].fillna(orbit_mode)
        
    # Impute Customer with 'Unknown'
    if 'Customer' in df_clean.columns:
        df_clean['Customer'] = df_clean['Customer'].fillna('Unknown')
        
    # Impute Payload with 'Unknown'
    if 'Payload' in df_clean.columns:
        df_clean['Payload'] = df_clean['Payload'].fillna('Unknown')
        
    # Suffix/Commas cleaning and numerical conversion for Payload mass
    if 'Payload mass' in df_clean.columns:
        df_clean['Payload mass'] = pd.to_numeric(
            df_clean['Payload mass'].astype(str).str.replace('~', '').str.replace(',', '').str.replace('kg', '').str.strip(), 
            errors='coerce'
        )
        # Impute missing mass values with the median mass
        mass_median = df_clean['Payload mass'].median()
        df_clean['Payload mass'] = df_clean['Payload mass'].fillna(mass_median)
        
    # Convert Flight No. to numeric
    if 'Flight No.' in df_clean.columns:
        df_clean['Flight No.'] = pd.to_numeric(df_clean['Flight No.'], errors='coerce')
        df_clean['Flight No.'] = df_clean['Flight No.'].fillna(0)
        
    # Clean categorical columns to remove brackets or typos
    for col in ['Version Booster', 'Orbit', 'Launch site']:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(str).str.replace('[', '', regex=False) \
                                                     .str.replace(']', '', regex=False) \
                                                     .str.replace('<', '', regex=False) \
                                                     .str.strip()
        
    print("Missing values after imputation:")
    print(df_clean.isnull().sum()[df_clean.isnull().sum() > 0])
    return df_clean

def remove_duplicates(df):
    """Step 3.2: Detect and remove duplicate rows."""
    print("\n--- Step 3.2: Handling Duplicates ---")
    duplicate_count = df.duplicated().sum()
    print(f"Number of duplicate rows found: {duplicate_count}")
    if duplicate_count > 0:
        df = df.drop_duplicates()
        print("Duplicates removed successfully.")
    return df

def correct_data_types(df):
    """Step 3.3: Correct columns stored with incorrect data types (e.g. Date to datetime)."""
    print("\n--- Step 3.3: Correcting Data Types ---")
    df_clean = df.copy()
    if 'Date' in df_clean.columns:
        df_clean['Date'] = pd.to_datetime(df_clean['Date'], errors='coerce')
        # Impute missing dates with forward fill
        df_clean['Date'] = df_clean['Date'].ffill()
        print("Converted 'Date' column to datetime.")
    return df_clean

def handle_outliers(df, columns):
    """Step 3.4: Handle outliers using the IQR capping method."""
    print("\n--- Step 3.4: Handling Outliers ---")
    df_clean = df.copy()
    for col in columns:
        if col in df_clean.columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Count outliers
            outliers = df_clean[(df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)]
            print(f"Feature '{col}': found {len(outliers)} outliers outside range [{lower_bound:.2f}, {upper_bound:.2f}]")
            
            # Cap outliers to bounds instead of deleting data (retains information)
            df_clean[col] = np.clip(df_clean[col], lower_bound, upper_bound)
            print(f"Capped outliers in '{col}' to range [{lower_bound:.2f}, {upper_bound:.2f}]")
    return df_clean

def encode_categorical(df, columns_to_encode):
    """Step 3.7: Encode categorical features using One-Hot Encoding."""
    print("\n--- Step 3.7: Encoding Categorical Features ---")
    # One-hot encode the specified features and drop the first category to avoid multicollinearity
    df_encoded = pd.get_dummies(df, columns=columns_to_encode, drop_first=True)
    # Convert booleans to 0/1 integers
    bool_cols = df_encoded.select_dtypes(include=['bool']).columns
    df_encoded[bool_cols] = df_encoded[bool_cols].astype(int)
    
    # Clean column names to make them compatible with XGBoost
    df_encoded.columns = df_encoded.columns.str.replace('[', '', regex=False) \
                                           .str.replace(']', '', regex=False) \
                                           .str.replace('<', '', regex=False)
                                           
    print(f"One-Hot encoded categorical columns: {columns_to_encode}")
    return df_encoded

def scale_numerical(X, columns_to_scale):
    """Step 3.8: Scale numerical features using StandardScaler."""
    print("\n--- Step 3.8: Scaling Numerical Features ---")
    X_scaled = X.copy()
    scaler = StandardScaler()
    X_scaled[columns_to_scale] = scaler.fit_transform(X_scaled[columns_to_scale])
    print(f"Standard-scaled features: {columns_to_scale}")
    return X_scaled, scaler

def handle_class_imbalance(X, y):
    """Step 3.9: Address class imbalance.
    Since we have 91% success class imbalance, we print stats and will configure
    classifiers to use class_weight='balanced' or compute class weights dynamically.
    """
    print("\n--- Step 3.9: Checking Class Imbalance ---")
    class_counts = np.bincount(y)
    print(f"Class 0 (Failure): {class_counts[0]} launches ({class_counts[0]/len(y):.2%})")
    print(f"Class 1 (Success): {class_counts[1]} launches ({class_counts[1]/len(y):.2%})")
    return X, y

def prepare_pipeline(filename='spacex_web_scraped.csv'):
    """Executes the full preprocessing pipeline."""
    # 1. Load data
    df = load_dataset(filename)
    
    # 2. Derive Target Class (Class = 1 if 'Success' in landing else 0)
    df['Class'] = df['Booster landing'].apply(lambda x: 1 if 'Success' in str(x) else 0)
    
    # 3. Preprocess
    df = handle_missing_values(df)
    df = remove_duplicates(df)
    df = correct_data_types(df)
    
    # 4. Outliers capping on numerical features
    df = handle_outliers(df, ['Payload mass'])
    
    # 5. Feature Selection
    # Target variable
    y = df['Class'].values
    
    # Select features for training
    features_to_keep = ['Flight No.', 'Payload mass', 'Launch site', 'Orbit', 'Version Booster']
    df_features = df[features_to_keep]
    
    # One-hot encode categorical features
    X = encode_categorical(df_features, ['Launch site', 'Orbit', 'Version Booster'])
    
    # Scale continuous numerical features
    X, scaler = scale_numerical(X, ['Flight No.', 'Payload mass'])
    
    # Handle Class Imbalance stats
    X, y = handle_class_imbalance(X, y)
    
    # 6. Train-test split (80% train, 20% test)
    print("\n--- Step 3.10: Train/Test Split ---")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    print(f"Training set: X_train shape {X_train.shape}, y_train shape {y_train.shape}")
    print(f"Testing set: X_test shape {X_test.shape}, y_test shape {y_test.shape}")
    
    # Save the prepared datasets to disk
    processed_path = os.path.join(DATA_DIR, 'processed_data.pkl')
    with open(processed_path, 'wb') as f:
        pickle.dump({
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': list(X.columns),
            'scaler': scaler
        }, f)
    print(f"Successfully saved processed dataset split to {processed_path}")
    
    return X_train, X_test, y_train, y_test, X.columns

if __name__ == '__main__':
    prepare_pipeline()
