import os
import pickle
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="SpaceX Falcon 9 Launch Dashboard", page_icon="🚀", layout="wide")

# Theme settings and CSS styling
st.markdown("""
<style>
    .main-title {
        font-family: 'Outfit', sans-serif;
        color: #ffffff;
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 25px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .metric-box {
        background-color: #1e293b;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #3b82f6;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'><h1>SpaceX Falcon 9 Landing Success Dashboard 🚀</h1></div>", unsafe_allow_html=True)

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best_model.pkl')
PROCESSED_PATH = os.path.join(BASE_DIR, 'data', 'processed_data.pkl')

@st.cache_data
def load_data():
    path = os.path.join(DATA_DIR, 'spacex_web_scraped.csv')
    if os.path.exists(path):
        df = pd.read_csv(path)
        # Clean data for EDA visuals to match inspect/training logic
        df['Class'] = df['Booster landing'].apply(lambda x: 1 if 'Success' in str(x) else 0)
        df['Payload mass'] = pd.to_numeric(
            df['Payload mass'].astype(str).str.replace('~', '').str.replace(',', '').str.replace('kg', '').str.strip(), 
            errors='coerce'
        )
        df['Payload mass'] = df['Payload mass'].fillna(df['Payload mass'].median())
        df['Flight No.'] = pd.to_numeric(df['Flight No.'], errors='coerce')
        # Remove brackets from categories
        for col in ['Version Booster', 'Orbit', 'Launch site']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace('[', '', regex=False) \
                                                         .str.replace(']', '', regex=False) \
                                                         .str.replace('<', '', regex=False) \
                                                         .str.strip()
        return df
    return None

df = load_data()

if df is not None:
    # Sidebar Navigation
    st.sidebar.header("Control Panel")
    options = st.sidebar.radio("Go to Page:", ["Overview & EDA Dashboard", "ML Predictive Engine"])
    
    if options == "Overview & EDA Dashboard":
        st.subheader("📊 SpaceX Launch Dataset Profile")
        
        # Display key metrics
        col1, col2, col3 = st.columns(3)
        total_launches = len(df)
        success_rate = df['Class'].mean() * 100
        total_boosters = df['Version Booster'].nunique()
        
        with col1:
            st.markdown(f"<div class='metric-box'><h3>Total Launches Analyzed</h3><h2>{total_launches}</h2></div>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<div class='metric-box'><h3>Average Landing Success</h3><h2>{success_rate:.2f}%</h2></div>", unsafe_allow_html=True)
        with col3:
            st.markdown(f"<div class='metric-box'><h3>Booster Versions</h3><h2>{total_boosters}</h2></div>", unsafe_allow_html=True)
            
        st.write("### 📋 Scraped Landing Records Preview")
        st.dataframe(df[['Flight No.', 'Launch site', 'Payload mass', 'Orbit', 'Version Booster', 'Booster landing', 'Class']].head(10))
        
        # Plots Grid
        st.write("### 📈 Visual EDA Charts")
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.write("#### Success Rate by Launch Site")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(data=df, x='Launch site', y='Class', errorbar=None, hue='Launch site', legend=False, palette='Blues_r', ax=ax)
            ax.set_ylabel('Success Rate')
            ax.set_ylim(0, 1.05)
            st.pyplot(fig)
            
        with col_right:
            st.write("#### Success Rate by Orbit Type")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(data=df, x='Orbit', y='Class', errorbar=None, hue='Orbit', legend=False, palette='viridis', ax=ax)
            ax.set_ylabel('Success Rate')
            ax.set_ylim(0, 1.05)
            plt.xticks(rotation=45)
            st.pyplot(fig)
            
    elif options == "ML Predictive Engine":
        st.subheader("🔮 Predict Falcon 9 Landing Success Probability")
        st.write("Configure details of a future Falcon 9 flight to estimate its landing success using our tuned XGBoost model.")
        
        # Load preprocessor
        if os.path.exists(MODEL_PATH) and os.path.exists(PROCESSED_PATH):
            with open(PROCESSED_PATH, 'rb') as f:
                meta = pickle.load(f)
            feature_names = meta['feature_names']
            scaler = meta['scaler']
            
            # Select boxes configured dynamically using unique values in dataset
            unique_sites = sorted(list(df['Launch site'].unique()))
            unique_orbits = sorted(list(df['Orbit'].unique()))
            unique_boosters = sorted(list(df['Version Booster'].unique()))
            
            col1, col2 = st.columns(2)
            with col1:
                max_flight = df['Flight No.'].dropna().max()
                default_flight_val = int(max_flight + 1) if not pd.isna(max_flight) else 100
                flight_num = st.number_input("Flight Number (Experience Index)", min_value=1, max_value=2000, value=default_flight_val)
                payload_mass = st.number_input("Payload Mass (kg)", min_value=0, max_value=20000, value=5000)
                orbit = st.selectbox("Destination Orbit", unique_orbits, index=unique_orbits.index('GTO') if 'GTO' in unique_orbits else 0)
            with col2:
                launch_site = st.selectbox("Launch Site", unique_sites)
                booster_version = st.selectbox("Booster Version Variant", unique_boosters, index=unique_boosters.index('F9 B5') if 'F9 B5' in unique_boosters else 0)
                
            if st.button("Calculate Success Probability"):
                # Prepare input DataFrame
                df_model = pd.DataFrame(0, index=[0], columns=feature_names)
                df_model['Flight No.'] = [flight_num]
                df_model['Payload mass'] = [payload_mass]
                
                # Scale
                df_model[['Flight No.', 'Payload mass']] = scaler.transform(df_model[['Flight No.', 'Payload mass']])
                
                # Encode dummy variables
                site_col = f"Launch site_{launch_site}"
                orbit_col = f"Orbit_{orbit}"
                booster_col = f"Version Booster_{booster_version}"
                
                if site_col in df_model.columns:
                    df_model[site_col] = 1
                if orbit_col in df_model.columns:
                    df_model[orbit_col] = 1
                if booster_col in df_model.columns:
                    df_model[booster_col] = 1
                
                # Run Inference
                with open(MODEL_PATH, 'rb') as f:
                    model = pickle.load(f)
                    
                prediction = int(model.predict(df_model)[0])
                prob = float(model.predict_proba(df_model)[0][1])
                
                st.markdown("---")
                if prediction == 1:
                    st.success(f"### 🚀 landing prediction: SUCCESS 🎉")
                    st.markdown(f"The model estimates a **{prob * 100:.2f}%** probability that the Falcon 9 first stage will land successfully.")
                else:
                    st.error(f"### 💥 landing prediction: FAILURE/NO ATTEMPT")
                    st.markdown(f"The model estimates a **{(1 - prob) * 100:.2f}%** probability of failure (landing success chance: **{prob * 100:.2f}%**).")
        else:
            st.warning("⚠️ Model and Preprocessor configurations not found. Please train models first using: `python -m src.model_development`.")
else:
    st.error("Dataset not found. Please run the download script first.")
