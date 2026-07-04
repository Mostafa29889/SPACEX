import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="SpaceX Falcon 9 Launch Dashboard", page_icon="🚀", layout="wide")

# Theme settings
st.markdown("""
<style>
    .main-title {
        font-family: 'Outfit', sans-serif;
        color: #ffffff;
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 25px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'><h1>SpaceX Falcon 9 Landing Prediction Dashboard 🚀</h1></div>", unsafe_allow_html=True)

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')

@st.cache_data
def load_data():
    path = os.path.join(DATA_DIR, 'spacex_web_scraped.csv')
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

df = load_data()

if df is not None:
    # Sidebar
    st.sidebar.header("Navigation")
    options = st.sidebar.radio("Go to:", ["Overview & EDA", "Interactive Predictor"])
    
    if options == "Overview & EDA":
        st.subheader("Exploratory Data Analysis")
        
        # Display key metrics
        col1, col2, col3 = st.columns(3)
        total_launches = len(df)
        success_rate = df['Class'].mean() * 100 if 'Class' in df.columns else 0.0
        
        col1.metric("Total Launches Analyzed", total_launches)
        col2.metric("Average Landing Success Rate", f"{success_rate:.1f}%")
        col3.metric("Number of Booster Versions", df['FlightNumber'].nunique() if 'FlightNumber' in df.columns else total_launches)
        
        st.write("### Launch Data Preview")
        st.dataframe(df.head(10))
        
        # Grid of visualizations
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.write("#### Success Rate by Launch Site")
            if 'LaunchSite' in df.columns and 'Class' in df.columns:
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.barplot(data=df, x='LaunchSite', y='Class', errorbar=None, palette='viridis', ax=ax)
                ax.set_ylabel('Success Rate')
                ax.set_ylim(0, 1.05)
                st.pyplot(fig)
                
        with col_right:
            st.write("#### Success Rate by Orbit Type")
            if 'Orbit' in df.columns and 'Class' in df.columns:
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.barplot(data=df, x='Orbit', y='Class', errorbar=None, palette='magma', ax=ax)
                ax.set_ylabel('Success Rate')
                ax.set_ylim(0, 1.05)
                plt.xticks(rotation=45)
                st.pyplot(fig)
                
    elif options == "Interactive Predictor":
        st.subheader("Predict Falcon 9 Landing Success")
        st.write("Enter launch details to estimate the probability of a successful landing.")
        
        # Layout forms
        col1, col2 = st.columns(2)
        with col1:
            flight_num = st.number_input("Flight Number", min_value=1, max_value=200, value=50)
            payload_mass = st.number_input("Payload Mass (kg)", min_value=0, max_value=20000, value=5000)
            orbit = st.selectbox("Orbit", ["LEO", "ISS", "PO", "GTO", "ES-L1", "SSO", "HEO", "MEO", "VLEO", "SO", "GEO"])
        with col2:
            launch_site = st.selectbox("Launch Site", ["CCSFS SLC 40", "KSC LC 39A", "VAFB SLC 4E"])
            flights = st.slider("Number of Flights with this Booster", 1, 15, 1)
            grid_fins = st.checkbox("Grid Fins Enabled", value=True)
            legs = st.checkbox("Landing Legs Enabled", value=True)
            
        if st.button("Predict Outcome"):
            # Mock model logic for demonstration
            # In production, load the trained model (.pkl) from models/
            base_prob = 0.5
            if legs: base_prob += 0.2
            if grid_fins: base_prob += 0.15
            if launch_site == "KSC LC 39A": base_prob += 0.05
            if payload_mass > 3000 and payload_mass < 6000: base_prob += 0.05
            
            final_prob = min(max(base_prob, 0.0), 1.0)
            success = final_prob >= 0.65
            
            st.write("---")
            if success:
                st.success(f"### Landing Prediction: SUCCESS 🎉 (Probability: {final_prob:.1%})")
            else:
                st.error(f"### Landing Prediction: FAILURE 💥 (Probability: {final_prob:.1%})")
else:
    st.warning("Please download the datasets first to populate the dashboard! Run `src/download_data.py` or use our script.")
