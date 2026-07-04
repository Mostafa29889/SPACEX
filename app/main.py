import os
import pickle
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(
    title="SpaceX Falcon 9 Landing Prediction API",
    description="FastAPI endpoint for predicting Falcon 9 first-stage landing outcomes.",
    version="1.0.0"
)

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best_model.pkl')
PROCESSED_PATH = os.path.join(BASE_DIR, 'data', 'processed_data.pkl')

class PredictionPayload(BaseModel):
    FlightNumber: int
    PayloadMass: float
    Orbit: str
    LaunchSite: str
    VersionBooster: str

@app.get("/")
def read_root():
    return {
        "message": "Welcome to the SpaceX Landing Prediction API!",
        "endpoints": {
            "/predict": "POST method for making predictions",
            "/health": "GET method for health checks"
        }
    }

@app.get("/health")
def health_check():
    model_loaded = os.path.exists(MODEL_PATH)
    processed_loaded = os.path.exists(PROCESSED_PATH)
    return {
        "status": "healthy",
        "model_loaded": model_loaded,
        "preprocessor_metadata_loaded": processed_loaded
    }

@app.post("/predict")
def predict(payload: PredictionPayload):
    if not os.path.exists(MODEL_PATH) or not os.path.exists(PROCESSED_PATH):
        raise HTTPException(
            status_code=503, 
            detail="Model or Preprocessor metadata not found. Please train the models first."
        )
    
    try:
        # Load best model
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
            
        # Load preprocessor metadata
        with open(PROCESSED_PATH, 'rb') as f:
            meta = pickle.load(f)
            
        feature_names = meta['feature_names']
        scaler = meta['scaler']
        
        # Prepare input data matching model features
        df_model = pd.DataFrame(0, index=[0], columns=feature_names)
        
        df_model['Flight No.'] = [payload.FlightNumber]
        df_model['Payload mass'] = [payload.PayloadMass]
        
        # Scale continuous features
        df_model[['Flight No.', 'Payload mass']] = scaler.transform(df_model[['Flight No.', 'Payload mass']])
        
        # Clean categories (remove brackets, spaces, etc.) to match encoder names
        cleaned_site = payload.LaunchSite.replace('[', '').replace(']', '').replace('<', '').strip()
        cleaned_orbit = payload.Orbit.replace('[', '').replace(']', '').replace('<', '').strip()
        cleaned_booster = payload.VersionBooster.replace('[', '').replace(']', '').replace('<', '').strip()
        
        site_col = f"Launch site_{cleaned_site}"
        orbit_col = f"Orbit_{cleaned_orbit}"
        booster_col = f"Version Booster_{cleaned_booster}"
        
        if site_col in df_model.columns:
            df_model[site_col] = 1
        if orbit_col in df_model.columns:
            df_model[orbit_col] = 1
        if booster_col in df_model.columns:
            df_model[booster_col] = 1
            
        # Predict
        prediction = int(model.predict(df_model)[0])
        prob = float(model.predict_proba(df_model)[0][1]) if hasattr(model, "predict_proba") else 0.5
        
        return {
            "prediction": prediction,
            "prediction_label": "Success" if prediction == 1 else "Failure",
            "probability": prob
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {str(e)}")
