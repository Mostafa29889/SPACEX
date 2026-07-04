import os
import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(
    title="SpaceX Falcon 9 Landing Prediction API",
    description="FastAPI endpoint for predicting Falcon 9 first-stage landing outcomes.",
    version="1.0.0"
)

# Define models path
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best_model.pkl')

class PredictionPayload(BaseModel):
    FlightNumber: int
    PayloadMass: float
    Orbit: str
    LaunchSite: str
    Flights: int
    GridFins: bool
    Reused: bool
    Legs: bool
    Block: float
    ReusedCount: int

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
    return {
        "status": "healthy",
        "model_loaded": model_loaded
    }

@app.post("/predict")
def predict(payload: PredictionPayload):
    # Try loading the model
    if not os.path.exists(MODEL_PATH):
        # Fallback to simple rule-based mock prediction if model is not trained yet
        print("Model file not found. Running rule-based inference fallback.")
        prob = 0.5
        if payload.Legs: prob += 0.2
        if payload.GridFins: prob += 0.15
        if payload.ReusedCount > 2: prob += 0.05
        
        prob = min(max(prob, 0.0), 1.0)
        prediction = 1 if prob >= 0.6 else 0
        return {
            "prediction": prediction,
            "prediction_label": "Success" if prediction == 1 else "Failure",
            "probability": prob,
            "fallback_model_used": True
        }
    
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
            
        # Format payload into DataFrame/array for the model
        # Modify feature preparation to match the training encoding/pipeline
        features = [[
            payload.FlightNumber, payload.PayloadMass, payload.Flights,
            int(payload.GridFins), int(payload.Reused), int(payload.Legs),
            payload.Block, payload.ReusedCount
        ]]
        
        prediction = int(model.predict(features)[0])
        prob = float(model.predict_proba(features)[0][1])
        
        return {
            "prediction": prediction,
            "prediction_label": "Success" if prediction == 1 else "Failure",
            "probability": prob,
            "fallback_model_used": False
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {str(e)}")
