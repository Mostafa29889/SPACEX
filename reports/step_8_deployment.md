# SpaceX Falcon 9 Landing Prediction - Step 8: Deployment Report

This report documents the implementation and local verification of **Step 8: Model Deployment** for the SpaceX Falcon 9 first-stage landing success prediction.

---

## 1. FastAPI Web Service Endpoint

We built a backend REST API using **FastAPI** to serve predictions to external clients.

- **FastAPI Code:** [app/main.py](file:///c:/Users/User/Desktop/SPACEX-main/SPACEX-main/app/main.py)
- **Status:** Active & running locally on port `8000`

### 1.1 How to Start the API Service
Run the following command in the project root:
```bash
python -m uvicorn app.main:app --port 8000
```

### 1.2 Health Check Endpoint
- **Request:** `GET http://127.0.0.1:8000/health`
- **Response:**
  ```json
  {
    "status": "healthy",
    "model_loaded": true,
    "preprocessor_metadata_loaded": true
  }
  ```

### 1.3 Inference Predict Endpoint
- **Request:** `POST http://127.0.0.1:8000/predict`
- **Headers:** `Content-Type: application/json`
- **Body:**
  ```json
  {
    "FlightNumber": 120,
    "PayloadMass": 6000.0,
    "Orbit": "GTO",
    "LaunchSite": "KSC LC 39A",
    "VersionBooster": "F9 B5"
  }
  ```
- **Response:**
  ```json
  {
    "prediction": 1,
    "prediction_label": "Success",
    "probability": 0.9130491614341736
  }
  ```

---

## 2. Streamlit Interactive Dashboard

We built a premium user-facing web dashboard using **Streamlit** to present Exploratory Data Analysis (EDA) and allow users to run predictions interactively.

- **Streamlit Code:** [app/dashboard.py](file:///c:/Users/User/Desktop/SPACEX-main/SPACEX-main/app/dashboard.py)
- **Status:** Active & running locally on port `8501`

### 2.1 How to Start the Dashboard App
Run the following command in the project root:
```bash
python -m streamlit run app/dashboard.py --server.port 8501
```

### 2.2 Features Included:
1. **Overview & EDA Dashboard:**
   - Displays real-time KPIs (Total launches analyzed, Average landing success rate, Unique booster versions).
   - Renders interactive bar charts showing success rate broken down by launch pad and orbit type.
   - Shows a table preview of the scraped raw database records.
2. **ML Predictive Engine:**
   - Provides input controls for Flight Number, Payload Mass, Destination Orbit, Launch Site, and Booster Version.
   - Values in select dropdowns are dynamically extracted from the dataset categories.
   - Outputs visual green `SUCCESS` or red `FAILURE` notifications with computed model probabilities.
