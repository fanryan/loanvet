import json
import logging
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import RootModel
from typing import Dict
from xgboost import XGBClassifier

from src.api.utils import predict_single, preprocess  # Assuming preprocess is in utils

# Configure logging
logging.basicConfig(level=logging.INFO)

# Paths to model and metadata files
MODEL_PATH = "models/final/xgb_final_model.joblib"
METADATA_PATH = "models/final/xgb_final_metadata.json"

# Load model
try:
    model: XGBClassifier = joblib.load(MODEL_PATH)
    logging.info("✅ Model loaded successfully.")
except Exception as e:
    logging.error(f"❌ Failed to load model: {e}")
    raise RuntimeError("Failed to load model.")

# Load metadata (features and threshold)
try:
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)
    threshold = metadata.get("threshold", 0.5)
    feature_list = metadata.get("features", [])
    logging.info(f"✅ Metadata loaded successfully with {len(feature_list)} features.")
except Exception as e:
    logging.error(f"❌ Failed to load metadata: {e}")
    raise RuntimeError("Failed to load metadata.")

# Pydantic model for validating raw input data (before preprocessing)
class RawInputRequest(RootModel[Dict[str, float]]):
    pass

app = FastAPI(title="LoanVet Credit Risk Model API")

# CORS settings for your frontend domain(s)
origins = [
    "https://fan-loanvet.streamlit.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict_endpoint(raw_input: RawInputRequest):
    """
    Accepts raw input features, preprocesses them into engineered features,
    validates feature completeness, then predicts credit risk using the loaded model.
    """
    raw_data = raw_input.root

    # Preprocess raw input to engineered features
    try:
        processed_data = preprocess(raw_data)
    except Exception as e:
        logging.error(f"Error during preprocessing: {e}")
        raise HTTPException(status_code=400, detail=f"Preprocessing error: {e}")

    # Check for missing or extra features after preprocessing
    missing = set(feature_list) - set(processed_data.keys())
    extra = set(processed_data.keys()) - set(feature_list)

    if missing:
        raise HTTPException(status_code=422, detail=f"Missing features after preprocessing: {missing}")
    if extra:
        raise HTTPException(status_code=422, detail=f"Unexpected features after preprocessing: {extra}")

    # Predict using the processed input features
    try:
        result = predict_single(processed_data, model, feature_list, threshold)
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed.")

    logging.info(f"✅ Prediction made: {result}")
    return {"prediction": result}