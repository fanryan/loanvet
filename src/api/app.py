import json
import logging
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import RootModel
from typing import Dict
from xgboost import XGBClassifier

from src.api.utils import predict_single

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load model and metadata
MODEL_PATH = "models/final/xgb_final_model.joblib"
METADATA_PATH = "models/final/xgb_final_metadata.json"

try:
    model: XGBClassifier = joblib.load(MODEL_PATH)
    logging.info("✅ Model loaded.")
except Exception as e:
    logging.error(f"❌ Failed to load model: {e}")
    raise RuntimeError("Failed to load model.")

try:
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)
    threshold = metadata.get("threshold", 0.5)
    feature_list = metadata.get("features", [])
    logging.info(f"✅ Metadata loaded. {len(feature_list)} features.")
except Exception as e:
    logging.error(f"❌ Failed to load metadata: {e}")
    raise RuntimeError("Failed to load metadata.")

# Pydantic model for input validation
class PredictionRequest(RootModel[Dict[str, float]]):
    pass

app = FastAPI(title="LoanVet Credit Risk Model API")

@app.post("/predict")
async def predict_endpoint(input_data: PredictionRequest):
    input_dict = input_data.root

    missing = set(feature_list) - set(input_dict.keys())
    extra = set(input_dict.keys()) - set(feature_list)

    if missing:
        raise HTTPException(status_code=422, detail=f"Missing features: {missing}")
    if extra:
        raise HTTPException(status_code=422, detail=f"Unexpected features: {extra}")

    result = predict_single(input_dict, model, feature_list, threshold)
    logging.info(f"✅ Prediction made: {result}")
    return {"prediction": result}