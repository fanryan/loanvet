import logging
import pandas as pd
from typing import Dict, Union

def predict_single(record: Dict[str, Union[float, int]], model, feature_list, threshold) -> Dict[str, Union[int, float]]:
    try:
        df = pd.DataFrame([record])
        df = df[feature_list]
        proba = model.predict_proba(df)[:, 1][0]
        label = int(proba >= threshold)
        return {"label": label, "probability": proba}
    except Exception as e:
        logging.error(f"❌ Prediction error: {e}")
        raise

def predict_batch(df: pd.DataFrame, model, feature_list, threshold) -> pd.DataFrame:
    try:
        df = df[feature_list]
        proba = model.predict_proba(df)[:, 1]
        labels = (proba >= threshold).astype(int)
        df_result = df.copy()
        df_result["probability"] = proba
        df_result["label"] = labels
        return df_result
    except Exception as e:
        logging.error(f"❌ Batch prediction error: {e}")
        raise