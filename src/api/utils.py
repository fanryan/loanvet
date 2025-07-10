import logging
import pandas as pd
import numpy as np
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

import numpy as np

def preprocess(raw_data: dict) -> dict:
    """
    Transform raw input features into the engineered features
    needed by the model pipeline.
    """
    processed = {}

    # Handle missing flags for monthly income and dependents
    monthly_income = raw_data.get("monthly_income")
    number_of_dependents = raw_data.get("number_of_dependents")
    
    processed["MonthlyIncome_missing_flag"] = 0 if monthly_income and monthly_income > 0 else 1
    processed["NumberOfDependents_missing_flag"] = 0 if number_of_dependents is not None else 1
    
    # Log transformations - add small epsilon to avoid log(0)
    processed["RevolvingUtilizationOfUnsecuredLines_log"] = np.log1p(raw_data.get("revolving_utilization_of_unsecured_lines", 0))
    processed["MonthlyIncome_log"] = np.log1p(monthly_income) if monthly_income else 0
    processed["DebtRatio_log"] = np.log1p(raw_data.get("debt_ratio", 0))
    processed["TotalDelinquencies_log"] = np.log1p(raw_data.get("total_delinquencies", 0))

    # Flags and engineered features
    processed["HighUtilizationFlag"] = 1 if raw_data.get("revolving_utilization_of_unsecured_lines", 0) > 0.75 else 0

    # IncomePerCreditLine = monthly_income / number_of_open_credit_lines_and_loans
    open_lines = raw_data.get("number_of_open_credit_lines_and_loans", 1)
    processed["IncomePerCreditLine"] = monthly_income / open_lines if open_lines > 0 else 0

    # Age groups: (one-hot encoding)
    age = raw_data.get("age", 0)
    processed["AgeGroup_MidAge"] = 1 if 35 <= age < 60 else 0
    processed["AgeGroup_Senior"] = 1 if age >= 60 else 0

    # Dependents groups: (one-hot encoding)
    dep = number_of_dependents if number_of_dependents is not None else 0
    processed["DependentsGroup_Small"] = 1 if 1 <= dep <= 3 else 0
    processed["DependentsGroup_Large"] = 1 if dep > 3 else 0

    # Interaction features
    processed["Util_x_Late"] = raw_data.get("revolving_utilization_of_unsecured_lines", 0) * raw_data.get("total_delinquencies", 0)
    processed["IncomePerDependent"] = (monthly_income / dep) if dep > 0 else 0
    processed["CreditLines_x_Delinquencies"] = open_lines * raw_data.get("total_delinquencies", 0)

    # Add raw numeric features required by the model
    processed["age"] = age
    processed["NumberOfOpenCreditLinesAndLoans"] = open_lines
    processed["NumberRealEstateLoansOrLines"] = raw_data.get("number_real_estate_loans_or_lines", 0)
    processed["NumberOfDependents"] = dep

    return processed