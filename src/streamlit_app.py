import streamlit as st
import requests

st.title("LoanVet Credit Risk Prediction")

input_data = {
    "age": st.number_input("Age", min_value=18, max_value=100, value=35),
    "monthly_income": st.number_input("Monthly Income", min_value=0.0, value=5000.0),
    "number_of_dependents": st.number_input("Number of Dependents", min_value=0, value=2),
    "number_of_open_credit_lines_and_loans": st.number_input("Open Credit Lines and Loans", min_value=0, value=5),
    "number_real_estate_loans_or_lines": st.number_input("Real Estate Loans or Lines", min_value=0, value=1),
    "debt_ratio": st.number_input("Debt Ratio", min_value=0.0, value=0.2),
    "revolving_utilization_of_unsecured_lines": st.number_input("Revolving Utilization of Unsecured Lines", min_value=0.0, max_value=10.0, value=0.12),
    "total_delinquencies": st.number_input("Total Delinquencies", min_value=0, value=0),
}

if st.button("Predict"):
    try:
        API_URL = "https://loanvet-b7d87d467e26.herokuapp.com/predict"
        response = requests.post(API_URL, json=input_data)
        response.raise_for_status()
        result = response.json()
        label = result["prediction"]["label"]
        proba = result["prediction"]["probability"]

        st.success(f"Prediction: {'High Risk' if label == 1 else 'Low Risk'}")
        st.info(f"Probability of Default: {proba:.4f}")
    except Exception as e:
        st.error(f"Error: {e}")