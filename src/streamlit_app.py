import streamlit as st
import requests

st.title("LoanVet Credit Risk Prediction")

# Input fields for all features (ensure keys exactly match feature_list)
input_data = {}
input_data["age"] = st.number_input("Age", min_value=18, max_value=100, value=35)
input_data["NumberOfOpenCreditLinesAndLoans"] = st.number_input("Open Credit Lines and Loans", min_value=0, value=5)
input_data["NumberRealEstateLoansOrLines"] = st.number_input("Real Estate Loans or Lines", min_value=0, value=1)
input_data["NumberOfDependents"] = st.number_input("Number of Dependents", min_value=0, value=2)
input_data["MonthlyIncome_missing_flag"] = st.number_input("Monthly Income Missing Flag (0 or 1)", min_value=0, max_value=1, value=0)
input_data["NumberOfDependents_missing_flag"] = st.number_input("Dependents Missing Flag (0 or 1)", min_value=0, max_value=1, value=0)
input_data["RevolvingUtilizationOfUnsecuredLines_log"] = st.number_input("Revolving Utilization (log)", value=0.12)
input_data["MonthlyIncome_log"] = st.number_input("Monthly Income (log)", value=10.5)
input_data["DebtRatio_log"] = st.number_input("Debt Ratio (log)", value=1.2)
input_data["TotalDelinquencies_log"] = st.number_input("Total Delinquencies (log)", value=0.0)
input_data["HighUtilizationFlag"] = st.number_input("High Utilization Flag (0 or 1)", min_value=0, max_value=1, value=0)
input_data["IncomePerCreditLine"] = st.number_input("Income per Credit Line", value=500.0)
input_data["AgeGroup_MidAge"] = st.number_input("Age Group MidAge (0 or 1)", min_value=0, max_value=1, value=1)
input_data["AgeGroup_Senior"] = st.number_input("Age Group Senior (0 or 1)", min_value=0, max_value=1, value=0)
input_data["DependentsGroup_Small"] = st.number_input("Dependents Group Small (0 or 1)", min_value=0, max_value=1, value=1)
input_data["DependentsGroup_Large"] = st.number_input("Dependents Group Large (0 or 1)", min_value=0, max_value=1, value=0)
input_data["Util_x_Late"] = st.number_input("Util x Late", value=0.0)
input_data["IncomePerDependent"] = st.number_input("Income per Dependent", value=250.0)
input_data["CreditLines_x_Delinquencies"] = st.number_input("Credit Lines x Delinquencies", value=0.1)

if st.button("Predict"):
    try:
        API_URL = "https://loanvet-api.up.railway.app/predict"
        response = requests.post(API_URL, json=input_data)
        response.raise_for_status()  # Raise for bad HTTP status
        result = response.json()
        label = result["prediction"]["label"]
        proba = result["prediction"]["probability"]

        st.success(f"Prediction: {'High Risk' if label == 1 else 'Low Risk'}")
        st.info(f"Probability of Default: {proba:.4f}")
    except Exception as e:
        st.error(f"Error: {e}")