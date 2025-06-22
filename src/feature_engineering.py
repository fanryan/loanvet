import numpy as np
import pandas as pd
import sqlite3
from eda_baseline import load_cleaned_data

def log_transform(df):
    skewed_features = [
        'RevolvingUtilizationOfUnsecuredLines',
        'MonthlyIncome',
        'DebtRatio',
        'NumberOfTime30-59DaysPastDueNotWorse',
        'NumberOfTimes90DaysLate',
        'NumberOfTime60-89DaysPastDueNotWorse',
        'TotalDelinquencies'
    ]
    for feat in skewed_features:
        df[f'{feat}_log'] = np.log1p(df[feat])
    return df

def aggregate_delinquencies(df):
    df['TotalDelinquencies'] = (
        df['NumberOfTime30-59DaysPastDueNotWorse'] +
        df['NumberOfTimes90DaysLate'] +
        df['NumberOfTime60-89DaysPastDueNotWorse']
    )
    return df

def bin_age(df):
    bins = [18, 30, 50, 100]
    labels = ['Young', 'MidAge', 'Senior']
    df['AgeGroup'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)
    return df

def flag_high_utilization(df):
    df['HighUtilizationFlag'] = (df['RevolvingUtilizationOfUnsecuredLines'] > 0.9).astype(int)
    return df

def income_per_credit_line(df):
    df['IncomePerCreditLine'] = df['MonthlyIncome'] / (df['NumberOfOpenCreditLinesAndLoans'] + 1)
    return df

def categorise_dependents(df):
    bins = [-1, 0, 2, 10]
    labels = ['None', 'Small', 'Large']
    df['DependentsGroup'] = pd.cut(df['NumberOfDependents'], bins=bins, labels=labels, right=True)
    return df

def encode_categorical_features(df, method='onehot'):
    categorical_cols = df.select_dtypes(include='category').columns.tolist()
    categorical_cols += df.select_dtypes(include='object').columns.tolist()
    
    if method == 'onehot':
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    elif method == 'label':
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        for col in categorical_cols:
            df[col] = le.fit_transform(df[col])
    else:
        raise ValueError("method must be 'onehot' or 'label'")
    
    return df

def add_interaction_features(df):
    df["Util_x_Late"] = df["RevolvingUtilizationOfUnsecuredLines_log"] * df["NumberOfTimes90DaysLate_log"]
    df["IncomePerDependent"] = df["MonthlyIncome"] / (df["NumberOfDependents"] + 1)
    df["CreditLines_x_Delinquencies"] = df["NumberOfOpenCreditLinesAndLoans"] * df["TotalDelinquencies"]
    return df

def preprocess_data(df):
    df = aggregate_delinquencies(df)
    df = log_transform(df)
    df = bin_age(df)
    df = flag_high_utilization(df)
    df = income_per_credit_line(df)
    df = categorise_dependents(df)
    df = encode_categorical_features(df, method='onehot')
    df = add_interaction_features(df)
    return df

def save_engineered_data(df):
    conn = sqlite3.connect("data/loanvet.db")
    df.to_sql("credit_risk_engineered", conn, if_exists='replace', index=False)
    conn.close()
    print("✅ Engineered data saved to 'credit_risk_engineered' table.")


if __name__ == "__main__":
    df = load_cleaned_data()
    df = preprocess_data(df)
    save_engineered_data(df)