import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sqlite3

def load_data():
    conn = sqlite3.connect("data/loanvet.db")
    df = pd.read_sql_query("SELECT * FROM credit_risk_raw", conn)
    conn.close()
    return df

def load_cleaned_data():
    conn = sqlite3.connect("data/loanvet.db")
    df = pd.read_sql_query("SELECT * FROM credit_risk_cleaned", conn)
    conn.close()
    return df

def eda_overview(df):
    print("Shape:", df.shape)
    print("\nColumns:\n", df.columns)
    print("\nℹInfo:")
    print(df.info())
    print("\nMissing values:\n", df.isnull().sum())
    print("\nDescribe:\n", df.describe(include='all'))

def plot_distributions(df):
    numeric_cols = df.select_dtypes(include='number').columns
    df[numeric_cols].hist(figsize=(12, 8))
    plt.tight_layout()
    plt.show()

    categorical_cols = df.select_dtypes(include='object').columns
    for col in categorical_cols:
        plt.figure(figsize=(8, 4))
        sns.countplot(data=df, x=col)
        plt.title(f'Distribution of {col}')
        plt.xticks(rotation=45)
        plt.show()

if __name__ == "__main__":
    df = load_data()
    eda_overview(df)
    plot_distributions(df)