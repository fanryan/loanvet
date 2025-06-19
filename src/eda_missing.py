import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from eda_baseline import load_data

def missing_values_summary(df):
    missing_count = df.isnull().sum()
    missing_pct = 100 * missing_count / len(df)
    missing_df = pd.DataFrame({'missing_count': missing_count, 'missing_pct': missing_pct})
    print("Missing values summary:\n", missing_df[missing_df['missing_count'] > 0])
    return missing_df

def plot_missing_heatmap(df):
    plt.figure(figsize=(12, 6))
    sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap='viridis')
    plt.title("Missing Values Heatmap")
    plt.show()

def missing_vs_target(df, target_col):
    for col in df.columns:
        if df[col].isnull().any():
            # Create flag column
            flag_col = f"{col}_missing_flag"
            df[flag_col] = np.where(df[col].isnull(), 1, 0)
            # Compare target mean for missing vs non-missing
            missing_target_mean = df.groupby(flag_col)[target_col].mean()
            print(f"\nTarget mean by missingness in {col}:\n", missing_target_mean)

if __name__ == "__main__":
    df = load_data()
    missing_values_summary(df)
    plot_missing_heatmap(df)
    missing_vs_target(df, target_col='SeriousDlqin2yrs')