import math
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
import seaborn as sns
from eda_baseline import load_cleaned_data

def plot_eda(df):
    numeric_cols = df.select_dtypes(include='number').columns

    # Define which numeric columns are binary
    binary_cols = [col for col in numeric_cols if df[col].nunique() == 2]
    continuous_cols = [col for col in numeric_cols if df[col].nunique() > 2]

    # Plot continuous numeric columns with boxplots
    n_cont = len(continuous_cols)
    cols = 3
    rows = math.ceil(n_cont / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = axes.flatten()

    for i, col in enumerate(continuous_cols):
        sns.boxplot(x=df[col], ax=axes[i])
        axes[i].set_title(f'Boxplot of {col}')

    # Remove unused axes for continuous plots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

    # Plot binary columns with countplots
    n_bin = len(binary_cols)
    if n_bin > 0:
        cols = 3
        rows = math.ceil(n_bin / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
        axes = axes.flatten()

        for i, col in enumerate(binary_cols):
            sns.countplot(x=df[col], ax=axes[i])
            axes[i].set_title(f'Countplot of {col}')
            axes[i].xaxis.set_major_locator(FixedLocator([0, 1]))
            axes[i].set_xticklabels(['0', '1'])

        # Remove unused axes for binary plots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()

def iqr_outlier_summary(df):
    numeric_cols = df.select_dtypes(include='number').columns
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        pct = len(outliers) / len(df) * 100
        print(f"{col}: {len(outliers)} outliers ({pct:.2f}%)")

if __name__ == "__main__":
    df = load_cleaned_data()
    plot_eda(df)
    iqr_outlier_summary(df)