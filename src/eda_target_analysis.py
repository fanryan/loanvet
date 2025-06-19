import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from eda_baseline import load_cleaned_data

def plot_numeric_distributions(df):
    numeric_cols = [
        col for col in df.select_dtypes(include='number').columns
        if not col.endswith("_missing_flag")
    ]

    n_cols = len(numeric_cols)
    cols = 4
    rows = math.ceil(n_cols / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = axes.flatten()

    for i, col in enumerate(numeric_cols):
        sns.histplot(df[col], kde=True, bins=30, ax=axes[i], color='skyblue', edgecolor='black')
        axes[i].set_title(f'Distribution of {col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Frequency")
        axes[i].grid(True, linestyle='--', alpha=0.5)

    # Turn off any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

def print_skewness(df):
    numeric_cols = df.select_dtypes(include='number').columns
    skewness = df[numeric_cols].skew().sort_values(ascending=False)
    print("Skewness of numeric features:\n", skewness)

def plot_target_distribution(df, target='SeriousDlqin2yrs'):
    print(f"Target variable '{target}' distribution (counts and proportions):")
    counts = df[target].value_counts()
    props = df[target].value_counts(normalize=True)
    print(pd.DataFrame({"count": counts, "proportion": props}))
    sns.countplot(x=target, data=df, color='C0')
    plt.title(f"Class distribution of {target}")
    plt.show()

def plot_numeric_by_target(df, target='SeriousDlqin2yrs'):
    numeric_cols = [col for col in df.select_dtypes(include='number').columns if not col.endswith('_missing_flag') and col != target]
    n_cols = 4
    rows = math.ceil(len(numeric_cols) / n_cols)
    fig, axes = plt.subplots(rows, n_cols, figsize=(n_cols * 5, rows * 4))
    axes = axes.flatten()

    for i, col in enumerate(numeric_cols):
        sns.boxplot(x=target, y=col, data=df, ax=axes[i], color='C1')
        axes[i].set_title(f"{col} by {target}")
        axes[i].grid(True, linestyle='--', alpha=0.5)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

def plot_correlation_matrix(df):
    numeric_cols = [col for col in df.select_dtypes(include='number').columns if not col.endswith('_missing_flag')]
    corr = df[numeric_cols].corr()
    plt.figure(figsize=(12,10))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", square=True, cbar_kws={"shrink": .75})
    plt.title('Correlation Matrix')
    plt.show()

def detect_multicollinearity(df, threshold=0.8):
    numeric_cols = [
        col for col in df.select_dtypes(include='number').columns
        if not col.endswith('_missing_flag')
    ]
    corr = df[numeric_cols].corr().abs()
    high_corr = []
    for i, fi in enumerate(numeric_cols):
        for fj in numeric_cols[i+1:]:
            if corr.at[fi, fj] > threshold:
                high_corr.append((fi, fj, corr.at[fi, fj]))
    if high_corr:
        print(f"Features with |corr| > {threshold}:")
        for fi, fj, val in high_corr:
            print(f"  {fi} ↔ {fj}: {val:.2f}")
    else:
        print(f"No feature pairs with |corr| > {threshold}.")
    return high_corr

if __name__ == "__main__":
    df = load_cleaned_data()
    plot_numeric_distributions(df)
    print_skewness(df)
    plot_target_distribution(df)
    plot_numeric_by_target(df)
    plot_correlation_matrix(df)
    detect_multicollinearity(df)