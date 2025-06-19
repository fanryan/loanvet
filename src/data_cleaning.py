import sqlite3
from eda_baseline import load_data

# Drop unnamed index column
def drop_first_col(df):
    return df.iloc[:, 1:]

# Drop duplicates (610)
def drop_duplicates(df, subset=None):
    before = len(df)
    df = df.drop_duplicates(subset=subset)
    after = len(df)
    print(f"Dropped {before - after} duplicate rows")
    return df

# == Deal with missing values ==

# Add missing flags for MonthlyIncome and NumberOfDependents
def create_missing_flags(df):
    df['MonthlyIncome_missing_flag'] = df['MonthlyIncome'].isnull().astype(int)
    df['NumberOfDependents_missing_flag'] = df['NumberOfDependents'].isnull().astype(int)
    return df

# Impute MonthlyIncome with median & NumberOfDependents with median or 0
def impute_missing_values(df):
    median_income = df['MonthlyIncome'].median()
    df['MonthlyIncome'] = df['MonthlyIncome'].fillna(median_income)

    median_dependents = df['NumberOfDependents'].median()
    df['NumberOfDependents'] = df['NumberOfDependents'].fillna(median_dependents)
    return df

# == Deal with outliers ==

custom_caps = {
    "RevolvingUtilizationOfUnsecuredLines": (0, 1),  # Utilization rate capped at 100%
    "age": (18, 100),  # Assuming credit-worthy age starts at 18; >100 likely data error
    "NumberOfTime30-59DaysPastDueNotWorse": (0, 12),  # More than once/month in 2 years is excessive; cap at monthly frequency
    "DebtRatio": (0, 5000),  # High cap allows debt-heavy individuals, but avoids ratios from near-zero income distortions
    "MonthlyIncome": (0, 50000),  # Allows high earners, but filters out noise (e.g. >$50k/month = top 0.1%)
    "NumberOfOpenCreditLinesAndLoans": (0, 30),  # >30 accounts is rare; cap to prevent skew
    "NumberOfTimes90DaysLate": (0, 12),  # Cap at one major delinquency per 2 months to preserve signal, avoid noise
    "NumberRealEstateLoansOrLines": (0, 10),  # 10+ mortgages or equity lines is rare, even for wealthy
    "NumberOfTime60-89DaysPastDueNotWorse": (0, 12),  # Same logic as other delinquency measures
    "NumberOfDependents": (0, 10)  # Large families allowed, >10 often unreliable or outlier
}

# Apply outlier caps
def cap_outliers(df, caps):
    for col, (lower, upper) in custom_caps.items():
        if col in df.columns:
            df[col] = df[col].clip(lower=lower, upper=upper)
    return df

#Save to a new SQLite table
def save_cleaned_data(df):
    conn = sqlite3.connect("data/loanvet.db")
    df.to_sql("credit_risk_cleaned", conn, if_exists='replace', index=False)
    conn.close()
    print("Cleaned data saved to 'credit_risk_cleaned' table.")

if __name__ == "__main__":
    df = load_data()
    df = drop_first_col(df)
    df = drop_duplicates(df)
    df = create_missing_flags(df)
    df = impute_missing_values(df)
    df = cap_outliers(df, custom_caps)
    save_cleaned_data(df)