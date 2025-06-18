import pandas as pd
from sqlalchemy import create_engine

# Load the credit risk training data
df = pd.read_csv("data/raw/credit_train.csv")

# Connect to the local SQLite database
engine = create_engine("sqlite:///data/loanvet.db")

# Save the DataFrame as a table in the database
df.to_sql("credit_risk_raw", con=engine, if_exists="replace", index=False)

print("✅ Data imported into SQLite.")
