import sqlite3
import pandas as pd

conn = sqlite3.connect("data/loanvet.db")
query = "SELECT * FROM credit_risk_raw LIMIT 5"
df_sample = pd.read_sql_query(query, conn)
print(df_sample)
conn.close()
