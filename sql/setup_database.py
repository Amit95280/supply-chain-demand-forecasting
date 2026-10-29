
import sqlite3
import pandas as pd
from pathlib import Path

db_path = Path("data/supply_chain.db")
csv_path = Path("data/raw/sales_data.csv")

df = pd.read_csv(csv_path)
conn = sqlite3.connect(db_path)
df.to_sql("sales", conn, if_exists="replace", index=False)
conn.close()

print(f"SQLite database created: {db_path}")
