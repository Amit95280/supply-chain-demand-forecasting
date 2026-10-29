
import pandas as pd

def load_data(path="data/raw/sales_data.csv"):
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.drop_duplicates().sort_values(["store", "item", "date"]).reset_index(drop=True)
    return df

def basic_report(df):
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "missing_values": int(df.isna().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
        "date_min": str(df["date"].min().date()),
        "date_max": str(df["date"].max().date()),
        "stores": int(df["store"].nunique()),
        "items": int(df["item"].nunique()),
    }
