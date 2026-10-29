
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from src.data_preprocessing import load_data
from src.inventory import safety_stock, reorder_point, inventory_status

def main():
    df = load_data()
    model = joblib.load("models/demand_model.pkl")
    latest = df["date"].max()
    recent = df[df["date"] >= latest - pd.Timedelta(days=27)]

    summary = (
        recent.groupby(["store","item"])
        .agg(
            avg_daily_demand=("sales","mean"),
            demand_std=("sales","std"),
            current_inventory=("inventory","last"),
            lead_time=("lead_time","last"),
            price=("price","last")
        )
        .reset_index()
    )

    summary["demand_std"] = summary["demand_std"].fillna(0)
    summary["safety_stock"] = summary.apply(
        lambda r: safety_stock(r["demand_std"], r["lead_time"]), axis=1
    )
    summary["reorder_point"] = summary.apply(
        lambda r: reorder_point(r["avg_daily_demand"], r["lead_time"], r["safety_stock"]), axis=1
    )
    summary["status"] = summary.apply(
        lambda r: inventory_status(r["current_inventory"], r["reorder_point"]), axis=1
    )
    summary["recommended_order"] = np.maximum(
        0, np.ceil(summary["reorder_point"] * 1.5 - summary["current_inventory"])
    ).astype(int)

    Path("data/processed").mkdir(exist_ok=True)
    summary.to_csv("data/processed/inventory_recommendations.csv", index=False)
    print("Created data/processed/inventory_recommendations.csv")

if __name__ == "__main__":
    main()
