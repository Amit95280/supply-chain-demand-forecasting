
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from src.data_preprocessing import load_data
from src.feature_engineering import make_features, FEATURES

DATA = "data/raw/sales_data.csv"
MODEL_PATH = "models/demand_model.pkl"

def main():
    df = load_data(DATA)
    feat = make_features(df)

    # Time-based validation: last 60 days
    cutoff = feat["date"].max() - pd.Timedelta(days=60)
    train = feat[feat["date"] < cutoff]
    valid = feat[feat["date"] >= cutoff]

    model = XGBRegressor(
        n_estimators=250,
        max_depth=7,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=4
    )
    model.fit(train[FEATURES], train["sales"])

    pred = model.predict(valid[FEATURES])
    mae = mean_absolute_error(valid["sales"], pred)
    rmse = mean_squared_error(valid["sales"], pred) ** 0.5
    mape = (abs(valid["sales"] - pred) / valid["sales"].clip(lower=1)).mean() * 100

    Path("models").mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    Path("models/metrics.json").write_text(json.dumps({
        "MAE": round(float(mae), 3),
        "RMSE": round(float(rmse), 3),
        "MAPE": round(float(mape), 3),
        "validation_days": 60
    }, indent=2))

    print("Model saved:", MODEL_PATH)
    print(f"MAE={mae:.2f}, RMSE={rmse:.2f}, MAPE={mape:.2f}%")

if __name__ == "__main__":
    main()
