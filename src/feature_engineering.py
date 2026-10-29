
import pandas as pd
import numpy as np

FEATURES = [
    "store", "item", "year", "month", "day", "day_of_week",
    "is_weekend", "day_of_year", "week_of_year",
    "lag_1", "lag_7", "lag_14", "lag_28",
    "rolling_7", "rolling_14", "rolling_28"
]

def make_features(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["store", "item", "date"])

    g = df.groupby(["store", "item"])["sales"]
    for lag in [1, 7, 14, 28]:
        df[f"lag_{lag}"] = g.shift(lag)

    shifted = g.shift(1)
    df["rolling_7"] = shifted.groupby([df["store"], df["item"]]).transform(
        lambda x: x.rolling(7, min_periods=1).mean()
    )
    df["rolling_14"] = shifted.groupby([df["store"], df["item"]]).transform(
        lambda x: x.rolling(14, min_periods=1).mean()
    )
    df["rolling_28"] = shifted.groupby([df["store"], df["item"]]).transform(
        lambda x: x.rolling(28, min_periods=1).mean()
    )

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["day_of_year"] = df["date"].dt.dayofyear
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)

    return df.dropna(subset=FEATURES + ["sales"]).reset_index(drop=True)
