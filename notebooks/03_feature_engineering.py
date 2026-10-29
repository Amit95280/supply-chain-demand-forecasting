
import sys
sys.path.append("..")
from src.data_preprocessing import load_data
from src.feature_engineering import make_features

df = load_data("../data/raw/sales_data.csv")
features = make_features(df)
print(features.head())
print(features.shape)
