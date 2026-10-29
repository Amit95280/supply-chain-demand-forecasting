
import sys
sys.path.append("..")
from src.data_preprocessing import load_data, basic_report

df = load_data("../data/raw/sales_data.csv")
print(df.head())
print(df.info())
print(basic_report(df))
