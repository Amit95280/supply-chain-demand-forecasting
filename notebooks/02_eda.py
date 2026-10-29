
import sys
sys.path.append("..")
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_preprocessing import load_data

df = load_data("../data/raw/sales_data.csv")

daily = df.groupby("date")["sales"].sum()
daily.plot(figsize=(12,5), title="Daily Total Sales")
plt.tight_layout()
plt.show()

sns.boxplot(data=df, x="promotion", y="sales")
plt.title("Promotion vs Sales")
plt.show()
