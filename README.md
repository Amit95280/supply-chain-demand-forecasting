
# Supply Chain Demand Forecasting & Inventory Optimization

## Placement Project

An end-to-end Data Science project that forecasts retail demand and converts forecasts into inventory decisions.

### Business objective

Reduce stockout risk and excess inventory by:
- analyzing historical demand
- forecasting future demand
- calculating safety stock
- calculating reorder points
- generating inventory recommendations

## Tech Stack

- Python
- Pandas / NumPy
- Matplotlib / Seaborn
- Scikit-learn
- XGBoost
- SQL / SQLite
- Streamlit
- Plotly

## Project Structure

```text
supply-chain-demand-forecasting/
├── data/
│   ├── raw/sales_data.csv
│   └── processed/
├── notebooks/
├── src/
├── models/
├── app/
├── sql/
├── requirements.txt
└── README.md
```

## Run locally

### 1. Create environment

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the model

```bash
python -m src.train_model
```

### 4. Create inventory recommendations

```bash
python -m src.build_outputs
```

### 5. Create SQLite database

```bash
python sql/setup_database.py
```

### 6. Launch dashboard

```bash
streamlit run app/app.py
```

## Dataset note

The included `sales_data.csv` is a synthetic but realistic retail dataset generated specifically for this project, so the ZIP is fully runnable without requiring a Kaggle account.

For a stronger portfolio version, the project can later be switched to the Kaggle Store Item Demand Forecasting dataset. That competition contains daily item-level sales across stores and is specifically designed for demand forecasting.

## Interview story

"I built an end-to-end supply chain analytics system that forecasts store-item demand using time-series features and XGBoost. I used time-based validation to avoid leakage and converted the demand forecast into safety-stock and reorder-point recommendations. Finally, I deployed the workflow through a Streamlit dashboard."

## Resume keywords

Demand Forecasting, Time Series, XGBoost, Feature Engineering, Inventory Optimization, Safety Stock, Reorder Point, SQL, EDA, Streamlit, Business Analytics
