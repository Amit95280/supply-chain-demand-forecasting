
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def regression_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    denom = np.maximum(np.abs(y_true), 1)
    mape = np.mean(np.abs((y_true - y_pred) / denom)) * 100
    return {"MAE": float(mae), "RMSE": float(rmse), "MAPE": float(mape)}
