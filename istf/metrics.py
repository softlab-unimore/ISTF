import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error


def compute_metrics(y_true, y_preds):
    return {
        'r2': r2_score(y_true, y_preds),
        'mae': mean_absolute_error(y_true, y_preds),
        'mse': mean_squared_error(y_true, y_preds),
        # 'mape': mean_absolute_percentage_error(y_true, y_preds)
    }
