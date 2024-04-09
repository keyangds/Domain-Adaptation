import numpy as np


def mae(y_hat, y):
    return np.abs(y_hat - y).mean()


def nmae(y_hat, y):
    delta = np.max(y) - np.min(y) + 1e-8
    return mae(y_hat, y) * 100 / delta


def mape(y_hat, y):
    return 100 * np.abs((y_hat - y) / (y + 1e-8)).mean()


def mse(y_hat, y):
    return np.square(y_hat - y).mean()


def rmse(y_hat, y):
    return np.sqrt(mse(y_hat, y))


def nrmse(y_hat, y):
    delta = np.max(y) - np.min(y) + 1e-8
    return rmse(y_hat, y) * 100 / delta


def nrmse_2(y_hat, y):
    nrmse_ = np.sqrt(np.square(y_hat - y).sum() / np.square(y).sum())
    return nrmse_ * 100


def r2(y_hat, y):
    return 1. - np.square(y_hat - y).sum() / (np.square(y.mean(0) - y).sum())


def masked_mae(y_hat, y, mask):
    err = np.abs(y_hat - y) * np.logical_not(mask).astype(int)
    return err.sum() / mask.sum()


def masked_mape(y_hat, y, mask):
    err = np.abs((y_hat - y) / (y + 1e-8)) * np.logical_not(mask).astype(int)
    return err.sum() / mask.sum()


def masked_mse(y_hat, y, mask):
    err = np.square(y_hat - y) * np.logical_not(mask).astype(int)
  
    return err.sum() / mask.sum()


def masked_rmse(y_hat, y, mask):
    err = np.square(y_hat - y) * np.logical_not(mask).astype(int)
    return np.sqrt(err.sum() / mask.sum())


def masked_mre(y_hat, y, mask):
    # print(mask.size)
    err = np.abs(y_hat - y) * np.logical_not(mask).astype(int)
    return err.sum() / ((y * np.logical_not(mask).astype(int)).sum() + 1e-8)

def nse(y_hat, y, mask):
    """
    Calculate Nash-Sutcliffe Efficiency (NSE) coefficient.

    Parameters:
    - y_hat: 1D NumPy array of predicted values
    - y: 1D NumPy array of observed values

    Returns:
    - NSE coefficient
    """
    mean_observed = np.mean(y) 
    numerator = np.sum((y - y_hat)**2) 
    denominator = np.sum((y - mean_observed)**2) 

    nse_coefficient = 1 - (numerator / denominator)
    return nse_coefficient


def kge(y_hat, y, mask):
    """
    Calculate Kling-Gupta Efficiency (KGE) coefficient.

    Parameters:
    - y_hat: 1D NumPy array of predicted values
    - y: 1D NumPy array of observed values

    Returns:
    - KGE coefficient
    """
    r = np.corrcoef(y_hat, y)[0, 1]  # Correlation coefficient
    beta = np.std(y_hat) / np.std(y)  # Ratio of standard deviation
    gamma = np.mean(y_hat) / np.mean(y)  # Ratio of means

    kge_coefficient = 1 - np.sqrt((r - 1)**2 + (beta - 1)**2 + (gamma - 1)**2)
    return kge_coefficient

