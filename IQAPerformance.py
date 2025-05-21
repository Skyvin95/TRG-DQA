import numpy as np
from scipy import stats

def IQAPerformance(y_pred, y):
    """
    Evaluation of IQA methods using SROCC, KROCC, PLCC, RMSE, MAE, OR.

    """

    sq = np.reshape(np.asarray(y), (-1,))
    sq_std = np.reshape(np.asarray(np.std(y)), (-1,))
    q = np.reshape(np.asarray(y_pred), (-1,))

    srocc = stats.spearmanr(sq, q)[0]
    krocc = stats.stats.kendalltau(sq, q)[0]
    plcc = stats.pearsonr(sq, q)[0]
    rmse = np.sqrt(((sq - q) ** 2).mean())
    mae = np.abs((sq - q)).mean()
    outlier_ratio = (np.abs(sq - q) > 2 * sq_std).mean()

    return srocc, krocc, plcc, rmse, mae, outlier_ratio