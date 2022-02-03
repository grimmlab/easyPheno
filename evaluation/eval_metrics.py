import sklearn
import numpy as np


def get_evaluation_report(y_pred: np.array, y_true: np.array, task: str, prefix: str = ''):
    """
    Get values for common evaluation metrics
    :param y_pred: predicted values
    :param y_true: true values
    :param task: ML task to solve
    :param prefix: prefix to be added to the key if multiple eval metrics are collected
    :return: dictionary with common metrics
    """
    if task == 'classification':
        eval_report_dict = {
            prefix + 'accuracy': sklearn.metrics.accuracy_score(y_true=y_true, y_pred=y_pred),
            prefix + 'f1_score': sklearn.metrics.f1_score(y_true=y_true, y_pred=y_pred),
            prefix + 'precision': sklearn.metrics.precision_score(y_true=y_true, y_pred=y_pred),
            prefix + 'recall': sklearn.metrics.recall_score(y_true=y_true, y_pred=y_pred),
            prefix + 'roc_auc': sklearn.metrics.roc_auc_score(y_true=y_true, y_pred=y_pred)
        }
    else:
        eval_report_dict = {
            prefix + 'mse': sklearn.metrics.mean_squared_error(y_true=y_true, y_pred=y_pred),
            prefix + 'rmse': sklearn.metrics.mean_squared_error(y_true=y_true, y_pred=y_pred, squared=False),
            prefix + 'r2_score': sklearn.metrics.r2_score(y_true=y_true, y_pred=y_pred),
            prefix + 'mape': sklearn.metrics.mean_absolute_percentage_error(y_true=y_true, y_pred=y_pred),
            prefix + 'explained_variance': sklearn.metrics.explained_variance_score(y_true=y_true, y_pred=y_pred)
        }

    return eval_report_dict
