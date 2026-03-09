import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score


def volatility_baseline(df, threshold_quantile=0.9):
    """
    Baseline simple :
    signal de stress si la volatilité 20 jours est élevée.
    """

    vol = df["vol_20"]

    threshold = vol.quantile(threshold_quantile)

    preds = (vol > threshold).astype(int)

    y_true = df["target"]

    metrics = {
        "roc_auc": roc_auc_score(y_true, preds),
        "precision": precision_score(y_true, preds),
        "recall": recall_score(y_true, preds),
        "f1": f1_score(y_true, preds),
    }

    return metrics