import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss


def walk_forward_validation(X, y, model, train_size=1000, test_size=250):
    """
    Walk-forward validation for time series.

    train_size : taille fenêtre d'entraînement
    test_size  : taille fenêtre test
    """

    metrics = []

    start = train_size

    while start + test_size < len(X):

        train_start = 0
        train_end = start

        test_end = start + test_size

        X_train = X.iloc[train_start:train_end]
        y_train = y.iloc[train_start:train_end]

        X_test = X.iloc[train_end:test_end]
        y_test = y.iloc[train_end:test_end]

        # scaling pour logistic
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model.fit(X_train_scaled, y_train)

        proba = model.predict_proba(X_test_scaled)[:, 1]

        auc = roc_auc_score(y_test, proba)
        brier = brier_score_loss(y_test, proba)

        metrics.append({
            "auc": auc,
            "brier": brier
        })

        start += test_size

    return pd.DataFrame(metrics)