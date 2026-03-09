import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
    confusion_matrix
)

from sklearn.preprocessing import StandardScaler


# ------------------------------------------------
# PREPARE DATA
# ------------------------------------------------

def prepare_data(df):

    feature_cols = [
        "ret_5","ret_10","ret_20","ret_60",
        "vol_5","vol_10","vol_20","ewma_vol",
        "drawdown_60","distance_to_peak",
        "sma_ratio_10","sma_ratio_20","sma_ratio_50",
        "slope_20",
        "skew_20","kurt_20"
    ]

    X = df[feature_cols]
    y = df["target"]

    return X, y


# ------------------------------------------------
# TRAIN TEST SPLIT (TEMPORAL)
# ------------------------------------------------

def time_split(X, y, train_ratio=0.7):

    n = len(X)

    split = int(n * train_ratio)

    X_train = X.iloc[:split]
    X_test = X.iloc[split:]

    y_train = y.iloc[:split]
    y_test = y.iloc[split:]

    return X_train, X_test, y_train, y_test


# ------------------------------------------------
# MODELS
# ------------------------------------------------

def get_models():

    models = {}

    models["logistic"] = LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    )

    models["random_forest"] = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        class_weight="balanced",
        random_state=42
    )

    models["xgboost"] = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss"
    )

    return models


# ------------------------------------------------
# TRAIN + EVALUATE
# ------------------------------------------------

def train_and_evaluate(X_train, X_test, y_train, y_test):

    models = get_models()

    results = {}

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    for name, model in models.items():

        if name == "logistic":
            model.fit(X_train_scaled, y_train)
            proba = model.predict_proba(X_test_scaled)[:,1]
            preds = model.predict(X_test_scaled)

        else:
            model.fit(X_train, y_train)
            proba = model.predict_proba(X_test)[:,1]
            preds = model.predict(X_test)

        results[name] = {

            "roc_auc": roc_auc_score(y_test, proba),
            "precision": precision_score(y_test, preds),
            "recall": recall_score(y_test, preds),
            "f1": f1_score(y_test, preds),
            "brier": brier_score_loss(y_test, proba),
            "confusion_matrix": confusion_matrix(y_test, preds)

        }

    return results