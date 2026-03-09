import numpy as np
import pandas as pd


# ------------------------------------------------
# RETURNS
# ------------------------------------------------

def compute_log_returns(df):

    df = df.copy()

    df["log_return"] = np.log(
    df["Close"] / df["Close"].shift(1)
)

    return df


# ------------------------------------------------
# TARGET
# ------------------------------------------------

def compute_target(df, horizon=5, threshold=0.98):

    df = df.copy()

    df["future_return"] = (
        df["log_return"]
        .shift(-1)
        .rolling(horizon)
        .sum()
    )

    log_threshold = np.log(threshold)

    df["target"] = (df["future_return"] < log_threshold).astype(int)

    return df


# ------------------------------------------------
# MOMENTUM FEATURES
# ------------------------------------------------

def momentum_features(df):

    df = df.copy()

    windows = [5, 10, 20, 60]

    for w in windows:

        df[f"ret_{w}"] = (
            np.log(df["Close"] / df["Close"].shift(w))
            .shift(1)
        )

    return df


# ------------------------------------------------
# VOLATILITY FEATURES
# ------------------------------------------------

def volatility_features(df):

    df = df.copy()

    for w in [5, 10, 20]:

        df[f"vol_{w}"] = (
            df["log_return"]
            .rolling(w)
            .std()
            .shift(1)
        )

    lambda_ = 0.94

    ewma_var = (
        df["log_return"]**2
    ).ewm(alpha=(1 - lambda_), adjust=False).mean()

    df["ewma_vol"] = np.sqrt(ewma_var).shift(1)

    return df


# ------------------------------------------------
# DRAWDOWN FEATURES
# ------------------------------------------------

def drawdown_features(df):

    df = df.copy()

    rolling_max = df["Close"].rolling(60).max().shift(1)
    df["drawdown_60"] = df["Close"] / rolling_max - 1

    df["distance_to_peak"] = -df["drawdown_60"]

    return df


# ------------------------------------------------
# TREND FEATURES
# ------------------------------------------------

def trend_features(df):

    df = df.copy()

    for w in [10, 20, 50]:

        sma = df["Close"].rolling(w).mean().shift(1)
        df[f"sma_ratio_{w}"] = df["Close"] / sma

        sma20 = df["Close"].rolling(20).mean().shift(1)

    df["slope_20"] = sma20 - sma20.shift(5)

    return df


# ------------------------------------------------
# DISTRIBUTION FEATURES
# ------------------------------------------------

def distribution_features(df):

    df = df.copy()

    df["skew_20"] = (
        df["log_return"]
        .rolling(20)
        .skew()
        .shift(1)
    )

    df["kurt_20"] = (
        df["log_return"]
        .rolling(20)
        .kurt()
        .shift(1)
    )

    return df


# ------------------------------------------------
# BUILD FINAL DATASET
# ------------------------------------------------

def build_dataset(df):

    df = compute_log_returns(df)

    df = momentum_features(df)

    df = volatility_features(df)

    df = drawdown_features(df)

    df = trend_features(df)

    df = distribution_features(df)

    df = compute_target(df)

    df = df.dropna()

    return df