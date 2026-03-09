import os
import pandas as pd
import yfinance as yf

DATA_PATH = "data/raw/SPY.csv"


def download_data(start="2004-01-01", end=None):
    """
    Download SPY data from Yahoo Finance
    and save it locally.
    """

    os.makedirs("data/raw", exist_ok=True)

    df = yf.download("SPY", start=start, end=end)

    df.to_csv(DATA_PATH)

    print("SPY data saved to data/raw/SPY.csv")

    return df


def load_data():
    """
    Load SPY dataset from local CSV.
    """

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            "SPY.csv not found. Run download_data() first."
        )

    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)

    # s'assurer que les colonnes numériques sont bien numériques
    numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    return df