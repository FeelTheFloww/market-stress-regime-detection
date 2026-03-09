import matplotlib.pyplot as plt


def plot_market_stress(df, predictions):

    plt.figure(figsize=(12,6))

    plt.plot(df.index, df["Close"], label="SPY")

    stress_dates = df.index[predictions == 1]

    plt.scatter(
        stress_dates,
        df.loc[stress_dates, "Close"],
        color="red",
        label="Detected stress",
        s=20
    )

    plt.legend()

    plt.title("Detected Market Stress Periods")

    plt.xlabel("Date")

    plt.ylabel("Price")

    plt.show()

def plot_stress_probability(df, proba, target):

    fig, ax1 = plt.subplots(figsize=(14,6))

    # Probabilité ML
    ax1.plot(df.index, proba, color="red", label="ML stress probability")
    ax1.set_ylabel("Stress Probability")

    # Prix SPY
    ax2 = ax1.twinx()
    ax2.plot(df.index, df["Close"], color="black", alpha=0.5, label="SPY price")
    ax2.set_ylabel("SPY price")

    # Stress réel
    stress_dates = df.index[target == 1]

    ax2.scatter(
        stress_dates,
        df.loc[stress_dates, "Close"],
        color="blue",
        label="Real stress",
        s=15
    )

    plt.title("Predicted Stress Probability vs Real Market Stress")

    fig.legend(loc="upper left")

    plt.show()

def plot_stress_prediction_analysis(dataset, proba, target):

    fig, ax1 = plt.subplots(figsize=(14,6))

    # -------------------------
    # PRICE
    # -------------------------

    ax1.plot(
        dataset.index,
        dataset["Close"],
        color="black",
        label="SPY Price"
    )

    ax1.set_ylabel("Price")
    ax1.set_xlabel("Date")

    # -------------------------
    # PROBABILITY
    # -------------------------

    ax2 = ax1.twinx()

    ax2.plot(
        dataset.index,
        proba,
        color="blue",
        alpha=0.6,
        label="Predicted stress probability"
    )

    ax2.set_ylabel("Stress probability")

    # -------------------------
    # REAL STRESS EVENTS
    # -------------------------

    stress_idx = dataset[target == 1].index

    ax1.scatter(
        stress_idx,
        dataset.loc[stress_idx]["Close"],
        color="red",
        s=15,
        label="Real stress events"
    )

    # -------------------------
    # LEGEND
    # -------------------------

    fig.legend(loc="upper left")

    plt.title("Market Stress Prediction vs Real Events")

    plt.show()