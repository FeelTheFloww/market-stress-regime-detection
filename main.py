from src.data_loader import download_data, load_data
from src.feature_engineering import build_dataset
from src.models import prepare_data, time_split, train_and_evaluate
from src.walk_forward import walk_forward_validation
from src.calibration import calibrate_model, plot_calibration
from src.baseline import volatility_baseline
from src.feature_importance import plot_feature_importance
from src.visualization import (
    plot_market_stress,
    plot_stress_probability,
    plot_stress_prediction_analysis
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import numpy as np

def main():

    # -----------------------------
    # DATA
    # -----------------------------

    download_data()

    df = load_data()

    dataset = build_dataset(df)

    dataset = dataset.dropna().copy()

    print("\nDataset shape:", dataset.shape)

    X  , y = prepare_data(dataset)

    X_train, X_test, y_train, y_test = time_split(X, y)
    print("\nDataset shape:", dataset.shape)


    # -----------------------------
    # BASELINE
    # -----------------------------

    baseline_metrics = volatility_baseline(dataset)

    print("\nVolatility baseline:")
    print(baseline_metrics)



    # -----------------------------
    # TRAIN MODELS
    # -----------------------------

    results = train_and_evaluate(
        X_train, X_test, y_train, y_test
    )

    print("\nModel Results\n")

    for model_name, metrics in results.items():

        print("\n", model_name)

        for k, v in metrics.items():

            if k != "confusion_matrix":
                print(k, ":", round(v, 4))

        print("confusion_matrix:\n", metrics["confusion_matrix"])


    # -----------------------------
    # WALK FORWARD VALIDATION
    # -----------------------------

    logistic_model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    )

    wf_results = walk_forward_validation(X, y, logistic_model)

    print("\nWalk-forward results:")
    print(wf_results)

    print("\nAverage metrics:")
    print(wf_results.mean())


    # -----------------------------
    # CALIBRATION
    # -----------------------------

    calibrated_model = calibrate_model(
        logistic_model,
        X_train,
        y_train
    )

    proba_calibrated = calibrated_model.predict_proba(X_test)[:,1]

    plot_calibration(y_test, proba_calibrated)


   # -----------------------------
    # RANDOM FOREST TRAINING
    # -----------------------------

    rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=6,
    random_state=42
    )

    rf_model.fit(X_train, y_train)
    proba_test = rf_model.predict_proba(X_test)[:,1]

    dataset["stress_proba"] = np.nan
    dataset.loc[X_test.index, "stress_proba"] = proba_test

    dataset["stress_proba_smooth"] = dataset["stress_proba"].rolling(10).mean()

    dataset["proba_lead"] = dataset["stress_proba_smooth"].shift(5)

    analysis_df = dataset.dropna(subset=["stress_proba_smooth"])

    plot_stress_prediction_analysis(
    analysis_df ,
    analysis_df ["stress_proba_smooth"],
    analysis_df ["target"]
    )
    # -----------------------------
    # FEATURE IMPORTANCE
    # -----------------------------

    feature_names = X.columns

    plot_feature_importance(
        rf_model,
        feature_names,
        path="results/plots/feature_importance.png"
    )


    # -----------------------------
    # MARKET STRESS VISUALIZATION
    # -----------------------------

    predictions = rf_model.predict(X_test)

    dataset["predicted_stress"] = np.nan
    dataset.loc[X_test.index, "predicted_stress"] = predictions

    plot_market_stress(dataset, dataset["predicted_stress"])


    # -----------------------------
    # STRESS PROBABILITY VISUALIZATION
    # -----------------------------

    plot_stress_probability(
    dataset,
    dataset["stress_proba_smooth"],
    dataset["target"]
    )
    print("\nAverage probability by future stress:")

    dataset["future_stress"] = dataset["target"].shift(-5)
    
    print(
    dataset.groupby("future_stress")["stress_proba_smooth"].mean()
    )

if __name__ == "__main__":
    main()