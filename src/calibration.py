import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.metrics import brier_score_loss


def calibrate_model(model, X_train, y_train, method="sigmoid"):
    """
    Calibrate a classifier using Platt scaling or isotonic regression.
    """

    calibrated = CalibratedClassifierCV(model, method=method, cv=3)

    calibrated.fit(X_train, y_train)

    return calibrated


def plot_calibration(y_true, y_prob):

    prob_true, prob_pred = calibration_curve(
        y_true, y_prob, n_bins=10
    )

    plt.figure()

    plt.plot(prob_pred, prob_true, marker="o", label="Model")
    plt.plot([0,1], [0,1], linestyle="--", label="Perfect calibration")

    plt.xlabel("Predicted probability")
    plt.ylabel("True probability")

    plt.title("Calibration Curve")

    plt.legend()

    plt.show()