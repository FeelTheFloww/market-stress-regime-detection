import shap
import matplotlib.pyplot as plt


def compute_shap_values(model, X):

    explainer = shap.Explainer(model)

    shap_values = explainer(X)

    return shap_values


def plot_shap_summary(shap_values):

    shap.summary_plot(shap_values)