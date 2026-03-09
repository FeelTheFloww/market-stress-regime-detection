import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_feature_importance(model, feature_names, path=None):

    importance = model.feature_importances_

    df_imp = pd.DataFrame({
        "feature": feature_names,
        "importance": importance
    }).sort_values("importance", ascending=False)

    plt.figure(figsize=(8,6))

    plt.barh(df_imp["feature"], df_imp["importance"])

    plt.gca().invert_yaxis()

    plt.title("Feature Importance")

    if path:

        os.makedirs(os.path.dirname(path), exist_ok=True)

        plt.savefig(path)

    else:

        plt.show()

    return df_imp