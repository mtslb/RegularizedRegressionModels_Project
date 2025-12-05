# src/domain/evaluation.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def plot_results(y_true, y_pred, title, save_scatter=None, save_hist=None):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    # Scatter
    plt.figure(figsize=(6,6))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.6)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
    plt.xlabel("Réel")
    plt.ylabel("Prédit")
    plt.title(title)
    if save_scatter:
        plt.savefig(save_scatter)
    plt.close()

    # Histogramme des erreurs
    plt.figure(figsize=(6,4))
    sns.histplot(y_pred - y_true, bins=50, kde=True)
    plt.title(f"Erreur : {title}")
    plt.xlabel("y_pred - y_true")
    if save_hist:
        plt.savefig(save_hist)
    plt.close()


def evaluate_model(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "R2": r2_score(y_true, y_pred)
    }
