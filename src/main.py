# src/main.py
import os
from domain.models.linear_model import run_model as run_linear
from domain.models.ridge_model import run_model as run_ridge
from domain.models.lasso_model import run_model as run_lasso
from domain.models.elasticnet_model import run_model as run_elasticnet
from domain import evaluation as ev

GRAPH_DIR = os.path.join(os.path.dirname(__file__), '..', 'graphs')
os.makedirs(GRAPH_DIR, exist_ok=True)

def save_plots(y_true, y_pred, model_name, target):
    title = f"{model_name} - {target}"
    file_scatter = os.path.join(GRAPH_DIR, f"{model_name}_{target}_scatter.png")
    file_hist = os.path.join(GRAPH_DIR, f"{model_name}_{target}_error_hist.png")
    ev.plot_results(y_true, y_pred, title, save_scatter=file_scatter, save_hist=file_hist)

def main():
    models = {
        "LinearRegression": run_linear,
        "Ridge": run_ridge,
        "Lasso": run_lasso,
        "ElasticNet": run_elasticnet
    }

    for model_name, run_fn in models.items():
        print(f"\n=== {model_name} ===")
        # Chaque run_model retourne : {"y_score": (y_true, y_pred), "y_members": (y_true, y_pred)}
        results = run_fn(n_splits=5)

        for target, (y_true, y_pred) in results.items():
            metrics = ev.evaluate_model(y_true, y_pred)
            print(f"{target} metrics: {metrics}")
            save_plots(y_true, y_pred, model_name, target)

if __name__ == "__main__":
    main()
