from utils.seed import set_seed
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
from sklearn.linear_model import lasso_path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.utils.paths import GRAPHS

def run_model(df=None, dataset_path: str = None, n_splits=5):
    """
    Lasso model pipeline (identique à Ridge pipeline structure)
    """
    import src.domain.data_processing as dp
    import src.domain.evaluation as ev

    set_seed(42)

    # Load
    if df is None:
        if dataset_path is None:
            raise ValueError("Il faut fournir df ou dataset_path")
        df = dp.load_data(dataset_path)

    X, y = dp.split_features_targets(df)
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    # Clean / to numeric
    X = X.apply(pd.to_numeric, errors="coerce")

    # Impute
    X_imputed = X.fillna(X.mean())

    # Drop constant features
    variances = X_imputed.var()
    cols_to_drop = variances[variances < 1e-6].index
    X_final = X_imputed.drop(columns=cols_to_drop)
    if len(cols_to_drop) > 0:
        print(f"⚠️ {len(cols_to_drop)} features constantes supprimées : {list(cols_to_drop)}")

    # Scaling
    scaler = dp.StandardScaler()
    X_scaled = scaler.fit_transform(X_final)

    # Targets
    y_score = y["y_score"]
    y_members_log = np.log1p(y["y_members"])

    model_score = Lasso(alpha=0.01, max_iter=50000)
    model_members = Lasso(alpha=0.01, max_iter=50000)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Collect predictions for regression plots
    all_y_true_score, all_y_pred_score = [], []
    all_y_true_members, all_y_pred_members = [], []

    # K-Fold y_score
    metrics_score_list = []
    for train_idx, val_idx in kf.split(X_scaled):
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y_score.iloc[train_idx], y_score.iloc[val_idx]

        model_score.fit(X_train, y_train)
        y_pred = model_score.predict(X_val)

        metrics_score_list.append(ev.evaluate_model(y_val, y_pred))

        all_y_true_score.append(y_val.to_numpy())
        all_y_pred_score.append(y_pred)

    metrics_score_avg = {k: np.mean([m[k] for m in metrics_score_list]) for k in metrics_score_list[0]}
    print("\n===== Lasso - y_score (K-FOLD) =====")
    print(metrics_score_avg)

    # K-Fold y_members (log)
    metrics_members_list = []
    for train_idx, val_idx in kf.split(X_scaled):
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        model_members.fit(X_train, y_members_log.iloc[train_idx])
        y_pred_log = model_members.predict(X_val)

        y_pred = np.expm1(y_pred_log)
        y_val_real = y["y_members"].iloc[val_idx]

        metrics_members_list.append(ev.evaluate_model(y_val_real, y_pred))

        all_y_true_members.append(y_val_real.to_numpy())
        all_y_pred_members.append(y_pred)

    metrics_members_avg = {k: np.mean([m[k] for m in metrics_members_list]) for k in metrics_members_list[0]}
    print("\n===== Lasso - y_members (LOG + K-FOLD) =====")
    print(metrics_members_avg)

    # Graphs per metric (folds)
    metric_names = list(metrics_score_list[0].keys())
    for metric in metric_names:
        plt.figure()
        plt.plot([m[metric] for m in metrics_score_list], marker="o")
        plt.title(f"{metric} - Lasso - y_score (KFold)")
        plt.xlabel("Fold")
        plt.ylabel(metric)
        plt.grid(True)
        plt.savefig(GRAPHS / f"lasso_y_score_{metric}.png", dpi=300)
        plt.close()

    for metric in metric_names:
        plt.figure()
        plt.plot([m[metric] for m in metrics_members_list], marker="o")
        plt.title(f"{metric} - Lasso - y_members (KFold)")
        plt.xlabel("Fold")
        plt.ylabel(metric)
        plt.grid(True)
        plt.savefig(GRAPHS / f"lasso_y_members_{metric}.png", dpi=300)
        plt.close()

    # Regression scatter + error hist (concat folds)
    y_true_score_full = np.concatenate(all_y_true_score) if all_y_true_score else np.array([])
    y_pred_score_full = np.concatenate(all_y_pred_score) if all_y_pred_score else np.array([])

    y_true_members_full = np.concatenate(all_y_true_members) if all_y_true_members else np.array([])
    y_pred_members_full = np.concatenate(all_y_pred_members) if all_y_pred_members else np.array([])

    if y_true_score_full.size:
        ev.plot_results(y_true_score_full, y_pred_score_full,
                        "Lasso - y_score",
                        save_scatter=GRAPHS / "lasso_regression_y_score.png",
                        save_hist=GRAPHS / "lasso_errors_y_score.png")

    if y_true_members_full.size:
        ev.plot_results(y_true_members_full, y_pred_members_full,
                        "Lasso - y_members",
                        save_scatter=GRAPHS / "lasso_regression_y_members.png",
                        save_hist=GRAPHS / "lasso_errors_y_members.png")
    
    alphas_lasso, coefs_lasso, _ = lasso_path(X_scaled, y_score, alphas=None)

    plt.figure(figsize=(12, 7))
    for coef in coefs_lasso:
        plt.plot(np.log10(alphas_lasso), coef)

    plt.xlabel("log10(alpha)")
    plt.ylabel("Coefficient value")
    plt.title("Lasso Regularization Path (y_score)")
    plt.grid(True)
    plt.savefig(GRAPHS / "lasso_regularization_path_y_score.png", dpi=300)
    plt.close()


    alphas_lasso2, coefs_lasso2, _ = lasso_path(X_scaled, y_members_log, alphas=None)

    plt.figure(figsize=(12, 7))
    for coef in coefs_lasso2:
        plt.plot(np.log10(alphas_lasso2), coef)

    plt.xlabel("log10(alpha)")
    plt.ylabel("Coefficient value")
    plt.title("Lasso Regularization Path (y_members_log)")
    plt.grid(True)
    plt.savefig(GRAPHS / "lasso_regularization_path_y_members.png", dpi=300)
    plt.close()

    return metrics_score_avg, metrics_members_avg
