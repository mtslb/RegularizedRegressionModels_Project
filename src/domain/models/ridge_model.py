# src/domain/models/ridge_model.py
from src.domain import data_processing as dp
from src.domain import evaluation as ev
from utils.seed import set_seed
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
import numpy as np

def run_model(n_splits=5):
    set_seed(42)

    # --- Charger et préparer les données ---
    df = dp.load_data("anime_features_.csv")
    X, y = dp.split_features_targets(df)
    scaler = dp.StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Log-transform pour y_members
    y_members_log = np.log1p(y["y_members"])

    # --- Modèles ---
    model_score = Ridge(alpha=1.0)
    model_members = Ridge(alpha=1.0)

    # --- K-Fold Cross Validation ---
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Scores pour y_score
    metrics_score_list = []
    for train_idx, val_idx in kf.split(X_scaled):
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y["y_score"].iloc[train_idx], y["y_score"].iloc[val_idx]
        model_score.fit(X_train, y_train)
        y_pred = model_score.predict(X_val)
        metrics_score_list.append(ev.evaluate_model(y_val, y_pred))

    metrics_score_avg = {k: np.mean([m[k] for m in metrics_score_list]) for k in metrics_score_list[0]}
    print("\n===== Ridge - y_score (K-Fold CV) =====")
    print(metrics_score_avg)

    # Scores pour y_members (log)
    metrics_members_list = []
    for train_idx, val_idx in kf.split(X_scaled):
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y_members_log.iloc[train_idx], y_members_log.iloc[val_idx]
        model_members.fit(X_train, y_train)
        y_pred_log = model_members.predict(X_val)
        y_pred = np.expm1(y_pred_log)
        metrics_members_list.append(ev.evaluate_model(y["y_members"].iloc[val_idx], y_pred))

    metrics_members_avg = {k: np.mean([m[k] for m in metrics_members_list]) for k in metrics_members_list[0]}
    print("\n===== Ridge - y_members (log, K-Fold CV) =====")
    print(metrics_members_avg)
