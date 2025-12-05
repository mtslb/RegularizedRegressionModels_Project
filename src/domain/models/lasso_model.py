from utils.seed import set_seed
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
import numpy as np

def run_model(df, dataset_path: str = None, n_splits=5):
    """
    Exécute un modèle Lasso pour prédire y_score et y_members.

    Args:
        df (pd.DataFrame, optional): DataFrame prêt à l'emploi.
        dataset_path (str, optional): Chemin vers le CSV si df n'est pas fourni.
        n_splits (int): Nombre de splits pour K-Fold.
    """
    import src.domain.data_processing as dp
    import src.domain.evaluation as ev

    set_seed(42)

    # --- Charger et préparer les données ---
    if df is None:
        if dataset_path is None:
            raise ValueError("Il faut fournir df ou dataset_path")
        df = dp.load_data(dataset_path)

    X, y = dp.split_features_targets(df)
    scaler = dp.StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Log-transform pour y_members
    y_members_log = np.log1p(y["y_members"])

    # --- Modèles ---
    model_score = Lasso(alpha=0.01, max_iter=50000)
    model_members = Lasso(alpha=0.1, max_iter=50000)

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
    print("\n===== Lasso - y_score (K-Fold CV) =====")
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
    print("\n===== Lasso - y_members (log, K-Fold CV) =====")
    print(metrics_members_avg)
