from utils.seed import set_seed
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd

def run_model(df=None, dataset_path: str = None, n_splits=5):
    """
    Exécute un modèle Ridge pour prédire y_score et y_members avec nettoyage des données.
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
    
    # S'assurer que X est un DataFrame pour faciliter le nettoyage
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    # 1. Nettoyage et Imputation des NaN (Cause N°1)
    # Convertir toutes les colonnes en numérique (les objets restants pourraient causer des problèmes)
    X = X.apply(pd.to_numeric, errors='coerce') 
    X_imputed = X.fillna(X.mean()) # Imputation simple par la moyenne
    
    # 2. Élimination des features constantes (Cause N°2)
    variances = X_imputed.var() 
    cols_to_drop = variances[variances < 1e-6].index
    X_final = X_imputed.drop(columns=cols_to_drop)
    
    if len(cols_to_drop) > 0:
        print(f"⚠️ {len(cols_to_drop)} features constantes supprimées avant scaling (variance < 1e-6).")
        
    # --- Scaling ---
    scaler = dp.StandardScaler()
    X_scaled = scaler.fit_transform(X_final) # Scaling sur les données nettoyées

    y_members_log = np.log1p(y["y_members"])

    model_score = Ridge(alpha=1.0)
    model_members = Ridge(alpha=1.0)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # --- y_score ---
    metrics_score_list = []
    # KFold.split() retourne les index pour le tableau NumPy (X_scaled)
    for train_idx, val_idx in kf.split(X_scaled): 
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y["y_score"].iloc[train_idx], y["y_score"].iloc[val_idx]
        
        # Vérification des NaN/Inf dans les sets d'entraînement (diagnostic de dernier recours)
        if np.isnan(X_train).any() or np.isinf(X_train).any():
             raise ValueError("NaN/Inf trouvés dans X_train après scaling et nettoyage.")

        model_score.fit(X_train, y_train)
        y_pred = model_score.predict(X_val)
        metrics_score_list.append(ev.evaluate_model(y_val, y_pred))

    metrics_score_avg = {k: np.mean([m[k] for m in metrics_score_list]) for k in metrics_score_list[0]}
    print("\n===== Ridge - y_score (K-Fold CV) =====")
    print(metrics_score_avg)

    # --- y_members ---
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
    
    return metrics_score_avg, metrics_members_avg