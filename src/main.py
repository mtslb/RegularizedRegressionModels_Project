# src/main.py
import sys
from pathlib import Path
import importlib

# Ajouter la racine du projet au path
sys.path.append(str(Path(__file__).resolve().parents[1]))

# ----------------------------
# Imports internes
# ----------------------------
from src.utils.paths import RAW, PROCESSED, GRAPHS
from src.data.preprocessing import preprocess_anime_dataset
from src.data.encoding import encode_features
from src.data.aggregate_features import aggregate_recommendation_stats, aggregate_staff_stats

# ----------------------------
# Créer les répertoires si nécessaire
# ----------------------------
PROCESSED.mkdir(exist_ok=True, parents=True)
GRAPHS.mkdir(exist_ok=True, parents=True)

# ----------------------------
# Pipeline principal
# ----------------------------
def run_pipeline(start_year=2010, models_to_run=None):
    
    print("=== Étape 1 : Prétraitement ===")
    df = preprocess_anime_dataset(start_year=start_year)

    print("=== Étape 2 : Encodage des features ===")
    df_encoded = encode_features(df)

    print("=== Étape 3 : Agrégation des features ===")
    df_reco = aggregate_recommendation_stats(df_encoded)
    df_final = aggregate_staff_stats(df_reco)

    final_path = PROCESSED / f"anime_features_{start_year}.csv"
    if not final_path.exists(): 
        df_final.to_csv(final_path, index=False)
    else: 
        print("Dataset final déjà existant, pas de réécriture.")


    print(f"Dataset final sauvegardé à : {final_path}")
    # ----------------------------
    # Exécution des modèles
    # ----------------------------
    print("=== Étape 4 : Exécution des modèles ===")

    # Dictionnaire des modèles disponibles
    available_models = {
        "linear": "src.domain.models.linear_model",
        "lasso": "src.domain.models.lasso_model",
        "ridge": "src.domain.models.ridge_model",
        "elasticnet": "src.domain.models.elasticnet_model"
    }

    # Si aucun modèle n'est précisé, exécuter tous
    if models_to_run is None:
        models_to_run = list(available_models.keys()) 

    for model_name in models_to_run:
        if model_name not in available_models:
            print(f"Modèle inconnu : {model_name}, skipping.")
            continue
        mod_path = available_models[model_name]
        print(f"\n--- Modèle : {model_name} ---")
        mod = importlib.import_module(mod_path)
        mod.run_model(df=df_final, n_splits=5)  # Utilise le DataFrame final directement


if __name__ == "__main__":
    run_pipeline(start_year=2010)
