# src/domain/recommendation_features.py
import pandas as pd
import sys
from pathlib import Path

# Ajouter la racine du projet au path pour les imports utils
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.utils.paths import RAW, PROCESSED

RAW_DIR = RAW
PROCESSED_DIR = PROCESSED
PROCESSED_DIR.mkdir(exist_ok=True)

def generate_reco_features(
    recommendations_file: str = "recommendations.csv",
    details_file: str = "details.csv",
    stats_file: str = "stats.csv"
) -> pd.DataFrame:
    """
    Génère un seul fichier contenant toutes les features des recommendations
    pour chaque anime : scores, membres et stats des watchers.
    
    Args:
        recommendations_file (str): fichier recommendations.csv
        details_file (str): fichier details.csv
        stats_file (str): fichier stats.csv
    
    Returns:
        pd.DataFrame: DataFrame complet avec toutes les stats agrégées
    """
    
    # Charger les fichiers
    reco = pd.read_csv(RAW_DIR / recommendations_file)
    details = pd.read_csv(RAW_DIR / details_file)
    stats = pd.read_csv(RAW_DIR / stats_file)
    
    # Ajouter score et members aux recommendations
    details_renamed = details.rename(columns={"mal_id": "recommendation_mal_id"})
    reco = reco.merge(
        details_renamed[["recommendation_mal_id", "score", "members"]],
        on="recommendation_mal_id",
        how="left"
    )


    # Ajouter stats des vues aux recommendations
    stats_renamed = stats.rename(columns={"mal_id": "recommendation_mal_id"})
    reco = reco.merge(stats_renamed, on="recommendation_mal_id", how="left")
    
    # Calculer toutes les stats agrégées par anime
    agg_score_members = reco.groupby("mal_id").agg(
        mean_score_reco=("score", "mean"),
        max_score_reco=("score", "max"),
        min_score_reco=("score", "min"),
        mean_members_reco=("members", "mean"),
        max_members_reco=("members", "max"),
        min_members_reco=("members", "min")
    )
    
    # Stats des watchers
    watcher_cols = ["watching", "completed", "on_hold", "dropped", "plan_to_watch", "total"]
    agg_watchers = reco.groupby("mal_id")[watcher_cols].agg(["mean", "max", "min"])
    agg_watchers.columns = [f"{col}_{stat}" for col, stat in agg_watchers.columns]
    
    # Fusionner toutes les stats
    reco_features = agg_score_members.merge(agg_watchers, left_index=True, right_index=True).reset_index()
    
    # Sauvegarde finale
    final_path = PROCESSED_DIR / "recommendation_features.csv"
    reco_features.to_csv(final_path, index=False)
    
    print(f"Fichier final généré : {final_path}")
    
    return reco_features

if __name__ == "__main__":
    generate_reco_features()
