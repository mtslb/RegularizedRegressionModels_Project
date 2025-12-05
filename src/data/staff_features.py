# src/domain/staff_features_gen.py
import pandas as pd
import sys
from pathlib import Path

# Ajouter la racine du projet au path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.utils.paths import RAW, PROCESSED  

RAW_DIR = RAW
PROCESSED_DIR = PROCESSED
PROCESSED_DIR.mkdir(exist_ok=True)

def generate_staff_stats(
    staff_works_file: str = "person_anime_works.csv",
    details_file: str = "details.csv",
    anime_stats_file: str = "stats.csv"
) -> pd.DataFrame:
    """
    Génère les statistiques par staffeur à partir des fichiers bruts.
    
    Args:
        staff_works_file (str): chemin vers person_anime_works.csv
        details_file (str): chemin vers details.csv
        anime_stats_file (str): chemin vers stats.csv
    
    Returns:
        pd.DataFrame: DataFrame avec les stats par staffeur et sauvegardé en CSV
    """
    
    # ----------------------------
    # 1️⃣ Charger les fichiers
    # ----------------------------
    staff_works = pd.read_csv(RAW_DIR / staff_works_file)
    anime_details = pd.read_csv(RAW_DIR / details_file)
    anime_stats = pd.read_csv(RAW_DIR / anime_stats_file)
    
    # ----------------------------
    # 2️⃣ Ajouter score et members à chaque ligne de staff_works
    # ----------------------------
    anime_details = anime_details.rename(columns={"mal_id": "anime_mal_id"})
    staff_works = staff_works.merge(
        anime_details[["anime_mal_id", "score", "members"]],
        on="anime_mal_id",
        how="left"
    )
    
    # ----------------------------
    # 3️⃣ Ajouter les stats à chaque ligne de staff_works
    # ----------------------------
    anime_stats = anime_stats.rename(columns={"mal_id": "anime_mal_id"})
    staff_works = staff_works.merge(anime_stats, on="anime_mal_id", how="left")
    
    # ----------------------------
    # 4️⃣ Calculer les stats par staffeur
    # ----------------------------
    agg_dict = {
        "score": "mean",
        "members": "mean",
        "anime_mal_id": "count"
    }
    
    staff_stats = staff_works.groupby("person_mal_id").agg(agg_dict).reset_index()
    
    # Renommer les colonnes
    staff_stats = staff_stats.rename(columns={
        "score": "mean_score_staff",
        "members": "mean_members_staff",
        "anime_mal_id": "count_anime_staff"
    })
    
    # ----------------------------
    # 5️⃣ Sauvegarder
    # ----------------------------
    staff_stats_path = PROCESSED_DIR / "staff_features.csv"
    staff_stats.to_csv(staff_stats_path, index=False)
    
    print(f"Fichier généré : {staff_stats_path}")
    print(staff_stats.head())
     
    return staff_stats

if __name__ == "__main__":  
    generate_staff_stats()
