# src/domain/data_processing.py
import pandas as pd
import sys
from pathlib import Path

# Ajouter la racine du projet au path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.utils import COLUMNS_TO_KEEP
from src.utils.paths import RAW, PROCESSED  


RAW_DIR = RAW
PROCESSED_DIR = PROCESSED
PROCESSED_DIR.mkdir(exist_ok=True)

def preprocess_anime_dataset(start_year=2010):
    """
    Prétraitement du dataset anime.
    Args:
        start_year (int): année minimale pour start_date
    Returns:
        pd.DataFrame: dataset prétraité
    """
    
    # Charger les fichiers
 
    details = pd.read_csv(RAW_DIR / "details.csv")
    characters = pd.read_csv(RAW_DIR / "character_anime_works.csv")
    recommendations = pd.read_csv(RAW_DIR / "recommendations.csv")
    staff_works = pd.read_csv(RAW_DIR / "person_anime_works.csv")
    
    # Filtrer colonnes et dropna
    details = details[COLUMNS_TO_KEEP]

    # Garder uniquement les animés commencés après start_year
    details["start_date"] = pd.to_datetime(details["start_date"], errors="coerce")
    details = details[details["start_date"].dt.year >= start_year]
    print(f"Après filtrage par date >= {start_year} :", len(details))

    # Fonction pour déduire la saison si manquante
    def infer_season(row):
        if pd.notna(row["season"]):
            return row["season"].lower()
        if pd.isna(row["start_date"]):
            return None
        month = row["start_date"].month
        day = row["start_date"].day
        if (month == 12 and day >= 21) or (month in [1, 2]) or (month == 3 and day < 21):
            return "winter"
        elif (month == 3 and day >= 21) or (month in [4, 5]) or (month == 6 and day < 21):
            return "spring"
        elif (month == 6 and day >= 21) or (month in [7, 8]) or (month == 9 and day < 21):
            return "summer"
        else:
            return "fall"

    details["season"] = details.apply(infer_season, axis=1)

    details = details.dropna(subset=["episodes", "season", "score", "members"])
    details = details.drop(columns=["start_date"])
    print("Après dropna :", len(details))

    # Ajouter nombre de personnages
    char_counts = characters.groupby("anime_mal_id")["role"].value_counts().unstack(fill_value=0)
    char_counts = char_counts.rename(columns={"Main": "main_characters", "Supporting": "supporting_characters"})
    details = details.merge(char_counts, left_on="mal_id", right_index=True, how="left")
    details[["main_characters", "supporting_characters"]] = details[["main_characters", "supporting_characters"]].fillna(0).astype(int)

    # Ajouter recommandations
    
    rec_grouped = recommendations.groupby("mal_id")["recommendation_mal_id"].apply(list).reset_index()
    details = details.merge(rec_grouped, on="mal_id", how="left")
    details["recommendation_mal_id"] = details["recommendation_mal_id"].apply(lambda x: x if isinstance(x, list) else [])

    # Ajouter liste des staffeurs
    staff_grouped = staff_works.groupby("anime_mal_id")["person_mal_id"].apply(list).reset_index()
    details = details.merge(staff_grouped, left_on="mal_id", right_on="anime_mal_id", how="left")
    details = details.drop(columns=["anime_mal_id"])
    details["staff"] = details["person_mal_id"].apply(lambda x: x if isinstance(x, list) else [])
    details = details.drop(columns=["person_mal_id"])

    # Supprimer lignes avec listes vides
    details = details[(details["staff"].map(len) > 0) & (details["recommendation_mal_id"].map(len) > 0)]

    # Renommer score et members
    details = details.rename(columns={"score": "y_score", "members": "y_members"})
    details.dropna()

    # Sauvegarder
    output_path = PROCESSED_DIR / f"anime_dataset_{start_year}.csv"

    # Générer le CSV uniquement si le fichier n'existe pas
    details.to_csv(output_path, index=False)
    if not output_path.exists():
        details.to_csv(output_path, index=False)
        print(f"Fichier généré : {output_path}")
    else:
        print(f"Fichier déjà existant : {output_path}")

   
    print("Dataset prétraité sauvegardé :", output_path)
    print(details.info())
    print("Nombre de lignes finales :", len(details))

    # Identifier les colonnes qui posent problème (liste ou autre type non hashable)
    colonnes_a_ignorer = []
    for col in details.columns:
        if details[col].apply(lambda x: isinstance(x, list)).any():
            colonnes_a_ignorer.append(col)

    # Afficher les colonnes traitées
    colonnes_a_traiter = [col for col in details.columns if col not in colonnes_a_ignorer]

    print("Nombre de valeurs uniques par colonne (sans les colonnes listes) :")
    print(details[colonnes_a_traiter].nunique())

    return details




if __name__ == "__main__":
    preprocess_anime_dataset()
