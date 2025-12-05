import pandas as pd
import ast
import numpy as np

# ----------------------------
# 1. Charger les fichiers
# ----------------------------
details = pd.read_csv("details.csv")
characters = pd.read_csv("character_anime_works.csv")
recommendations = pd.read_csv("recommendations.csv")
staff_works = pd.read_csv("person_anime_works.csv")

# ----------------------------
# 2. Filtrer colonnes et dropna
# ----------------------------
keep_cols = ["mal_id", "type", "status", "genres", "studios", "themes",
             "start_date", "demographics", "source", "rating", "episodes",
             "season", "score", "members"]
details = details[keep_cols]
# Garder uniquement les animés commencés après 2010
details["start_date"] = pd.to_datetime(details["start_date"], errors="coerce")
details = details[details["start_date"].dt.year >= 2010]
print("Après filtrage par date :", len(details))
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
    else:  # 21 septembre – 20 décembre
        return "fall"


details["season"] = details.apply(infer_season, axis=1)

details = details.dropna(subset=[ "episodes","season", "score", "members"])

details = details.drop(columns=["start_date", "status"])
print("Après dropna :", len(details))
# ----------------------------
# 3. Ajouter nombre de personnages
# ----------------------------
char_counts = characters.groupby("anime_mal_id")["role"].value_counts().unstack(fill_value=0)
char_counts = char_counts.rename(columns={"Main": "main_characters", "Supporting": "supporting_characters"})
details = details.merge(char_counts, left_on="mal_id", right_index=True, how="left")
details[["main_characters", "supporting_characters"]] = details[["main_characters", "supporting_characters"]].fillna(0).astype(int)

# ----------------------------
# 4. Ajouter recommandations
# ----------------------------
rec_grouped = recommendations.groupby("mal_id")["recommendation_mal_id"].apply(list).reset_index()
details = details.merge(rec_grouped, on="mal_id", how="left")
details["recommendation_mal_id"] = details["recommendation_mal_id"].apply(lambda x: x if isinstance(x, list) else [])

# ----------------------------
# 5. Ajouter liste des staffeurs
# ----------------------------
staff_grouped = staff_works.groupby("anime_mal_id")["person_mal_id"].apply(list).reset_index()
details = details.merge(staff_grouped, left_on="mal_id", right_on="anime_mal_id", how="left")
details = details.drop(columns=["anime_mal_id"])
details["staff"] = details["person_mal_id"].apply(lambda x: x if isinstance(x, list) else [])
details = details.drop(columns=["person_mal_id"])

# ----------------------------
# 6. Supprimer lignes avec listes vides
# ----------------------------
details = details[(details["staff"].map(len) > 0) & (details["recommendation_mal_id"].map(len) > 0)]

# ----------------------------
# 7. Renommer score et members
# ----------------------------
details = details.rename(columns={"score": "y_score", "members": "y_members"})

# ----------------------------
# 8. Sauvegarder
# ----------------------------
details.to_csv("anime_dataset.csv", index=False)
print(details.info())
print(len(details))