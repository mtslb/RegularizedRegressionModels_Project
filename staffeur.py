import pandas as pd
import numpy as np

# --- Charger les fichiers ---
staff_works = pd.read_csv("person_anime_works.csv")
anime_details = pd.read_csv("details.csv")  # contient mal_id, score, members
anime_stats = pd.read_csv("stats.csv")      # stats par anime : mal_id, watching, completed, etc.

# --- Ajouter score et members à chaque ligne de staff_works ---
anime_details = anime_details.rename(columns={"mal_id": "anime_mal_id"})
staff_works = staff_works.merge(
    anime_details[["anime_mal_id", "score", "members"]],
    on="anime_mal_id",
    how="left"
)

# --- Ajouter les stats à chaque ligne de staff_works ---
anime_stats = anime_stats.rename(columns={"mal_id": "anime_mal_id"})
staff_works = staff_works.merge(anime_stats, on="anime_mal_id", how="left")

# --- Calculer les stats par staffeur ---
agg_dict = {
    "score": "mean",
    "members": "mean",
    "anime_mal_id": "count"   # <- NOUVEAU : compteur d'animes
}

staff_stats = staff_works.groupby("person_mal_id").agg(agg_dict).reset_index()

# Renommer les colonnes
staff_stats = staff_stats.rename(columns={
    "score": "mean_score_staff",
    "members": "mean_members_staff",
    "anime_mal_id": "count_anime_staff"     # <- NOUVEAU
})

# --- Export ---
print(staff_stats.head())
staff_stats.to_csv("staff_stats.csv", index=False)
