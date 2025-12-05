import pandas as pd
import numpy as np

# --- Charger les fichiers ---
reco = pd.read_csv("recommendations.csv")  # contient mal_id et recommendation_mal_id
anime_details = pd.read_csv("details.csv")        # contient mal_id, score, members

# --- Ajouter score et members à chaque recommendation ---
anime_details = anime_details.rename(columns={"mal_id": "recommendation_mal_id"})
reco = reco.merge(
    anime_details[["recommendation_mal_id", "score", "members"]],
    on="recommendation_mal_id",
    how="left"
)

# --- Calculer les stats par anime de base ---
reco_stats = reco.groupby("mal_id").agg(
    mean_score_reco = ("score", "mean"),
    max_score_reco = ("score", "max"),
    min_score_reco = ("score", "min"),
    mean_members_reco = ("members", "mean"),
    max_members_reco = ("members", "max"),
    min_members_reco = ("members", "min")
).reset_index()

# --- Sauvegarder ---
reco_stats.to_csv("reco_stats.csv", index=False)

print(reco_stats.head())
