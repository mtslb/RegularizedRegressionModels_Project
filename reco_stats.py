import pandas as pd

# --- Charger le dataset principal ---
df = pd.read_csv("anime_features_ENCODE.csv")

# --- Charger les stats des recommandations ---
reco_stats = pd.read_csv("reco_stats.csv")              # score et members
reco_stats_extended = pd.read_csv("reco_stats_extended.csv")  # watching, completed, etc.

# --- Merge simple sur mal_id ---
df = df.merge(reco_stats, on="mal_id", how="left")
df = df.merge(reco_stats_extended, on="mal_id", how="left")


# --- Supprimer la colonne recommendation_mal_id ---
df = df.drop(columns=["recommendation_mal_id"])

# --- Sauvegarder le dataset enrichi ---
df.to_csv("anime_features.csv", index=False)

print(df.head())
