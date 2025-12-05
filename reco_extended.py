import pandas as pd

# --- Charger les fichiers ---
reco = pd.read_csv("recommendations.csv")  # contient mal_id et recommendation_mal_id
stats = pd.read_csv("stats.csv")            # contient mal_id, watching, completed, on_hold, dropped, plan_to_watch, total

# --- Merge les stats dans les recommendations ---
reco = reco.merge(
    stats.rename(columns={"mal_id": "recommendation_mal_id"}),
    on="recommendation_mal_id",
    how="left"
)

# --- Colonnes à agréger ---
agg_columns = ["watching", "completed", "on_hold", "dropped", "plan_to_watch", "total"]
agg_dict = {col: ["mean", "max", "min"] for col in agg_columns}

# --- Calculer les stats agrégées par anime de base ---
reco_stats_extended = reco.groupby("mal_id").agg(agg_dict)

# --- Aplatir les colonnes multi-index ---
reco_stats_extended.columns = [f"{col}_{stat}" for col, stat in reco_stats_extended.columns]
reco_stats_extended = reco_stats_extended.reset_index()

# --- Sauvegarder ---
reco_stats_extended.to_csv("reco_stats_extended.csv", index=False)
print(reco_stats_extended.head())
