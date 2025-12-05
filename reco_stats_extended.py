import pandas as pd
import ast
import numpy as np

# --- Charger les fichiers ---
df = pd.read_csv("anime_features.csv")              # Dataset principal
reco_stats_ext = pd.read_csv("reco_stats_extended.csv")  # Stats étendues des recommandations

# --- Convertir la colonne recommendation_mal_id en liste Python ---
df["recommendation_mal_id"] = df["recommendation_mal_id"].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) else []
)

# --- Créer un dictionnaire mal_id -> stats étendues ---
reco_stats_ext_dict = reco_stats_ext.set_index("mal_id").to_dict(orient="index")

# --- Fonction pour agréger les stats étendues des recommandations ---
def aggregate_reco_stats_ext(reco_list):
    if not reco_list:
        return pd.Series({f"{col}_{stat}_agg": np.nan
                          for col in ["watching", "completed", "on_hold", "dropped", "plan_to_watch", "total"]
                          for stat in ["mean","max","min"]})
    
    agg_data = {col: [] for col in ["watching", "completed", "on_hold", "dropped", "plan_to_watch", "total"]}
    
    for mal in reco_list:
        stats = reco_stats_ext_dict.get(mal)
        if stats:
            for col in agg_data.keys():
                agg_data[col].append(stats.get(f"{col}_mean", np.nan))  # On prend la moyenne pour chaque reco
    
    result = {}
    for col, values in agg_data.items():
        values = [v for v in values if not pd.isna(v)]
        result[f"{col}_mean_agg"] = np.mean(values) if values else np.nan
        result[f"{col}_max_agg"] = np.max(values) if values else np.nan
        result[f"{col}_min_agg"] = np.min(values) if values else np.nan
    
    return pd.Series(result)

# --- Appliquer aux lignes ---
df_reco_ext_agg = df["recommendation_mal_id"].apply(aggregate_reco_stats_ext)

# --- Ajouter les colonnes au dataframe ---
df = pd.concat([df, df_reco_ext_agg], axis=1)

# --- Remplacer les NaN par 0 ---
df = df.fillna(0)

# --- Supprimer la colonne recommendation_mal_id ---
df = df.drop(columns=["recommendation_mal_id"])

# --- Sauvegarder le dataset enrichi ---
df.to_csv("anime_features_final.csv", index=False)
print(df.head())
