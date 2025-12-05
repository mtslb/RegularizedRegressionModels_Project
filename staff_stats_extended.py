import pandas as pd
import ast
import numpy as np

# --- Charger les fichiers ---
df = pd.read_csv("anime_features.csv")
staff_stats = pd.read_csv("staff_stats_extended.csv")  # stats par staffeur

# --- Convertir la colonne staff en liste Python ---
df["staff"] = df["staff"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])

# --- Créer un dictionnaire staff_id -> stats ---
staff_stats_dict = staff_stats.set_index("person_mal_id").to_dict(orient="index")

# --- Colonnes de stats à agréger ---
stat_cols = [col for col in staff_stats.columns if col != "person_mal_id"]

# --- Fonction pour agréger les stats des staffeurs d'un anime ---
def aggregate_staff_stats(staff_list):
    if not staff_list:
        # Si pas de staff, retourner NaN pour chaque stat
        return pd.Series({f"{col}_agg": np.nan for col in stat_cols})
    
    # Récupérer toutes les stats des staffeurs existants
    stats_per_staff = []
    for staff_id in staff_list:
        if staff_id in staff_stats_dict:
            stats_per_staff.append(staff_stats_dict[staff_id])
    
    if not stats_per_staff:
        return pd.Series({f"{col}_agg": np.nan for col in stat_cols})
    
    # Pour chaque colonne de stats, calculer mean, max, min
    aggregated = {}
    for col in stat_cols:
        values = [s[col] for s in stats_per_staff if pd.notna(s[col])]
        if values:
            aggregated[f"{col}_staff"] = np.mean(values)
            aggregated[f"{col}_staff"] = np.max(values)
            aggregated[f"{col}_staff"] = np.min(values)
        else:
            aggregated[f"{col}_staff"] = np.nan
            aggregated[f"{col}_staff"] = np.nan
            aggregated[f"{col}_staff"] = np.nan
            
    return pd.Series(aggregated)

# --- Appliquer aux lignes ---
df_staff_agg = df["staff"].apply(aggregate_staff_stats)

# --- Ajouter les colonnes agrégées au dataframe ---
df = pd.concat([df, df_staff_agg], axis=1)

# --- Optionnel : supprimer la colonne staff ---
df = df.drop(columns=["staff"])

# --- Sauvegarder ---
df.to_csv("anime_features_final.csv", index=False)
print(df.head())
