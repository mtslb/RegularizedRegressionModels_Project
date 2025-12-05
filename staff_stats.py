import pandas as pd
import ast
import numpy as np

# --- Charger les fichiers ---
df = pd.read_csv("anime_features.csv")
staff_stats = pd.read_csv("staff_stats.csv")  # toutes les stats des staffeurs

# --- Convertir la colonne staff en listes Python ---
df["staff"] = df["staff"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])

# --- Créer un dictionnaire staff_id -> stats ---
staff_stats_dict = staff_stats.set_index("person_mal_id").to_dict(orient="index")

# --- Colonnes de stats à agréger ---
stat_cols = [col for col in staff_stats.columns if col != "person_mal_id"]

# --- Fonction pour agréger les stats des staffeurs d'un anime ---
def aggregate_staff_stats(staff_list, y_score, y_members):
    """
    Pour un anime :
    - récupère les stats moyennes de chaque staffeur
    - retire la contribution du film actuel (score / count, members / count)
    - calcule la moyenne des staffeurs corrigés
    """
    if not staff_list:
        return pd.Series({col: np.nan for col in stat_cols})

    stats_per_staff = [staff_stats_dict[s] for s in staff_list if s in staff_stats_dict]

    if not stats_per_staff:
        return pd.Series({col: np.nan for col in stat_cols})

    aggregated = {}

    # Pour chaque statistique à traiter
    for col in stat_cols:

        values = []
        for s in stats_per_staff:

            # On ignore si pas de valeur
            if pd.isna(s[col]):
                continue

            # Correction uniquement pour score et members
            if col == "mean_score_staff":
                corrected = s[col] - (y_score / s["count_anime_staff"])
            elif col == "mean_members_staff":
                corrected = s[col] - (y_members / s["count_anime_staff"])
            else:
                corrected = s[col]

            values.append(corrected)

        aggregated[col] = np.mean(values) if values else np.nan

    return pd.Series(aggregated)


# --- Appliquer aux lignes ---
df_staff_agg = df.apply(lambda row: aggregate_staff_stats(row["staff"], row["y_score"], row["y_members"]), axis=1)

# --- Ajouter les colonnes agrégées au dataframe ---
df = pd.concat([df, df_staff_agg], axis=1)


# --- Optionnel : supprimer la colonne staff ---
df = df.drop(columns=["staff", "count_anime_staff"])


# --- Sauvegarder ---
df.to_csv("anime_features_.csv", index=False)
print(df.head())
