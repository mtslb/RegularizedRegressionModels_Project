import pandas as pd
import ast
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.paths import PROCESSED

def aggregate_recommendation_stats(
    df: pd.DataFrame,
    stats_csv: str = PROCESSED / "recommendation_features.csv",
    list_column: str ="recommendation_mal_id",
    id_column_in_csv: str = "mal_id",
    agg_cols: list = None
) -> pd.DataFrame:
    """
    Agrège des stats depuis un CSV pour une colonne de listes dans le DataFrame.

    Args:
        df (pd.DataFrame): DataFrame principal
        stats_csv (str): chemin vers le CSV contenant les stats
        list_column (str): colonne du df contenant des listes d'IDs
        id_column_in_csv (str): colonne du CSV correspondant à l'ID
        agg_cols (list, optional): colonnes du CSV à agréger. Si None, prend toutes sauf l'ID

    Returns:
        pd.DataFrame: df enrichi avec les colonnes agrégées
    """

    # --- Charger le CSV ---
    stats = pd.read_csv(stats_csv)
    if agg_cols is None:
        agg_cols = [col for col in stats.columns if col != id_column_in_csv]

    # --- Convertir la colonne en liste Python ---
    df[list_column] = df[list_column].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])

    # --- Créer un dictionnaire ID -> stats ---
    stats_dict = stats.set_index(id_column_in_csv).to_dict(orient="index")

    # --- Fonction d'agrégation ---
    def aggregate_ids(id_list):
        if not id_list:
            return pd.Series({f"{col}_mean_agg": np.nan for col in agg_cols} |
                             {f"{col}_max_agg": np.nan for col in agg_cols} |
                             {f"{col}_min_agg": np.nan for col in agg_cols})
        
        agg_data = {col: [] for col in agg_cols}
        for i in id_list:
            s = stats_dict.get(i)
            if s:
                for col in agg_cols:
                    val = s.get(col, np.nan)
                    if val is not None:
                        agg_data[col].append(val)
        
        result = {}
        for col, values in agg_data.items():
            values = [v for v in values if not pd.isna(v)]
            result[f"{col}_mean_agg"] = np.mean(values) if values else np.nan
            result[f"{col}_max_agg"] = np.max(values) if values else np.nan
            result[f"{col}_min_agg"] = np.min(values) if values else np.nan
        
        return pd.Series(result)

    # --- Appliquer l'agrégation ---
    df_agg = df[list_column].apply(aggregate_ids)

    # --- Ajouter les colonnes au df ---
    df = pd.concat([df, df_agg], axis=1)

    # --- Remplacer NaN par 0 ---
    df = df.fillna(0)

    return df





def aggregate_staff_stats(
    df: pd.DataFrame,
    staff_stats_csv: str = PROCESSED / "staff_features.csv",
    y_score_col: str = "y_score",
    y_members_col: str = "y_members",
    staff_col: str = "staff",
    drop_staff_col: bool = True
) -> pd.DataFrame:
    """
    Agrège les statistiques des staffeurs pour chaque anime.

    Args:
        df (pd.DataFrame): DataFrame avec colonne staff contenant des listes d'IDs
        staff_stats_csv (str): chemin vers staff_stats.csv
        y_score_col (str): colonne score de l'anime pour correction
        y_members_col (str): colonne members de l'anime pour correction
        staff_col (str): nom de la colonne staff dans df
        drop_staff_col (bool): supprime la colonne staff si True

    Returns:
        pd.DataFrame: DataFrame enrichi avec les stats agrégées des staffeurs
    """

    # --- Charger les stats des staffeurs ---
    staff_stats = pd.read_csv(staff_stats_csv)
    staff_stats_dict = staff_stats.set_index("person_mal_id").to_dict(orient="index")

    # --- Colonnes de stats à agréger ---
    stat_cols = [col for col in staff_stats.columns if col != "person_mal_id"]

    # --- Convertir la colonne staff en listes Python ---
    df[staff_col] = df[staff_col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])

    # --- Fonction pour agréger les stats des staffeurs d'un anime ---
    def aggregate_staff_stats_row(staff_list, y_score, y_members):
        if not staff_list:
            return pd.Series({col: np.nan for col in stat_cols})

        stats_per_staff = [staff_stats_dict[s] for s in staff_list if s in staff_stats_dict]
        if not stats_per_staff:
            return pd.Series({col: np.nan for col in stat_cols})

        aggregated = {}
        for col in stat_cols:
            values = []
            for s in stats_per_staff:
                if pd.isna(s[col]):
                    continue
                if col == "mean_score_staff":
                    corrected = s[col] - (y_score / s["count_anime_staff"])
                elif col == "mean_members_staff":
                    corrected = s[col] - (y_members / s["count_anime_staff"])
                else:
                    corrected = s[col]
                values.append(corrected)
            aggregated[col] = np.mean(values) if values else np.nan
        return pd.Series(aggregated)

    # --- Appliquer la fonction ---
    df_staff_agg = df.apply(lambda row: aggregate_staff_stats_row(row[staff_col],
                                                                 row[y_score_col],
                                                                 row[y_members_col]), axis=1)

    # --- Ajouter les colonnes agrégées au dataframe ---
    df = pd.concat([df, df_staff_agg], axis=1)

    # --- Optionnel : supprimer la colonne staff ---
    if drop_staff_col:
        df = df.drop(columns=[staff_col, "count_anime_staff"], errors='ignore')

    return df
