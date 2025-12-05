# src/domain/aggregate_features.py
import pandas as pd
import ast
import numpy as np
from pathlib import Path

def aggregate_list_stats(
    df: pd.DataFrame,
    id_column: str,
    stats_path: str,
    corrections: dict = None,
    drop_id_column: bool = True
) -> pd.DataFrame:
    """
    Agrège les statistiques depuis un CSV de stats pour une liste d'IDs dans un DataFrame.

    Args:
        df (pd.DataFrame): DataFrame principal.
        id_column (str): Nom de la colonne contenant des listes d'IDs.
        stats_path (str): Chemin vers le CSV contenant les stats des IDs.
        corrections (dict, optional): Colonnes à corriger avec une fonction lambda.
            Exemple : {"mean_score_staff": lambda val, row: val - row["y_score"]/row["count_anime_staff"]}
        drop_id_column (bool): Supprime la colonne id_column si True.

    Returns:
        pd.DataFrame: DataFrame enrichi avec les colonnes agrégées.
    """

    # --- Charger les stats ---
    stats = pd.read_csv(stats_path)
    id_stats_dict = stats.set_index(stats.columns[0]).to_dict(orient="index")
    stat_cols = [col for col in stats.columns if col != stats.columns[0]]

    # --- Convertir les listes ---
    df[id_column] = df[id_column].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])

    # --- Fonction d'agrégation ---
    def aggregate_ids(id_list, row):
        if not id_list:
            return pd.Series({col: np.nan for col in stat_cols})

        stats_per_id = [id_stats_dict[i] for i in id_list if i in id_stats_dict]
        if not stats_per_id:
            return pd.Series({col: np.nan for col in stat_cols})

        aggregated = {}
        for col in stat_cols:
            values = []
            for s in stats_per_id:
                if pd.isna(s[col]):
                    continue
                val = s[col]
                if corrections and col in corrections:
                    val = corrections[col](val, row)
                values.append(val)
            aggregated[col] = np.mean(values) if values else np.nan
        return pd.Series(aggregated)

    # --- Appliquer la fonction ---
    df_agg = df.apply(lambda row: aggregate_ids(row[id_column], row), axis=1)
    df = pd.concat([df, df_agg], axis=1)

    if drop_id_column:
        df = df.drop(columns=[id_column], errors='ignore')

    return df
