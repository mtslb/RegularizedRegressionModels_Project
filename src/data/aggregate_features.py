import pandas as pd
import ast
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.paths import PROCESSED

def aggregate_recommendation_stats(
    df: pd.DataFrame,
    reco_stats_csv: str = PROCESSED / "recommendation_features.csv",
    reco_col: str = "recommendation_mal_id",
    id_column_in_csv: str = "mal_id",
    drop_reco_col: bool = True
) -> pd.DataFrame:
    """
    Agrège les statistiques des recommandations pour chaque anime.

    Args:
        df (pd.DataFrame): DataFrame avec colonne reco_col contenant des listes d'IDs
        reco_stats_csv (str): chemin vers recommendation_features.csv
        reco_col (str): nom de la colonne de listes d'IDs dans df
        id_column_in_csv (str): nom de la colonne ID dans le CSV
        drop_reco_col (bool): supprime la colonne reco_col si True

    Returns:
        pd.DataFrame: DataFrame enrichi avec les stats agrégées
    """

    # --- Charger les stats ---
    reco_stats = pd.read_csv(reco_stats_csv)
    agg_cols = [col for col in reco_stats.columns if col != id_column_in_csv]

    # --- Convertir la colonne en listes d'int ---
    def parse_list(x):
        if isinstance(x, str):
            try:
                return [int(i) for i in ast.literal_eval(x)]
            except:
                return []
        elif isinstance(x, list):
            return [int(i) for i in x if isinstance(i, (int, float, str))]
        return []

    df[reco_col] = df[reco_col].apply(parse_list)

    # --- Dictionnaire ID -> stats ---
    reco_stats_dict = reco_stats.set_index(id_column_in_csv).to_dict(orient="index")

    # --- Fonction d'agrégation ---
    def aggregate_reco_row(reco_list):
        if not reco_list:
            return pd.Series({f"{col}_mean_agg": np.nan for col in agg_cols} |
                             {f"{col}_max_agg": np.nan for col in agg_cols} |
                             {f"{col}_min_agg": np.nan for col in agg_cols})

        agg_data = {col: [] for col in agg_cols}
        for rid in reco_list:
            stats = reco_stats_dict.get(rid)
            if stats:
                for col in agg_cols:
                    val = stats.get(col, np.nan)
                    if val is not None:
                        agg_data[col].append(val)

        result = {}
        for col, values in agg_data.items():
            values = [v for v in values if not pd.isna(v)]
            result[f"{col}_mean_agg"] = np.mean(values) if values else np.nan
            result[f"{col}_max_agg"] = np.max(values) if values else np.nan
            result[f"{col}_min_agg"] = np.min(values) if values else np.nan

        return pd.Series(result)

    # --- Appliquer aux lignes ---
    df_reco_agg = df[reco_col].apply(aggregate_reco_row)

    # --- Ajouter les colonnes au DataFrame ---
    df = pd.concat([df, df_reco_agg], axis=1)

    # --- Supprimer la colonne originale si demandé ---
    if drop_reco_col:
        df = df.drop(columns=[reco_col], errors='ignore')

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

    # --- Convertir la colonne staff en listes d'int ---
    def parse_staff_list(x):
        if isinstance(x, str):
            try:
                return [int(i) for i in ast.literal_eval(x)]
            except:
                return []
        elif isinstance(x, list):
            return [int(i) for i in x if isinstance(i, (int, float, str))]
        return []

    df[staff_col] = df[staff_col].apply(parse_staff_list)

    # --- Fonction pour agréger les stats d'un anime ---
    def aggregate_staff_stats_row(staff_list, y_score, y_members):
        stats_per_staff = [staff_stats_dict[s] for s in staff_list if s in staff_stats_dict]
        if not stats_per_staff:
            return pd.Series({col: np.nan for col in stat_cols})

        aggregated = {}
        for col in stat_cols:
            values = []
            for s in stats_per_staff:
                if pd.isna(s[col]):
                    continue
                divisor = s.get("count_anime_staff", 1)
                if divisor == 0:
                    divisor = 1
                if col == "mean_score_staff":
                    corrected = s[col] - (y_score / divisor)
                elif col == "mean_members_staff":
                    corrected = s[col] - (y_members / divisor)
                else:
                    corrected = s[col]
                values.append(corrected)
            aggregated[col] = np.mean(values) if values else np.nan
        return pd.Series(aggregated)

    # --- Appliquer aux lignes ---
    df_staff_agg = df.apply(
        lambda row: aggregate_staff_stats_row(
            row[staff_col], row[y_score_col], row[y_members_col]
        ),
        axis=1
    )

    # --- Ajouter les colonnes au DataFrame ---
    df = pd.concat([df, df_staff_agg], axis=1)

    # --- Supprimer la colonne staff si nécessaire ---
    if drop_staff_col:
        df = df.drop(columns=[staff_col, "count_anime_staff"], errors='ignore')

    return df