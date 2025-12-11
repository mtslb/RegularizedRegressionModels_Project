import pandas as pd
import ast
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.paths import PROCESSED

def aggregate_recommendation_stats(
    df: pd.DataFrame,
    reco_stats_csv: Path = PROCESSED / "recommendation_features.csv",
    join_column: str = "mal_id",
    drop_recommendation_cols: bool = True
) -> pd.DataFrame:
    """
    Jointure simple des statistiques agrégées de recommandations (issues de 
    recommendation_features.csv) avec le DataFrame principal.
    
    Args:
        df (pd.DataFrame): DataFrame principal.
        reco_stats_csv (Path): Chemin vers le fichier CSV des caractéristiques 
                                déjà agrégées.
        join_column (str): Nom de la colonne ID commune aux deux DataFrames.
        suffix (str): Suffixe à ajouter aux colonnes du fichier de stats 
                      si elles existent déjà dans df.

    Returns:
        pd.DataFrame: Le DataFrame d'origine enrichi des colonnes de stats.
    """

    # Chargement des stats de référence

    try:
        reco_stats = pd.read_csv(reco_stats_csv)
    except FileNotFoundError:
        print(f"Erreur: Fichier de stats non trouvé à {reco_stats_csv}")
        return df
        
    # S'assurer que la colonne de jointure est bien typée
    reco_stats[join_column] = pd.to_numeric(reco_stats[join_column], errors="coerce").astype("Int64")

   
    # Jointure (Merge)

    
    # Effectuer un LEFT MERGE : on garde toutes les lignes de df
    # et on ajoute les colonnes correspondantes de reco_stats
    df = df.merge(
        reco_stats,
        on=join_column,
        how="left",
    )
    if drop_recommendation_cols:
        df = df.drop(columns=["recommendation_mal_id"], errors="ignore")

    return df


def aggregate_staff_stats(
    df: pd.DataFrame,
    staff_stats_csv: Path = PROCESSED / "staff_features.csv",
    y_score_col: str = "y_score",
    y_members_col: str = "y_members",
    staff_col: str = "staff",
    drop_staff_col: bool = True
):

    staff_stats = pd.read_csv(staff_stats_csv)
    staff_stats["person_mal_id"] = pd.to_numeric(staff_stats["person_mal_id"], errors='coerce').astype("Int64")

    stat_cols = [c for c in staff_stats.columns if c != "person_mal_id"]
    staff_dict = staff_stats.set_index("person_mal_id").to_dict(orient="index")

    def parse_list(x):
        if isinstance(x, str):
            try:
                return list(map(int, ast.literal_eval(x)))
            except:
                return []
        if isinstance(x, list):
            return [int(v) for v in x if pd.notna(v)]
        return []

    df[staff_col] = df[staff_col].apply(parse_list)
    df[y_score_col]   = pd.to_numeric(df[y_score_col], errors="coerce")
    df[y_members_col] = pd.to_numeric(df[y_members_col], errors="coerce")

    def aggregate_row(lst, parent_score, parent_members):
        stats_list = [staff_dict.get(int(s)) for s in lst if int(s) in staff_dict]

        if not stats_list:
            return pd.Series({c: np.nan for c in stat_cols})

        out = {}
        for col in stat_cols:
            values = []
            for s in stats_list:
                v = s[col]
                if pd.isna(v):
                    continue

                divisor = s.get("count_anime_staff", 1) or 1

                if col == "mean_score_staff" and pd.notna(parent_score):
                    v -= parent_score / divisor
                if col == "mean_members_staff" and pd.notna(parent_members):
                    v -= parent_members / divisor

                values.append(v)

            out[col] = np.mean(values) if values else np.nan

        return pd.Series(out)

    df_agg = df.apply(lambda r: aggregate_row(r[staff_col], r[y_score_col], r[y_members_col]), axis=1)

    df = pd.concat([df, df_agg], axis=1)

    if drop_staff_col:
        df = df.drop(columns=[staff_col, "count_anime_staff"], errors="ignore")

    return df
if __name__ == "__main__":            
    # Exemple d'utilisation
    df = pd.read_csv(PROCESSED / "anime_details_encoded.csv")
    df_reco = aggregate_recommendation_stats(df)
    df_final = aggregate_staff_stats(df_reco)
    df_final.to_csv(PROCESSED / "anime_details_aggregated.csv", index=False)
    print(df_final.head())