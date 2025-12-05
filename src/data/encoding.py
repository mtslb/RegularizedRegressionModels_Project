# src/domain/encoding.py
import sys
from pathlib import Path
import pandas as pd
import ast
from sklearn.preprocessing import MultiLabelBinarizer

# Ajouter la racine du projet au path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.utils import CATEGORICAL_SIMPLE, MULTI_HOT_COLS

def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode les colonnes catégorielles et multi-label d'un dataset anime.

    Args:
        df (pd.DataFrame): dataset prétraité (output de data_processing.py)

    Returns:
        pd.DataFrame: dataset avec features encodées
    """

    # --- One-hot encoding des colonnes simples ---
    df = pd.get_dummies(df, columns=CATEGORICAL_SIMPLE, dtype=int)

    # --- Fonction pour convertir les strings JSON en listes Python ---
    def parse_list_column(col: pd.Series) -> pd.Series:
        return col.apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith("[") else [])

    # --- Fonction multi-hot encoding ---
    def multilabel_encode(df: pd.DataFrame, column_name: str, prefix: str) -> pd.DataFrame:
        parsed = parse_list_column(df[column_name])
        mlb = MultiLabelBinarizer()
        encoded = mlb.fit_transform(parsed)
        cols = [f"{prefix}_{cls}" for cls in mlb.classes_]
        return pd.DataFrame(encoded, columns=cols, index=df.index)

    # --- Colonnes multi-label à encoder ---
    for col in MULTI_HOT_COLS:
        mh = multilabel_encode(df, col, prefix=col)
        df = pd.concat([df, mh], axis=1)
        df.drop(columns=[col], inplace=True)

    return df



if __name__ == "__main__":     
    # Exemple d'utilisation
    df = pd.read_csv("anime_details_preprocessed.csv")
    df_encoded = encode_features(df)
    df_encoded.to_csv("anime_details_encoded.csv", index=False)
    print(df_encoded.head())
