import pandas as pd
import ast
from sklearn.preprocessing import MultiLabelBinarizer

# --- Lecture du dataset ---
df = pd.read_csv("anime_dataset.csv")

# --- One-hot encoding des colonnes simples ---
categorical_simple = [
    "type",
    #"status",
    "source",
    "rating",
    "season",
    "demographics"
]

df = pd.get_dummies(df, columns=categorical_simple, dtype=int)

# --- Fonction pour convertir les strings JSON en listes Python ---
def parse_list_column(col):
    return col.apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith("[") else [])

# --- Fonction multi-hot encoding ---
def multilabel_encode(df, column_name, prefix):
    mlb = MultiLabelBinarizer()
    parsed = parse_list_column(df[column_name])
    encoded = mlb.fit_transform(parsed)
    cols = [f"{prefix}_{cls}" for cls in mlb.classes_]
    return pd.DataFrame(encoded, columns=cols, index=df.index)

# --- Colonnes multi-label à encoder ---
multi_hot_cols = ["genres", "studios", "themes"]

for col in multi_hot_cols:
    mh = multilabel_encode(df, col, prefix=col)
    df = pd.concat([df, mh], axis=1)
    df.drop(columns=[col], inplace=True)

# --- Vérification finale ---
df.head()
df.info()
# --- Sauvegarde du dataset
df.to_csv("anime_features_ENCODE.csv", index=False)