# src/utils/utils.py

# Colonnes à conserver dans le dataset
COLUMNS_TO_KEEP = [
    "mal_id", "type", "status", "genres", "studios", "themes",
    "start_date", "demographics", "source", "rating", "episodes",
    "season", "score", "members"
]

# Colonnes pour one-hot encoding simple
CATEGORICAL_SIMPLE = [
    "type",
    "source",
    "rating",
    "season",
    "demographics"
]

# Colonnes multi-label à encoder
MULTI_HOT_COLS = [
    "genres",
    "studios",
    "themes"
]
