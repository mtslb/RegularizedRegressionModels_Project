# src/utils/config.py

class Config:
    RANDOM_STATE = 42
    TEST_SIZE = 0.2

    # Hyperparamètres modèles
    RIDGE_ALPHA = 1.0
    LASSO_ALPHA = 0.1
    ELASTICNET_ALPHA = 0.5

    # Chemins relatifs (à compléter selon ton projet)
    DATA_RAW = "data/raw/"
    DATA_PROCESSED = "data/processed/"
    MODEL_DIR = "models/"
    REPORTS_DIR = "reports/"
