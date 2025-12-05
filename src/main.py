# src/main.py
import importlib
from pathlib import Path

# Liste de tous les fichiers modèles
model_files = [
    "src.domain.models.lasso_model",
    "src.domain.models.ridge_model",
    "src.domain.models.elasticnet_model",
    "src.domain.models.linear_model"
]

GRAPH_DIR = Path("graphs")
GRAPH_DIR.mkdir(exist_ok=True)

def run_all_models():
    for mod_path in model_files:
        print(f"\n===== Execution du modèle: {mod_path.split('.')[-1]} =====")
        mod = importlib.import_module(mod_path)
        if hasattr(mod, "run_model"):
            mod.run_model(n_splits=5)
        else:
            print(f"Aucune fonction run_model trouvée dans {mod_path}")

if __name__ == "__main__":
    run_all_models()
