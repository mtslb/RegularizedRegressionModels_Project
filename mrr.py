import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- Charger le dataset ---
df = pd.read_csv("anime_features_.csv")
df = df.fillna(0)
if "staff" in df.columns:
    df = df.drop(columns=["staff"])

targets = ["y_score", "y_members"]
X = df.drop(columns=targets + ["mal_id"])
y = df[targets]

# --- Split train/test ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- Standardiser les features ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Log-transform pour y_members ---
y_train_log = np.log1p(y_train["y_members"])
y_test_log = np.log1p(y_test["y_members"])

# --- Modèles ---
models_score = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.01, max_iter=50000),
    "ElasticNet": ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=50000)
}

models_members = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.1, max_iter=50000),
    "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=50000)
}

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# --- Fonction pour plotter les résultats ---
def plot_results(y_true, y_pred, title):
    plt.figure(figsize=(6,6))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.6)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel("Réel")
    plt.ylabel("Prédit")
    plt.title(title)
    plt.show()
    
    plt.figure(figsize=(6,4))
    sns.histplot(y_pred - y_true, bins=50, kde=True)
    plt.title(f"Distribution des erreurs : {title}")
    plt.xlabel("Erreur (y_pred - y_true)")
    plt.show()

# --- Evaluation y_score ---
print("\n===== Prediction de y_score =====")
for name, model in models_score.items():
    model.fit(X_train_scaled, y_train["y_score"])
    y_pred = model.predict(X_test_scaled)
    
    mae = mean_absolute_error(y_test["y_score"], y_pred)
    rms = rmse(y_test["y_score"], y_pred)
    r2 = r2_score(y_test["y_score"], y_pred)
    
    print(f"{name} | MAE: {mae:.4f} | RMSE: {rms:.4f} | R²: {r2:.4f}")
    plot_results(y_test["y_score"], y_pred, f"{name} - y_score")

# --- Evaluation y_members (log-transform) ---
print("\n===== Prediction de y_members (log-transformé) =====")
for name, model in models_members.items():
    model.fit(X_train_scaled, y_train_log)
    y_pred_log = model.predict(X_test_scaled)
    y_pred = np.expm1(y_pred_log)
    
    mae = mean_absolute_error(y_test["y_members"], y_pred)
    rms = rmse(y_test["y_members"], y_pred)
    r2 = r2_score(y_test["y_members"], y_pred)
    
    print(f"{name} | MAE: {mae:.4f} | RMSE: {rms:.4f} | R²: {r2:.4f}")
    plot_results(y_test["y_members"], y_pred, f"{name} - y_members (log)")

# --- Evaluation y_members sans log-transform (Linear + Ridge) ---
print("\n===== Prediction de y_members (sans log-transform) =====")
models_members_simple = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0)
}

for name, model in models_members_simple.items():
    model.fit(X_train_scaled, y_train["y_members"])
    y_pred = model.predict(X_test_scaled)
    
    mae = mean_absolute_error(y_test["y_members"], y_pred)
    rms = rmse(y_test["y_members"], y_pred)
    r2 = r2_score(y_test["y_members"], y_pred)
    
    print(f"{name} | MAE: {mae:.4f} | RMSE: {rms:.4f} | R²: {r2:.4f}")
    plot_results(y_test["y_members"], y_pred, f"{name} - y_members (sans log)")
