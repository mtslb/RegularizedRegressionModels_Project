# src/domain/data_processing.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(path):
    df = pd.read_csv(path)
    df = df.fillna(0)
    if "staff" in df.columns:
        df = df.drop(columns=["staff"])
    return df

def split_features_targets(df, targets=["y_score", "y_members"]):
    X = df.drop(columns=targets + ["mal_id"])
    y = df[targets]
    return X, y

def train_test_split_scaled(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
