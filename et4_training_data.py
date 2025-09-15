#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 16:05:10 2025

This script trains logistic regression models using historical CSV files
from the main_indicators_test folder. Each trained model is saved as "lr_model_{ticker}.pkl".
If a model already exists for a ticker, it is loaded instead of trained.

Changes to address ConvergenceWarnings and improve modeling:
  1) Data is scaled with StandardScaler.
  2) max_iter is increased.
  3) Solver is included in hyperparameter grid search.
  4) A stricter future_return threshold is used for labeling.
"""

import os
import glob
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import pytz

# scikit-learn and ML imports
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# -------------------------
# Configuration
# -------------------------
HISTORICAL_FOLDER = "main_indicators_test"  # Folder with training CSV files
MODEL_FOLDER = "lr_model"                   # Directory to save/load the models
TIMEZONE = "Asia/Kolkata"

# -------------------------
# 1) Utility: Time Normalization
# -------------------------
def normalize_time(df, tz=TIMEZONE):
    """Converts the 'date' column to timezone-aware datetime objects."""
    df = df.copy()
    if 'date' not in df.columns:
        raise KeyError("DataFrame missing 'date' column.")
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df.dropna(subset=['date'], inplace=True)
    if df['date'].dt.tz is None:
        df['date'] = df['date'].dt.tz_localize('UTC')
    df['date'] = df['date'].dt.tz_convert(tz)
    return df

# -------------------------
# 2) Feature Extraction
# -------------------------
def extract_features(row):
    """
    Given a row (pandas Series) of indicator values, extract a feature vector.
    Adjust or add features as per your strategy.
    """
    return [
        row.get("close", 0),
        row.get("volume", 0),
        row.get("VWAP", 0),
        row.get("ADX", 0),
        row.get("MACD", 0),
        row.get("Stochastic", 0),
        row.get("20_SMA", 0),
        row.get("SMA20_Slope", 0)
    ]

# -------------------------
# 3) Create Training Data
# -------------------------
def create_training_data(file):
    """
    Reads a single CSV file, normalizes time, computes future_return,
    sets a stricter threshold (0.02 => 2%), and extracts features/labels.
    """
    df = pd.read_csv(file)
    df = normalize_time(df, tz=TIMEZONE)

    # Calculate future return using next 5 rows
    df['future_return'] = df['close'].shift(-5) / df['close'] - 1
    df.dropna(inplace=True)

    # Stricter condition => future_return > 2%
    df['label'] = (df['future_return'] > 0.02).astype(int)

    data, labels = [], []
    for _, row in df.iterrows():
        data.append(extract_features(row))
        labels.append(row['label'])
    return np.array(data), np.array(labels)

# -------------------------
# 4) Train & Save Model for a Ticker
# -------------------------
def train_logistic_regression_model(ticker):
    """
    For a given ticker:
      - Searches for {ticker}_main_indicators.csv in HISTORICAL_FOLDER.
      - Reads data, scales features, runs a hyperparameter grid search with cross-validation.
      - Saves the best model to lr_model_{ticker}.pkl.
    """
    file_pattern = os.path.join(HISTORICAL_FOLDER, f"{ticker}_main_indicators.csv")
    files = glob.glob(file_pattern)
    if not files:
        print(f"No data found for {ticker}. Skipping.")
        return None

    model_path = os.path.join(MODEL_FOLDER, f"lr_model_{ticker}.pkl")
    if os.path.exists(model_path):
        # Already trained => load and return
        print(f"Model for {ticker} already exists. Loading...")
        with open(model_path, "rb") as f:
            lr_model = pickle.load(f)
        return lr_model
    else:
        # Need to train a new model
        print(f"Training model for {ticker}...")
        X, y = create_training_data(files[0])  # Assume single file
        if X.shape[0] == 0:
            raise ValueError(f"No training data found for {ticker}. Check data files.")

        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Scale data => helps solver converge & improves numeric stability
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Hyperparameter grid
        param_grid = {
            'solver':       ['lbfgs', 'saga'],  # include alternative solver
            'C':            [0.01, 0.1, 1, 10],
            'class_weight': [None, 'balanced'],
            # We'll fix max_iter to a higher value to avoid ConvergenceWarnings
        }
        base_lr = LogisticRegression(
            max_iter=5000  # increased from default 100 => helps lbfgs converge
        )
        # 3-fold cross-validation
        grid_search = GridSearchCV(base_lr, param_grid, cv=3, scoring='accuracy')
        grid_search.fit(X_train_scaled, y_train)

        best_model = grid_search.best_estimator_

        # Evaluate best model
        y_pred = best_model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        print(f"[{ticker}] Best Params: {grid_search.best_params_}")
        print(f"[{ticker}] Accuracy: {acc:.2f}")
        print(classification_report(y_test, y_pred))

        # Save the best model + the scaler (optional if needed at inference)
        with open(model_path, "wb") as f:
            pickle.dump(best_model, f)

        return best_model

# -------------------------
# 5) Load Model for a Ticker
# -------------------------
def load_lr_model(ticker):
    """
    Loads lr_model_{ticker}.pkl from MODEL_FOLDER. 
    If missing, triggers training.
    """
    model_path = os.path.join(MODEL_FOLDER, f"lr_model_{ticker}.pkl")
    if not os.path.exists(model_path):
        print(f"Model for {ticker} not found. Training model...")
        return train_logistic_regression_model(ticker)
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        print(f"Model for {ticker} loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model for {ticker}: {e}")
        return None

# -------------------------
# 6) Main
# -------------------------
def main():
    """
    1. Looks in HISTORICAL_FOLDER for files named {ticker}_main_indicators.csv.
    2. For each ticker, tries to load or train a logistic regression model 
       and saves it to {MODEL_FOLDER}/lr_model_{ticker}.pkl.
    """
    if not os.path.exists(MODEL_FOLDER):
        os.makedirs(MODEL_FOLDER)

    # Identify tickers by scanning the folder
    all_files = os.listdir(HISTORICAL_FOLDER)
    # e.g., "AARTIIND_main_indicators.csv" => ticker = "AARTIIND"
    tickers = set(os.path.splitext(f)[0].split('_')[0] for f in all_files if f.endswith(".csv"))

    for ticker in tickers:
        try:
            model = load_lr_model(ticker)
            if model:
                print(f"Model for {ticker} is ready for use.\n")
        except ValueError as e:
            # If training data isn't found, log/warn and move on
            print(f"{e} Skipping {ticker}...\n")
            continue

# -------------------------
# Script Entry
# -------------------------
if __name__ == "__main__":
    main()
