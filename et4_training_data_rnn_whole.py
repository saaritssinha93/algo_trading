#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 16:05:10 2025

This script trains RNN models (using an LSTM layer) using historical CSV files
from the main_indicators_test folder. Each trained model is saved as "rnn_model_{ticker}.h5".
If a model already exists for a ticker, it is loaded instead of being retrained.

Key differences from the logistic regression version:
  1) Data sequences are created using a sliding window (default length = 10).
  2) Features are scaled with StandardScaler (applied across all timesteps).
  3) A simple LSTM-based RNN is used for binary classification.
  4) Future returns are computed and used to generate binary labels as before.
"""

import os
import glob
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import pytz

# Scikit-learn imports for splitting and scaling
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# TensorFlow/Keras imports for building the RNN model
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# -------------------------
# Configuration
# -------------------------
HISTORICAL_FOLDER = "main_indicators_test"  # Folder with training CSV files
MODEL_FOLDER = "rnn_model"                   # Directory to save/load the models
TIMEZONE = "Asia/Kolkata"
SEQ_LENGTH = 10  # Number of timesteps in each sequence
EPOCHS = 10      # Number of training epochs (adjust as needed)
BATCH_SIZE = 32  # Batch size for training

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
# 3) Create Training Data Sequences
# -------------------------
def create_training_data(file, seq_length=SEQ_LENGTH):
    """
    Reads a single CSV file, normalizes time, computes future_return,
    sets a threshold (0.02 => 2%), and extracts sequences of features/labels.
    Each sequence is of length `seq_length` and the label is the classification label 
    for the last row in the sequence.
    """
    df = pd.read_csv(file)
    df = normalize_time(df, tz=TIMEZONE)

    # Calculate future return using next 5 rows
    df['future_return'] = df['close'].shift(-5) / df['close'] - 1
    df.dropna(inplace=True)

    # Label: 1 if future_return > 2%, else 0
    df['label'] = (df['future_return'] > 0.02).astype(int)

    # Extract feature vectors for each row
    feature_list = df.apply(extract_features, axis=1).tolist()
    labels = df['label'].tolist()

    # Create sequences using a sliding window approach
    sequences = []
    seq_labels = []
    for i in range(seq_length, len(feature_list)):
        sequences.append(feature_list[i-seq_length:i])
        seq_labels.append(labels[i])
    return np.array(sequences), np.array(seq_labels)

# -------------------------
# 4) Train & Save RNN Model for a Ticker
# -------------------------
def train_rnn_model(ticker, seq_length=SEQ_LENGTH, epochs=EPOCHS, batch_size=BATCH_SIZE):
    """
    For a given ticker:
      - Searches for {ticker}_main_indicators.csv in HISTORICAL_FOLDER.
      - Reads data, creates sequences, scales features, and trains an LSTM-based RNN.
      - Saves the trained model to rnn_model_{ticker}.h5.
    """
    file_pattern = os.path.join(HISTORICAL_FOLDER, f"{ticker}_main_indicators.csv")
    files = glob.glob(file_pattern)
    if not files:
        print(f"No data found for {ticker}. Skipping.")
        return None

    model_path = os.path.join(MODEL_FOLDER, f"rnn_model_{ticker}.h5")
    if os.path.exists(model_path):
        # Model already trained; load and return it.
        print(f"Model for {ticker} already exists. Loading...")
        return load_model(model_path)
    else:
        print(f"Training RNN model for {ticker}...")
        X, y = create_training_data(files[0], seq_length=seq_length)
        if X.shape[0] == 0:
            raise ValueError(f"No training data found for {ticker}. Check data files.")

        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Scale the data.
        # Reshape from (samples, seq_length, num_features) to (-1, num_features) for scaling,
        # then reshape back to original dimensions.
        num_features = X_train.shape[2]
        scaler = StandardScaler()
        X_train_reshaped = X_train.reshape(-1, num_features)
        X_train_scaled = scaler.fit_transform(X_train_reshaped).reshape(X_train.shape)
        X_test_reshaped = X_test.reshape(-1, num_features)
        X_test_scaled = scaler.transform(X_test_reshaped).reshape(X_test.shape)

        # Build the LSTM-based RNN model
        model = Sequential()
        model.add(LSTM(50, input_shape=(seq_length, num_features), return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X_train_scaled, y_train, epochs=epochs, batch_size=batch_size,
                  validation_data=(X_test_scaled, y_test), verbose=1)

        # Evaluate the model
        loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
        print(f"[{ticker}] Test Accuracy: {accuracy:.2f}")

        # Save the model
        if not os.path.exists(MODEL_FOLDER):
            os.makedirs(MODEL_FOLDER)
        model.save(model_path)

        # Optionally, save the scaler for later inference
        scaler_path = os.path.join(MODEL_FOLDER, f"scaler_{ticker}.pkl")
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)

        return model

# -------------------------
# 5) Load RNN Model for a Ticker
# -------------------------
def load_rnn_model(ticker):
    """
    Loads rnn_model_{ticker}.h5 from MODEL_FOLDER. 
    If the model does not exist, it triggers training.
    """
    model_path = os.path.join(MODEL_FOLDER, f"rnn_model_{ticker}.h5")
    if not os.path.exists(model_path):
        print(f"Model for {ticker} not found. Training model...")
        return train_rnn_model(ticker)
    try:
        model = load_model(model_path)
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
    1. Scans the HISTORICAL_FOLDER for files named {ticker}_main_indicators.csv.
    2. For each ticker, loads or trains an RNN model and saves it to MODEL_FOLDER.
    """
    if not os.path.exists(MODEL_FOLDER):
        os.makedirs(MODEL_FOLDER)

    # Identify tickers by scanning the folder (e.g., "AARTIIND_main_indicators.csv" gives ticker "AARTIIND")
    all_files = os.listdir(HISTORICAL_FOLDER)
    tickers = set(os.path.splitext(f)[0].split('_')[0] for f in all_files if f.endswith(".csv"))

    for ticker in tickers:
        try:
            model = load_rnn_model(ticker)
            if model:
                print(f"RNN model for {ticker} is ready for use.\n")
        except ValueError as e:
            print(f"{e} Skipping {ticker}...\n")
            continue

# -------------------------
# Script Entry
# -------------------------
if __name__ == "__main__":
    main()
