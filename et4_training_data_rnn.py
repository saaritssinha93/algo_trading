#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 16:05:10 2025

This script trains RNN (LSTM) models using historical CSV files
from the main_indicators_test folder. For each ticker, four separate models are created,
one for each strategy:
  - Bullish Breakout
  - Bullish Pullback
  - Bearish Breakdown
  - Bearish Pullback
  
Each trained model is saved as "rnn_model_{ticker}_{strategy}.h5" in the MODEL_FOLDER.
If a model already exists for a ticker and strategy, it is loaded instead of retrained.

It uses:
  1) LSTM-based RNN architecture
  2) A sequence length (SEQ_LEN) for building training samples
  3) Condition-based labeling for each of the 4 strategies
  4) Data scaling for stable training
  5) Basic train/test split

@author: Saarit
"""

import os
import glob
import numpy as np
import pandas as pd
import pickle
from datetime import datetime
import pytz
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# -------------------------
# Configuration
# -------------------------
HISTORICAL_FOLDER = "main_indicators_test"  # Folder with training CSV files
MODEL_FOLDER = "rnn_model"                  # Directory to save/load the models
TIMEZONE = "Asia/Kolkata"

SEQ_LEN = 30     # how many consecutive rows form one sequence
FUTURE_SHIFT = 0 # label the last time step in each sequence
EPOCHS = 30      # number of training epochs (adjust as needed)
BATCH_SIZE = 32  # adjust as needed

if not os.path.exists(MODEL_FOLDER):
    os.makedirs(MODEL_FOLDER)

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
# 2) Feature Extraction (for each row)
# -------------------------
def extract_features_row(row):
    """
    Extracts features from a single row. We'll stack these row-features into a sequence.
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
# 3) Condition Checks for Each Strategy
# -------------------------
def check_bullish_breakout(daily_change, adx_val, price, high_10,
                           vol, macd_val, roll_vol,
                           macd_above, stoch_diff, slope_okay_for_bullish,
                           is_ich_bull, vwap):
    def valid_adx(adx):
        return 29 <= adx <= 45
    return ((2.0 <= daily_change <= 10.0) and
            valid_adx(adx_val) and
            (price >= 0.997 * high_10) and
            (macd_val >= 4.0) and
            (vol >= 3.0 * roll_vol) and
            macd_above and
            (stoch_diff > 0) and
            slope_okay_for_bullish and
            is_ich_bull and
            (price > vwap))

def check_bullish_pullback(daily_change, adx_val, macd_above,
                           stoch_diff, slope_okay_for_bullish,
                           pullback_pct, vwap, price):
    def valid_adx(adx):
        return 29 <= adx <= 45
    return ((daily_change >= -0.5) and
            valid_adx(adx_val) and
            macd_above and
            (stoch_diff > 0) and
            slope_okay_for_bullish and
            (pullback_pct >= 0.025) and
            (price > vwap))

def check_bearish_breakdown(daily_change, adx_val, price, low_10,
                            vol, roll_vol, macd_val,
                            macd_above, stoch_diff, slope_okay_for_bearish,
                            is_ich_bear, vwap):
    def valid_adx(adx):
        return 29 <= adx <= 45
    return ((-10.0 <= daily_change <= -2.0) and
            valid_adx(adx_val) and
            (price <= 1.003 * low_10) and
            (vol >= 3.0 * roll_vol) and
            (macd_val <= -4.0) and
            (not macd_above) and
            (stoch_diff < 0) and
            slope_okay_for_bearish and
            is_ich_bear and
            (price < vwap))

def check_bearish_pullback(daily_change, adx_val, macd_above,
                           stoch_diff, slope_okay_for_bearish,
                           bounce_pct, vwap, price):
    def valid_adx(adx):
        return 29 <= adx <= 45
    return ((daily_change <= 0.5) and
            valid_adx(adx_val) and
            (not macd_above) and
            (stoch_diff < 0) and
            slope_okay_for_bearish and
            (bounce_pct >= 0.025) and
            (price < vwap))

# -------------------------
# 3) Compute Additional Indicators
# -------------------------
def compute_additional_indicators(df):
    """
    Computes additional columns needed for strategy conditions:
      - rolling_high_10, rolling_low_10, rolling_vol_10.
      - daily_change (in percentage) if not present.
      - SMA20_Slope and StochDiff.
      - Rolling 5-bar high and low for pullback/bounce calculations.
      - Ichimoku indicators.
    """
    df = df.copy()
    # Ensure date is normalized and sorted
    df = normalize_time(df)
    df.sort_values('date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # Compute rolling metrics (using a window of 10 for breakout/breakdown)
    df['rolling_high_10'] = df['high'].rolling(window=10, min_periods=10).max()
    df['rolling_low_10']  = df['low'].rolling(window=10, min_periods=10).min()
    df['rolling_vol_10']  = df['volume'].rolling(window=10, min_periods=10).mean()
    
    # Compute daily change if not present (percentage change from open)
    if 'daily_change' not in df.columns:
        df['daily_change'] = ((df['close'] - df['open']) / df['open']) * 100
    
    # Compute SMA20_Slope if not present (difference of 20_SMA)
    if 'SMA20_Slope' not in df.columns and '20_SMA' in df.columns:
        df['SMA20_Slope'] = df['20_SMA'].diff()
    
    # Compute StochDiff (difference of Stochastic)
    if 'StochDiff' not in df.columns and 'Stochastic' in df.columns:
        df['StochDiff'] = df['Stochastic'].diff()
    
    # Rolling 5-bar high/low for pullback calculations
    df['rolling_high_5'] = df['high'].rolling(window=5, min_periods=5).max()
    df['rolling_low_5']  = df['low'].rolling(window=5, min_periods=5).min()
    # Pullback percentage: how far below the recent 5-bar high is the current price
    df['pullback_pct'] = (df['rolling_high_5'] - df['close']) / df['rolling_high_5']
    # Bounce percentage: how far above the recent 5-bar low is the current price
    df['bounce_pct'] = (df['close'] - df['rolling_low_5']) / df['rolling_low_5']
    
    # Compute Ichimoku indicators (simplified version)
    # Tenkan-sen: average of highest high and lowest low over 9 periods
    df['tenkan_sen'] = (df['high'].rolling(window=9, min_periods=9).max() +
                        df['low'].rolling(window=9, min_periods=9).min()) / 2
    # Kijun-sen: average of highest high and lowest low over 26 periods
    df['kijun_sen'] = (df['high'].rolling(window=26, min_periods=26).max() +
                       df['low'].rolling(window=26, min_periods=26).min()) / 2
    # Senkou Span A and B (shifted 26 periods ahead)
    df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)
    df['senkou_span_b'] = ((df['high'].rolling(window=52, min_periods=52).max() +
                             df['low'].rolling(window=52, min_periods=52).min()) / 2).shift(26)
    
    # Define functions to check Ichimoku conditions per row
    def is_ichimoku_bullish(row, tol=0.001):
        if pd.isna(row.get('senkou_span_a')) or pd.isna(row.get('senkou_span_b')):
            return False
        lower_span = min(row['senkou_span_a'], row['senkou_span_b'])
        return (row['close'] >= lower_span * (1 - tol) and row['tenkan_sen'] >= row['kijun_sen'] * (1 - tol))
    
    def is_ichimoku_bearish(row, tol=0.001):
        if pd.isna(row.get('senkou_span_a')) or pd.isna(row.get('senkou_span_b')):
            return False
        higher_span = max(row['senkou_span_a'], row['senkou_span_b'])
        return (row['close'] <= higher_span * (1 + tol) and row['tenkan_sen'] <= row['kijun_sen'] * (1 + tol))
    
    df['is_ich_bull'] = df.apply(is_ichimoku_bullish, axis=1)
    df['is_ich_bear'] = df.apply(is_ichimoku_bearish, axis=1)
    
    # Assume Signal_Line is already in df for MACD signal comparison
    if 'Signal_Line' in df.columns:
        df['macd_above'] = df['MACD'] > df['Signal_Line']
    else:
        df['macd_above'] = True

    return df

# -------------------------
# 5) Create RNN Training Data for a Strategy
# -------------------------
def create_rnn_training_data_strategy(file, strategy, seq_len=SEQ_LEN, future_shift=FUTURE_SHIFT):
    """
    Reads a CSV file, computes indicators, then forms sequences of length seq_len.
    We label each sequence as 1 if the *last row* of that sequence
    meets the condition for the given strategy, else 0.
    """
    df = pd.read_csv(file)
    df = compute_additional_indicators(df)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # We'll build a feature matrix for each row
    feature_matrix = df.apply(extract_features_row, axis=1).to_list()
    feature_matrix = np.array(feature_matrix)  # shape: (num_rows, num_features)

    X_sequences = []
    y_labels = []
    
    # Slide over the dataframe with a window of seq_len
    for i in range(len(df) - seq_len - future_shift + 1):
        # Extract sequence of length seq_len
        seq = feature_matrix[i:i+seq_len]
        # Label is determined by the row at [i + seq_len + future_shift - 1]
        label_row_idx = i + seq_len + future_shift - 1
        if label_row_idx >= len(df):
            break

        row_label = df.iloc[label_row_idx]  # the final row for labeling

        # We'll check the condition for that final row
        slope_bull = (row_label['SMA20_Slope'] > 0)
        slope_bear = (row_label['SMA20_Slope'] < 0)
        daily_change = row_label['daily_change']
        adx_val = row_label['ADX']
        price = row_label['close']
        high_10 = row_label['rolling_high_10']
        low_10  = row_label['rolling_low_10']
        vol = row_label['volume']
        roll_vol = row_label['rolling_vol_10']
        macd_val = row_label['MACD']
        macd_above = row_label['macd_above']
        stoch_diff = row_label['StochDiff']
        vwap = row_label['VWAP']
        is_ich_bull = row_label['is_ich_bull']
        is_ich_bear = row_label['is_ich_bear']
        pullback_pct = row_label['pullback_pct']
        bounce_pct   = row_label['bounce_pct']
        
        if strategy == 'bull_breakout':
            condition_met = check_bullish_breakout(daily_change, adx_val, price, high_10,
                                                   vol, macd_val, roll_vol, macd_above,
                                                   stoch_diff, slope_bull, is_ich_bull, vwap)
        elif strategy == 'bull_pullback':
            condition_met = check_bullish_pullback(daily_change, adx_val, macd_above,
                                                   stoch_diff, slope_bull, pullback_pct, vwap, price)
        elif strategy == 'bear_breakdown':
            condition_met = check_bearish_breakdown(daily_change, adx_val, price, low_10,
                                                    vol, roll_vol, macd_val, macd_above,
                                                    stoch_diff, slope_bear, is_ich_bear, vwap)
        elif strategy == 'bear_pullback':
            condition_met = check_bearish_pullback(daily_change, adx_val, macd_above,
                                                   stoch_diff, slope_bear, bounce_pct, vwap, price)
        else:
            raise ValueError("Unknown strategy specified.")
        
        label = 1 if condition_met else 0
        X_sequences.append(seq)
        y_labels.append(label)
    
    X_sequences = np.array(X_sequences)  # shape: (num_samples, seq_len, num_features)
    y_labels = np.array(y_labels)
    return X_sequences, y_labels

# -------------------------
# 6) Build & Train RNN Model for a Ticker & Strategy
# -------------------------
def build_rnn_model(input_shape):
    """
    Builds a simple LSTM-based RNN model for binary classification.
    Adjust architecture as needed.
    """
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(32))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def train_rnn_model_strategy(ticker, strategy):
    """
    For a given ticker and strategy:
      - Looks for {ticker}_main_indicators.csv in HISTORICAL_FOLDER
      - Builds sequences using create_rnn_training_data_strategy
      - Trains an LSTM-based RNN
      - Saves the model to rnn_model_{ticker}_{strategy}.h5
    """
    file_pattern = os.path.join(HISTORICAL_FOLDER, f"{ticker}_main_indicators.csv")
    files = glob.glob(file_pattern)
    if not files:
        print(f"No data found for {ticker}. Skipping {strategy}...")
        return None
    
    model_path = os.path.join(MODEL_FOLDER, f"rnn_model_{ticker}_{strategy}.h5")
    if os.path.exists(model_path):
        print(f"Model for {ticker} ({strategy}) already exists. Loading it instead of training.")
        return load_model(model_path)
    else:
        print(f"Training RNN model for {ticker} ({strategy})...")
        X, y = create_rnn_training_data_strategy(files[0], strategy, seq_len=SEQ_LEN, future_shift=FUTURE_SHIFT)
        if len(X) == 0:
            print(f"No training sequences for {ticker}, strategy={strategy}. Check data.")
            return None
        
        # We might need at least 2 classes => else we skip or augment
        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            print(f"Only one class present for {ticker} ({strategy}). We'll skip or fallback to conditions only.")
            return None
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Flatten sequences to scale features
        num_samples, seq_len, num_features = X_train.shape
        X_train_reshaped = X_train.reshape(num_samples * seq_len, num_features)
        X_test_reshaped  = X_test.reshape(X_test.shape[0] * X_test.shape[1], num_features)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_reshaped)
        X_test_scaled  = scaler.transform(X_test_reshaped)
        
        # Reshape back to sequence form
        X_train_scaled = X_train_scaled.reshape(num_samples, seq_len, num_features)
        X_test_scaled  = X_test_scaled.reshape(X_test.shape[0], seq_len, num_features)
        
        # Build model
        input_shape = (seq_len, num_features)
        model = build_rnn_model(input_shape=input_shape)
        
        # Fit model
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        history = model.fit(X_train_scaled, y_train,
                            validation_data=(X_test_scaled, y_test),
                            epochs=EPOCHS,
                            batch_size=BATCH_SIZE,
                            callbacks=[early_stop],
                            verbose=1)
        
        # Evaluate
        loss, acc = model.evaluate(X_test_scaled, y_test)
        print(f"[{ticker} {strategy}] => Test Accuracy: {acc:.2f}")
        
        # Save
        model.save(model_path)
        print(f"Saved model to {model_path}")
        return model

# -------------------------
# 7) Load RNN Model for a Ticker & Strategy
# -------------------------
def load_rnn_model_strategy(ticker, strategy):
    """
    Loads rnn_model_{ticker}_{strategy}.h5 from MODEL_FOLDER.
    If missing, triggers training.
    """
    import tensorflow as tf
    model_path = os.path.join(MODEL_FOLDER, f"rnn_model_{ticker}_{strategy}.h5")
    if not os.path.exists(model_path):
        print(f"Model for {ticker} ({strategy}) not found. Training new model.")
        return train_rnn_model_strategy(ticker, strategy)
    try:
        model = load_model(model_path)
        print(f"Model for {ticker} ({strategy}) loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model for {ticker} ({strategy}): {e}")
        return None

# -------------------------
# 8) Main
# -------------------------
def main():
    """
    1. Scans HISTORICAL_FOLDER for {ticker}_main_indicators.csv.
    2. For each ticker, trains or loads an RNN model for each strategy:
         - bull_breakout
         - bull_pullback
         - bear_breakdown
         - bear_pullback
       Each model is saved as rnn_model_{ticker}_{strategy}.h5 in MODEL_FOLDER.
    """
    if not os.path.exists(MODEL_FOLDER):
        os.makedirs(MODEL_FOLDER)

    all_files = os.listdir(HISTORICAL_FOLDER)
    tickers = set(os.path.splitext(f)[0].split('_')[0] for f in all_files if f.endswith(".csv"))
    strategies = ['bull_breakout', 'bull_pullback', 'bear_breakdown', 'bear_pullback']
    
    for ticker in tickers:
        for strategy in strategies:
            model = load_rnn_model_strategy(ticker, strategy)
            if model:
                print(f"RNN model for {ticker} ({strategy}) is ready.\n")
            else:
                print(f"No model (or no data) for {ticker} ({strategy}). Conditions-only fallback.\n")

# -------------------------
# Script Entry
# -------------------------
if __name__ == "__main__":
    main()
