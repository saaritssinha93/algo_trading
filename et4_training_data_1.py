#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 16:05:10 2025

This script trains logistic regression models using historical CSV files
from the main_indicators_test folder. For each ticker, four separate models are created,
one for each strategy:
  - Bullish Breakout
  - Bullish Pullback
  - Bearish Breakdown
  - Bearish Pullback
  
Each trained model is saved as "lr_model_{ticker}_{strategy}.pkl" in the MODEL_FOLDER.
If a model already exists for a ticker and strategy, it is loaded instead of retrained.

The training uses:
  1) Data scaling with StandardScaler.
  2) An increased max_iter to avoid convergence warnings.
  3) A hyperparameter grid search over solver, C, and class_weight.
  4) Custom labeling of rows based on the strategy-specific conditions.
  
@author: Saarit
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
# 4a) Strategy Conditions Functions
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
# 5) Create Training Data for a Strategy
# -------------------------
def create_training_data_strategy(file, strategy):
    """
    Reads a CSV file, computes additional indicators,
    and then labels each row based on the specified strategy's conditions.
    
    strategy: one of 'bull_breakout', 'bull_pullback', 'bear_breakdown', 'bear_pullback'
    """
    df = pd.read_csv(file)
    df = compute_additional_indicators(df)
    df.dropna(inplace=True)
    
    data, labels = [], []
    
    for idx, row in df.iterrows():
        if idx < 30:
            continue
        
        slope_bull = row['SMA20_Slope'] > 0
        slope_bear = row['SMA20_Slope'] < 0
        
        daily_change = row['daily_change']
        adx_val = row['ADX']
        price = row['close']
        high_10 = row['rolling_high_10']
        low_10 = row['rolling_low_10']
        vol = row['volume']
        roll_vol = row['rolling_vol_10']
        macd_val = row['MACD']
        macd_above = row['macd_above']
        stoch_diff = row['StochDiff']
        vwap = row['VWAP']
        
        pullback_pct = row['pullback_pct']
        bounce_pct = row['bounce_pct']
        
        is_ich_bull = row['is_ich_bull']
        is_ich_bear = row['is_ich_bear']
        
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
        data.append(extract_features(row))
        labels.append(label)
    
    return np.array(data), np.array(labels)

# -------------------------
# 6) Train & Save LR Model for a Ticker & Strategy
# -------------------------
def train_logistic_regression_model_strategy(ticker, strategy):
    """
    For a given ticker and strategy:
      - Searches for {ticker}_main_indicators.csv in HISTORICAL_FOLDER.
      - Creates training data based on the strategy-specific conditions.
      - Augments data if only one class is present.
      - Scales features, performs grid search with cross-validation.
      - Saves the best model to MODEL_FOLDER as lr_model_{ticker}_{strategy}.pkl.
    """
    file_pattern = os.path.join(HISTORICAL_FOLDER, f"{ticker}_main_indicators.csv")
    files = glob.glob(file_pattern)
    if not files:
        print(f"No data found for {ticker}. Skipping.")
        return None

    model_path = os.path.join(MODEL_FOLDER, f"lr_model_{ticker}_{strategy}.pkl")
    if os.path.exists(model_path):
        print(f"Model for {ticker} ({strategy}) already exists. Loading...")
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        return model
    else:
        print(f"Training model for {ticker} ({strategy})...")
        X, y = create_training_data_strategy(files[0], strategy)
        if X.shape[0] == 0:
            raise ValueError(f"No training data found for {ticker} with strategy {strategy}.")
        
        # Check if at least two classes exist; if not, augment data with a synthetic sample.
        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            print(f"Only one class present for {ticker} ({strategy}). Augmenting training data.")
            dummy_feature = np.mean(X, axis=0, keepdims=True)
            dummy_label = 1 - unique_classes[0]
            X = np.concatenate([X, dummy_feature], axis=0)
            y = np.concatenate([y, [dummy_label]])
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        param_grid = {
            'solver': ['lbfgs', 'saga'],
            'C': [0.01, 0.1, 1, 10],
            'class_weight': [None, 'balanced']
        }
        base_lr = LogisticRegression(max_iter=5000)
        grid_search = GridSearchCV(base_lr, param_grid, cv=3, scoring='accuracy')
        grid_search.fit(X_train_scaled, y_train)
        best_model = grid_search.best_estimator_
        
        y_pred = best_model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        print(f"[{ticker} {strategy}] Best Params: {grid_search.best_params_}")
        print(f"[{ticker} {strategy}] Accuracy: {acc:.2f}")
        print(classification_report(y_test, y_pred))
        
        with open(model_path, "wb") as f:
            pickle.dump(best_model, f)
        
        return best_model

# -------------------------
# 7) Load LR Model for a Ticker & Strategy
# -------------------------
def load_lr_model_strategy(ticker, strategy):
    """
    Loads lr_model_{ticker}_{strategy}.pkl from MODEL_FOLDER.
    If missing, triggers training.
    """
    model_path = os.path.join(MODEL_FOLDER, f"lr_model_{ticker}_{strategy}.pkl")
    if not os.path.exists(model_path):
        print(f"Model for {ticker} ({strategy}) not found. Training model...")
        return train_logistic_regression_model_strategy(ticker, strategy)
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
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
    1. Scans HISTORICAL_FOLDER for files named {ticker}_main_indicators.csv.
    2. For each ticker, trains or loads separate logistic regression models for:
         - Bullish Breakout
         - Bullish Pullback
         - Bearish Breakdown
         - Bearish Pullback
       Each model is saved as lr_model_{ticker}_{strategy}.pkl in MODEL_FOLDER.
    """
    all_files = os.listdir(HISTORICAL_FOLDER)
    tickers = set(os.path.splitext(f)[0].split('_')[0] for f in all_files if f.endswith(".csv"))
    strategies = ['bull_breakout', 'bull_pullback', 'bear_breakdown', 'bear_pullback']
    
    for ticker in tickers:
        for strategy in strategies:
            try:
                model = load_lr_model_strategy(ticker, strategy)
                if model:
                    print(f"Model for {ticker} ({strategy}) is ready for use.\n")
            except ValueError as e:
                print(f"{e} Skipping {ticker} ({strategy})...\n")
                continue

# -------------------------
# Script Entry
# -------------------------
if __name__ == "__main__":
    main()
