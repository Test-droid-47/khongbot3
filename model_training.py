#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
model_training_dual.py - BTC/USDT 1H with Advanced Features
- TP=0.3%, SL=0.3% (symmetrical)
- 7 new quant features (no lookahead)
- Trains Long & Short XGBoost
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# CONFIG
# ==========================================
CSV_FILE = "ohlcv.csv"
TP = 0.003
SL = 0.003
LOOKAHEAD = 12
SPLIT_RATIO = 0.8

MODEL_LONG = "xgboost_long_btc.json"
MODEL_SHORT = "xgboost_short_btc.json"

# ==========================================
# ADVANCED FEATURE ENGINEERING
# ==========================================
def engineer_features(df):
    """
    Adds all 7 new features to the DataFrame.
    Uses only past data (rolling, shifting, resampling).
    """
    df = df.copy()
    close = df['close'].astype(float)
    high = df['high'].astype(float)
    low = df['low'].astype(float)
    volume = df['volume'].astype(float)
    
    # ---- 1. ATR & NATR% ----
    tr = np.maximum(high - low,
                    np.maximum(abs(high - close.shift(1)),
                               abs(low - close.shift(1))))
    atr = tr.rolling(14).mean()
    df['natr_percent'] = (atr / close) * 100   # Feature 2

    # ---- 2. Candle Structure (existing) ----
    df['range'] = high - low + 1e-9
    df['close_position'] = (close - low) / df['range']
    df['avg_range_20'] = df['range'].rolling(20).mean()
    df['range_ratio'] = df['range'] / (df['avg_range_20'] + 1e-9)

    # ---- 3. Wick‑to‑Body Multiplier (Feature 3) ----
    body = abs(close - df['open'])
    upper_wick = high - df[['close', 'open']].max(axis=1)
    lower_wick = df[['close', 'open']].min(axis=1) - low
    df['wick_body_ratio'] = (upper_wick - lower_wick) / (body + 1e-5)

    # ---- 4. Session Time (Feature 4) ----
    if isinstance(df.index, pd.DatetimeIndex):
        hour = df.index.hour
        df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * hour / 24)

    # ---- 5. Rolling Momentum (Feature 5) ----
    for lag in [1, 2, 3]:
        df[f'mom_{lag}'] = np.log(close / close.shift(lag))

    # ---- 6. Efficiency Ratio (Feature 6) ----
    # Kaufman Efficiency Ratio over 5 candles
    net_move = abs(close - close.shift(5))
    path = (abs(close - close.shift(1)) +
            abs(close.shift(1) - close.shift(2)) +
            abs(close.shift(2) - close.shift(3)) +
            abs(close.shift(3) - close.shift(4)) +
            abs(close.shift(4) - close.shift(5)))
    df['efficiency_ratio'] = net_move / (path + 1e-9)

    # ---- 7. Daily Pivot Distance (Feature 7) ----
    # Compute yesterday's pivot: (High + Low + Close) / 3
    # Resample to daily, get previous day's pivot
    daily = df.resample('D').agg({'high': 'max', 'low': 'min', 'close': 'last'})
    daily['pivot'] = (daily['high'] + daily['low'] + daily['close']) / 3
    daily['pivot_shift'] = daily['pivot'].shift(1)   # yesterday's pivot
    # Merge back to hourly
    df['daily_pivot'] = daily['pivot_shift'].reindex(df.index, method='ffill')
    df['pivot_distance'] = (close - df['daily_pivot']) / df['daily_pivot']  # percentage

    # ---- 8. Relative Volume Z‑Score (Feature 1) ----
    # Compute z‑score of volume against same hour over last 20 days
    if isinstance(df.index, pd.DatetimeIndex):
        df['hour'] = df.index.hour
        # For each hour, rolling mean/std of volume over last 20 occurrences (shifted)
        df['vol_mean_same_hour'] = (
            df.groupby('hour')['volume']
            .transform(lambda x: x.rolling(20, min_periods=1).mean().shift(1))
        )
        df['vol_std_same_hour'] = (
            df.groupby('hour')['volume']
            .transform(lambda x: x.rolling(20, min_periods=1).std(ddof=0).shift(1))
        )
        df['vol_zscore'] = (volume - df['vol_mean_same_hour']) / (df['vol_std_same_hour'] + 1e-9)

    # Drop intermediate columns (keep only final features)
    drop_cols = ['range', 'avg_range_20', 'upper_wick', 'lower_wick', 'hour', 
                 'vol_mean_same_hour', 'vol_std_same_hour', 'daily_pivot']
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True, errors='ignore')

    df = df.dropna()
    return df

# ==========================================
# TARGET CREATION
# ==========================================
def create_targets(df):
    long_labels, short_labels = [], []
    for i in range(len(df) - LOOKAHEAD - 1):
        entry = df.iloc[i + 1]['open']
        long_tp = entry * (1 + TP)
        long_sl = entry * (1 - SL)
        short_tp = entry * (1 - TP)
        short_sl = entry * (1 + SL)

        long_win, short_win = 0, 0

        # Long: check SL first
        for j in range(1, LOOKAHEAD + 1):
            idx = i + j
            high = df.iloc[idx]['high']
            low = df.iloc[idx]['low']
            if low <= long_sl:
                break
            if high >= long_tp:
                long_win = 1
                break

        # Short: check SL first
        for j in range(1, LOOKAHEAD + 1):
            idx = i + j
            high = df.iloc[idx]['high']
            low = df.iloc[idx]['low']
            if high >= short_sl:
                break
            if low <= short_tp:
                short_win = 1
                break

        long_labels.append(long_win)
        short_labels.append(short_win)

    return long_labels, short_labels

# ==========================================
# MAIN
# ==========================================
print("📊 Loading BTC/USDT 1h data...")
df_raw = pd.read_csv(CSV_FILE)
df_raw.columns = [col.lower() for col in df_raw.columns]

if 'timestamp' in df_raw.columns:
    df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp'])
    df_raw.set_index('timestamp', inplace=True)

print(f"✅ Total 1h candles: {len(df_raw)}")
print(f"📅 Period: {df_raw.index.min().date()} to {df_raw.index.max().date()}")

total_rows = len(df_raw)
split_idx = int(total_rows * SPLIT_RATIO)

train_raw = df_raw.iloc[:split_idx].copy()
test_raw = df_raw.iloc[split_idx:].copy()

print(f"\n📅 TRAIN: {train_raw.index.min().date()} to {train_raw.index.max().date()} ({len(train_raw)} candles)")
print(f"📅 TEST : {test_raw.index.min().date()} to {test_raw.index.max().date()} ({len(test_raw)} candles)\n")

print("🔄 Creating targets on train data...")
long_labels, short_labels = create_targets(train_raw)
print(f"🔍 Long labels : {pd.Series(long_labels).value_counts().to_dict()}")
print(f"🔍 Short labels: {pd.Series(short_labels).value_counts().to_dict()}")

train_labeled = train_raw.iloc[:len(long_labels)].copy()
train_labeled['long_label'] = long_labels
train_labeled['short_label'] = short_labels

print("🛠️ Engineering features on train set...")
train_feat = engineer_features(train_labeled)
print(f"✅ Train rows after features: {len(train_feat)}")

exclude = ['open', 'high', 'low', 'close', 'volume',
           'long_label', 'short_label']
X_train = train_feat.drop(columns=[c for c in exclude if c in train_feat.columns])
y_long = train_feat['long_label']
y_short = train_feat['short_label']

print(f"✅ Final feature columns: {list(X_train.columns)}\n")

# ---------- TRAIN LONG ----------
if y_long.nunique() > 1:
    ratio_long = (y_long == 0).sum() / ((y_long == 1).sum() + 1e-9)
    print(f"🤖 Training LONG (scale_pos_weight={ratio_long:.2f})...")
    model_long = xgb.XGBClassifier(
        n_estimators=150, max_depth=5, learning_rate=0.03,
        subsample=0.7, colsample_bytree=0.7,
        reg_alpha=0.5, reg_lambda=1.0, min_child_weight=3,
        objective='binary:logistic', scale_pos_weight=ratio_long,
        random_state=42, eval_metric='logloss'
    )
    model_long.fit(X_train, y_long)
    model_long.save_model(MODEL_LONG)
    y_pred = model_long.predict(X_train)
    y_proba = model_long.predict_proba(X_train)[:, 1]
    print(f"✅ Long saved. Accuracy: {accuracy_score(y_long, y_pred)*100:.2f}%")
    print(f"📊 AUC: {roc_auc_score(y_long, y_proba):.4f}")
    print("📊 Confusion Matrix:")
    print(confusion_matrix(y_long, y_pred))
    print(classification_report(y_long, y_pred, target_names=['Loss', 'Win']))
    print("\n🔑 Top 10 Features (Long):")
    for i, f in sorted(zip(model_long.feature_importances_, X_train.columns), reverse=True)[:10]:
        print(f"   {f}: {i:.4f}")
else:
    print("⚠️ Long target has one class. Skipping.")

print("\n" + "-"*50)

# ---------- TRAIN SHORT ----------
if y_short.nunique() > 1:
    ratio_short = (y_short == 0).sum() / ((y_short == 1).sum() + 1e-9)
    print(f"🤖 Training SHORT (scale_pos_weight={ratio_short:.2f})...")
    model_short = xgb.XGBClassifier(
        n_estimators=150, max_depth=5, learning_rate=0.03,
        subsample=0.7, colsample_bytree=0.7,
        reg_alpha=0.5, reg_lambda=1.0, min_child_weight=3,
        objective='binary:logistic', scale_pos_weight=ratio_short,
        random_state=42, eval_metric='logloss'
    )
    model_short.fit(X_train, y_short)
    model_short.save_model(MODEL_SHORT)
    y_pred = model_short.predict(X_train)
    y_proba = model_short.predict_proba(X_train)[:, 1]
    print(f"✅ Short saved. Accuracy: {accuracy_score(y_short, y_pred)*100:.2f}%")
    print(f"📊 AUC: {roc_auc_score(y_short, y_proba):.4f}")
    print("📊 Confusion Matrix:")
    print(confusion_matrix(y_short, y_pred))
    print(classification_report(y_short, y_pred, target_names=['Loss', 'Win']))
    print("\n🔑 Top 10 Features (Short):")
    for i, f in sorted(zip(model_short.feature_importances_, X_train.columns), reverse=True)[:10]:
        print(f"   {f}: {i:.4f}")
else:
    print("⚠️ Short target has one class. Skipping.")

print("\n✅ Training complete! Run `backtest_dual.py`")
