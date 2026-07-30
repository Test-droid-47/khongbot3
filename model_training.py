#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
model_training.py
Trains dual XGBoost models (Long & Short) on OHLCV dataset.
Saves xgboost_long.json and xgboost_short.json.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURATION
# ==========================================
CSV_FILE = "ohlcv.csv"
TP = 0.003         # 0.3%
SL = 0.0015        # 0.15%
LOOKAHEAD = 4
MODEL_LONG = "xgboost_long.json"
MODEL_SHORT = "xgboost_short.json"

# ==========================================
# FEATURE ENGINEERING (COMMON)
# ==========================================
def engineer_features(df):
    df = df.copy()
    df['range'] = df['high'] - df['low'] + 1e-9
    df['close_position'] = (df['close'] - df['low']) / df['range']
    df['avg_range_20'] = df['range'].rolling(20).mean()
    df['range_ratio'] = df['range'] / (df['avg_range_20'] + 1e-9)

    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    df['atr5'] = df['tr'].rolling(5).mean()
    df['atr20'] = df['tr'].rolling(20).mean()
    df['atr_expansion'] = df['atr5'] / (df['atr20'] + 1e-9)

    df['daily_high'] = df['high'].rolling(24).max()
    df['daily_low'] = df['low'].rolling(24).min()
    df['dist_to_daily_high'] = (df['daily_high'] - df['close']) / df['close']
    df['dist_to_daily_low'] = (df['close'] - df['daily_low']) / df['close']

    df['upper_wick'] = df['high'] - df[['close', 'open']].max(axis=1)
    df['lower_wick'] = df[['close', 'open']].min(axis=1) - df['low']
    df['wick_imbalance'] = (df['upper_wick'] - df['lower_wick']) / (df['range'] + 1e-9)
    df['body_ratio'] = abs(df['close'] - df['open']) / df['range']

    drop_cols = ['range', 'avg_range_20', 'tr', 'atr5', 'atr20',
                 'daily_high', 'daily_low', 'upper_wick', 'lower_wick']
    df = df.drop(columns=drop_cols)
    df = df.dropna()
    return df

# ==========================================
# 1. LOAD DATA & CREATE DUAL TARGETS
# ==========================================
print("📊 Loading data...")
df_raw = pd.read_csv(CSV_FILE)
df_raw.columns = [col.lower() for col in df_raw.columns]

print("🔄 Creating targets for Long and Short models...")
long_labels, short_labels = [], []
total_rows = len(df_raw)

for i in range(total_rows - LOOKAHEAD):
    entry = df_raw.loc[i, 'open']
    long_tp = entry * (1 + TP)
    long_sl = entry * (1 - SL)
    short_tp = entry * (1 - TP)
    short_sl = entry * (1 + SL)

    long_win, short_win = 0, 0

    for j in range(1, LOOKAHEAD + 1):
        high = df_raw.loc[i + j, 'high']
        low = df_raw.loc[i + j, 'low']

        if long_win == 0 and high >= long_tp:
            min_low = df_raw.loc[i+1:i+j, 'low'].min()
            if min_low > long_sl:
                long_win = 1

        if short_win == 0 and low <= short_tp:
            max_high = df_raw.loc[i+1:i+j, 'high'].max()
            if max_high < short_sl:
                short_win = 1

    long_labels.append(long_win)
    short_labels.append(short_win)

df_labeled = df_raw.iloc[:len(long_labels)].copy()
df_labeled['long_label'] = long_labels
df_labeled['short_label'] = short_labels

# ==========================================
# 2. ENGINEER FEATURES
# ==========================================
print("🛠️ Engineering features...")
df_feat = engineer_features(df_labeled)
print(f"✅ Total rows after feature eng: {len(df_feat)}")

exclude = ['open', 'high', 'low', 'close', 'long_label', 'short_label']
X = df_feat.drop(columns=[c for c in exclude if c in df_feat.columns])
y_long = df_feat['long_label']
y_short = df_feat['short_label']

print(f"📈 Long  distribution: {y_long.value_counts().to_dict()}")
print(f"📈 Short distribution: {y_short.value_counts().to_dict()}")

# ==========================================
# 3. TRAIN LONG MODEL
# ==========================================
print("\n🤖 Training LONG model...")
ratio_long = (y_long == 0).sum() / ((y_long == 1).sum() + 1e-9)
model_long = xgb.XGBClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    objective='binary:logistic',
    scale_pos_weight=ratio_long,
    random_state=42, eval_metric='logloss'
)
model_long.fit(X, y_long)
model_long.save_model(MODEL_LONG)
print(f"✅ Long model saved to {MODEL_LONG} (scale_pos_weight={ratio_long:.2f})")

# ==========================================
# 4. TRAIN SHORT MODEL
# ==========================================
print("\n🤖 Training SHORT model...")
ratio_short = (y_short == 0).sum() / ((y_short == 1).sum() + 1e-9)
model_short = xgb.XGBClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    objective='binary:logistic',
    scale_pos_weight=ratio_short,
    random_state=42, eval_metric='logloss'
)
model_short.fit(X, y_short)
model_short.save_model(MODEL_SHORT)
print(f"✅ Short model saved to {MODEL_SHORT} (scale_pos_weight={ratio_short:.2f})")

# ==========================================
# 5. FEATURE IMPORTANCE
# ==========================================
print("\n🔑 Top 5 features (Long):")
for i, f in enumerate(X.columns[:5]):
    print(f"   {i+1}. {f}: {model_long.feature_importances_[i]:.4f}")

print("\n🔑 Top 5 features (Short):")
for i, f in enumerate(X.columns[:5]):
    print(f"   {i+1}. {f}: {model_short.feature_importances_[i]:.4f}")

print("\n✅ Training complete! Models ready for backtesting and live trading.")