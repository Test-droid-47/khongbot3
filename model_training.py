#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
model_training.py
- Chronological split (80% train, 20% test for backtest).
- Features computed only on train set.
- Trains Long & Short models.
- Prints train metrics + feature importance + split dates.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# CONFIG
# ==========================================
CSV_FILE = "ohlcv.csv"
TP = 0.003
SL = 0.0015
LOOKAHEAD = 4
SPLIT_RATIO = 0.8
MODEL_LONG = "xgboost_long.json"
MODEL_SHORT = "xgboost_short.json"

# ==========================================
# FEATURE ENGINEERING (same)
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
# TARGET CREATION
# ==========================================
def create_targets(df):
    long_labels, short_labels = [], []
    for i in range(len(df) - LOOKAHEAD):
        entry = df.iloc[i]['open']
        long_tp = entry * (1 + TP)
        long_sl = entry * (1 - SL)
        short_tp = entry * (1 - TP)
        short_sl = entry * (1 + SL)

        long_win, short_win = 0, 0
        for j in range(1, LOOKAHEAD + 1):
            high = df.iloc[i + j]['high']
            low = df.iloc[i + j]['low']

            if long_win == 0 and high >= long_tp:
                if df.iloc[i+1:i+j]['low'].min() > long_sl:
                    long_win = 1
            if short_win == 0 and low <= short_tp:
                if df.iloc[i+1:i+j]['high'].max() < short_sl:
                    short_win = 1
        long_labels.append(long_win)
        short_labels.append(short_win)
    return long_labels, short_labels

# ==========================================
# 1. LOAD & SPLIT
# ==========================================
print("📊 Loading data...")
df_raw = pd.read_csv(CSV_FILE)
df_raw.columns = [col.lower() for col in df_raw.columns]
df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp'])
df_raw.set_index('timestamp', inplace=True)

total_rows = len(df_raw)
split_idx = int(total_rows * SPLIT_RATIO)

train_raw = df_raw.iloc[:split_idx].copy()
test_raw = df_raw.iloc[split_idx:].copy()

print(f"\n📅 TRAIN period : {train_raw.index[0].date()} to {train_raw.index[-1].date()} ({len(train_raw)} candles)")
print(f"📅 TEST period  : {test_raw.index[0].date()} to {test_raw.index[-1].date()} ({len(test_raw)} candles)\n")

# ==========================================
# 2. TARGET & FEATURES ON TRAIN
# ==========================================
print("🔄 Creating targets on train data...")
long_labels, short_labels = create_targets(train_raw)
train_labeled = train_raw.iloc[:len(long_labels)].copy()
train_labeled['long_label'] = long_labels
train_labeled['short_label'] = short_labels

print("🛠️ Engineering features on train set...")
train_feat = engineer_features(train_labeled)
print(f"✅ Train rows after features: {len(train_feat)}")
print(f"📈 Long train dist: {train_feat['long_label'].value_counts().to_dict()}")
print(f"📈 Short train dist: {train_feat['short_label'].value_counts().to_dict()}\n")

# ==========================================
# 3. TRAIN MODELS
# ==========================================
exclude = ['open', 'high', 'low', 'close', 'long_label', 'short_label', 'fear_greed', 'volume']
X_train = train_feat.drop(columns=[c for c in exclude if c in train_feat.columns])
y_long = train_feat['long_label']
y_short = train_feat['short_label']

print("🤖 Training LONG model...")
ratio_long = (y_long == 0).sum() / ((y_long == 1).sum() + 1e-9)
model_long = xgb.XGBClassifier(
    n_estimators=100, max_depth=4, learning_rate=0.03,
    subsample=0.6, colsample_bytree=0.6,
    objective='binary:logistic',
    scale_pos_weight=ratio_long,
    random_state=42, eval_metric='logloss'
)
model_long.fit(X_train, y_long)
model_long.save_model(MODEL_LONG)

# --- Metrics on train ---
y_pred_long = model_long.predict(X_train)
acc_long = accuracy_score(y_long, y_pred_long)
print(f"✅ Long model saved (scale_pos_weight={ratio_long:.2f})")
print(f"📊 Long Train Accuracy : {acc_long*100:.2f}%")
print("Classification Report (Long):\n", classification_report(y_long, y_pred_long, target_names=['Loss', 'Win']))

print("\n🤖 Training SHORT model...")
ratio_short = (y_short == 0).sum() / ((y_short == 1).sum() + 1e-9)
model_short = xgb.XGBClassifier(
    n_estimators=100, max_depth=4, learning_rate=0.03,
    subsample=0.6, colsample_bytree=0.6,
    objective='binary:logistic',
    scale_pos_weight=ratio_short,
    random_state=42, eval_metric='logloss'
)
model_short.fit(X_train, y_short)
model_short.save_model(MODEL_SHORT)

y_pred_short = model_short.predict(X_train)
acc_short = accuracy_score(y_short, y_pred_short)
print(f"✅ Short model saved (scale_pos_weight={ratio_short:.2f})")
print(f"📊 Short Train Accuracy : {acc_short*100:.2f}%")
print("Classification Report (Short):\n", classification_report(y_short, y_pred_short, target_names=['Loss', 'Win']))

# ==========================================
# 4. FEATURE IMPORTANCE
# ==========================================
print("\n🔑 Top 5 features (Long):")
for i, f in enumerate(X_train.columns[:5]):
    print(f"   {i+1}. {f}: {model_long.feature_importances_[i]:.4f}")

print("\n🔑 Top 5 features (Short):")
for i, f in enumerate(X_train.columns[:5]):
    print(f"   {i+1}. {f}: {model_short.feature_importances_[i]:.4f}")

print("\n✅ Training complete! Models are ready for backtesting on unseen 20% data.")
