#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
model_training.py - Robust version with debug prints and error handling
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
# FEATURE ENGINEERING (New Set)
# ==========================================
def engineer_features(df):
    df = df.copy()
    close = df['close']; high = df['high']; low = df['low']; volume = df['volume']
    
    def hurst(series, lags=100):
        if len(series) < lags:
            return np.nan
        tau = []
        lagvec = np.arange(1, lags)
        for lag in lagvec:
            pp = np.subtract(series[lag:], series[:-lag])
            tau.append(np.std(pp))
        poly = np.polyfit(np.log(lagvec), np.log(tau), 1)
        return poly[0] * 2.0
    
    df['hurst_exp'] = close.rolling(100).apply(lambda x: hurst(x.values), raw=True)
    df['vol_aggression'] = volume * (high - low) / close
    vwap = (volume * close).rolling(50).sum() / volume.rolling(50).sum()
    df['vwap_ema_spread'] = vwap - close.ewm(span=20).mean()
    df['price_accel'] = close - 2 * close.shift(1) + close.shift(2)
    tr = np.maximum(high - low, 
                    np.maximum(abs(high - close.shift(1)),
                               abs(low - close.shift(1))))
    atr = tr.rolling(14).mean()
    df['natr'] = atr / close
    ret = close.pct_change()
    df['amihud_illiq'] = abs(ret) / (volume + 1e-9) * 1e9
    high_24 = high.rolling(24).max()
    low_24 = low.rolling(24).min()
    df['stop_buy_dist'] = (high_24 - low_24) / close
    
    df = df.dropna()
    return df

# ==========================================
# TARGET CREATION (Corrected with debug)
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
        for j in range(1, LOOKAHEAD + 1):
            idx = i + j
            high = df.iloc[idx]['high']
            low = df.iloc[idx]['low']

            if long_win == 0:
                if low <= long_sl:
                    long_win = 0
                    break
                if high >= long_tp:
                    long_win = 1
                    break

            if short_win == 0:
                if high >= short_sl:
                    short_win = 0
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

print("🔄 Creating targets on train data...")
long_labels, short_labels = create_targets(train_raw)

# --- DEBUG: print target distribution before feature engineering ---
print(f"🔍 Long labels before FE: {pd.Series(long_labels).value_counts().to_dict()}")
print(f"🔍 Short labels before FE: {pd.Series(short_labels).value_counts().to_dict()}")

train_labeled = train_raw.iloc[:len(long_labels)].copy()
train_labeled['long_label'] = long_labels
train_labeled['short_label'] = short_labels

print("🛠️ Engineering features on train set...")
train_feat = engineer_features(train_labeled)
print(f"✅ Train rows after features: {len(train_feat)}")
print(f"📈 Long train dist: {train_feat['long_label'].value_counts().to_dict()}")
print(f"📈 Short train dist: {train_feat['short_label'].value_counts().to_dict()}\n")

exclude = ['open', 'high', 'low', 'close', 'long_label', 'short_label']
X_train = train_feat.drop(columns=[c for c in exclude if c in train_feat.columns])
y_long = train_feat['long_label']
y_short = train_feat['short_label']

# --- Handle case where one class is missing ---
if len(y_long.unique()) < 2:
    print("⚠️ Long training data has only one class! Model will not learn. Check data.")
if len(y_short.unique()) < 2:
    print("⚠️ Short training data has only one class! Model will not learn. Check data.")

print("🤖 Training LONG model...")
model_long = xgb.XGBClassifier(
    n_estimators=80, max_depth=3, learning_rate=0.02,
    subsample=0.6, colsample_bytree=0.6,
    reg_alpha=1.0, reg_lambda=2.0, min_child_weight=5,
    objective='binary:logistic', scale_pos_weight=1.0,
    random_state=42, eval_metric='logloss'
)
model_long.fit(X_train, y_long)
model_long.save_model(MODEL_LONG)

y_pred_long = model_long.predict(X_train)
print(f"✅ Long model saved")
print(f"📊 Long Train Accuracy : {accuracy_score(y_long, y_pred_long)*100:.2f}%")
# Only print classification report if both classes present
if len(y_long.unique()) == 2:
    print(classification_report(y_long, y_pred_long, target_names=['Loss', 'Win']))
else:
    print("⚠️ Skipping classification report for Long (only one class).")

print("\n🤖 Training SHORT model...")
model_short = xgb.XGBClassifier(
    n_estimators=80, max_depth=3, learning_rate=0.02,
    subsample=0.6, colsample_bytree=0.6,
    reg_alpha=1.0, reg_lambda=2.0, min_child_weight=5,
    objective='binary:logistic', scale_pos_weight=1.0,
    random_state=42, eval_metric='logloss'
)
model_short.fit(X_train, y_short)
model_short.save_model(MODEL_SHORT)

y_pred_short = model_short.predict(X_train)
print(f"✅ Short model saved")
print(f"📊 Short Train Accuracy : {accuracy_score(y_short, y_pred_short)*100:.2f}%")
if len(y_short.unique()) == 2:
    print(classification_report(y_short, y_pred_short, target_names=['Loss', 'Win']))
else:
    print("⚠️ Skipping classification report for Short (only one class).")

print("\n🔑 Top 5 features (Long):")
for i, f in enumerate(X_train.columns[:5]):
    print(f"   {i+1}. {f}: {model_long.feature_importances_[i]:.4f}")

print("\n🔑 Top 5 features (Short):")
for i, f in enumerate(X_train.columns[:5]):
    print(f"   {i+1}. {f}: {model_short.feature_importances_[i]:.4f}")

print("\n✅ Training complete! Now run backtesting.py")
