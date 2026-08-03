#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
model_training_dual.py - Final Version with Dataframe Export
Trains Long & Short models.
Excludes OHLC, volume, fear_greed, timestamp.
Prints: Accuracy, AUC, Confusion Matrix, Feature Importance.
Saves first 1000 rows of training features+labels to 'training_sample.csv'.
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
SL = 0.0015
LOOKAHEAD = 4
SPLIT_RATIO = 0.8
MODEL_LONG = "xgboost_long.json"
MODEL_SHORT = "xgboost_short.json"

# ==========================================
# FEATURES (same for both)
# ==========================================
def hurst(series):
    if len(series) < 10:
        return np.nan
    max_lag = min(100, len(series)//2)
    if max_lag < 2:
        return np.nan
    lags = np.arange(2, max_lag)
    tau = []
    for lag in lags:
        pp = np.subtract(series[lag:], series[:-lag])
        tau.append(np.std(pp))
    if len(tau) < 2:
        return np.nan
    try:
        poly = np.polyfit(np.log(lags[:len(tau)]), np.log(tau), 1)
        return poly[0] * 2.0
    except:
        return np.nan

def engineer_features(df):
    df = df.copy()
    close = df['close'].astype(float)
    high = df['high'].astype(float)
    low = df['low'].astype(float)
    volume = df['volume'].astype(float)

    hurst_vals = close.rolling(100).apply(lambda x: hurst(x), raw=True)
    df['hurst_exp'] = hurst_vals.fillna(0.5)

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
# TARGETS (Long and Short)
# ==========================================
def create_targets(df):
    long_labels, short_labels = [], []
    for i in range(len(df) - LOOKAHEAD - 1):
        entry = df.iloc[i + 1]['open']
        long_tp = entry * (1 + TP)
        long_sl = entry * (1 - SL)
        short_tp = entry * (1 - TP)
        short_sl = entry * (1 + SL)

        # Long
        long_win = 0
        for j in range(1, LOOKAHEAD + 1):
            idx = i + j
            high = df.iloc[idx]['high']
            low = df.iloc[idx]['low']
            if low <= long_sl:
                break
            if high >= long_tp:
                long_win = 1
                break

        # Short
        short_win = 0
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
print("📊 Loading data...")
df_raw = pd.read_csv(CSV_FILE)
df_raw.columns = [col.lower() for col in df_raw.columns]
df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp'])
df_raw.set_index('timestamp', inplace=True)

split_idx = int(len(df_raw) * SPLIT_RATIO)
train_raw = df_raw.iloc[:split_idx].copy()
test_raw = df_raw.iloc[split_idx:].copy()

print(f"\n📅 TRAIN: {train_raw.index[0].date()} to {train_raw.index[-1].date()} ({len(train_raw)} candles)")
print(f"📅 TEST : {test_raw.index[0].date()} to {test_raw.index[-1].date()} ({len(test_raw)} candles)\n")

print("🔄 Creating targets...")
long_labels, short_labels = create_targets(train_raw)
print(f"🔍 Long labels : {pd.Series(long_labels).value_counts().to_dict()}")
print(f"🔍 Short labels: {pd.Series(short_labels).value_counts().to_dict()}")

train_labeled = train_raw.iloc[:len(long_labels)].copy()
train_labeled['long_label'] = long_labels
train_labeled['short_label'] = short_labels

print("🛠️ Engineering features...")
train_feat = engineer_features(train_labeled)
print(f"✅ Train rows: {len(train_feat)}")

# ==========================================
# PERMANENT EXCLUDE LIST
# ==========================================
exclude = ['open', 'high', 'low', 'close', 'long_label', 'short_label',
           'volume', 'fear_greed', 'timestamp']

X_train = train_feat.drop(columns=[c for c in exclude if c in train_feat.columns])
y_long = train_feat['long_label']
y_short = train_feat['short_label']

# ---- NEW: SAVE TRAINING DATAFRAME SAMPLE (first 1000 rows) ----
sample_df = train_feat.head(1000).copy()
sample_df.to_csv("training_sample.csv", index=True)  # index is timestamp
print("✅ Saved first 1000 rows of training data (features + labels) to 'training_sample.csv'")
print("   Columns: timestamp (index), features, long_label, short_label")

print(f"✅ Final feature columns: {list(X_train.columns)}")

# ---------- LONG ----------
if y_long.nunique() > 1:
    ratio_long = (y_long == 0).sum() / ((y_long == 1).sum() + 1e-9)
    print(f"\n🤖 Training LONG (scale_pos_weight={ratio_long:.2f})...")
    model_long = xgb.XGBClassifier(
        n_estimators=120, max_depth=4, learning_rate=0.03,
        subsample=0.7, colsample_bytree=0.7,
        reg_alpha=0.3, reg_lambda=0.5, min_child_weight=2,
        objective='binary:logistic', scale_pos_weight=ratio_long,
        random_state=42, eval_metric='logloss'
    )
    model_long.fit(X_train, y_long)
    model_long.save_model(MODEL_LONG)
    y_pred = model_long.predict(X_train)
    y_proba = model_long.predict_proba(X_train)[:, 1]
    
    print(f"✅ Long saved.")
    print(f"📊 Long Accuracy: {accuracy_score(y_long, y_pred)*100:.2f}%")
    print(f"📊 Long AUC: {roc_auc_score(y_long, y_proba):.4f}")
    print("📊 Confusion Matrix:")
    print(confusion_matrix(y_long, y_pred))
    print("📊 Classification Report:")
    print(classification_report(y_long, y_pred, target_names=['Loss', 'Win']))
    
    print("\n🔑 Top 10 Features (Long):")
    imp = model_long.feature_importances_
    for i, f in sorted(zip(imp, X_train.columns), reverse=True)[:10]:
        print(f"   {f}: {i:.4f}")

# ---------- SHORT ----------
if y_short.nunique() > 1:
    ratio_short = (y_short == 0).sum() / ((y_short == 1).sum() + 1e-9)
    print(f"\n🤖 Training SHORT (scale_pos_weight={ratio_short:.2f})...")
    model_short = xgb.XGBClassifier(
        n_estimators=120, max_depth=4, learning_rate=0.03,
        subsample=0.7, colsample_bytree=0.7,
        reg_alpha=0.3, reg_lambda=0.5, min_child_weight=2,
        objective='binary:logistic', scale_pos_weight=ratio_short,
        random_state=42, eval_metric='logloss'
    )
    model_short.fit(X_train, y_short)
    model_short.save_model(MODEL_SHORT)
    y_pred = model_short.predict(X_train)
    y_proba = model_short.predict_proba(X_train)[:, 1]
    
    print(f"✅ Short saved.")
    print(f"📊 Short Accuracy: {accuracy_score(y_short, y_pred)*100:.2f}%")
    print(f"📊 Short AUC: {roc_auc_score(y_short, y_proba):.4f}")
    print("📊 Confusion Matrix:")
    print(confusion_matrix(y_short, y_pred))
    print("📊 Classification Report:")
    print(classification_report(y_short, y_pred, target_names=['Loss', 'Win']))
    
    print("\n🔑 Top 10 Features (Short):")
    imp = model_short.feature_importances_
    for i, f in sorted(zip(imp, X_train.columns), reverse=True)[:10]:
        print(f"   {f}: {i:.4f}")

print("\n✅ Training done. Now run `backtest_dual.py`")
