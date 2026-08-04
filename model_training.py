#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
model_training_dual.py - Direction Based Target
- Long win: +0.3% touches BEFORE -0.3%
- Short win: -0.3% touches BEFORE +0.3%
- No SL involved in target creation (SL only for execution)
- Preprocessing: drops constant features, log1p amihud, clips vol_aggression
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
LOOKAHEAD = 4
SPLIT_RATIO = 0.8
MODEL_LONG = "xgboost_long.json"
MODEL_SHORT = "xgboost_short.json"

# ==========================================
# FEATURE ENGINEERING
# ==========================================
def engineer_features(df):
    df = df.copy()
    close = df['close'].astype(float)
    high = df['high'].astype(float)
    low = df['low'].astype(float)
    volume = df['volume'].astype(float)

    # Hurst (may become constant, will be dropped)
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

    # --- PREPROCESSING FIXES ---
    # Drop constant columns
    constant_cols = [col for col in df.columns if df[col].nunique() == 1]
    if constant_cols:
        df.drop(columns=constant_cols, inplace=True)
        print(f"   Dropped constant columns: {constant_cols}")

    # Log transform amihud
    if 'amihud_illiq' in df.columns:
        df['amihud_illiq'] = np.log1p(df['amihud_illiq'])

    # Cap vol_aggression at 99th percentile
    if 'vol_aggression' in df.columns:
        p99 = df['vol_aggression'].quantile(0.99)
        df['vol_aggression'] = df['vol_aggression'].clip(upper=p99)

    df = df.dropna()
    return df

# ==========================================
# DIRECTION-BASED TARGET (No SL check)
# ==========================================
def create_targets(df):
    long_labels, short_labels = [], []
    for i in range(len(df) - LOOKAHEAD - 1):
        entry = df.iloc[i + 1]['open']
        up_0_3 = entry * (1 + TP)
        down_0_3 = entry * (1 - TP)

        long_win, short_win = 0, 0
        # Scan next 4 bars, find the FIRST 0.3% move
        for j in range(1, LOOKAHEAD + 1):
            idx = i + j
            high = df.iloc[idx]['high']
            low = df.iloc[idx]['low']

            # Agar upar 0.3% pehle touch ho -> Long win
            if high >= up_0_3:
                long_win = 1
                short_win = 0
                break
            # Agar neeche 0.3% pehle touch ho -> Short win
            if low <= down_0_3:
                long_win = 0
                short_win = 1
                break
            # Agar kuch nahi, loop continues

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

print("🔄 Creating direction-based targets...")
long_labels, short_labels = create_targets(train_raw)
print(f"🔍 Long labels : {pd.Series(long_labels).value_counts().to_dict()}")
print(f"🔍 Short labels: {pd.Series(short_labels).value_counts().to_dict()}")

train_labeled = train_raw.iloc[:len(long_labels)].copy()
train_labeled['long_label'] = long_labels
train_labeled['short_label'] = short_labels

print("🛠️ Engineering features...")
train_feat = engineer_features(train_labeled)
print(f"✅ Train rows: {len(train_feat)}")

# Exclude OHLC, volume, fear_greed, timestamp
exclude = ['open', 'high', 'low', 'close', 'long_label', 'short_label',
           'volume', 'timestamp']

X_train = train_feat.drop(columns=[c for c in exclude if c in train_feat.columns])
y_long = train_feat['long_label']
y_short = train_feat['short_label']

# Save sample for inspection
train_feat.head(1000).to_csv("training_sample.csv", index=True)
print(f"✅ Saved training sample to 'training_sample.csv'")
print(f"✅ Final features: {list(X_train.columns)}")

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
