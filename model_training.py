#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
model_training.py - High Confidence Trading
- Fixed Hurst exponent (NaN filled with 0.5)
- All other NaNs dropped (lookback warmup)
- Trains binary classifier for TP before SL
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# CONFIG
# ==========================================
CSV_FILE = "ohlcv.csv"
TP = 0.003          # 0.3% Take Profit
SL = 0.0015         # 0.15% Stop Loss
LOOKAHEAD = 4       # Scan next 4 candles
SPLIT_RATIO = 0.8   # 80% train, 20% test
MODEL_PATH = "xgboost_high_confidence.json"

# ==========================================
# FEATURE ENGINEERING (with NaN handling)
# ==========================================
def hurst(series):
    """Compute Hurst exponent on a numpy array. Returns NaN if fails."""
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
    
    # --- Hurst with fallback ---
    hurst_vals = close.rolling(100).apply(lambda x: hurst(x.values), raw=True)
    df['hurst_exp'] = hurst_vals.fillna(0.5)   # random walk assumption
    
    # --- Other features (no fill, will be dropped if NaN) ---
    df['vol_aggression'] = volume * (high - low) / close
    vwap = (volume * close).rolling(50).sum() / volume.rolling(50).sum()
    df['vwap_ema_spread'] = vwap - close.ewm(span=20).mean()
    df['price_accel'] = close - 2 * close.shift(1) + close.shift(2)
    
    # True Range for ATR
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
    
    # --- Drop all rows with any remaining NaN (lookback warmup) ---
    df = df.dropna()
    return df

# ==========================================
# TARGET CREATION (TP before SL)
# ==========================================
def create_target(df):
    labels = []
    for i in range(len(df) - LOOKAHEAD - 1):
        entry = df.iloc[i + 1]['open']
        tp = entry * (1 + TP)
        sl = entry * (1 - SL)
        
        win = 0
        for j in range(1, LOOKAHEAD + 1):
            idx = i + j
            high = df.iloc[idx]['high']
            low = df.iloc[idx]['low']
            
            if low <= sl:
                break
            if high >= tp:
                win = 1
                break
        
        labels.append(win)
    return labels

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

print("🔄 Creating target (TP before SL)...")
train_labels = create_target(train_raw)
print(f"🔍 Labels before FE: {pd.Series(train_labels).value_counts().to_dict()}")

train_labeled = train_raw.iloc[:len(train_labels)].copy()
train_labeled['label'] = train_labels

print("🛠️ Engineering features on train set...")
train_feat = engineer_features(train_labeled)
print(f"✅ Train rows after features: {len(train_feat)}")
print(f"📈 Label distribution: {train_feat['label'].value_counts().to_dict()}\n")

if len(train_feat) == 0:
    print("❌ No training data after feature engineering. Check your data or adjust lookback windows.")
    exit(1)

exclude = ['open', 'high', 'low', 'close', 'label']
X_train = train_feat.drop(columns=[c for c in exclude if c in train_feat.columns])
y_train = train_feat['label']

print("🤖 Training XGBoost...")
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.6,
    colsample_bytree=0.6,
    reg_alpha=1.0,
    reg_lambda=2.0,
    min_child_weight=5,
    objective='binary:logistic',
    scale_pos_weight=1.0,
    random_state=42,
    eval_metric='logloss'
)
model.fit(X_train, y_train)
model.save_model(MODEL_PATH)

y_pred = model.predict(X_train)
y_proba = model.predict_proba(X_train)[:, 1]
acc = accuracy_score(y_train, y_pred)
auc = roc_auc_score(y_train, y_proba)

print(f"✅ Model saved to {MODEL_PATH}")
print(f"📊 Train Accuracy: {acc*100:.2f}%")
print(f"📊 AUC Score: {auc:.3f}")
print("Classification Report:\n", classification_report(y_train, y_pred, target_names=['Loss', 'Win']))

print("\n🔑 Top 5 features:")
for i, f in enumerate(X_train.columns[:5]):
    print(f"   {i+1}. {f}: {model.feature_importances_[i]:.4f}")

print("\n✅ Training complete! Now run confidence_backtest.py")
