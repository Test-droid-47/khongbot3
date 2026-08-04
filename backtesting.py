#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
backtest_dual.py - Top 1 Signal Backtest
- Scans the entire test set, collects all potential signals.
- Selects the SINGLE best signal (highest confidence between Long/Short).
- Enters that single trade.
- Uses SL = 0.15% and TP = 0.3%.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
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

CAPITAL = 100
LEVERAGE = 10
SLIPPAGE = 0.0005
FEE = 0.0007

# ==========================================
# FEATURES (Same as training)
# ==========================================
def engineer_features(df):
    df = df.copy()
    close = df['close'].astype(float)
    high = df['high'].astype(float)
    low = df['low'].astype(float)
    volume = df['volume'].astype(float)

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

    # Preprocessing
    constant_cols = [col for col in df.columns if df[col].nunique() == 1]
    if constant_cols:
        df.drop(columns=constant_cols, inplace=True)
    if 'amihud_illiq' in df.columns:
        df['amihud_illiq'] = np.log1p(df['amihud_illiq'])
    if 'vol_aggression' in df.columns:
        p99 = df['vol_aggression'].quantile(0.99)
        df['vol_aggression'] = df['vol_aggression'].clip(upper=p99)

    df = df.dropna()
    return df

# ==========================================
# MAIN
# ==========================================
print("📊 Loading data...")
df_raw = pd.read_csv(CSV_FILE)
df_raw.columns = [col.lower() for col in df_raw.columns]
df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp'])
df_raw.set_index('timestamp', inplace=True)

split_idx = int(len(df_raw) * SPLIT_RATIO)
test_raw = df_raw.iloc[split_idx:].copy()
print(f"\n📅 BACKTEST: {test_raw.index[0].date()} to {test_raw.index[-1].date()} ({len(test_raw)} candles)")

print("🛠️ Engineering features...")
test_feat = engineer_features(test_raw)
print(f"✅ Test rows: {len(test_feat)}")

exclude = ['open', 'high', 'low', 'close', 'volume', 'timestamp']
X_test = test_feat.drop(columns=[c for c in exclude if c in test_feat.columns])
df_test = test_feat

print("🔮 Loading models...")
model_long = xgb.XGBClassifier()
model_short = xgb.XGBClassifier()
model_long.load_model(MODEL_LONG)
model_short.load_model(MODEL_SHORT)

# ==========================================
# COLLECT ALL SIGNALS (No Threshold)
# ==========================================
print("💻 Scanning for the TOP 1 signal...")
signals = []
max_index = len(df_test) - LOOKAHEAD - 1

for i in range(max_index):
    features = X_test.iloc[i:i+1]
    conf_long = model_long.predict_proba(features)[0][1]
    conf_short = model_short.predict_proba(features)[0][1]
    
    # Choose direction based on higher confidence (no threshold)
    if conf_long >= conf_short:
        direction = 'Long'
        confidence = conf_long
    else:
        direction = 'Short'
        confidence = conf_short
    
    signals.append({
        'index': i,
        'direction': direction,
        'confidence': confidence,
        'entry_time': df_test.index[i+1]
    })

# Sort by confidence descending and pick the TOP 1
signals_sorted = sorted(signals, key=lambda x: x['confidence'], reverse=True)
top_signal = signals_sorted[0]

print(f"🏆 Top Signal: {top_signal['direction']} with confidence {top_signal['confidence']*100:.2f}%")
print(f"   Entry Time: {top_signal['entry_time']}")

# ==========================================
# EXECUTE THE TOP 1 TRADE
# ==========================================
i = top_signal['index']
direction = top_signal['direction']
confidence = top_signal['confidence']

# Entry at next candle's open (i+1)
entry_price = df_test.iloc[i+1]['open']
if direction == 'Long':
    tp_price = entry_price * (1 + TP)
    sl_price = entry_price * (1 - SL)
else:
    tp_price = entry_price * (1 - TP)
    sl_price = entry_price * (1 + SL)

exit_price, exit_time, result = None, 0, 'No Exit'
for j in range(1, LOOKAHEAD + 1):
    idx = i + j
    if idx >= len(df_test):
        break
    high = df_test.iloc[idx]['high']
    low = df_test.iloc[idx]['low']

    if direction == 'Long':
        if low <= sl_price:
            exit_price, exit_time, result = sl_price, j, 'Loss'
            break
        if high >= tp_price:
            exit_price, exit_time, result = tp_price, j, 'Win'
            break
    else:
        if high >= sl_price:
            exit_price, exit_time, result = sl_price, j, 'Loss'
            break
        if low <= tp_price:
            exit_price, exit_time, result = tp_price, j, 'Win'
            break

if exit_price is None:
    exit_idx = i + LOOKAHEAD
    if exit_idx < len(df_test):
        exit_price = df_test.iloc[exit_idx]['close']
    else:
        exit_price = entry_price
    exit_time = LOOKAHEAD
    result = 'No Exit'
else:
    exit_idx = i + exit_time

pnl_pct = (exit_price - entry_price) / entry_price if direction == 'Long' else (entry_price - exit_price) / entry_price
notional = CAPITAL * LEVERAGE
trade_pnl = notional * pnl_pct - notional * (SLIPPAGE + FEE)
final_capital = CAPITAL + trade_pnl

# ==========================================
# RESULTS
# ==========================================
print("\n" + "="*60)
print("📊 TOP 1 SIGNAL BACKTEST RESULT")
print("="*60)
print(f"📅 Entry: {df_test.index[i+1]}")
print(f"📅 Exit : {df_test.index[exit_idx] if exit_time>0 else df_test.index[i]}")
print(f"🎯 Direction: {direction}")
print(f"📊 Confidence: {confidence*100:.2f}%")
print(f"📈 Entry Price: {entry_price:.4f}")
print(f"📉 Exit Price : {exit_price:.4f}")
print(f"📊 Result: {result}")
print(f"📊 Bars Held: {exit_time}")
print(f"📊 PnL % (Leveraged): {pnl_pct*LEVERAGE*100:.2f}%")
print(f"💵 Final Capital: ${final_capital:.2f} (Initial: ${CAPITAL})")
print("="*60)
print("💾 Trade log saved to 'top1_signal_trade.csv'")
