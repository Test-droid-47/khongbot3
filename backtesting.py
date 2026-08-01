#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
confidence_backtest.py - High confidence trading
Only takes trades when confidence > 0.75
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

MODEL_PATH = "xgboost_high_confidence.json"
CONFIDENCE_THRESHOLD = 0.75   # only trade when > 75%

CAPITAL = 100
LEVERAGE = 10
SLIPPAGE = 0.0005
FEE = 0.0007

# ==========================================
# FEATURE ENGINEERING (same)
# ==========================================
def hurst(series):
    lags = min(100, len(series))
    if lags < 2:
        return np.nan
    tau = []
    lagvec = np.arange(1, lags)
    for lag in lagvec:
        pp = np.subtract(series[lag:], series[:-lag])
        tau.append(np.std(pp))
    if len(tau) < 2:
        return np.nan
    try:
        poly = np.polyfit(np.log(lagvec[:len(tau)]), np.log(tau), 1)
        return poly[0] * 2.0
    except:
        return np.nan

def engineer_features(df):
    df = df.copy()
    close = df['close']; high = df['high']; low = df['low']; volume = df['volume']
    
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
# MAIN
# ==========================================
print("📊 Loading data...")
df_raw = pd.read_csv(CSV_FILE)
df_raw.columns = [col.lower() for col in df_raw.columns]
df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp'])
df_raw.set_index('timestamp', inplace=True)

split_idx = int(len(df_raw) * SPLIT_RATIO)
test_raw = df_raw.iloc[split_idx:].copy()

print(f"\n📅 BACKTEST PERIOD : {test_raw.index[0].date()} to {test_raw.index[-1].date()} ({len(test_raw)} candles)")

print("🛠️ Engineering features on test set...")
test_feat = engineer_features(test_raw)
print(f"✅ Test rows after features: {len(test_feat)}")

exclude = ['open', 'high', 'low', 'close']
X_test = test_feat.drop(columns=[c for c in exclude if c in test_feat.columns])
df_test = test_feat

print("🔮 Loading model...")
model = xgb.XGBClassifier()
model.load_model(MODEL_PATH)

print("💻 Simulating high-confidence trades...")
trades = []
capital_curve = [CAPITAL]
max_index = len(df_test) - LOOKAHEAD - 1

in_trade_until = -1

for i in range(max_index):
    if i < in_trade_until:
        capital_curve.append(capital_curve[-1])
        continue

    features = X_test.iloc[i:i+1]
    confidence = model.predict_proba(features)[0][1]   # probability of win

    # Only trade if confidence is high
    if confidence < CONFIDENCE_THRESHOLD:
        capital_curve.append(capital_curve[-1])
        continue

    # Enter trade at next candle's open
    entry_price = df_test.iloc[i+1]['open']
    tp_price = entry_price * (1 + TP)
    sl_price = entry_price * (1 - SL)

    exit_price, exit_time, result = None, 0, 'No Exit'

    for j in range(1, LOOKAHEAD + 1):
        idx = i + j
        if idx >= len(df_test):
            break
        high = df_test.iloc[idx]['high']
        low = df_test.iloc[idx]['low']

        if low <= sl_price:
            exit_price, exit_time, result = sl_price, j, 'Loss'
            break
        if high >= tp_price:
            exit_price, exit_time, result = tp_price, j, 'Win'
            break

    if exit_price is None:
        exit_price = df_test.iloc[i+LOOKAHEAD]['close']
        exit_time = LOOKAHEAD
        result = 'No Exit'

    in_trade_until = i + exit_time

    pnl_pct = (exit_price - entry_price) / entry_price
    notional = capital_curve[-1] * LEVERAGE
    trade_pnl = notional * pnl_pct - notional * (SLIPPAGE + FEE)
    new_capital = max(0, capital_curve[-1] + trade_pnl)
    capital_curve.append(new_capital)

    trades.append({
        'Entry_Time': df_test.index[i+1],
        'Entry': entry_price,
        'Exit': exit_price,
        'Exit_Time': df_test.index[i+exit_time] if exit_time>0 else df_test.index[i],
        'Result': result,
        'Bars_Held': exit_time,
        'Net_PnL': trade_pnl,
        'Capital_After': new_capital,
        'Confidence': confidence
    })

# ==========================================
# METRICS
# ==========================================
if len(trades) == 0:
    print("\n❌ No trades! Confidence threshold too high ({}). Try lowering to 0.60.".format(CONFIDENCE_THRESHOLD))
    exit(0)

df_trades = pd.DataFrame(trades)
df_trades.to_csv("backtest_trades_high_confidence.csv", index=False)

total_trades = len(df_trades)
wins = len(df_trades[df_trades['Result'] == 'Win'])
losses = len(df_trades[df_trades['Result'] == 'Loss'])
no_exit = len(df_trades[df_trades['Result'] == 'No Exit'])
win_rate = wins / total_trades * 100 if total_trades > 0 else 0
final_capital = capital_curve[-1]
total_return = (final_capital - CAPITAL) / CAPITAL * 100

print("\n" + "="*60)
print("📊 HIGH CONFIDENCE BACKTEST")
print("="*60)
print(f"📅 Period: {df_test.index[0].date()} to {df_test.index[-1].date()}")
print(f"🎯 Confidence Threshold: {CONFIDENCE_THRESHOLD*100:.0f}%")
print(f"💵 Initial Capital: ${CAPITAL}")
print(f"📈 Final Capital  : ${final_capital:.2f}")
print(f"📈 Total Return   : {total_return:.2f}%")
print("-" * 60)
print(f"📊 Total Trades   : {total_trades}")
print(f"📊 Wins           : {wins}")
print(f"📊 Losses         : {losses}")
print(f"📊 No Exit (flat) : {no_exit}")
print(f"📊 Win Rate (%)   : {win_rate:.2f}%")
print(f"📊 Avg Win ($)    : ${df_trades[df_trades['Net_PnL']>0]['Net_PnL'].mean():.2f}" if wins>0 else "N/A")
print(f"📊 Avg Loss ($)   : ${df_trades[df_trades['Net_PnL']<0]['Net_PnL'].mean():.2f}" if losses>0 else "N/A")
print("="*60)
print("💾 Detailed trade log saved to 'backtest_trades_high_confidence.csv'")
