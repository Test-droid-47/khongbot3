#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
backtesting.py
- Uses only the last 20% (unseen) data.
- Computes features on test set using ONLY past data (no future).
- Simulates trades with realistic costs.
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

LONG_CONF_THRESHOLD = 0.65
SHORT_CONF_THRESHOLD = 0.62

CAPITAL = 100
LEVERAGE = 10
SLIPPAGE = 0.0005
FEE = 0.0007

# ==========================================
# FEATURE ENGINEERING (same as training)
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
# 1. SPLIT RAW DATA (80/20) AND FETCH TEST
# ==========================================
print("📊 Loading data...")
df_raw = pd.read_csv(CSV_FILE)
df_raw.columns = [col.lower() for col in df_raw.columns]
df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp'])
df_raw.set_index('timestamp', inplace=True)

split_idx = int(len(df_raw) * SPLIT_RATIO)
test_raw = df_raw.iloc[split_idx:].copy()
print(f"📅 Backtest period: {test_raw.index[0]} to {test_raw.index[-1]}")
print(f"📊 Total test candles: {len(test_raw)}")

if len(test_raw) < LOOKAHEAD + 30:
    print("❌ Not enough test data. Increase dataset.")
    exit(0)

# ==========================================
# 2. ENGINEER FEATURES ON TEST (using only test data, but rolling uses past within test)
# ==========================================
print("🛠️ Engineering features on test set...")
test_feat = engineer_features(test_raw)  # rolling windows use only previous rows inside test set
print(f"✅ Test rows after features: {len(test_feat)}")

# ==========================================
# 3. LOAD MODELS
# ==========================================
print("🔮 Loading models...")
model_long = xgb.XGBClassifier()
model_long.load_model(MODEL_LONG)
model_short = xgb.XGBClassifier()
model_short.load_model(MODEL_SHORT)

# ==========================================
# 4. BACKTEST SIMULATION ON TEST FEATURES
# ==========================================
exclude = ['open', 'high', 'low', 'close']
X_test = test_feat.drop(columns=[c for c in exclude if c in test_feat.columns])
df_test = test_feat  # contains all columns, we'll use index for time

print("💻 Simulating trades on unseen test data...")
trades = []
capital_curve = [CAPITAL]
max_index = len(df_test) - LOOKAHEAD - 1

for i in range(max_index):
    features = X_test.iloc[i:i+1]
    p_long = model_long.predict_proba(features)[0][1]
    p_short = model_short.predict_proba(features)[0][1]

    long_signal = p_long >= LONG_CONF_THRESHOLD
    short_signal = p_short >= SHORT_CONF_THRESHOLD

    if long_signal and short_signal:
        if p_long >= p_short:
            direction = 'Long'; confidence = p_long
        else:
            direction = 'Short'; confidence = p_short
    elif long_signal:
        direction = 'Long'; confidence = p_long
    elif short_signal:
        direction = 'Short'; confidence = p_short
    else:
        direction = None; confidence = 0

    if direction is None:
        capital_curve.append(capital_curve[-1])
        continue

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
                exit_price = sl_price; exit_time = j; result = 'Loss'; break
            if high >= tp_price:
                exit_price = tp_price; exit_time = j; result = 'Win'; break
        else:
            if high >= sl_price:
                exit_price = sl_price; exit_time = j; result = 'Loss'; break
            if low <= tp_price:
                exit_price = tp_price; exit_time = j; result = 'Win'; break

    if exit_price is None:
        exit_price = entry_price
        exit_time = LOOKAHEAD
        result = 'No Exit'

    pnl_pct = (exit_price - entry_price) / entry_price if direction == 'Long' else (entry_price - exit_price) / entry_price
    notional = capital_curve[-1] * LEVERAGE
    raw_pnl = notional * pnl_pct
    total_cost = notional * (SLIPPAGE + FEE)
    trade_pnl = raw_pnl - total_cost
    new_capital = capital_curve[-1] + trade_pnl
    capital_curve.append(new_capital)

    trades.append({
        'Entry_Time': df_test.index[i],
        'Direction': direction,
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
# 5. METRICS
# ==========================================
if len(trades) == 0:
    print("❌ No trades. Lower thresholds.")
    exit(0)

df_trades = pd.DataFrame(trades)
df_trades.to_csv("backtest_trades.csv", index=False)

total_trades = len(df_trades)
wins = len(df_trades[df_trades['Result'] == 'Win'])
losses = len(df_trades[df_trades['Result'] == 'Loss'])
win_rate = wins / total_trades * 100 if total_trades > 0 else 0
final_capital = capital_curve[-1]
total_return = (final_capital - CAPITAL) / CAPITAL * 100
gross_wins = df_trades[df_trades['Net_PnL'] > 0]['Net_PnL'].sum()
gross_losses = abs(df_trades[df_trades['Net_PnL'] < 0]['Net_PnL'].sum())
profit_factor = gross_wins / gross_losses if gross_losses > 0 else np.inf
peak = np.maximum.accumulate(capital_curve)
drawdown = (peak - capital_curve) / peak * 100
max_dd = drawdown.max()
returns = np.diff(capital_curve) / capital_curve[:-1]
sharpe = (np.mean(returns) / (np.std(returns) + 1e-9)) * np.sqrt(365 * 24)

print("\n" + "="*60)
print("📊 BACKTEST RESULTS (UNSEEN 20%)")
print("="*60)
print(f"💵 Period: {df_test.index[0]} to {df_test.index[-1]}")
print(f"💵 Initial Capital: ${CAPITAL}")
print(f"📈 Final Capital: ${final_capital:.2f}")
print(f"📈 Total Return: {total_return:.2f}%")
print(f"📈 Sharpe: {sharpe:.2f}")
print(f"📈 Max DD: {max_dd:.2f}%")
print("-" * 60)
print(f"📊 Total Trades: {total_trades}")
print(f"📊 Wins: {wins}, Losses: {losses}")
print(f"📊 Win Rate: {win_rate:.2f}%")
print(f"📊 Profit Factor: {profit_factor:.2f}")
print("="*60)
long_trades = len(df_trades[df_trades['Direction'] == 'Long'])
short_trades = len(df_trades[df_trades['Direction'] == 'Short'])
long_wins = len(df_trades[(df_trades['Direction']=='Long') & (df_trades['Result']=='Win')])
short_wins = len(df_trades[(df_trades['Direction']=='Short') & (df_trades['Result']=='Win')])
print(f"🔹 Long: {long_trades} (Win rate: {long_wins/max(1,long_trades)*100:.2f}%)")
print(f"🔹 Short: {short_trades} (Win rate: {short_wins/max(1,short_trades)*100:.2f}%)")
print("="*60)
print("💾 Detailed trade log saved to 'backtest_trades.csv'")
