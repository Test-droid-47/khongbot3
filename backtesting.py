#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
backtesting.py
Loads trained Long & Short XGBoost models and backtests on historical data.
Includes realistic costs: Slippage 0.05% (entry+exit) and Fee 0.07% (maker+taker).
Default Capital: $100, Leverage: 10x.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURATION (WITH COSTS)
# ==========================================
CSV_FILE = "ohlcv.csv"
LOOKAHEAD = 4
TP = 0.003
SL = 0.0015

MODEL_LONG = "xgboost_long.json"
MODEL_SHORT = "xgboost_short.json"

LONG_CONF_THRESHOLD = 0.65
SHORT_CONF_THRESHOLD = 0.62

# ===== CAPITAL & LEVERAGE =====
CAPITAL = 100                 # Starting capital (can change to 1000)
LEVERAGE = 10                 # 10x Leverage

# ===== TRADING COSTS (Round Trip) =====
SLIPPAGE = 0.0005             # 0.05% (worse entry/exit price)
FEE = 0.0007                  # 0.07% (Binance futures maker+taker)

# ==========================================
# FEATURE ENGINEERING
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
# 1. LOAD MODELS & DATA
# ==========================================
print("📊 Loading data...")
df_raw = pd.read_csv(CSV_FILE)
df_raw.columns = [col.lower() for col in df_raw.columns]

print("🔮 Loading models...")
model_long = xgb.XGBClassifier()
model_long.load_model(MODEL_LONG)
model_short = xgb.XGBClassifier()
model_short.load_model(MODEL_SHORT)

print("🛠️ Engineering features...")
df_feat = engineer_features(df_raw)
print(f"✅ Backtest rows: {len(df_feat)}")

exclude = ['open', 'high', 'low', 'close']
X = df_feat.drop(columns=[c for c in exclude if c in df_feat.columns])

# ==========================================
# 2. SIMULATE TRADES (Forward walk)
# ==========================================
print("💻 Simulating trades with costs...")
trades = []
capital_curve = [CAPITAL]
max_index = len(df_feat) - LOOKAHEAD - 1

for i in range(max_index):
    features = X.iloc[i:i+1]
    p_long = model_long.predict_proba(features)[0][1]
    p_short = model_short.predict_proba(features)[0][1]

    # Decision
    long_signal = p_long >= LONG_CONF_THRESHOLD
    short_signal = p_short >= SHORT_CONF_THRESHOLD

    if long_signal and short_signal:
        if p_long >= p_short:
            direction = 'Long'
            confidence = p_long
        else:
            direction = 'Short'
            confidence = p_short
    elif long_signal:
        direction = 'Long'
        confidence = p_long
    elif short_signal:
        direction = 'Short'
        confidence = p_short
    else:
        direction = None
        confidence = 0

    if direction is None:
        capital_curve.append(capital_curve[-1])
        continue

    entry_price = df_feat.loc[df_feat.index[i+1], 'open']

    if direction == 'Long':
        tp_price = entry_price * (1 + TP)
        sl_price = entry_price * (1 - SL)
    else:
        tp_price = entry_price * (1 - TP)
        sl_price = entry_price * (1 + SL)

    exit_price = None
    exit_time = 0
    result = 'No Exit'

    for j in range(1, LOOKAHEAD + 1):
        idx = i + j
        if idx >= len(df_feat):
            break
        high = df_feat.loc[df_feat.index[idx], 'high']
        low = df_feat.loc[df_feat.index[idx], 'low']

        if direction == 'Long':
            if low <= sl_price:
                exit_price = sl_price
                exit_time = j
                result = 'Loss'
                break
            if high >= tp_price:
                exit_price = tp_price
                exit_time = j
                result = 'Win'
                break
        else:
            if high >= sl_price:
                exit_price = sl_price
                exit_time = j
                result = 'Loss'
                break
            if low <= tp_price:
                exit_price = tp_price
                exit_time = j
                result = 'Win'
                break

    if exit_price is None:
        exit_price = entry_price
        exit_time = LOOKAHEAD
        result = 'No Exit'

    # Calculate raw PnL %
    if direction == 'Long':
        pnl_pct = (exit_price - entry_price) / entry_price
    else:
        pnl_pct = (entry_price - exit_price) / entry_price

    # --- COSTS CALCULATION (UPDATED) ---
    notional = capital_curve[-1] * LEVERAGE
    raw_pnl = notional * pnl_pct
    
    slippage_cost = notional * SLIPPAGE
    fee_cost = notional * FEE
    total_cost = slippage_cost + fee_cost
    
    trade_pnl = raw_pnl - total_cost
    new_capital = capital_curve[-1] + trade_pnl
    capital_curve.append(new_capital)

    trades.append({
        'Entry_Time': df_feat.index[i],
        'Direction': direction,
        'Entry': entry_price,
        'Exit': exit_price,
        'Exit_Time': df_feat.index[i + exit_time] if exit_time > 0 else df_feat.index[i],
        'Result': result,
        'Bars_Held': exit_time,
        'Raw_PnL': raw_pnl,
        'Slippage_Cost': slippage_cost,
        'Fee_Cost': fee_cost,
        'Net_PnL': trade_pnl,
        'Capital_After': new_capital,
        'Confidence': confidence
    })

# ==========================================
# 3. PERFORMANCE METRICS (Unchanged)
# ==========================================
if len(trades) == 0:
    print("❌ No trades executed! Lower your thresholds.")
    exit(0)

df_trades = pd.DataFrame(trades)
df_trades.to_csv("backtest_trades.csv", index=False)

total_trades = len(df_trades)
wins = len(df_trades[df_trades['Result'] == 'Win'])
losses = len(df_trades[df_trades['Result'] == 'Loss'])
no_exit = len(df_trades[df_trades['Result'] == 'No Exit'])
win_rate = wins / total_trades * 100 if total_trades > 0 else 0

total_pnl = df_trades['Net_PnL'].sum()
final_capital = capital_curve[-1]
total_return = (final_capital - CAPITAL) / CAPITAL * 100

avg_win = df_trades[df_trades['Net_PnL'] > 0]['Net_PnL'].mean() if wins > 0 else 0
avg_loss = df_trades[df_trades['Net_PnL'] < 0]['Net_PnL'].mean() if losses > 0 else 0
gross_wins = df_trades[df_trades['Net_PnL'] > 0]['Net_PnL'].sum()
gross_losses = abs(df_trades[df_trades['Net_PnL'] < 0]['Net_PnL'].sum())
profit_factor = gross_wins / gross_losses if gross_losses > 0 else np.inf

peak = np.maximum.accumulate(capital_curve)
drawdown = (peak - capital_curve) / peak * 100
max_dd = drawdown.max()

returns = np.diff(capital_curve) / capital_curve[:-1]
sharpe = (np.mean(returns) / (np.std(returns) + 1e-9)) * np.sqrt(365 * 24)

print("\n" + "="*60)
print("📊 BACKTEST RESULTS (With Slippage & Fees)")
print("="*60)
print(f"💵 Initial Capital     : ${CAPITAL}")
print(f"📈 Final Capital       : ${final_capital:.2f}")
print(f"📈 Total Return (%)    : {total_return:.2f}%")
print(f"📈 Sharpe Ratio (hrly) : {sharpe:.2f}")
print(f"📈 Max Drawdown (%)    : {max_dd:.2f}%")
print("-" * 60)
print(f"📊 Total Trades        : {total_trades}")
print(f"📊 Wins                : {wins}")
print(f"📊 Losses              : {losses}")
print(f"📊 No Exit (Flat)      : {no_exit}")
print(f"📊 Win Rate (%)        : {win_rate:.2f}%")
print(f"📊 Avg Win ($)         : ${avg_win:.2f}")
print(f"📊 Avg Loss ($)        : ${avg_loss:.2f}")
print(f"📊 Profit Factor       : {profit_factor:.2f}")
print("="*60)

long_trades = len(df_trades[df_trades['Direction'] == 'Long'])
short_trades = len(df_trades[df_trades['Direction'] == 'Short'])
long_wins = len(df_trades[(df_trades['Direction']=='Long') & (df_trades['Result']=='Win')])
short_wins = len(df_trades[(df_trades['Direction']=='Short') & (df_trades['Result']=='Win')])
print(f"🔹 Long Trades  : {long_trades} (Win Rate: {long_wins/max(1,long_trades)*100:.2f}%)")
print(f"🔹 Short Trades : {short_trades} (Win Rate: {short_wins/max(1,short_trades)*100:.2f}%)")
print("="*60)
print("💾 Detailed trade log saved to 'backtest_trades.csv'")
print("🔍 Cost applied: Slippage=0.05% (round trip), Fees=0.07% (round trip)")