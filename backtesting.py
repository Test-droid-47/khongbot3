#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
backtesting.py - New Features + Correct Exit
- Uses same feature set as training
- Entry at i+1 open, exit at last close if no TP/SL
- Prevents overlapping trades
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

LONG_CONF_THRESHOLD = 0.50
SHORT_CONF_THRESHOLD = 0.50

CAPITAL = 100
LEVERAGE = 10
SLIPPAGE = 0.0005
FEE = 0.0007

# ==========================================
# FEATURE ENGINEERING (Same as training)
# ==========================================
def engineer_features(df):
    df = df.copy()
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    # Hurst
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

if len(test_raw) < LOOKAHEAD + 30:
    print("❌ Not enough test data.")
    exit(0)

print("🛠️ Engineering features on test set...")
test_feat = engineer_features(test_raw)
print(f"✅ Test rows after features: {len(test_feat)}")

exclude = ['open', 'high', 'low', 'close']
X_test = test_feat.drop(columns=[c for c in exclude if c in test_feat.columns])
df_test = test_feat

print("🔮 Loading models...")
model_long = xgb.XGBClassifier()
model_long.load_model(MODEL_LONG)
model_short = xgb.XGBClassifier()
model_short.load_model(MODEL_SHORT)

print("💻 Simulating trades on unseen test data...")
trades = []
capital_curve = [CAPITAL]
max_index = len(df_test) - LOOKAHEAD - 1

in_trade_until = -1

for i in range(max_index):
    if i < in_trade_until:
        capital_curve.append(capital_curve[-1])
        continue

    features = X_test.iloc[i:i+1]
    p_long = model_long.predict_proba(features)[0][1]
    p_short = model_short.predict_proba(features)[0][1]

    long_signal = p_long >= LONG_CONF_THRESHOLD
    short_signal = p_short >= SHORT_CONF_THRESHOLD

    if long_signal and short_signal:
        direction = 'Long' if p_long >= p_short else 'Short'
        confidence = max(p_long, p_short)
    elif long_signal:
        direction, confidence = 'Long', p_long
    elif short_signal:
        direction, confidence = 'Short', p_short
    else:
        direction, confidence = None, 0

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

    # If no exit by LOOKAHEAD, exit at close of candle i+LOOKAHEAD
    if exit_price is None:
        exit_idx = i + LOOKAHEAD
        if exit_idx < len(df_test):
            exit_price = df_test.iloc[exit_idx]['close']
        else:
            exit_price = entry_price  # fallback
        exit_time = LOOKAHEAD
        result = 'No Exit'

    in_trade_until = i + exit_time

    pnl_pct = (exit_price - entry_price) / entry_price if direction == 'Long' else (entry_price - exit_price) / entry_price
    notional = capital_curve[-1] * LEVERAGE
    trade_pnl = notional * pnl_pct - notional * (SLIPPAGE + FEE)
    new_capital = max(0, capital_curve[-1] + trade_pnl)
    capital_curve.append(new_capital)

    trades.append({
        'Entry_Time': df_test.index[i+1],
        'Direction': direction,
        'Entry': entry_price,
        'Exit': exit_price,
        'Exit_Time': df_test.index[exit_idx] if exit_time>0 else df_test.index[i],
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
    print("❌ No trades. Lower thresholds.")
    exit(0)

df_trades = pd.DataFrame(trades)
df_trades.to_csv("backtest_trades.csv", index=False)

total_trades = len(df_trades)
wins = len(df_trades[df_trades['Result'] == 'Win'])
losses = len(df_trades[df_trades['Result'] == 'Loss'])
no_exit = len(df_trades[df_trades['Result'] == 'No Exit'])
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
print(f"📅 Period: {df_test.index[0].date()} to {df_test.index[-1].date()}")
print(f"💵 Initial Capital: ${CAPITAL}")
print(f"📈 Final Capital  : ${final_capital:.2f}")
print(f"📈 Total Return   : {total_return:.2f}%")
print(f"📈 Sharpe Ratio   : {sharpe:.2f}")
print(f"📈 Max Drawdown   : {max_dd:.2f}%")
print("-" * 60)
print(f"📊 Total Trades   : {total_trades}")
print(f"📊 Wins           : {wins}")
print(f"📊 Losses         : {losses}")
print(f"📊 No Exit (flat) : {no_exit}")
print(f"📊 Win Rate (%)   : {win_rate:.2f}%")
print(f"📊 Avg Win ($)    : ${df_trades[df_trades['Net_PnL']>0]['Net_PnL'].mean():.2f}" if wins>0 else "N/A")
print(f"📊 Avg Loss ($)   : ${df_trades[df_trades['Net_PnL']<0]['Net_PnL'].mean():.2f}" if losses>0 else "N/A")
print(f"📊 Profit Factor  : {profit_factor:.2f}")
print("="*60)

long_trades = len(df_trades[df_trades['Direction'] == 'Long'])
short_trades = len(df_trades[df_trades['Direction'] == 'Short'])
long_wins = len(df_trades[(df_trades['Direction']=='Long') & (df_trades['Result']=='Win')])
short_wins = len(df_trades[(df_trades['Direction']=='Short') & (df_trades['Result']=='Win')])
print(f"🔹 Long Trades  : {long_trades} (Win rate: {long_wins/max(1,long_trades)*100:.2f}%)")
print(f"🔹 Short Trades : {short_trades} (Win rate: {short_wins/max(1,short_trades)*100:.2f}%)")
print("="*60)
print("💾 Detailed trade log saved to 'backtest_trades.csv'")