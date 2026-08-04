#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
backtest_dual.py - BTC/USDT 1H with Advanced Features
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
SL = 0.003
LOOKAHEAD = 12
SPLIT_RATIO = 0.8

MODEL_LONG = "xgboost_long_btc.json"
MODEL_SHORT = "xgboost_short_btc.json"

CONFIDENCE_THRESHOLD = 0.60

CAPITAL = 1000
LEVERAGE = 10
SLIPPAGE = 0.0005
FEE = 0.0007

# ==========================================
# FEATURE ENGINEERING (EXACT SAME)
# ==========================================
def engineer_features(df):
    """Same as training – copy‑paste the function from model_training."""
    # (copy the exact engineer_features function here to avoid duplication)
    # I'll repeat it for completeness – in practice you'd import it.
    df = df.copy()
    close = df['close'].astype(float)
    high = df['high'].astype(float)
    low = df['low'].astype(float)
    volume = df['volume'].astype(float)
    
    # ATR
    tr = np.maximum(high - low,
                    np.maximum(abs(high - close.shift(1)),
                               abs(low - close.shift(1))))
    atr = tr.rolling(14).mean()
    df['natr_percent'] = (atr / close) * 100

    df['range'] = high - low + 1e-9
    df['close_position'] = (close - low) / df['range']
    df['avg_range_20'] = df['range'].rolling(20).mean()
    df['range_ratio'] = df['range'] / (df['avg_range_20'] + 1e-9)

    body = abs(close - df['open'])
    upper_wick = high - df[['close', 'open']].max(axis=1)
    lower_wick = df[['close', 'open']].min(axis=1) - low
    df['wick_body_ratio'] = (upper_wick - lower_wick) / (body + 1e-5)

    if isinstance(df.index, pd.DatetimeIndex):
        hour = df.index.hour
        df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * hour / 24)

    for lag in [1, 2, 3]:
        df[f'mom_{lag}'] = np.log(close / close.shift(lag))

    net_move = abs(close - close.shift(5))
    path = (abs(close - close.shift(1)) +
            abs(close.shift(1) - close.shift(2)) +
            abs(close.shift(2) - close.shift(3)) +
            abs(close.shift(3) - close.shift(4)) +
            abs(close.shift(4) - close.shift(5)))
    df['efficiency_ratio'] = net_move / (path + 1e-9)

    # Daily pivot
    daily = df.resample('D').agg({'high': 'max', 'low': 'min', 'close': 'last'})
    daily['pivot'] = (daily['high'] + daily['low'] + daily['close']) / 3
    daily['pivot_shift'] = daily['pivot'].shift(1)
    df['daily_pivot'] = daily['pivot_shift'].reindex(df.index, method='ffill')
    df['pivot_distance'] = (close - df['daily_pivot']) / df['daily_pivot']

    # Relative Volume Z‑Score
    if isinstance(df.index, pd.DatetimeIndex):
        df['hour'] = df.index.hour
        df['vol_mean_same_hour'] = (
            df.groupby('hour')['volume']
            .transform(lambda x: x.rolling(20, min_periods=1).mean().shift(1))
        )
        df['vol_std_same_hour'] = (
            df.groupby('hour')['volume']
            .transform(lambda x: x.rolling(20, min_periods=1).std(ddof=0).shift(1))
        )
        df['vol_zscore'] = (volume - df['vol_mean_same_hour']) / (df['vol_std_same_hour'] + 1e-9)

    drop_cols = ['range', 'avg_range_20', 'upper_wick', 'lower_wick', 'hour',
                 'vol_mean_same_hour', 'vol_std_same_hour', 'daily_pivot']
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True, errors='ignore')
    df = df.dropna()
    return df

# ==========================================
# MAIN
# ==========================================
print("📊 Loading BTC/USDT 1h data...")
df_raw = pd.read_csv(CSV_FILE)
df_raw.columns = [col.lower() for col in df_raw.columns]

if 'timestamp' in df_raw.columns:
    df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp'])
    df_raw.set_index('timestamp', inplace=True)

total_rows = len(df_raw)
split_idx = int(total_rows * SPLIT_RATIO)
test_raw = df_raw.iloc[split_idx:].copy()

print(f"\n📅 BACKTEST: {test_raw.index.min().date()} to {test_raw.index.max().date()} ({len(test_raw)} candles)")

print("🛠️ Engineering features on test set...")
test_feat = engineer_features(test_raw)
print(f"✅ Test rows: {len(test_feat)}")

exclude = ['open', 'high', 'low', 'close', 'volume']
X_test = test_feat.drop(columns=[c for c in exclude if c in test_feat.columns])
df_test = test_feat

print("🔮 Loading models...")
model_long = xgb.XGBClassifier()
model_short = xgb.XGBClassifier()
try:
    model_long.load_model(MODEL_LONG)
    has_long = True
except:
    has_long = False
    print("⚠️ Long model missing.")
try:
    model_short.load_model(MODEL_SHORT)
    has_short = True
except:
    has_short = False
    print("⚠️ Short model missing.")

if not has_long and not has_short:
    print("❌ No models. Exiting.")
    exit(1)

print(f"💻 Simulating with threshold = {CONFIDENCE_THRESHOLD*100:.0f}%...")
trades = []
capital_curve = [CAPITAL]
max_index = len(df_test) - LOOKAHEAD - 1
in_trade_until = -1

for i in range(max_index):
    if i < in_trade_until:
        capital_curve.append(capital_curve[-1])
        continue

    features = X_test.iloc[i:i+1]
    conf_long = model_long.predict_proba(features)[0][1] if has_long else 0.0
    conf_short = model_short.predict_proba(features)[0][1] if has_short else 0.0

    if conf_long >= conf_short:
        conf = conf_long
        direction = 'Long'
    else:
        conf = conf_short
        direction = 'Short'

    if conf < CONFIDENCE_THRESHOLD:
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

    if exit_price is None:
        exit_idx = i + LOOKAHEAD
        exit_price = df_test.iloc[exit_idx]['close'] if exit_idx < len(df_test) else entry_price
        exit_time = LOOKAHEAD
        result = 'No Exit'
    else:
        exit_idx = i + exit_time

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
        'Exit_Time': df_test.index[exit_idx] if exit_time > 0 else df_test.index[i],
        'Result': result,
        'Bars_Held': exit_time,
        'Net_PnL': trade_pnl,
        'Capital_After': new_capital,
        'Confidence': conf
    })

if len(trades) == 0:
    print(f"❌ No trades at threshold {CONFIDENCE_THRESHOLD}. Try lowering to 0.50.")
    exit(0)

df_trades = pd.DataFrame(trades)
df_trades.to_csv("backtest_btc_1h_advanced.csv", index=False)

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
returns = np.diff(capital_curve) / capital_curve[:-1]
sharpe = (np.mean(returns) / (np.std(returns) + 1e-9)) * np.sqrt(365 * 24)
peak = np.maximum.accumulate(capital_curve)
max_dd = ((peak - capital_curve) / peak * 100).max()

print("\n" + "="*70)
print("📊 BTC 1H BACKTEST (ADVANCED FEATURES)")
print("="*70)
print(f"📅 Period: {df_test.index[0].date()} to {df_test.index[-1].date()}")
print(f"🎯 TP/SL: {TP*100:.1f}% / {SL*100:.1f}% (1:1)")
print(f"🎯 Confidence Threshold: {CONFIDENCE_THRESHOLD*100:.0f}%")
print(f"💵 Initial Capital: ${CAPITAL}")
print(f"📈 Final Capital  : ${final_capital:.2f}")
print(f"📈 Total Return   : {total_return:.2f}%")
print(f"📈 Sharpe Ratio   : {sharpe:.2f}")
print(f"📈 Max Drawdown   : {max_dd:.2f}%")
print("-" * 70)
print(f"📊 Total Trades   : {total_trades}")
print(f"📊 Wins           : {wins}")
print(f"📊 Losses         : {losses}")
print(f"📊 No Exit (flat) : {no_exit}")
print(f"📊 Win Rate (%)   : {win_rate:.2f}%")
print(f"📊 Avg Win ($)    : ${df_trades[df_trades['Net_PnL']>0]['Net_PnL'].mean():.2f}" if wins>0 else "N/A")
print(f"📊 Avg Loss ($)   : ${df_trades[df_trades['Net_PnL']<0]['Net_PnL'].mean():.2f}" if losses>0 else "N/A")
print(f"📊 Profit Factor  : {profit_factor:.2f}")
print("="*70)

longs = len(df_trades[df_trades['Direction']=='Long'])
shorts = len(df_trades[df_trades['Direction']=='Short'])
long_wins = len(df_trades[(df_trades['Direction']=='Long') & (df_trades['Result']=='Win')])
short_wins = len(df_trades[(df_trades['Direction']=='Short') & (df_trades['Result']=='Win')])
print(f"🔹 Long trades : {longs} (Win rate: {long_wins/max(1,longs)*100:.2f}%)")
print(f"🔹 Short trades: {shorts} (Win rate: {short_wins/max(1,shorts)*100:.2f}%)")
print("="*70)
print("💾 Trade log saved to 'backtest_btc_1h_advanced.csv'")
