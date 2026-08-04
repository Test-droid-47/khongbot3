#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BTC/USDT 1H RULE-BASED STRATEGY (KAMA + Breakeven) - Softer Entry
- Multiplier reduced to 0.3
- Volatility filter (ATR > 0.5 * ATR_MA) to skip dead markets
- TP = 0.30%, SL = 0.45%, Breakeven at 0.12%
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# CONFIG (softer entry)
# ==========================================
CSV_FILE = "ohlcv.csv"
TP = 0.003                      # 0.30%
SL_INITIAL = 0.0045             # 0.45%
BREAKEVEN_TRIGGER = 0.0012      # 0.12%
LOOKAHEAD = 12

CAPITAL = 100
LEVERAGE = 10
SLIPPAGE = 0.0005
FEE = 0.0007

# Entry parameters
ATR_MULTIPLIER = 0.3            # <-- lowered from 0.5
MIN_VOLATILITY_RATIO = 0.5      # ATR must be > 0.5 * ATR_MA(50)

# ==========================================
# KAUFMAN'S ADAPTIVE MOVING AVERAGE
# ==========================================
def kama(close, n=20):
    er = abs(close - close.shift(n)) / (abs(close.diff()).rolling(n).sum() + 1e-9)
    sc = (er * (2.0/(2.0+1.0) - 2.0/(30.0+1.0)) + 2.0/(30.0+1.0)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    return pd.Series(kama, index=close.index)

# ==========================================
# LOAD DATA
# ==========================================
print("📊 Loading BTC/USDT 1h data...")
df = pd.read_csv(CSV_FILE)
df.columns = [c.lower() for c in df.columns]
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)

print(f"✅ Loaded {len(df)} 1h candles.")
print(f"📅 Period: {df.index.min().date()} to {df.index.max().date()}")

# ==========================================
# INDICATORS
# ==========================================
close = df['close']
high = df['high']
low = df['low']

df['kama'] = kama(close, n=20)

tr = np.maximum(high - low,
                np.maximum(abs(high - close.shift(1)),
                           abs(low - close.shift(1))))
df['atr'] = tr.rolling(14).mean()
df['atr_ma'] = df['atr'].rolling(50).mean()

# Volatility filter: avoid very quiet periods
df['vol_ok'] = df['atr'] > (MIN_VOLATILITY_RATIO * df['atr_ma'])

# Entry signals: cross KAMA + ATR*multiplier AND volatility OK
df['long_signal'] = (close > df['kama'] + df['atr'] * ATR_MULTIPLIER) & \
                    (close.shift(1) <= df['kama'].shift(1) + df['atr'].shift(1) * ATR_MULTIPLIER) & \
                    df['vol_ok']

df['short_signal'] = (close < df['kama'] - df['atr'] * ATR_MULTIPLIER) & \
                     (close.shift(1) >= df['kama'].shift(1) - df['atr'].shift(1) * ATR_MULTIPLIER) & \
                     df['vol_ok']

# ==========================================
# BACKTEST SIMULATION (with Breakeven)
# ==========================================
print("💻 Simulating trades...")
trades = []
capital_curve = [CAPITAL]
in_trade_until = -1
max_index = len(df) - LOOKAHEAD - 1

for i in range(max_index):
    if i < in_trade_until:
        capital_curve.append(capital_curve[-1])
        continue

    if df.iloc[i]['long_signal']:
        direction = 'Long'
        entry_price = df.iloc[i+1]['open']
    elif df.iloc[i]['short_signal']:
        direction = 'Short'
        entry_price = df.iloc[i+1]['open']
    else:
        capital_curve.append(capital_curve[-1])
        continue

    if direction == 'Long':
        tp_price = entry_price * (1 + TP)
        sl_price = entry_price * (1 - SL_INITIAL)
        be_price = entry_price * (1 + BREAKEVEN_TRIGGER)
    else:
        tp_price = entry_price * (1 - TP)
        sl_price = entry_price * (1 + SL_INITIAL)
        be_price = entry_price * (1 - BREAKEVEN_TRIGGER)

    exit_price, exit_time, result = None, 0, 'No Exit'
    breakeven_activated = False

    for j in range(1, LOOKAHEAD + 1):
        idx = i + j
        if idx >= len(df):
            break
        high = df.iloc[idx]['high']
        low = df.iloc[idx]['low']

        if direction == 'Long' and not breakeven_activated:
            if high >= be_price:
                breakeven_activated = True
                sl_price = entry_price * (1 + FEE)
        elif direction == 'Short' and not breakeven_activated:
            if low <= be_price:
                breakeven_activated = True
                sl_price = entry_price * (1 - FEE)

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
        exit_price = df.iloc[exit_idx]['close'] if exit_idx < len(df) else entry_price
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
        'Entry_Time': df.index[i+1],
        'Direction': direction,
        'Entry': entry_price,
        'Exit': exit_price,
        'Result': result,
        'Bars_Held': exit_time,
        'Net_PnL': trade_pnl,
        'Capital_After': new_capital,
        'Breakeven_Activated': breakeven_activated
    })

# ==========================================
# REPORT
# ==========================================
if len(trades) == 0:
    print("❌ Still no trades. Try lowering ATR_MULTIPLIER to 0.2 or removing volatility filter.")
    exit(0)

df_trades = pd.DataFrame(trades)
df_trades.to_csv("btc_1h_breakeven_trades.csv", index=False)

total_trades = len(df_trades)
wins = len(df_trades[df_trades['Result'] == 'Win'])
losses = len(df_trades[df_trades['Result'] == 'Loss'])
no_exit = len(df_trades[df_trades['Result'] == 'No Exit'])
win_rate = wins / total_trades * 100
final_capital = capital_curve[-1]
total_return = (final_capital - CAPITAL) / CAPITAL * 100
gross_wins = df_trades[df_trades['Net_PnL'] > 0]['Net_PnL'].sum()
gross_losses = abs(df_trades[df_trades['Net_PnL'] < 0]['Net_PnL'].sum())
profit_factor = gross_wins / gross_losses if gross_losses > 0 else np.inf

print("\n" + "="*70)
print("📊 BTC 1H BREAKEVEN (SOFTER) BACKTEST")
print("="*70)
print(f"📅 Period: {df.index[0].date()} to {df.index[-1].date()}")
print(f"🎯 TP: {TP*100:.2f}% | SL: {SL_INITIAL*100:.2f}% | BE: {BREAKEVEN_TRIGGER*100:.2f}%")
print(f"🎯 Entry multiplier: {ATR_MULTIPLIER} | Vol filter: {MIN_VOLATILITY_RATIO}")
print(f"💵 Initial Capital: ${CAPITAL}")
print(f"📈 Final Capital  : ${final_capital:.2f}")
print(f"📈 Total Return   : {total_return:.2f}%")
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
be_activated = df_trades['Breakeven_Activated'].sum()
print(f"🔹 Breakeven activated on {be_activated} trades")
print("="*70)
print("💾 Trade log saved to 'btc_1h_breakeven_trades.csv'")
