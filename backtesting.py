#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PhD-Level Quantitative Strategy for SOL/USDT 1h Futures
- Regime Detection (HMM)
- Dual Signal Ensemble (Momentum + Mean Reversion)
- Adaptive Kelly Position Sizing
- Realistic Costs (Slippage + Fees)
"""

import pandas as pd
import numpy as np
from scipy import stats
from hmmlearn import hmm
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURATION
# ==========================================
TP = 0.003          # 0.3% Take Profit
SL = 0.0015         # 0.15% Stop Loss
LOOKAHEAD = 4       # Max hold bars
CAPITAL = 100
LEVERAGE = 10
SLIPPAGE = 0.0005
FEE = 0.0007

# Signal thresholds
ENTRY_THRESHOLD = 0.5
EXIT_THRESHOLD = 0.1

# Kelly parameters
MAX_KELLY_FRACTION = 0.25
KELLY_WINDOW = 100

# ==========================================
# CORE STRATEGY CLASS
# ==========================================

class QuantStrategy:
    def __init__(self, df, btc_df=None):
        """
        df: SOL/USDT OHLCV DataFrame with columns: timestamp, open, high, low, close, volume
        btc_df: BTC/USDT OHLCV DataFrame (optional, for cross-sectional momentum)
        """
        self.df = df.copy()
        self.btc_df = btc_df
        self.signals = pd.DataFrame(index=df.index)
        self._prepare_data()
        self._compute_features()
        self._fit_hmm()
        
    def _prepare_data(self):
        """Compute base features"""
        self.df['returns'] = self.df['close'].pct_change()
        self.df['log_returns'] = np.log(self.df['close'] / self.df['close'].shift(1))
        self.df['high_low'] = self.df['high'] - self.df['low']
        self.df['atr'] = self.df['high_low'].rolling(14).mean()
        
    def _compute_features(self):
        """Compute all strategy features"""
        close = self.df['close']
        
        # 1. Moving Averages
        self.df['ema20'] = close.ewm(span=20).mean()
        self.df['ema50'] = close.ewm(span=50).mean()
        self.df['ema200'] = close.ewm(span=200).mean()
        
        # 2. Volatility
        self.df['volatility'] = self.df['returns'].rolling(20).std()
        self.df['vol_ratio'] = self.df['volatility'] / self.df['volatility'].rolling(100).mean()
        
        # 3. Z-Score (Mean Reversion Signal)
        self.df['zscore'] = (close - self.df['ema50']) / (close.rolling(50).std() + 1e-9)
        
        # 4. Cross-Sectional Momentum (if BTC data available)
        if self.btc_df is not None:
            btc_returns = self.btc_df['close'].pct_change(periods=24)
            sol_returns = self.df['close'].pct_change(periods=24)
            alpha = sol_returns - btc_returns
            self.df['alpha'] = (alpha - alpha.rolling(168).mean()) / (alpha.rolling(168).std() + 1e-9)
        else:
            # Fallback: simple momentum
            self.df['alpha'] = (close / close.shift(24) - 1) / (close.rolling(24).std() + 1e-9)
        
        # 5. Regime features
        self.df['trend_strength'] = (self.df['ema20'] - self.df['ema50']) / self.df['ema50']
        self.df['is_trending'] = abs(self.df['trend_strength']) > 0.01
        
        # 6. Bollinger Bands (reversion signal)
        bb_mid = self.df['ema20']
        bb_std = close.rolling(20).std()
        self.df['bb_upper'] = bb_mid + 2 * bb_std
        self.df['bb_lower'] = bb_mid - 2 * bb_std
        self.df['bb_position'] = (close - bb_mid) / (bb_std + 1e-9)
        
    def _fit_hmm(self):
        """Fit Hidden Markov Model for regime detection"""
        features = ['returns', 'volatility', 'trend_strength']
        X = self.df[features].dropna().values
        
        # Scale features
        X_scaled = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)
        
        model = hmm.GaussianHMM(n_components=3, covariance_type="full", n_iter=1000, random_state=42)
        model.fit(X_scaled)
        
        # Predict states
        states = model.predict(X_scaled)
        self.df['regime'] = np.nan
        self.df.loc[self.df[features].dropna().index, 'regime'] = states
        
        # Label regimes (based on mean returns in each state)
        regime_returns = self.df.groupby('regime')['returns'].mean()
        sorted_regimes = regime_returns.sort_values().index
        self.regime_map = {
            sorted_regimes[0]: 'Bearish',
            sorted_regimes[1]: 'Ranging',
            sorted_regimes[2]: 'Bullish'
        }
        self.df['regime_label'] = self.df['regime'].map(self.regime_map)
        
    def generate_signals(self):
        """Generate trading signals"""
        # Signal A: Momentum (cross-sectional)
        self.df['signal_momentum'] = self.df['alpha']
        
        # Signal B: Mean Reversion
        self.df['signal_reversion'] = -self.df['zscore']
        
        # Regime-based weighting
        self.df['momentum_weight'] = np.where(
            self.df['regime_label'].isin(['Bullish', 'Bearish']), 0.70, 0.20
        )
        self.df['reversion_weight'] = 1 - self.df['momentum_weight']
        
        # Ensemble signal
        self.df['signal_ensemble'] = (
            self.df['momentum_weight'] * self.df['signal_momentum'] +
            self.df['reversion_weight'] * self.df['signal_reversion']
        )
        
        # Volatility-adjusted signal
        self.df['signal'] = self.df['signal_ensemble'] / (self.df['vol_ratio'] + 0.5)
        
        return self.df
    
    def compute_position_size(self, i, capital):
        """Adaptive Kelly position sizing"""
        # Get recent trade history
        if not hasattr(self, '_trade_history'):
            self._trade_history = []
            
        if len(self._trade_history) >= KELLY_WINDOW:
            recent = self._trade_history[-KELLY_WINDOW:]
            wins = [t for t in recent if t > 0]
            losses = [t for t in recent if t < 0]
            
            win_rate = len(wins) / len(recent) if recent else 0.5
            avg_win = np.mean(wins) if wins else 0.01
            avg_loss = abs(np.mean(losses)) if losses else 0.01
            
            # Kelly formula
            kelly = win_rate - (1 - win_rate) / (avg_win / avg_loss) if avg_loss > 0 else 0
            kelly = np.clip(kelly, 0, MAX_KELLY_FRACTION)
        else:
            kelly = 0.10  # initial conservative
            
        # Volatility scaling
        atr = self.df.iloc[i]['atr']
        atr_mean = self.df['atr'].rolling(100).mean().iloc[i] if i > 100 else atr
        vol_scalar = atr / (atr_mean + 1e-9)
        vol_scalar = np.clip(vol_scalar, 0.5, 2.0)
        
        position_size = capital * LEVERAGE * kelly * (1 / vol_scalar)
        return position_size / self.df.iloc[i]['close']  # return quantity

    def backtest(self):
        """Run full backtest with realistic costs"""
        self.generate_signals()
        df = self.df.copy()
        df = df.dropna()
        
        trades = []
        capital_curve = [CAPITAL]
        in_trade_until = -1
        position = 0
        entry_price = 0
        direction = None
        
        for i in range(len(df) - LOOKAHEAD - 1):
            # Skip if in trade
            if i < in_trade_until:
                capital_curve.append(capital_curve[-1])
                continue
                
            # Generate signal
            signal = df.iloc[i]['signal']
            
            # Entry conditions
            if position == 0:
                if signal > ENTRY_THRESHOLD:
                    direction = 'Long'
                    position = self.compute_position_size(i, capital_curve[-1])
                    entry_price = df.iloc[i+1]['open']
                    in_trade_until = -1
                elif signal < -ENTRY_THRESHOLD:
                    direction = 'Short'
                    position = self.compute_position_size(i, capital_curve[-1])
                    entry_price = df.iloc[i+1]['open']
                    in_trade_until = -1
                else:
                    capital_curve.append(capital_curve[-1])
                    continue
                    
            # Exit logic
            if position != 0:
                tp = entry_price * (1 + TP) if direction == 'Long' else entry_price * (1 - TP)
                sl = entry_price * (1 - SL) if direction == 'Long' else entry_price * (1 + SL)
                
                for j in range(1, LOOKAHEAD + 1):
                    idx = i + j
                    if idx >= len(df):
                        break
                        
                    high = df.iloc[idx]['high']
                    low = df.iloc[idx]['low']
                    
                    if direction == 'Long':
                        if low <= sl:
                            exit_price = sl
                            result = 'Loss'
                            exit_time = j
                            break
                        if high >= tp:
                            exit_price = tp
                            result = 'Win'
                            exit_time = j
                            break
                    else:
                        if high >= sl:
                            exit_price = sl
                            result = 'Loss'
                            exit_time = j
                            break
                        if low <= tp:
                            exit_price = tp
                            result = 'Win'
                            exit_time = j
                            break
                
                if 'exit_price' not in locals():
                    exit_price = df.iloc[i+LOOKAHEAD]['close'] if i+LOOKAHEAD < len(df) else entry_price
                    exit_time = LOOKAHEAD
                    result = 'No Exit'
                    
                in_trade_until = i + exit_time
                
                # PnL calculation with costs
                if direction == 'Long':
                    pnl_pct = (exit_price - entry_price) / entry_price
                else:
                    pnl_pct = (entry_price - exit_price) / entry_price
                    
                notional = capital_curve[-1] * LEVERAGE
                raw_pnl = notional * pnl_pct
                total_cost = notional * (SLIPPAGE + FEE)
                trade_pnl = raw_pnl - total_cost
                
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
                    'Signal': signal,
                    'Regime': df.iloc[i]['regime_label']
                })
                
                # Reset
                position = 0
                entry_price = 0
                direction = None
                exit_price = None
                
        self._trade_history = [t['Net_PnL'] for t in trades]  # for next run
        return trades, capital_curve

# ==========================================
# BACKTEST EXECUTION
# ==========================================

# Load data (replace with your file path)
df = pd.read_csv("ohlcv.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)

# Optional: load BTC data
try:
    btc_df = pd.read_csv("btc_ohlcv.csv")
    btc_df['timestamp'] = pd.to_datetime(btc_df['timestamp'])
    btc_df.set_index('timestamp', inplace=True)
except:
    btc_df = None
    print("⚠️ No BTC data found. Using single-asset momentum.")

# Run strategy
strategy = QuantStrategy(df, btc_df)
trades, capital_curve = strategy.backtest()

# ==========================================
# PERFORMANCE METRICS
# ==========================================

if len(trades) == 0:
    print("❌ No trades executed.")
    exit(0)

df_trades = pd.DataFrame(trades)

total_trades = len(df_trades)
wins = len(df_trades[df_trades['Result'] == 'Win'])
losses = len(df_trades[df_trades['Result'] == 'Loss'])
no_exit = len(df_trades[df_trades['Result'] == 'No Exit'])
win_rate = wins / total_trades * 100

final_capital = capital_curve[-1]
total_return = (final_capital - CAPITAL) / CAPITAL * 100

# Sharpe Ratio
returns = np.diff(capital_curve) / capital_curve[:-1]
sharpe = (np.mean(returns) / (np.std(returns) + 1e-9)) * np.sqrt(365 * 24)

# Profit Factor
gross_wins = df_trades[df_trades['Net_PnL'] > 0]['Net_PnL'].sum()
gross_losses = abs(df_trades[df_trades['Net_PnL'] < 0]['Net_PnL'].sum())
profit_factor = gross_wins / gross_losses if gross_losses > 0 else np.inf

# Max Drawdown
peak = np.maximum.accumulate(capital_curve)
drawdown = (peak - capital_curve) / peak * 100
max_dd = drawdown.max()

# Regime breakdown
regime_performance = df_trades.groupby('Regime').agg({
    'Net_PnL': ['count', 'sum', 'mean']
})

print("\n" + "="*70)
print("📊 PhD-LEVEL QUANT STRATEGY BACKTEST RESULTS")
print("="*70)
print(f"📅 Period: {df.index[0].date()} to {df.index[-1].date()}")
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
print(f"📊 Avg Win ($)    : ${df_trades[df_trades['Net_PnL']>0]['Net_PnL'].mean():.2f}" if wins > 0 else "N/A")
print(f"📊 Avg Loss ($)   : ${df_trades[df_trades['Net_PnL']<0]['Net_PnL'].mean():.2f}" if losses > 0 else "N/A")
print(f"📊 Profit Factor  : {profit_factor:.2f}")
print("="*70)

# Regime breakdown
print("\n📊 Performance by Regime:")
print(regime_performance)

# Save trade log
df_trades.to_csv("quant_strategy_trades.csv")
print("\n💾 Trade log saved to 'quant_strategy_trades.csv'")
