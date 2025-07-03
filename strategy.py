#!/usr/bin/env python3
"""
tesla - Momentum, cross_sectional_momentum, trend_following Trading Strategy

Strategy Type: momentum, cross_sectional_momentum, trend_following
Description: tesla
Created: 2025-07-03T14:36:13.816Z

WARNING: This is a template implementation. Thoroughly backtest before live trading.
"""

import os
import sys
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('strategy.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class teslaStrategy:
    """
    tesla Implementation
    
    Strategy Type: momentum, cross_sectional_momentum, trend_following
    Risk Level: Monitor drawdowns and position sizes carefully
    """
    
    def __init__(self, config=None):
        self.config = config or self.get_default_config()
        self.positions = {}
        self.performance_metrics = {}
        logger.info(f"Initialized tesla strategy")
        
    def get_default_config(self):
        """Default configuration parameters"""
        return {
            'max_position_size': 0.05,  # 5% max position size
            'stop_loss_pct': 0.05,      # 5% stop loss
            'lookback_period': 20,       # 20-day lookback
            'rebalance_freq': 'daily',   # Rebalancing frequency
            'transaction_costs': 0.001,  # 0.1% transaction costs
        }
    
    def load_data(self, symbols, start_date, end_date):
        """Load market data for analysis"""
        try:
            import yfinance as yf
            data = yf.download(symbols, start=start_date, end=end_date)
            logger.info(f"Loaded data for {len(symbols)} symbols")
            return data
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return None

# =============================================================================
# USER'S STRATEGY IMPLEMENTATION
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Strategy class definition
class TeslaStrategy:
    def __init__(self, price_data, lookback_mom=20, lookback_trend=50, top_n=2, risk_per_trade=0.01, initial_capital=100000):
        # price_data: DataFrame with columns as tickers, rows as dates, values as prices
        self.price_data = price_data
        self.lookback_mom = lookback_mom
        self.lookback_trend = lookback_trend
        self.top_n = top_n
        self.risk_per_trade = risk_per_trade
        self.initial_capital = initial_capital
        self.positions = pd.DataFrame(index=price_data.index, columns=price_data.columns).fillna(0)
        self.returns = price_data.pct_change().fillna(0)
        self.portfolio_value = pd.Series(index=price_data.index)
        self.signal_log = []

    def generate_signals(self):
        # Momentum signal: past lookback_mom return
        momentum = self.price_data.pct_change(self.lookback_mom)
        # Trend-following: price above moving average
        ma = self.price_data.rolling(self.lookback_trend).mean()
        trend = self.price_data > ma
        # Cross-sectional momentum: rank assets by momentum
        ranks = momentum.rank(axis=1, ascending=False)
        signals = pd.DataFrame(0, index=self.price_data.index, columns=self.price_data.columns)
        for date in self.price_data.index:
            # Select top N assets by momentum
            if date not in ranks.index or date not in trend.index:
                continue
            top_assets = ranks.loc[date].nsmallest(self.top_n).index
            for asset in self.price_data.columns:
                if asset in top_assets and trend.loc[date, asset]:
                    signals.loc[date, asset] = 1
                else:
                    signals.loc[date, asset] = 0
        self.signals = signals
        logging.info("Signals generated")
        return signals

    def position_sizing(self):
        # Equal weight among selected assets, risk managed by volatility
        vol = self.returns.rolling(20).std()
        weights = pd.DataFrame(0, index=self.price_data.index, columns=self.price_data.columns)
        for date in self.signals.index:
            selected = self.signals.loc[date][self.signals.loc[date] == 1].index
            if len(selected) == 0:
                continue
            inv_vol = 1 / (vol.loc[date, selected] + 1e-8)
            inv_vol = inv_vol / inv_vol.sum()
            for asset in selected:
                weights.loc[date, asset] = inv_vol[asset]
        self.weights = weights
        logging.info("Position sizing completed")
        return weights

    def backtest(self):
        # Calculate daily portfolio returns
        portfolio_returns = (self.weights.shift(1) * self.returns).sum(axis=1)
        self.portfolio_value = (1 + portfolio_returns).cumprod() * self.initial_capital
        self.portfolio_returns = portfolio_returns
        logging.info("Backtest completed")
        return self.portfolio_value

    def calculate_performance(self):
        # Sharpe ratio
        sharpe = np.sqrt(252) * self.portfolio_returns.mean() / (self.portfolio_returns.std() + 1e-8)
        # Max drawdown
        cum_returns = (1 + self.portfolio_returns).cumprod()
        running_max = np.maximum.accumulate(cum_returns)
        drawdown = (cum_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        # CAGR
        n_years = (self.portfolio_returns.index[-1] - self.portfolio_returns.index[0]).days / 365.25
        total_return = cum_returns.iloc[-1]
        cagr = total_return ** (1 / n_years) - 1 if n_years > 0 else np.nan
        perf = {
            "Sharpe Ratio": sharpe,
            "Max Drawdown": max_drawdown,
            "CAGR": cagr
        }
        logging.info("Performance calculated: %s" % perf)
        return perf

    def plot_results(self):
        plt.figure(figsize=(12,6))
        plt.plot(self.portfolio_value, label="Portfolio Value")
        plt.title("Tesla Strategy Backtest")
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value")
        plt.legend()
        plt.show()

# Sample data generation
np.random.seed(42)
dates = pd.date_range(start="2020-01-01", end="2023-01-01", freq="B")
tickers = ["TSLA", "AAPL", "MSFT", "GOOG", "AMZN"]
price_data = pd.DataFrame(index=dates, columns=tickers)
for ticker in tickers:
    # Simulate random walk prices
    price = 100 + np.cumsum(np.random.randn(len(dates)) * 2 + 0.1)
    price_data[ticker] = price

# Main execution block
if __name__ == "__main__":
    try:
        # Instantiate strategy
        strategy = TeslaStrategy(price_data, lookback_mom=20, lookback_trend=50, top_n=2, risk_per_trade=0.01, initial_capital=100000)
        # Generate signals
        signals = strategy.generate_signals()
        print("Signals head:")
        print(signals.head())
        # Position sizing
        weights = strategy.position_sizing()
        print("Weights head:")
        print(weights.head())
        # Backtest
        portfolio_value = strategy.backtest()
        print("Portfolio value tail:")
        print(portfolio_value.tail())
        # Performance metrics
        perf = strategy.calculate_performance()
        print("Performance metrics:")
        for k, v in perf.items():
            print("%s: %s" % (k, v))
        # Plot results
        strategy.plot_results()
    except Exception as e:
        logging.error("Error in strategy execution: %s" % str(e))

# =============================================================================
# STRATEGY EXECUTION AND TESTING
# =============================================================================

if __name__ == "__main__":
    # Example usage and testing
    strategy = teslaStrategy()
    print(f"Strategy '{strategyName}' initialized successfully!")
    
    # Example data loading
    symbols = ['SPY', 'QQQ', 'IWM']
    start_date = '2020-01-01'
    end_date = '2023-12-31'
    
    print(f"Loading data for symbols: {symbols}")
    data = strategy.load_data(symbols, start_date, end_date)
    
    if data is not None:
        print(f"Data loaded successfully. Shape: {data.shape}")
        print("Strategy ready for backtesting!")
    else:
        print("Failed to load data. Check your internet connection.")
