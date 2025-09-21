"""
FX Trading Platform with URL Routing
Features:
- URL-based routing: /login, /register, /dashboard
- Dashboard with moving averages and real-time market data
- Trading algorithm visualization (SMA, RSI, Bollinger Bands)
- Strategy comparison and efficiency analysis
"""

import streamlit as st

# MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="FX Trading Platform",
    page_icon="�",
    layout="wide"
)

import pandas as pd
import numpy as np
import requests
from datetime import datetime
import sqlite3
from sqlalchemy import create_engine, text
import plotly.graph_objs as go
import plotly.express as px
import hashlib
import time
import threading
from collections import deque

# Machine Learning imports with fallback
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report, accuracy_score
    ML_AVAILABLE = True
except ImportError:
    print("Warning: scikit-learn not available. ML features will be disabled.")
    ML_AVAILABLE = False
    RandomForestClassifier = None
    train_test_split = None
    StandardScaler = None
    classification_report = None
    accuracy_score = None

import warnings
warnings.filterwarnings('ignore')

# ---------------------------
# Configuration
# ---------------------------
import os
from pathlib import Path

# Create data directory if it doesn't exist
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# Database file path
DB_FILE = DATA_DIR / "trades.db"
SUPPORTED_CURRENCIES = ["USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD", "SEK", "NOK"]

# ---------------------------
# Streaming Data Management
# ---------------------------
class StreamingDataManager:
    """Manages real-time streaming data for FX prices and indicators"""
    
    def __init__(self, max_points=100):
        self.max_points = max_points
        self.price_data = {}
        self.indicator_data = {}
        self.last_update = {}
        
    def initialize_symbol(self, symbol):
        """Initialize data structures for a new symbol"""
        if symbol not in self.price_data:
            self.price_data[symbol] = deque(maxlen=self.max_points)
            self.indicator_data[symbol] = {
                'sma_20': deque(maxlen=self.max_points),
                'sma_50': deque(maxlen=self.max_points),
                'rsi': deque(maxlen=self.max_points),
                'bb_upper': deque(maxlen=self.max_points),
                'bb_lower': deque(maxlen=self.max_points),
                'timestamps': deque(maxlen=self.max_points)
            }
            self.last_update[symbol] = datetime.now()
            
            # Pre-populate with some historical data for better visualization
            self.populate_historical_data(symbol)
            
    def populate_historical_data(self, symbol):
        """Pre-populate with realistic historical data"""
        base_price = 1.2345
        now = datetime.now()
        
        # Generate 60 historical points (1 minute intervals)
        for i in range(60, 0, -1):
            timestamp = now - pd.Timedelta(minutes=i)
            # Generate realistic price movement
            price_change = np.random.normal(0, 0.003)
            trend = np.sin(i / 10) * 0.001  # Add some wave pattern
            price = base_price + price_change + trend + (np.random.random() - 0.5) * 0.01
            price = max(0.5, min(2.0, price))  # Keep in range
            self.add_price_point(symbol, price, timestamp)
            
    def add_price_point(self, symbol, price, timestamp=None):
        """Add a new price point and calculate indicators"""
        if timestamp is None:
            timestamp = datetime.now()
            
        self.initialize_symbol(symbol)
        
        # Add new price point
        self.price_data[symbol].append(price)
        self.indicator_data[symbol]['timestamps'].append(timestamp)
        self.last_update[symbol] = timestamp
        
        # Calculate indicators if we have enough data
        prices = list(self.price_data[symbol])
        if len(prices) >= 20:
            # Simple Moving Averages
            sma_20 = np.mean(prices[-20:])
            self.indicator_data[symbol]['sma_20'].append(sma_20)
            
            if len(prices) >= 50:
                sma_50 = np.mean(prices[-50:])
                self.indicator_data[symbol]['sma_50'].append(sma_50)
            else:
                self.indicator_data[symbol]['sma_50'].append(sma_20)
                
            # RSI calculation
            if len(prices) >= 15:
                rsi_val = self.calculate_rsi(prices[-15:])
                self.indicator_data[symbol]['rsi'].append(rsi_val)
            else:
                self.indicator_data[symbol]['rsi'].append(50)
                
            # Bollinger Bands
            bb_upper, bb_lower = self.calculate_bollinger_bands(prices[-20:])
            self.indicator_data[symbol]['bb_upper'].append(bb_upper)
            self.indicator_data[symbol]['bb_lower'].append(bb_lower)
        else:
            # Not enough data for indicators
            self.indicator_data[symbol]['sma_20'].append(price)
            self.indicator_data[symbol]['sma_50'].append(price)
            self.indicator_data[symbol]['rsi'].append(50)
            self.indicator_data[symbol]['bb_upper'].append(price * 1.02)
            self.indicator_data[symbol]['bb_lower'].append(price * 0.98)
            
    def calculate_rsi(self, prices, periods=14):
        """Calculate RSI indicator"""
        if len(prices) < periods + 1:
            return 50
            
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-periods:])
        avg_loss = np.mean(losses[-periods:])
        
        if avg_loss == 0:
            return 100
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
        
    def calculate_bollinger_bands(self, prices, periods=20, std_mult=2):
        """Calculate Bollinger Bands"""
        if len(prices) < periods:
            mean_price = np.mean(prices)
            return mean_price * 1.02, mean_price * 0.98
            
        mean = np.mean(prices[-periods:])
        std = np.std(prices[-periods:])
        upper = mean + (std * std_mult)
        lower = mean - (std * std_mult)
        return upper, lower
        
    def get_dataframe(self, symbol):
        """Get current data as DataFrame for plotting"""
        if symbol not in self.price_data:
            return pd.DataFrame()
            
        data = {
            'timestamp': list(self.indicator_data[symbol]['timestamps']),
            'price': list(self.price_data[symbol]),
            'sma_20': list(self.indicator_data[symbol]['sma_20']),
            'sma_50': list(self.indicator_data[symbol]['sma_50']),
            'rsi': list(self.indicator_data[symbol]['rsi']),
            'bb_upper': list(self.indicator_data[symbol]['bb_upper']),
            'bb_lower': list(self.indicator_data[symbol]['bb_lower'])
        }
        
        return pd.DataFrame(data)

# Global streaming data manager
if 'streaming_manager' not in st.session_state:
    st.session_state.streaming_manager = StreamingDataManager()

# ---------------------------
# Automated Trading System
# ---------------------------
class AutomatedTradingManager:
    """Manages automated trading for institutional accounts"""
    
    def __init__(self):
        self.is_active = False
        self.trading_capital = 10_000_000  # $10M USD
        self.max_daily_loss = 1_000_000    # $1M USD loss limit
        self.daily_pnl = 0
        self.trade_size_percentage = 0.01   # 1% of capital per trade
        self.max_trades_per_day = 50
        self.trades_today = 0
        self.monitored_pairs = [
            "USD/EUR", "USD/GBP", "USD/JPY", "USD/AUD", "USD/CAD",
            "EUR/GBP", "EUR/JPY", "GBP/JPY", "AUD/USD", "NZD/USD"
        ]
        self.active_signals = {}  # Store signals for each pair
        self.ml_model = None  # Random Forest model
        self.ml_scaler = None  # Feature scaler
        self.ml_enabled = False  # ML trading toggle
        self.ml_accuracy = 0.0  # Model accuracy
        
    def create_training_dataset(self):
        """Create training dataset from historical price data"""
        st.info("🤖 Generating ML training dataset...")
        
        # Generate comprehensive historical data for training
        np.random.seed(42)
        n_samples = 2000  # 2000 data points
        
        data = []
        for i in range(n_samples):
            # Base price with trend
            base_price = 1.2 + (i / n_samples) * 0.3 + np.random.normal(0, 0.02)
            
            # Generate price series around this point
            prices = []
            for j in range(60):  # 60 periods for indicators
                price = base_price + np.random.normal(0, 0.01) + np.sin(j/10) * 0.005
                prices.append(price)
            
            current_price = prices[-1]
            
            # Calculate technical indicators
            sma_20 = np.mean(prices[-20:])
            sma_50 = np.mean(prices[-50:]) if len(prices) >= 50 else sma_20
            
            # RSI calculation
            deltas = np.diff(prices[-15:])
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            avg_gain = np.mean(gains)
            avg_loss = np.mean(losses) if np.mean(losses) > 0 else 0.0001
            rsi = 100 - (100 / (1 + avg_gain / avg_loss))
            
            # Bollinger Bands
            bb_middle = np.mean(prices[-20:])
            bb_std = np.std(prices[-20:])
            bb_upper = bb_middle + (bb_std * 2)
            bb_lower = bb_middle - (bb_std * 2)
            
            # Additional features
            volatility = np.std(prices[-10:])
            price_momentum = (prices[-1] - prices[-5]) / prices[-5]
            volume_proxy = abs(np.random.normal(1000000, 200000))  # Simulated volume
            
            # Future price for labeling (look ahead 5 periods)
            future_prices = []
            for k in range(5):
                future_price = current_price + np.random.normal(0, 0.01)
                future_prices.append(future_price)
            
            future_return = (np.mean(future_prices) - current_price) / current_price
            
            # Create label: 1 for BUY, 0 for HOLD, -1 for SELL
            if future_return > 0.005:  # >0.5% return = BUY
                label = 1
            elif future_return < -0.005:  # <-0.5% return = SELL
                label = -1
            else:
                label = 0  # HOLD
            
            data.append({
                'price': current_price,
                'sma_20': sma_20,
                'sma_50': sma_50,
                'rsi': rsi,
                'bb_upper': bb_upper,
                'bb_lower': bb_lower,
                'bb_middle': bb_middle,
                'volatility': volatility,
                'momentum': price_momentum,
                'volume': volume_proxy,
                'sma_ratio': sma_20 / sma_50,
                'bb_position': (current_price - bb_lower) / (bb_upper - bb_lower),
                'price_to_sma20': current_price / sma_20,
                'price_to_sma50': current_price / sma_50,
                'label': label
            })
        
        df = pd.DataFrame(data)
        st.success(f"✅ Generated {len(df)} training samples")
        return df
    
    def train_ml_model(self):
        """Train Random Forest model for trading decisions"""
        if not ML_AVAILABLE:
            st.error("❌ Machine Learning libraries not available. Please install scikit-learn: pip install scikit-learn")
            return None, None
            
        st.info("🧠 Training Random Forest model...")
        
        # Create training dataset
        df = self.create_training_dataset()
        
        # Prepare features and labels
        feature_columns = [
            'sma_20', 'sma_50', 'rsi', 'bb_upper', 'bb_lower', 'volatility', 
            'momentum', 'volume', 'sma_ratio', 'bb_position', 'price_to_sma20', 'price_to_sma50'
        ]
        
        X = df[feature_columns].values
        y = df['label'].values
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        self.ml_scaler = StandardScaler()
        X_train_scaled = self.ml_scaler.fit_transform(X_train)
        X_test_scaled = self.ml_scaler.transform(X_test)
        
        # Train Random Forest
        self.ml_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        
        self.ml_model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = self.ml_model.predict(X_test_scaled)
        self.ml_accuracy = accuracy_score(y_test, y_pred)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': self.ml_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        st.success(f"🎯 Model trained! Accuracy: {self.ml_accuracy:.2%}")
        
        # Display feature importance
        st.markdown("##### 📊 Feature Importance")
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(feature_importance.head(6), hide_index=True)
        with col2:
            fig_importance = px.line(
                feature_importance.head(6),
                x='importance',
                y='feature',
                title='Top 6 Feature Importance',
                markers=True
            )
            fig_importance.update_layout(
                yaxis={'categoryorder': 'total ascending'}
            )
            st.plotly_chart(fig_importance, use_container_width=True)
        
        # Classification report
        with st.expander("📈 Detailed Model Performance"):
            class_report = classification_report(y_test, y_pred, output_dict=True)
            st.json(class_report)
        
        self.ml_enabled = True
        return self.ml_model, self.ml_accuracy
    
    def generate_ml_trading_signal(self, symbol, stream_df):
        """Generate trading signal using ML model"""
        if self.ml_model is None or len(stream_df) < 50:
            return None, None
            
        latest_idx = len(stream_df) - 1
        
        # Extract features
        current_price = stream_df['price'].iloc[latest_idx]
        sma_20 = stream_df['sma_20'].iloc[latest_idx]
        sma_50 = stream_df['sma_50'].iloc[latest_idx]
        rsi = stream_df['rsi'].iloc[latest_idx]
        bb_upper = stream_df['bb_upper'].iloc[latest_idx]
        bb_lower = stream_df['bb_lower'].iloc[latest_idx]
        bb_middle = (bb_upper + bb_lower) / 2
        
        # Calculate additional features
        prices = list(stream_df['price'].tail(10))
        volatility = np.std(prices)
        momentum = (prices[-1] - prices[-5]) / prices[-5] if len(prices) >= 5 else 0
        volume_proxy = abs(np.random.normal(1000000, 200000))  # Simulated
        
        features = np.array([[
            sma_20, sma_50, rsi, bb_upper, bb_lower, volatility,
            momentum, volume_proxy, sma_20/sma_50, 
            (current_price - bb_lower) / (bb_upper - bb_lower),
            current_price / sma_20, current_price / sma_50
        ]])
        
        # Scale features and predict
        features_scaled = self.ml_scaler.transform(features)
        prediction = self.ml_model.predict(features_scaled)[0]
        prediction_proba = self.ml_model.predict_proba(features_scaled)[0]
        
        # Convert prediction to trading signal
        if prediction == 1:  # BUY
            confidence = prediction_proba[2] if len(prediction_proba) > 2 else prediction_proba[1]
            return {
                'action': 'BUY',
                'price': current_price,
                'strength': confidence,
                'reasons': ['ML Buy Signal', f'Confidence: {confidence:.1%}', 'Random Forest'],
                'model': 'Random Forest'
            }
        elif prediction == -1:  # SELL
            confidence = prediction_proba[0] if len(prediction_proba) > 2 else prediction_proba[0]
            return {
                'action': 'SELL',
                'price': current_price,
                'strength': confidence,
                'reasons': ['ML Sell Signal', f'Confidence: {confidence:.1%}', 'Random Forest'],
                'model': 'Random Forest'
            }
        
        return None  # HOLD
        
    def get_all_currency_rates(self):
        """Get rates for all monitored currency pairs"""
        rates = {}
        base_rates, _ = fetch_live_exchange_rates("USD")
        
        if base_rates:
            for pair in self.monitored_pairs:
                base, quote = pair.split("/")
                if base == "USD" and quote in base_rates:
                    rates[pair] = base_rates[quote]
                elif quote == "USD" and base in base_rates:
                    rates[pair] = 1.0 / base_rates[base] if base_rates[base] != 0 else 1.0
                elif base in base_rates and quote in base_rates:
                    rates[pair] = base_rates[quote] / base_rates[base] if base_rates[base] != 0 else 1.0
                else:
                    # Simulate rate if not available
                    rates[pair] = 1.2345 + np.random.normal(0, 0.01)
        else:
            # Simulate all rates if API fails
            for pair in self.monitored_pairs:
                rates[pair] = 1.2345 + np.random.normal(0, 0.01)
                
        return rates
        
    def monitor_all_currency_pairs(self, user_id):
        """Monitor all currency pairs for trading opportunities"""
        if not self.can_trade():
            return []
            
        executed_trades = []
        currency_rates = self.get_all_currency_rates()
        
        for pair in self.monitored_pairs:
            if pair not in currency_rates:
                continue
            
            # Add current rate to streaming data
            current_rate = currency_rates[pair]
            # Add some volatility for realistic simulation
            current_rate += np.random.normal(0, current_rate * 0.001)
            
            # Initialize streaming data for this pair if needed
            st.session_state.streaming_manager.add_price_point(pair, current_rate)
            
            # Get streaming data for analysis
            stream_df = st.session_state.streaming_manager.get_dataframe(pair)
            
            if not stream_df.empty and len(stream_df) > 20:
                # Prioritize ML signal if model is available and enabled
                signal = None
                if self.ml_enabled and self.ml_model is not None:
                    ml_result = self.generate_ml_trading_signal(pair, stream_df)
                    if ml_result and ml_result.get('action') != "HOLD":
                        # Use ML signal directly (it's already in the correct format)
                        signal = ml_result
                
                # Fallback to rule-based signal if ML not available
                if signal is None:
                    signal = self.generate_trading_signal(pair, stream_df)
                
                if signal and signal['strength'] > 0.65:  # Higher threshold for auto-execution
                    # Check if we haven't traded this pair recently
                    if self.should_execute_signal(pair, signal):
                        trade_size = self.calculate_trade_size()
                        
                        success, order_status = insert_trade(
                            user_id,
                            pair,
                            signal['action'],
                            signal['price'],
                            trade_size,
                            f"AUTO-{pair}: " + ", ".join(signal['reasons'][:2])  # Limit reason length
                        )
                        
                        if success and order_status == "Filled":
                            executed_trades.append({
                                'pair': pair,
                                'action': signal['action'],
                                'price': signal['price'],
                                'size': trade_size,
                                'reasons': signal['reasons'],
                                'strength': signal['strength']
                            })
                            self.trades_today += 1
                            
                            # Store signal to avoid immediate re-trading
                            self.active_signals[pair] = {
                                'timestamp': datetime.now(),
                                'action': signal['action'],
                                'price': signal['price']
                            }
        
        return executed_trades
    
    def should_execute_signal(self, pair, signal):
        """Check if we should execute this signal (avoid over-trading same pair)"""
        if pair not in self.active_signals:
            return True
            
        last_signal = self.active_signals[pair]
        time_since_last = datetime.now() - last_signal['timestamp']
        
        # Don't trade same pair within 5 minutes
        if time_since_last.total_seconds() < 300:
            return False
            
        # Don't reverse position too quickly
        if last_signal['action'] != signal['action']:
            price_change = abs(signal['price'] - last_signal['price']) / last_signal['price']
            if price_change < 0.002:  # Less than 0.2% price change
                return False
                
        return True
        
    def calculate_daily_pnl(self, user_id):
        """Calculate today's P&L from trades"""
        today = datetime.now().strftime('%Y-%m-%d')
        engine = get_db_engine()
        
        try:
            with engine.connect() as conn:
                result = conn.execute(text("""
                SELECT SUM(
                    CASE 
                        WHEN side = 'BUY' THEN -price * quantity 
                        WHEN side = 'SELL' THEN price * quantity 
                        ELSE 0 
                    END
                ) as daily_pnl,
                COUNT(*) as trade_count
                FROM trades 
                WHERE user_id = :user_id 
                AND DATE(timestamp) = :today
                AND status = 'Filled'
                """), {"user_id": user_id, "today": today})
                
                row = result.fetchone()
                if row and row[0] is not None:
                    self.daily_pnl = float(row[0])
                    self.trades_today = int(row[1])
                else:
                    self.daily_pnl = 0
                    self.trades_today = 0
                    
        except Exception as e:
            st.error(f"Error calculating daily P&L: {e}")
            self.daily_pnl = 0
            self.trades_today = 0
            
        return self.daily_pnl, self.trades_today
    
    def should_stop_trading(self):
        """Check if trading should be stopped due to loss limits"""
        return self.daily_pnl <= -self.max_daily_loss
    
    def can_trade(self):
        """Check if automated trading is allowed"""
        return (self.is_active and 
                not self.should_stop_trading() and 
                self.trades_today < self.max_trades_per_day)
    
    def calculate_trade_size(self):
        """Calculate trade size based on capital and risk management"""
        base_size = self.trading_capital * self.trade_size_percentage
        
        # Reduce trade size if we're in loss
        if self.daily_pnl < 0:
            loss_factor = min(abs(self.daily_pnl) / self.max_daily_loss, 0.8)
            base_size *= (1 - loss_factor)
        
        return max(100, int(base_size / 100) * 100)  # Round to nearest 100
    
    def generate_trading_signal(self, symbol, stream_df):
        """Generate automated trading signals based on multiple indicators"""
        if len(stream_df) < 20:
            return None
            
        latest_idx = len(stream_df) - 1
        prev_idx = latest_idx - 1
        
        current_price = stream_df['price'].iloc[latest_idx]
        sma_20 = stream_df['sma_20'].iloc[latest_idx]
        sma_50 = stream_df['sma_50'].iloc[latest_idx]
        rsi = stream_df['rsi'].iloc[latest_idx]
        bb_upper = stream_df['bb_upper'].iloc[latest_idx]
        bb_lower = stream_df['bb_lower'].iloc[latest_idx]
        
        # Previous values for crossover detection
        prev_sma_20 = stream_df['sma_20'].iloc[prev_idx]
        prev_sma_50 = stream_df['sma_50'].iloc[prev_idx]
        prev_price = stream_df['price'].iloc[prev_idx]
        
        signals = []
        
        # SMA Crossover Signal
        if sma_20 > sma_50 and prev_sma_20 <= prev_sma_50:
            signals.append({'type': 'BUY', 'reason': 'SMA Crossover', 'strength': 0.7})
        elif sma_20 < sma_50 and prev_sma_20 >= prev_sma_50:
            signals.append({'type': 'SELL', 'reason': 'SMA Crossover', 'strength': 0.7})
        
        # RSI Signal
        if rsi < 25:  # More aggressive than manual (30)
            signals.append({'type': 'BUY', 'reason': 'RSI Oversold', 'strength': 0.8})
        elif rsi > 75:  # More aggressive than manual (70)
            signals.append({'type': 'SELL', 'reason': 'RSI Overbought', 'strength': 0.8})
        
        # Bollinger Bands Signal
        if current_price <= bb_lower and prev_price > bb_lower:
            signals.append({'type': 'BUY', 'reason': 'BB Lower Touch', 'strength': 0.6})
        elif current_price >= bb_upper and prev_price < bb_upper:
            signals.append({'type': 'SELL', 'reason': 'BB Upper Touch', 'strength': 0.6})
        
        # Combine signals for final decision
        if len(signals) >= 2:  # Require multiple confirmations
            buy_signals = [s for s in signals if s['type'] == 'BUY']
            sell_signals = [s for s in signals if s['type'] == 'SELL']
            
            if len(buy_signals) >= 2:
                avg_strength = sum(s['strength'] for s in buy_signals) / len(buy_signals)
                return {
                    'action': 'BUY',
                    'price': current_price,
                    'strength': avg_strength,
                    'reasons': [s['reason'] for s in buy_signals]
                }
            elif len(sell_signals) >= 2:
                avg_strength = sum(s['strength'] for s in sell_signals) / len(sell_signals)
                return {
                    'action': 'SELL',
                    'price': current_price,
                    'strength': avg_strength,
                    'reasons': [s['reason'] for s in sell_signals]
                }
        
        return None

# Global automated trading manager for institutions
if 'auto_trading_manager' not in st.session_state:
    st.session_state.auto_trading_manager = AutomatedTradingManager()

# ---------------------------
# Database Functions
# ---------------------------
def get_db_engine():
    """Create database engine with proper error handling"""
    try:
        # Ensure the directory exists
        DB_FILE.parent.mkdir(exist_ok=True)
        
        # Create engine with absolute path
        db_url = f"sqlite:///{DB_FILE.absolute()}"
        engine = create_engine(db_url, echo=False)
        
        # Test connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            
        return engine
    except Exception as e:
        st.error(f"Database connection error: {e}")
        st.error(f"Trying to connect to: {DB_FILE.absolute()}")
        # Fallback to in-memory database
        st.warning("Falling back to temporary in-memory database")
        return create_engine("sqlite:///:memory:", echo=False)

def init_db():
    """Initialize the database with users and trades tables"""
    try:
        engine = get_db_engine()
        with engine.connect() as conn:
            # Create users table
            conn.execute(text("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                user_type TEXT NOT NULL DEFAULT 'user'
            )
            """))
            
            # Check if trades table exists and get its structure
            result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table' AND name='trades'"))
            table_exists = result.fetchone() is not None
            
            if table_exists:
                # Check if user_id column exists in trades table
                try:
                    result = conn.execute(text("PRAGMA table_info(trades)"))
                    columns = [row[1] for row in result.fetchall()]
                
                    if 'user_id' not in columns:
                        # Need to migrate: add user_id column
                        try:
                            conn.execute(text("ALTER TABLE trades ADD COLUMN user_id INTEGER DEFAULT 1"))
                            conn.commit()
                            st.info("Database migrated: Added user_id column")
                        except Exception as migrate_error:
                            # If ALTER fails, recreate table
                            st.warning("Recreating trades table with proper schema...")
                            conn.execute(text("DROP TABLE IF EXISTS trades"))
                            conn.execute(text("""
                            CREATE TABLE trades (
                                id INTEGER PRIMARY KEY AUTOINCREMENT,
                                user_id INTEGER NOT NULL DEFAULT 1,
                                timestamp TEXT NOT NULL,
                                symbol TEXT NOT NULL,
                                side TEXT NOT NULL,
                                price REAL NOT NULL,
                                quantity REAL NOT NULL,
                                status TEXT NOT NULL,
                                strategy TEXT NOT NULL,
                                pnl REAL DEFAULT 0.0,
                                FOREIGN KEY (user_id) REFERENCES users (id)
                            )
                            """))
                            st.info("Trades table recreated with proper schema")
                except Exception as schema_error:
                    st.error(f"Error checking database schema: {schema_error}")
            else:
                # Create new trades table with proper schema
                conn.execute(text("""
                CREATE TABLE trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL DEFAULT 1,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    price REAL NOT NULL,
                    quantity REAL NOT NULL,
                    status TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    pnl REAL DEFAULT 0.0,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
                """))
                st.info("Created trades table with proper schema")
            
            # Create default users
            admin_password = hash_password("admin123")
            conn.execute(text("""
            INSERT OR IGNORE INTO users (username, password_hash, user_type)
            VALUES ('admin', :password, 'institution')
            """), {"password": admin_password})
            
            user_password = hash_password("karthi123")
            conn.execute(text("""
            INSERT OR IGNORE INTO users (username, password_hash, user_type)
            VALUES ('karthi', :password, 'user')
            """), {"password": user_password})
            
            conn.commit()
            return True
            
    except Exception as e:
        st.error(f"Failed to initialize database: {e}")
        st.error(f"Database path: {DB_FILE.absolute()}")
        return False

def hash_password(password):
    """Hash password with salt"""
    salt = "fx_trading_salt"
    return hashlib.sha256((password + salt).encode()).hexdigest()

def authenticate_user(username, password):
    """Authenticate user and return user info"""
    engine = get_db_engine()
    password_hash = hash_password(password)
    
    with engine.connect() as conn:
        result = conn.execute(text("""
        SELECT id, username, user_type FROM users 
        WHERE username = :username AND password_hash = :password
        """), {"username": username, "password": password_hash})
        
        user = result.fetchone()
        if user:
            return {
                'id': user[0],
                'username': user[1],
                'user_type': user[2]
            }
    return None

def create_user(username, password, user_type):
    """Create a new user"""
    engine = get_db_engine()
    password_hash = hash_password(password)
    
    try:
        with engine.connect() as conn:
            conn.execute(text("""
            INSERT INTO users (username, password_hash, user_type)
            VALUES (:username, :password, :user_type)
            """), {"username": username, "password": password_hash, "user_type": user_type})
            conn.commit()
            return True
    except Exception:
        return False

# ---------------------------
# FX Data Functions
# ---------------------------
def fetch_live_exchange_rates(base_currency="USD"):
    """Fetch live exchange rates from API"""
    try:
        url = f"https://api.exchangerate-api.com/v4/latest/{base_currency}"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            return data.get("rates", {}), data.get("date", "")
    except Exception as e:
        st.error(f"API Error: {e}")
    return None, None

def get_currency_pair_rate(base, quote, rates):
    """Get exchange rate for currency pair"""
    if base == quote:
        return 1.0
    if quote in rates:
        return rates[quote]
    return None

def generate_price_series(symbol, start_price=1.0, days=30):
    """Generate realistic price series with indicators"""
    np.random.seed(42)
    n_points = days * 24  # Hourly data
    
    # Generate price using geometric Brownian motion
    dt = 1/24  # 1 hour
    mu = 0.0001  # drift
    sigma = 0.02  # volatility
    
    prices = [start_price]
    for i in range(n_points-1):
        drift = mu * dt
        shock = sigma * np.sqrt(dt) * np.random.normal()
        price = prices[-1] * np.exp(drift + shock)
        prices.append(price)
    
    # Create DataFrame
    dates = pd.date_range(start=datetime.now() - pd.Timedelta(days=days), periods=n_points, freq='H')
    df = pd.DataFrame({'timestamp': dates, 'price': prices})
    df.set_index('timestamp', inplace=True)
    
    # Add technical indicators
    df['sma_20'] = df['price'].rolling(window=20).mean()
    df['sma_50'] = df['price'].rolling(window=50).mean()
    
    # RSI
    delta = df['price'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['bb_middle'] = df['price'].rolling(window=20).mean()
    bb_std = df['price'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    
    return df

# ---------------------------
# Trading Strategy Functions
# ---------------------------
def generate_trading_signals(df, strategy='SMA'):
    """Generate trading signals based on strategy"""
    signals = []
    
    if strategy == 'SMA':
        # SMA Crossover
        for i in range(1, len(df)):
            if df['sma_20'].iloc[i] > df['sma_50'].iloc[i] and df['sma_20'].iloc[i-1] <= df['sma_50'].iloc[i-1]:
                signals.append({'timestamp': df.index[i], 'signal': 'BUY', 'price': df['price'].iloc[i]})
            elif df['sma_20'].iloc[i] < df['sma_50'].iloc[i] and df['sma_20'].iloc[i-1] >= df['sma_50'].iloc[i-1]:
                signals.append({'timestamp': df.index[i], 'signal': 'SELL', 'price': df['price'].iloc[i]})
    
    elif strategy == 'RSI':
        # RSI Overbought/Oversold
        for i in range(1, len(df)):
            if df['rsi'].iloc[i] < 30 and df['rsi'].iloc[i-1] >= 30:
                signals.append({'timestamp': df.index[i], 'signal': 'BUY', 'price': df['price'].iloc[i]})
            elif df['rsi'].iloc[i] > 70 and df['rsi'].iloc[i-1] <= 70:
                signals.append({'timestamp': df.index[i], 'signal': 'SELL', 'price': df['price'].iloc[i]})
    
    elif strategy == 'Bollinger':
        # Bollinger Bands Mean Reversion
        for i in range(1, len(df)):
            if df['price'].iloc[i] <= df['bb_lower'].iloc[i] and df['price'].iloc[i-1] > df['bb_lower'].iloc[i-1]:
                signals.append({'timestamp': df.index[i], 'signal': 'BUY', 'price': df['price'].iloc[i]})
            elif df['price'].iloc[i] >= df['bb_upper'].iloc[i] and df['price'].iloc[i-1] < df['bb_upper'].iloc[i-1]:
                signals.append({'timestamp': df.index[i], 'signal': 'SELL', 'price': df['price'].iloc[i]})
    
    return signals

def calculate_strategy_performance(signals, df):
    """Calculate strategy performance metrics"""
    if not signals:
        return {'total_trades': 0, 'winning_trades': 0, 'total_pnl': 0, 'win_rate': 0}
    
    trades = []
    position = None
    
    for signal in signals:
        if signal['signal'] == 'BUY' and position is None:
            position = {'entry_price': signal['price'], 'entry_time': signal['timestamp']}
        elif signal['signal'] == 'SELL' and position is not None:
            pnl = signal['price'] - position['entry_price']
            trades.append(pnl)
            position = None
    
    if not trades:
        return {'total_trades': 0, 'winning_trades': 0, 'total_pnl': 0, 'win_rate': 0}
    
    total_trades = len(trades)
    winning_trades = len([t for t in trades if t > 0])
    total_pnl = sum(trades)
    win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
    
    return {
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'total_pnl': total_pnl,
        'win_rate': win_rate
    }

# ---------------------------
# Trade Database Functions
# ---------------------------
def insert_trade(user_id, symbol, side, price, quantity, strategy):
    """Insert a trade into database"""
    engine = get_db_engine()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Simulate realistic order processing
    import random
    
    # 95% success rate for market orders
    if random.random() < 0.95:
        status = "Filled"
    
    
    try:
        with engine.connect() as conn:
            # Debug: Check if user_id exists
            
            conn.execute(text("""
            INSERT INTO trades (user_id, timestamp, symbol, side, price, quantity, status, strategy)
            VALUES (:user_id, :timestamp, :symbol, :side, :price, :quantity, :status, :strategy)
            """), {
                "user_id": user_id,
                "timestamp": timestamp,
                "symbol": symbol,
                "side": side,
                "price": price,
                "quantity": quantity,
                "status": status,
                "strategy": strategy
            })
            conn.commit()
            
            # Verify insertion
            result = conn.execute(text("SELECT COUNT(*) FROM trades WHERE user_id = :user_id"), {"user_id": user_id})
            count = result.fetchone()[0]
            
            return True, status
    except Exception as e:
        st.error(f"Error inserting trade: {e}")
        st.write(f"DEBUG: Failed to insert trade for user_id: {user_id}")
        return False, "Error"

def get_user_position(user_id, symbol):
    """Get current position for a user in a specific symbol"""
    engine = get_db_engine()
    try:
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT side, SUM(quantity) as total_quantity 
                FROM trades 
                WHERE user_id = :user_id AND symbol = :symbol AND status = 'Filled'
                GROUP BY side
            """), {"user_id": user_id, "symbol": symbol})
            
            positions = result.fetchall()
            buy_quantity = 0
            sell_quantity = 0
            
            for row in positions:
                if row[0] == 'BUY':
                    buy_quantity = row[1]
                elif row[0] == 'SELL':
                    sell_quantity = row[1]
            
            # Net position = buys - sells
            net_position = buy_quantity - sell_quantity
            return net_position, buy_quantity, sell_quantity
    except Exception as e:
        st.error(f"Error getting position: {e}")
        return 0, 0, 0

def get_user_balance(user_id):
    """Calculate user's available balance based on trades"""
    engine = get_db_engine()
    try:
        with engine.connect() as conn:
            # Get all filled trades to calculate P&L
            result = conn.execute(text("""
                SELECT symbol, side, price, quantity, timestamp
                FROM trades 
                WHERE user_id = :user_id AND status = 'Filled'
                ORDER BY timestamp
            """), {"user_id": user_id})
            
            trades = result.fetchall()
            
            # Start with initial balance
            initial_balance = 10000  # Default starting balance
            
            # Calculate realized P&L from closed positions
            positions = {}  # symbol -> list of open positions
            realized_pnl = 0
            
            for trade in trades:
                symbol, side, price, quantity, timestamp = trade
                
                if symbol not in positions:
                    positions[symbol] = []
                
                if side == 'BUY':
                    positions[symbol].append({'price': price, 'quantity': quantity})
                elif side == 'SELL' and positions[symbol]:
                    # Close positions FIFO
                    remaining_sell = quantity
                    while remaining_sell > 0 and positions[symbol]:
                        open_pos = positions[symbol][0]
                        close_qty = min(remaining_sell, open_pos['quantity'])
                        
                        # Calculate P&L
                        pnl = (price - open_pos['price']) * close_qty
                        realized_pnl += pnl
                        
                        # Update positions
                        open_pos['quantity'] -= close_qty
                        remaining_sell -= close_qty
                        
                        if open_pos['quantity'] == 0:
                            positions[symbol].pop(0)
            
            # Calculate margin used by open positions (Forex uses leverage)
            margin_used = 0
            for symbol, open_positions in positions.items():
                for pos in open_positions:
                    # Use 1% margin requirement (100:1 leverage) instead of 100%
                    position_margin = pos['price'] * pos['quantity'] * 0.01
                    margin_used += position_margin
            
            available_balance = initial_balance + realized_pnl - margin_used
            return max(0, available_balance), realized_pnl, margin_used
            
    except Exception as e:
        st.error(f"Error calculating balance: {e}")
        return 10000, 0, 0

def validate_trade(user_id, symbol, side, price, quantity):
    """Validate if a trade can be executed - Forex trading allows short positions"""
    
    # Get current position and balance
    net_position, buy_qty, sell_qty = get_user_position(user_id, symbol)
    available_balance, realized_pnl, margin_used = get_user_balance(user_id)
    
    # Calculate required margin for the trade
    required_margin = price * quantity * 0.01  # 1% margin requirement for Forex
    
    if side == 'BUY':
        # For BUY orders, check available balance for margin
        if required_margin > available_balance:
            return False, f"Insufficient margin. Required: ${required_margin:.2f}, Available: ${available_balance:.2f}"
            
    elif side == 'SELL':
        # For SELL orders in Forex, we allow short selling (creating negative positions)
        # Check if user has enough margin for the position
        if required_margin > available_balance:
            return False, f"Insufficient margin for short position. Required: ${required_margin:.2f}, Available: ${available_balance:.2f}"
        
        # Optional: Set maximum short position limits (e.g., -100,000 units max)
        max_short_position = -100000
        new_position = net_position - quantity
        if new_position < max_short_position:
            return False, f"Maximum short position exceeded. Current: {net_position:,.0f}, After trade: {new_position:,.0f}, Limit: {max_short_position:,.0f}"
    
    return True, "Trade validated"

def check_db_health():
    """Check database connection and table structure"""
    try:
        engine = get_db_engine()
        with engine.connect() as conn:
            # Test basic connection
            conn.execute(text("SELECT 1"))
            
            # Check tables exist
            result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
            tables = [row[0] for row in result.fetchall()]
            
            if 'users' not in tables or 'trades' not in tables:
                return False, "Missing required tables"
                
            # Check trades table structure
            result = conn.execute(text("PRAGMA table_info(trades)"))
            columns = [row[1] for row in result.fetchall()]
            required_columns = ['user_id', 'timestamp', 'symbol', 'side', 'price', 'quantity', 'status', 'strategy']
            
            for col in required_columns:
                if col not in columns:
                    return False, f"Missing column: {col}"
            
            return True, "Database healthy"
            
    except Exception as e:
        return False, f"Database error: {e}"

def insert_trade(user_id, symbol, side, price, quantity, strategy):
    """Insert a trade into database with validation and error handling"""
    
    # Check database health first
    db_healthy, health_message = check_db_health()
    if not db_healthy:
        st.error(f"Database health check failed: {health_message}")
        st.error(f"Database location: {DB_FILE.absolute()}")
        return False, f"DATABASE ERROR: {health_message}"
    
    # Validate trade
    is_valid, message = validate_trade(user_id, symbol, side, price, quantity)
    if not is_valid:
        return False, f"REJECTED: {message}"
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Simulate realistic order processing
    import random
    
    # 95% success rate for market orders (after validation)
    if random.random() < 0.95:
        status = "Filled"
    else:
        status = random.choice(["Rejected", "Cancelled"])
    
    try:
        engine = get_db_engine()
        with engine.connect() as conn:
            # Insert trade with explicit error handling
            conn.execute(text("""
            INSERT INTO trades (user_id, timestamp, symbol, side, price, quantity, status, strategy)
            VALUES (:user_id, :timestamp, :symbol, :side, :price, :quantity, :status, :strategy)
            """), {
                "user_id": user_id,
                "timestamp": timestamp,
                "symbol": symbol,
                "side": side,
                "price": price,
                "quantity": quantity,
                "status": status,
                "strategy": strategy
            })
            conn.commit()
            
            return True, status
            
    except Exception as e:
        error_msg = f"Database insert failed: {str(e)}"
        st.error(error_msg)
        st.error(f"Database location: {DB_FILE.absolute()}")
        return False, f"DATABASE_ERROR: {str(e)}"

def load_trades(user_id):
    """Load trades for a specific user"""
    engine = get_db_engine()
    try:
        # Debug info
        with engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM trades WHERE user_id = :user_id"), {"user_id": user_id})
            count = result.fetchone()[0]
            st.write(f"DEBUG: Loading trades for user_id {user_id}, found {count} trades")
        
        # Use SQLAlchemy text() for the query instead of raw SQL
        with engine.connect() as conn:
            result = conn.execute(text("""
            SELECT * FROM trades WHERE user_id = :user_id ORDER BY timestamp DESC
            """), {"user_id": user_id})
            
            # Convert to DataFrame manually
            rows = result.fetchall()
            if rows:
                columns = result.keys()
                df = pd.DataFrame(rows, columns=columns)
                return df
            else:
                return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading trades: {e}")
        return pd.DataFrame()

# ---------------------------
# URL Routing Functions
# ---------------------------
def get_current_route():
    """Get current route from URL parameters"""
    query_params = st.query_params
    return query_params.get("page", "login")

def navigate_to(page):
    """Navigate to a specific page"""
    st.query_params["page"] = page

# ---------------------------
# Page Functions
# ---------------------------
def login_page():
    """Login page"""
    st.title("FX Trading Platform - Login")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.form("login_form", clear_on_submit=False):
            st.markdown("### Login to Your Account")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login", use_container_width=True)
            
            if submitted:
                if not username or not password:
                    st.error("Please fill in all fields")
                else:
                    user = authenticate_user(username, password)
                    if user:
                        st.session_state.logged_in = True
                        st.session_state.user_id = user['id']
                        st.session_state.username = user['username']
                        st.session_state.user_type = user['user_type']
                        navigate_to("dashboard")
                        st.success("Login successful! Redirecting...")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("Invalid username or password")
        
        st.markdown("---")
        if st.button("Don't have an account? Register", use_container_width=True):
            navigate_to("register")
            st.rerun()

def register_page():
    """Registration page"""
    st.title("FX Trading Platform - Register")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.form("register_form", clear_on_submit=False):
            st.markdown("### Create New Account")
            new_username = st.text_input("Username")
            new_password = st.text_input("Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            account_type = st.selectbox("Account Type", ["user", "institution"])
            submitted = st.form_submit_button("Create Account", use_container_width=True)
            
            if submitted:
                if not new_username or not new_password:
                    st.error("Please fill in all fields")
                elif new_password != confirm_password:
                    st.error("Passwords don't match")
                elif len(new_password) < 6:
                    st.error("Password must be at least 6 characters")
                elif create_user(new_username, new_password, account_type):
                    st.success("Account created successfully!")
                    st.info("Please go to login page to sign in")
                else:
                    st.error("Username already exists or registration failed")
        
        st.markdown("---")
        if st.button("Already have an account? Login", use_container_width=True):
            navigate_to("login")
            st.rerun()

def dashboard_page():
    """Main dashboard page"""
    # Header with user info
    st.title("FX Trading Dashboard")
    
    col1, col2, col3 = st.columns([3, 3, 2])
    with col1:
        st.info(f"**Welcome, {st.session_state.username}**")
    with col2:
        st.info(f"**Account Type:** {st.session_state.user_type.title()}")
    with col3:
        if st.button("Logout", type="secondary"):
            for key in ['logged_in', 'user_id', 'username', 'user_type']:
                if key in st.session_state:
                    del st.session_state[key]
            navigate_to("login")
            st.rerun()
    
    st.markdown("---")
    
    # Main dashboard tabs
    tab1, tab2, tab3 = st.tabs(["Live Trading", "Strategy Analysis", "Strategy Comparison"])
    
    with tab1:
        live_trading_section()
    
    with tab2:
        strategy_analysis_section()
    
    with tab3:
        strategy_comparison_section()

def live_trading_section():
    """Live trading section with real-time streaming data"""
    st.subheader("LIVE FX Trading - Real-Time Streaming")
    
    # Manual refresh control
    col_refresh, col_status, col_refresh_btn = st.columns([2, 1, 1])
    with col_refresh:
        st.info("� Live data updates on page refresh")
    with col_status:
        st.metric("Status", "LIVE", delta="Manual")
    with col_refresh_btn:
        if st.button("🔄 Refresh Data"):
            st.rerun()
    
    col1, col2 = st.columns([1, 2])
    
    # Check if user is institution for automated trading
    is_institution = st.session_state.get('user_type', '') == 'institution'
    
    with col1:
        st.markdown("#### Live Trading Controls")
        
        # Institutional Automated Trading Section
        if is_institution:
            st.markdown("---")
            st.markdown("#### 🏦 Institutional Automated Trading")
            
            # Calculate current daily P&L
            auto_manager = st.session_state.auto_trading_manager
            daily_pnl, trades_today = auto_manager.calculate_daily_pnl(st.session_state.user_id)
            
            # Risk Status Display
            col_pnl, col_trades = st.columns(2)
            with col_pnl:
                pnl_color = "normal" if daily_pnl >= 0 else ("inverse" if daily_pnl > -500000 else "off")
                st.metric("Daily P&L", f"${daily_pnl:,.0f}", delta=pnl_color)
            with col_trades:
                st.metric("Trades Today", f"{trades_today}/{auto_manager.max_trades_per_day}")
            
            # Trading Status
            if auto_manager.should_stop_trading():
                st.error("� AUTOMATED TRADING HALTED - Daily loss limit exceeded!")
                st.warning("Trading switched to MANUAL mode for risk management")
                auto_manager.is_active = False
                
            # Database Health Check (for troubleshooting deployment issues)
            with st.expander("🔧 Database Diagnostics", expanded=False):
                db_healthy, db_message = check_db_health()
                if db_healthy:
                    st.success(f"✅ {db_message}")
                else:
                    st.error(f"❌ {db_message}")
                    st.error("⚠️ Database issues may prevent trade recording")
                
                # Show database details
                st.info(f"📁 **Database Location:** `{DB_FILE.absolute()}`")
                
                try:
                    if DB_FILE.exists():
                        size_bytes = DB_FILE.stat().st_size
                        size_mb = size_bytes / (1024 * 1024)
                        st.success(f"📊 **Database Size:** {size_mb:.2f} MB")
                        
                        # Test database connection
                        engine = get_db_engine()
                        with engine.connect() as conn:
                            # Count total trades
                            result = conn.execute(text("SELECT COUNT(*) FROM trades"))
                            total_trades = result.fetchone()[0]
                            st.info(f"📈 **Total Trades Recorded:** {total_trades:,}")
                            
                            # Count today's trades  
                            today = datetime.now().strftime("%Y-%m-%d")
                            result = conn.execute(text("SELECT COUNT(*) FROM trades WHERE DATE(timestamp) = ?"), (today,))
                            today_trades = result.fetchone()[0]
                            st.info(f"🗓️ **Today's Trades:** {today_trades}")
                            
                    else:
                        st.warning("⚠️ Database file not found - will be created on first trade")
                        
                except Exception as e:
                    st.error(f"❌ Database Error: {str(e)}")
                    st.error("This may cause trade recording failures. Check logs for details.")
            
            st.markdown("---")
            col_toggle, col_capital = st.columns(2)
            with col_toggle:
                auto_enabled = st.checkbox(
                    "Enable Auto Trading",
                    value=auto_manager.is_active and not auto_manager.should_stop_trading(),
                    disabled=auto_manager.should_stop_trading()
                )
                auto_manager.is_active = auto_enabled
                
            with col_capital:
                st.metric("Trading Capital", f"${auto_manager.trading_capital:,.0f}")
            
            # Auto Trading Settings
            if auto_enabled:
                st.markdown("**⚙️ Auto Trading Settings**")
                
                # Automatically train ML model if not already trained
                if auto_manager.ml_model is None and ML_AVAILABLE:
                    with st.spinner("🧠 Initializing ML trading model..."):
                        auto_manager.train_ml_model()
                    st.success("🤖 ML Model Ready - Auto-training complete!")
                
                # Enable ML by default when auto trading is active
                auto_manager.ml_enabled = True if auto_manager.ml_model is not None else False
                
                # Show ML model status
                if auto_manager.ml_model is not None:
                    st.success(f"🤖 ML-Powered Trading Active - Accuracy: {auto_manager.ml_accuracy:.1%}")
                else:
                    if not ML_AVAILABLE:
                        st.warning("⚠️ ML libraries not available. Using rule-based trading.")
                    else:
                        st.info("🔄 Preparing ML trading model...")
                
                # Trading parameters
                trade_size_pct = st.slider("Trade Size (%)", 0.1, 2.0, auto_manager.trade_size_percentage * 100, 0.1) / 100
                auto_manager.trade_size_percentage = trade_size_pct
                
                max_trades = st.slider("Max Trades/Day", 10, 100, auto_manager.max_trades_per_day, 5)
                auto_manager.max_trades_per_day = max_trades
                
                # Calculate current trade size
                trade_size = auto_manager.calculate_trade_size()
                st.info(f"💰 Current Trade Size: ${trade_size:,}")
            
            st.markdown("---")
        
        # Currency pair selection
        base_currency = st.selectbox("Base Currency", SUPPORTED_CURRENCIES, index=0, key="live_base")
        quote_currency = st.selectbox("Quote Currency", SUPPORTED_CURRENCIES, index=1, key="live_quote")
        symbol = f"{base_currency}/{quote_currency}"
        
        # Get live rate and stream it
        rates, rate_date = fetch_live_exchange_rates(base_currency)
        current_rate = None
        if rates:
            current_rate = get_currency_pair_rate(base_currency, quote_currency, rates)
            if current_rate:
                # Add some realistic volatility to live rates for better visualization
                volatility = np.random.normal(0, current_rate * 0.001)  # 0.1% volatility
                current_rate = current_rate + volatility
                
                # Add to streaming data
                st.session_state.streaming_manager.add_price_point(symbol, current_rate)
                st.success(f"🟢 LIVE Rate: **{current_rate:.4f}**")
                st.caption(f"Updated: {rate_date}")
            else:
                st.error("Currency pair not available")
        else:
            st.warning("🟡 Using simulated data")
            # Generate more realistic simulated streaming data
            if 'last_simulated_price' not in st.session_state:
                st.session_state.last_simulated_price = 1.2345
            
            # Create more volatility for visible price movements
            price_change = np.random.normal(0, 0.005)  # Increased volatility
            trend = np.random.choice([-0.002, 0.002], p=[0.3, 0.7])  # Add trending
            current_rate = st.session_state.last_simulated_price + price_change + trend
            
            # Keep price in reasonable range
            current_rate = max(0.5, min(2.0, current_rate))
            st.session_state.last_simulated_price = current_rate
            
            st.session_state.streaming_manager.add_price_point(symbol, current_rate)
        
        st.markdown("---")
        
        # Live Price Display
        if current_rate:
            st.metric("💱 Current Price", f"{current_rate:.5f}", delta=f"{np.random.uniform(-0.002, 0.002):.5f}")
            
            # Get streaming data for indicators
            stream_df = st.session_state.streaming_manager.get_dataframe(symbol)
            if not stream_df.empty and len(stream_df) > 0:
                latest_idx = len(stream_df) - 1
                
                # Live Moving Averages
                st.markdown("**📈 Live Moving Averages**")
                col_sma1, col_sma2 = st.columns(2)
                with col_sma1:
                    sma_20_val = stream_df['sma_20'].iloc[latest_idx]
                    st.metric("SMA 20", f"{sma_20_val:.5f}")
                with col_sma2:
                    sma_50_val = stream_df['sma_50'].iloc[latest_idx]
                    st.metric("SMA 50", f"{sma_50_val:.5f}")
                
                # Live RSI
                rsi_val = stream_df['rsi'].iloc[latest_idx]
                st.metric("🎯 RSI (14)", f"{rsi_val:.1f}", 
                         delta="Overbought" if rsi_val > 70 else ("Oversold" if rsi_val < 30 else "Neutral"))
                
                # Live Bollinger Bands
                st.markdown("**📊 Live Bollinger Bands**")
                col_bb1, col_bb2 = st.columns(2)
                with col_bb1:
                    bb_upper_val = stream_df['bb_upper'].iloc[latest_idx]
                    st.metric("BB Upper", f"{bb_upper_val:.5f}")
                with col_bb2:
                    bb_lower_val = stream_df['bb_lower'].iloc[latest_idx]
                    st.metric("BB Lower", f"{bb_lower_val:.5f}")
        
        st.markdown("---")
        
        # Automated Trading Signal Detection (for institutions)
        if is_institution and auto_manager.is_active and auto_manager.can_trade():
            st.markdown("#### 🤖 Multi-Currency Automated Trading")
            
            # Execute automated trading across all pairs
            executed_trades = auto_manager.monitor_all_currency_pairs(st.session_state.user_id)
            
            if executed_trades:
                st.success(f"🎯 Executed {len(executed_trades)} automated trades!")
                
                # Display executed trades
                for trade in executed_trades:
                    col_t1, col_t2, col_t3 = st.columns(3)
                    with col_t1:
                        st.metric(f"{trade['pair']}", f"{trade['action']}")
                    with col_t2:
                        st.metric("Price", f"{trade['price']:.5f}")
                    with col_t3:
                        st.metric("Size", f"${trade['size']:,}")
                    
                    st.caption(f"Reasons: {', '.join(trade['reasons'])} | Strength: {trade['strength']:.1%}")
                    st.markdown("---")
            else:
                # Show current monitoring status
                st.info(f"🔍 Monitoring {len(auto_manager.monitored_pairs)} currency pairs for trading opportunities...")
                
                # Display monitored pairs in a compact format
                pairs_text = " • ".join(auto_manager.monitored_pairs)
                st.caption(f"**Monitored Pairs**: {pairs_text}")
                
                # Show any pending signals that don't meet execution criteria
                st.caption("⏳ Waiting for strong multi-indicator confirmation signals...")
            
            # Show current signal strength for selected pair
            stream_df = st.session_state.streaming_manager.get_dataframe(symbol)
            if not stream_df.empty and len(stream_df) > 20:
                st.markdown(f"#### 📡 Active Trading Signal: {symbol}")
                
                # Get the active signal (prioritize ML if available)
                active_signal = None
                signal_type = "Rule-Based"
                
                if auto_manager.ml_enabled and auto_manager.ml_model is not None:
                    ml_result = auto_manager.generate_ml_trading_signal(symbol, stream_df)
                    if ml_result and ml_result.get('action') != "HOLD":
                        active_signal = ml_result
                        signal_type = "🤖 ML-Based"
                
                # Fallback to rule-based if ML signal is HOLD or not available
                if active_signal is None:
                    active_signal = auto_manager.generate_trading_signal(symbol, stream_df)
                    signal_type = "📊 Rule-Based"
                
                # Display the active signal
                if active_signal:
                    col_s1, col_s2, col_s3 = st.columns(3)
                    with col_s1:
                        action_color = "🟢" if active_signal['action'] == "BUY" else "🔴" if active_signal['action'] == "SELL" else "⚪"
                        st.metric(f"{action_color} Signal", active_signal['action'])
                    with col_s2:
                        st.metric("Strength", f"{active_signal['strength']:.1%}")
                    with col_s3:
                        execute_threshold = "✅ EXECUTE" if active_signal['strength'] > 0.65 else "⏸️ WAIT"
                        st.metric("Status", execute_threshold)
                    
                    st.caption(f"**{signal_type}**: {', '.join(active_signal['reasons'][:3])}")
                    
                    # Show model accuracy if using ML
                    if signal_type == "🤖 ML-Based" and auto_manager.ml_accuracy:
                        st.caption(f"🎯 Model Accuracy: {auto_manager.ml_accuracy:.1%}")
                else:
                    st.metric("⚪ Signal", "HOLD")
                    st.caption("Waiting for trading opportunity...")
            
            st.markdown("---")
        
        # Database Health Check (available for all user types)
        if not is_institution:  # Show for retail users too
            with st.expander("🔧 Database Diagnostics", expanded=False):
                db_healthy, db_message = check_db_health()
                if db_healthy:
                    st.success(f"✅ {db_message}")
                else:
                    st.error(f"❌ {db_message}")
                    st.error("⚠️ Database issues may prevent trade recording")
                
                # Show database details
                st.info(f"📁 **Database Location:** `{DB_FILE.absolute()}`")
                
                try:
                    if DB_FILE.exists():
                        size_bytes = DB_FILE.stat().st_size
                        size_mb = size_bytes / (1024 * 1024)
                        st.success(f"📊 **Database Size:** {size_mb:.2f} MB")
                        
                        # Test database connection and show trade counts
                        engine = get_db_engine()
                        with engine.connect() as conn:
                            # Count user's trades
                            result = conn.execute(text("SELECT COUNT(*) FROM trades WHERE user_id = ?"), (st.session_state.user_id,))
                            user_trades = result.fetchone()[0]
                            st.info(f"📈 **Your Total Trades:** {user_trades:,}")
                            
                    else:
                        st.warning("⚠️ Database file not found - will be created on first trade")
                        
                except Exception as e:
                    st.error(f"❌ Database Error: {str(e)}")
                    st.error("This may cause trade recording failures. Check logs for details.")
        
        # Manual Trading form
        trading_mode = "🤖 AUTO" if (is_institution and auto_manager.is_active) else "👤 MANUAL"
        
        # Show current position and balance
        st.markdown("**💰 Account Status**")
        net_position, buy_qty, sell_qty = get_user_position(st.session_state.user_id, symbol)
        available_balance, realized_pnl, margin_used = get_user_balance(st.session_state.user_id)
        
        col_pos1, col_pos2, col_pos3 = st.columns(3)
        with col_pos1:
            st.metric("Available Balance", f"${available_balance:,.2f}")
        with col_pos2:
            position_color = "🟢" if net_position > 0 else "🔴" if net_position < 0 else "⚪"
            st.metric(f"{position_color} Net Position", f"{net_position:,.0f} units")
        with col_pos3:
            pnl_color = "🟢" if realized_pnl > 0 else "🔴" if realized_pnl < 0 else "⚪"
            st.metric(f"{pnl_color} Realized P&L", f"${realized_pnl:,.2f}")
        
        st.markdown("---")
        
        with st.form("trade_form", clear_on_submit=False):
            st.markdown(f"**⚡ Execute Trade ({trading_mode})**")
            trade_side = st.selectbox("Trade Type", ["BUY", "SELL"])
            
            # Auto-suggest trade size for institutions
            if is_institution and auto_manager.is_active:
                suggested_size = auto_manager.calculate_trade_size()
                quantity = st.number_input("Quantity", min_value=100, value=suggested_size, step=100)
            else:
                quantity = st.number_input("Quantity", min_value=100, value=1000, step=100)
            
            # Show validation info before trade
            if trade_side == "BUY":
                required_margin = current_rate * quantity * 0.01 if current_rate else 0  # 1% margin
                st.caption(f"💸 Required margin (1% leverage): ${required_margin:,.2f}")
                if required_margin > available_balance:
                    st.error(f"⚠️ Insufficient margin! Need ${required_margin:,.2f}, have ${available_balance:,.2f}")
            elif trade_side == "SELL":
                required_margin = current_rate * quantity * 0.01 if current_rate else 0  # 1% margin
                new_position = net_position - quantity
                position_type = "🟢 LONG" if net_position > 0 else "🔴 SHORT" if net_position < 0 else "⚪ FLAT"
                
                st.caption(f"📊 Current position: {net_position:,.0f} units ({position_type})")
                st.caption(f"📉 After trade: {new_position:,.0f} units")
                st.caption(f"💸 Required margin: ${required_margin:,.2f}")
                
                # Show warning for large short positions
                if new_position < -50000:
                    st.warning(f"⚠️ Large short position: {new_position:,.0f} units")
                    
                if required_margin > available_balance:
                    st.error(f"⚠️ Insufficient margin! Need ${required_margin:,.2f}, have ${available_balance:,.2f}")
            
            strategy = st.selectbox("Strategy", ["Manual", "SMA", "RSI", "Bollinger"])
            
            if st.form_submit_button("Execute Trade", type="primary"):
                success, order_status = insert_trade(
                    st.session_state.user_id,
                    symbol,
                    trade_side,
                    current_rate,
                    quantity,
                    strategy
                )
                if success:
                    if order_status == "Filled":
                        st.success(f"🟢 Order {order_status}: {trade_side} {quantity} {symbol} @ {current_rate:.5f}")
                        st.balloons()
                        st.rerun()  # Refresh to update positions
                    elif "REJECTED" in order_status:
                        st.error(f"🔴 {order_status}")
                    elif order_status == "Cancelled":
                        st.warning(f"🟡 Order {order_status}: {trade_side} {quantity} {symbol} - Market closed")
                else:
                    st.error("❌ Failed to submit order")
    
    with col2:
        st.markdown("#### 📈 Live Price Chart with Streaming Indicators")
        
        # Get streaming data
        stream_df = st.session_state.streaming_manager.get_dataframe(symbol)
        
        if not stream_df.empty:
            # Create streaming chart
            fig = go.Figure()
            
            # Price line
            fig.add_trace(go.Scatter(
                x=stream_df['timestamp'], 
                y=stream_df['price'], 
                name='Live Price', 
                line=dict(color='#00ff41', width=3),
                mode='lines+markers'
            ))
            
            # Moving averages
            fig.add_trace(go.Scatter(
                x=stream_df['timestamp'], 
                y=stream_df['sma_20'], 
                name='SMA 20', 
                line=dict(color='orange', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=stream_df['timestamp'], 
                y=stream_df['sma_50'], 
                name='SMA 50', 
                line=dict(color='red', width=2)
            ))
            
            # Bollinger Bands - Add lower first, then upper with fill
            fig.add_trace(go.Scatter(
                x=stream_df['timestamp'], 
                y=stream_df['bb_lower'], 
                name='BB Lower', 
                line=dict(color='purple', width=2, dash='dash'),
                fill=None
            ))
            
            fig.add_trace(go.Scatter(
                x=stream_df['timestamp'], 
                y=stream_df['bb_upper'], 
                name='BB Upper', 
                line=dict(color='purple', width=2, dash='dash'),
                fill='tonexty',
                fillcolor='rgba(128,0,128,0.15)'
            ))
            
            fig.update_layout(
                title=f"🔴 LIVE: {symbol} - Real-Time Streaming",
                xaxis_title="Time",
                yaxis_title="Price",
                height=400,
                showlegend=True,
                template="plotly_dark",
                plot_bgcolor='rgba(0,0,0,0.1)'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Live RSI Chart
            if len(stream_df) > 0:
                fig_rsi = go.Figure()
                fig_rsi.add_trace(go.Scatter(
                    x=stream_df['timestamp'], 
                    y=stream_df['rsi'], 
                    name='RSI', 
                    line=dict(color='purple', width=2)
                ))
                fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
                fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
                fig_rsi.add_hline(y=50, line_dash="dot", line_color="white", annotation_text="Neutral")
                
                fig_rsi.update_layout(
                    title="🎯 Live RSI Indicator",
                    yaxis=dict(range=[0, 100]),
                    height=200,
                    template="plotly_dark"
                )
                st.plotly_chart(fig_rsi, use_container_width=True)
        else:
            st.info("🔄 Initializing live data stream...")
            # Show placeholder chart
            fig = go.Figure()
            fig.add_annotation(text="Waiting for live data...", 
                             xref="paper", yref="paper",
                             x=0.5, y=0.5, showarrow=False)
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    # Live Order Status Monitor
    st.markdown("---")
    st.markdown("#### 💼 Account Overview & Trading History")
    
    # Account Summary
    st.markdown("**📊 Account Summary**")
    available_balance, realized_pnl, margin_used = get_user_balance(st.session_state.user_id)
    
    col_bal1, col_bal2, col_bal3, col_bal4 = st.columns(4)
    with col_bal1:
        st.metric("💰 Available Balance", f"${available_balance:,.2f}")
    with col_bal2:
        pnl_color = "🟢" if realized_pnl >= 0 else "🔴"
        st.metric(f"{pnl_color} Total P&L", f"${realized_pnl:,.2f}")
    with col_bal3:
        st.metric("📊 Margin Used", f"${margin_used:,.2f}")
    with col_bal4:
        equity = available_balance + margin_used
        st.metric("💎 Total Equity", f"${equity:,.2f}")
    
    # Position Summary by Currency Pair
    st.markdown("**🎯 Current Positions**")
    positions_data = []
    
    # Check positions for all major currency pairs
    major_pairs = ["EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF", "AUD/USD", "USD/CAD", "NZD/USD"]
    for pair in major_pairs:
        net_pos, buy_qty, sell_qty = get_user_position(st.session_state.user_id, pair)
        if net_pos != 0:  # Only show pairs with positions
            positions_data.append({
                "Currency Pair": pair,
                "Net Position": f"{net_pos:,.0f}",
                "Total Buys": f"{buy_qty:,.0f}",
                "Total Sells": f"{sell_qty:,.0f}",
                "Status": "🟢 LONG" if net_pos > 0 else "🔴 SHORT"
            })
    
    if positions_data:
        positions_df = pd.DataFrame(positions_data)
        st.dataframe(positions_df, hide_index=True, use_container_width=True)
    else:
        st.info("📭 No open positions")
    
    st.markdown("---")
    st.markdown("#### 📋 Live Order Status Monitor")
    
    # Enhanced filtering for institutional users
    if is_institution:
        col_filter, col_pair_filter, col_strategy_filter, col_refresh_orders = st.columns([2, 2, 2, 1])
        with col_filter:
            status_filter = st.selectbox(
                "Filter by Status:",
                ["All", "Filled", "Rejected", "Cancelled"],
                key="status_filter"
            )
        with col_pair_filter:
            # Get unique currency pairs from trades
            all_pairs = ["All"] + auto_manager.monitored_pairs if is_institution else ["All"]
            pair_filter = st.selectbox(
                "Filter by Pair:",
                all_pairs,
                key="pair_filter"
            )
        with col_strategy_filter:
            strategy_filter = st.selectbox(
                "Filter by Strategy:",
                ["All", "Manual", "AUTO-SMA", "AUTO-RSI", "AUTO-BB", "AUTO-Multi"],
                key="strategy_filter"
            )
        with col_refresh_orders:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Refresh Orders"):
                st.rerun()
    else:
        # Simple filtering for regular users
        col_filter, col_refresh_orders = st.columns([3, 1])
        with col_filter:
            status_filter = st.selectbox(
                "Filter by Status:",
                ["All", "Filled", "Rejected", "Cancelled"],
                key="status_filter"
            )
            pair_filter = "All"
            strategy_filter = "All"
        with col_refresh_orders:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Refresh Orders"):
                st.rerun()
    
    trades_df = load_trades(st.session_state.user_id)
    if not trades_df.empty:
        # Apply filters
        filtered_trades = trades_df.copy()
        
        if status_filter != "All":
            filtered_trades = filtered_trades[filtered_trades['status'] == status_filter]
            
        if pair_filter != "All":
            filtered_trades = filtered_trades[filtered_trades['symbol'] == pair_filter]
            
        if strategy_filter != "All":
            if strategy_filter.startswith("AUTO-"):
                filtered_trades = filtered_trades[filtered_trades['strategy'].str.contains(strategy_filter, na=False)]
            else:
                filtered_trades = filtered_trades[filtered_trades['strategy'] == strategy_filter]
            
        # Show latest trades first
        recent_trades = filtered_trades.head(15)  # Show more trades for institutions
        
        # Add status and strategy indicators
        def get_status_display(row):
            side = row['side']
            status = row['status']
            strategy = row.get('strategy', 'Manual')
            
            # Color coding for different strategies
            if 'AUTO-' in strategy:
                emoji = "🤖"
            else:
                emoji = "👤"
            
            if status == "Filled":
                return f"🟢 {emoji} {side} - Filled" if side == "BUY" else f"🔴 {emoji} {side} - Filled"
            elif status == "Rejected":
                return f"❌ {emoji} {side} - Rejected"
            elif status == "Cancelled":
                return f"🟡 {emoji} {side} - Cancelled"
            else:
                return f"⚪ {emoji} {side} - {status}"
        
        def get_strategy_display(strategy):
            if 'AUTO-' in strategy:
                return f"🤖 {strategy}"
            else:
                return f"👤 {strategy}"
        
        recent_trades['Order Status'] = recent_trades.apply(get_status_display, axis=1)
        recent_trades['Strategy Type'] = recent_trades['strategy'].apply(get_strategy_display)
        
        # Display with formatting - show more columns for institutions
        if is_institution:
            display_columns = ['timestamp', 'Order Status', 'symbol', 'price', 'quantity', 'Strategy Type', 'status']
        else:
            display_columns = ['timestamp', 'Order Status', 'symbol', 'price', 'quantity', 'strategy', 'status']
            
        st.dataframe(
            recent_trades[display_columns],
            use_container_width=True,
            hide_index=True
        )
        
        # Enhanced Order Status Metrics
        if is_institution:
            st.markdown("#### 📊 Multi-Currency Trading Dashboard")
            
            # Automated vs Manual trades breakdown
            auto_trades = trades_df[trades_df['strategy'].str.contains('AUTO-', na=False)]
            manual_trades = trades_df[~trades_df['strategy'].str.contains('AUTO-', na=False)]
            
            col_auto, col_manual, col_pairs = st.columns(3)
            with col_auto:
                st.metric("🤖 Automated Trades", len(auto_trades))
            with col_manual:
                st.metric("👤 Manual Trades", len(manual_trades))  
            with col_pairs:
                unique_pairs = trades_df['symbol'].nunique()
                st.metric("💱 Currency Pairs Traded", unique_pairs)
            
            # Currency pair performance
            if len(auto_trades) > 0:
                st.markdown("##### 🌍 Multi-Currency Performance")
                pair_stats = auto_trades.groupby('symbol').agg({
                    'id': 'count',
                    'quantity': 'mean',
                    'price': 'mean'
                }).round(2)
                pair_stats.columns = ['Trades', 'Avg Size', 'Avg Price']
                pair_stats = pair_stats.sort_values('Trades', ascending=False)
                
                # Display top performing pairs
                st.dataframe(pair_stats.head(8), use_container_width=True)
            
            # Daily Risk Metrics
            col_r1, col_r2, col_r3 = st.columns(3)
            with col_r1:
                loss_pct = (abs(daily_pnl) / auto_manager.max_daily_loss * 100) if daily_pnl < 0 else 0
                st.metric("Loss Limit Used", f"{loss_pct:.1f}%", delta="Risk Level")
            with col_r2:
                remaining_capital = auto_manager.trading_capital + daily_pnl
                st.metric("Available Capital", f"${remaining_capital:,.0f}")
            with col_r3:
                trade_pct = (trades_today / auto_manager.max_trades_per_day * 100)
                st.metric("Trade Limit Used", f"{trade_pct:.1f}%")
            
            # Risk Status Bar
            if loss_pct > 90:
                st.error("🚨 CRITICAL: Approaching daily loss limit!")
            elif loss_pct > 75:
                st.warning("⚠️ HIGH RISK: Close to loss limit")
            elif loss_pct > 50:
                st.info("🔶 MODERATE RISK: Monitor closely")
            
            st.markdown("---")
        
        st.markdown("#### 📊 Order Status Summary")
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        with col1:
            st.metric("Total Orders", len(trades_df))
        with col2:
            filled_orders = len(trades_df[trades_df['status'] == 'Filled'])
            st.metric("🟢 Filled", filled_orders)
        with col3:
            rejected_orders = len(trades_df[trades_df['status'] == 'Rejected'])
            st.metric("🔴 Rejected", rejected_orders)
        with col4:
            cancelled_orders = len(trades_df[trades_df['status'] == 'Cancelled'])
            st.metric("� Cancelled", cancelled_orders)
        with col5:
            fill_rate = (filled_orders / len(trades_df) * 100) if len(trades_df) > 0 else 0
            st.metric("Fill Rate", f"{fill_rate:.1f}%")
        with col6:
            avg_quantity = trades_df['quantity'].mean()
            st.metric("Avg Size", f"{avg_quantity:.0f}")
        
        if len(filtered_trades) == 0:
            st.info(f"No orders with status '{status_filter}' found.")
        else:
            st.caption(f"Showing {len(recent_trades)} of {len(filtered_trades)} {status_filter.lower()} orders")
        with col5:
            latest_trade = trades_df.iloc[0]['timestamp'] if len(trades_df) > 0 else "None"
            st.metric("⏰ Last Trade", latest_trade.split(' ')[1] if ' ' in str(latest_trade) else "None")
    else:
        st.info("No orders yet. Execute your first order above to see the live order status monitor!")
        
    # Performance summary
    if not trades_df.empty:
        st.markdown("#### 📊 Live Trading Performance")
        
        # Calculate daily P&L (simplified)
        trades_today = trades_df[trades_df['timestamp'].str.contains(datetime.now().strftime('%Y-%m-%d'))]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📅 Today's Trades", len(trades_today))
        with col2:
            if len(trades_today) > 0:
                avg_price_today = trades_today['price'].mean()
                st.metric("💰 Avg Price Today", f"{avg_price_today:.5f}")
            else:
                st.metric("💰 Avg Price Today", "0.00000")
        with col3:
            total_volume_today = trades_today['quantity'].sum() if len(trades_today) > 0 else 0
            st.metric("📈 Volume Today", f"{total_volume_today:,.0f}")

def strategy_analysis_section():
    """Strategy analysis with algorithm visualization"""
    st.subheader("Trading Strategy Analysis")
    
    # Strategy selection
    selected_strategy = st.selectbox(
        "Select Strategy to Analyze",
        ["SMA Crossover", "RSI Overbought/Oversold", "Bollinger Bands Mean Reversion"]
    )
    
    # Generate sample data
    symbol = "EUR/USD"
    df = generate_price_series(symbol, 1.0850, days=30)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Strategy Signals")
        
        strategy_map = {
            "SMA Crossover": "SMA",
            "RSI Overbought/Oversold": "RSI", 
            "Bollinger Bands Mean Reversion": "Bollinger"
        }
        
        signals = generate_trading_signals(df, strategy_map[selected_strategy])
        performance = calculate_strategy_performance(signals, df)
        
        # Performance metrics
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Total Trades", performance['total_trades'])
            st.metric("Winning Trades", performance['winning_trades'])
        with col_b:
            st.metric("Win Rate", f"{performance['win_rate']:.1f}%")
            st.metric("Total P&L", f"{performance['total_pnl']:.4f}")
    
    with col2:
        st.markdown("#### Signal Visualization")
        
        # Create chart with signals
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['price'], name='Price', line=dict(color='blue')))
        
        if strategy_map[selected_strategy] == 'SMA':
            fig.add_trace(go.Scatter(x=df.index, y=df['sma_20'], name='SMA 20', line=dict(color='orange')))
            fig.add_trace(go.Scatter(x=df.index, y=df['sma_50'], name='SMA 50', line=dict(color='red')))
        elif strategy_map[selected_strategy] == 'Bollinger':
            fig.add_trace(go.Scatter(x=df.index, y=df['bb_upper'], name='BB Upper', line=dict(color='gray')))
            fig.add_trace(go.Scatter(x=df.index, y=df['bb_lower'], name='BB Lower', line=dict(color='gray')))
        
        # Add signals
        if signals:
            buy_signals = [s for s in signals if s['signal'] == 'BUY']
            sell_signals = [s for s in signals if s['signal'] == 'SELL']
            
            if buy_signals:
                fig.add_trace(go.Scatter(
                    x=[s['timestamp'] for s in buy_signals],
                    y=[s['price'] for s in buy_signals],
                    mode='markers',
                    marker=dict(color='green', size=10, symbol='triangle-up'),
                    name='Buy Signals'
                ))
            
            if sell_signals:
                fig.add_trace(go.Scatter(
                    x=[s['timestamp'] for s in sell_signals],
                    y=[s['price'] for s in sell_signals],
                    mode='markers',
                    marker=dict(color='red', size=10, symbol='triangle-down'),
                    name='Sell Signals'
                ))
        
        fig.update_layout(
            title=f"{selected_strategy} - {symbol}",
            xaxis_title="Time",
            yaxis_title="Price",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Additional indicator chart
    if strategy_map[selected_strategy] == 'RSI':
        st.markdown("#### RSI Indicator")
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=df.index, y=df['rsi'], name='RSI', line=dict(color='purple')))
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
        fig_rsi.update_layout(title="RSI (14 periods)", yaxis=dict(range=[0, 100]), height=300)
        st.plotly_chart(fig_rsi, use_container_width=True)

def strategy_comparison_section():
    """Strategy comparison for efficiency analysis"""
    st.subheader("Trading Strategy Comparison")
    
    # Generate data for comparison
    symbol = "GBP/USD"
    df = generate_price_series(symbol, 1.2650, days=30)
    
    strategies = ["SMA", "RSI", "Bollinger"]
    results = {}
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### Strategy Performance Metrics")
        
        # Calculate performance for each strategy
        comparison_data = []
        for strategy in strategies:
            signals = generate_trading_signals(df, strategy)
            performance = calculate_strategy_performance(signals, df)
            results[strategy] = performance
            
            comparison_data.append({
                'Strategy': strategy,
                'Total Trades': performance['total_trades'],
                'Win Rate (%)': performance['win_rate'],
                'Total P&L': performance['total_pnl']
            })
        
        # Display comparison table
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
        
        # Best strategy
        if comparison_data:
            best_strategy = max(comparison_data, key=lambda x: x['Total P&L'])
            st.success(f"🏆 **Best Strategy:** {best_strategy['Strategy']}")
            st.info(f"P&L: {best_strategy['Total P&L']:.4f} | Win Rate: {best_strategy['Win Rate (%)']:.1f}%")
    
    with col2:
        st.markdown("#### Performance Comparison Charts")
        
        # P&L comparison chart
        fig_pnl = px.line(
            comparison_df, 
            x='Strategy', 
            y='Total P&L',
            title="Total P&L Comparison",
            markers=True,
            line_shape='linear'
        )
        fig_pnl.update_traces(line=dict(width=3))
        fig_pnl.update_layout(
            xaxis_title="Strategy",
            yaxis_title="Total P&L"
        )
        st.plotly_chart(fig_pnl, use_container_width=True)
        
        # Win rate comparison
        fig_winrate = px.line(
            comparison_df,
            x='Strategy',
            y='Win Rate (%)',
            title="Win Rate Comparison",
            markers=True,
            line_shape='linear'
        )
        fig_winrate.update_traces(line=dict(width=3, color='blue'))
        fig_winrate.update_layout(
            xaxis_title="Strategy",
            yaxis_title="Win Rate (%)"
        )
        st.plotly_chart(fig_winrate, use_container_width=True)

# ---------------------------
# Main Application
# ---------------------------
def main():
    """Main application entry point"""
    # Initialize database
    init_db()
    
    # Check database health and display status in sidebar
    with st.sidebar:
        with st.expander("System Status", expanded=False):
            db_healthy, db_message = check_db_health()
            if db_healthy:
                st.success(f"✅ Database: {db_message}")
            else:
                st.error(f"❌ Database: {db_message}")
            
            st.info(f"📁 Database Location: {DB_FILE.absolute()}")
            
            # Show database file size if it exists
            try:
                if DB_FILE.exists():
                    size_bytes = DB_FILE.stat().st_size
                    size_mb = size_bytes / (1024 * 1024)
                    st.info(f"📊 Database Size: {size_mb:.2f} MB")
                else:
                    st.warning("⚠️ Database file not found")
            except Exception as e:
                st.warning(f"⚠️ Cannot read database file: {e}")
    
    # Initialize session state
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    
    # Get current route
    current_page = get_current_route()
    
    # Route to appropriate page
    if st.session_state.logged_in and current_page == "dashboard":
        dashboard_page()
    elif current_page == "register":
        register_page()
    elif current_page == "login" or not st.session_state.logged_in:
        login_page()
    else:
        # Default to login
        navigate_to("login")
        login_page()

if __name__ == "__main__":
    main()