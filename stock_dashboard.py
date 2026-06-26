import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from datetime import datetime, timedelta
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import yaml
import warnings
warnings.filterwarnings('ignore')

# Config files keep the MVP watchlist and scoring weights editable without
# changing application code.
CONFIG_DIR = Path(__file__).parent / "config"
WATCHLIST_PATH = CONFIG_DIR / "watchlist.yaml"
WEIGHTS_PATH = CONFIG_DIR / "scoring_weights.yaml"
OUTLOOK_WINDOWS = {
    "1 Week (5 trading days)": 5,
    "2 Weeks (10 trading days)": 10,
    "4 Weeks (20 trading days)": 20,
}

# Configure Streamlit page
st.set_page_config(
    page_title="AI Stock Market Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

class StockAnalyzer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        
    def fetch_stock_data(self, symbol, period="1y"):
        """Fetch stock data with error handling"""
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period=period)
            info = stock.info
            return data, info
        except Exception as e:
            st.error(f"Error fetching data for {symbol}: {str(e)}")
            return None, None
    
    def calculate_technical_indicators(self, data):
        """Calculate comprehensive technical indicators using pure pandas/numpy"""
        df = data.copy()
        
        # Simple Moving Averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        
        # Exponential Moving Averages
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_histogram'] = df['MACD'] - df['MACD_signal']
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['BB_middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
        df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
        
        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=30).mean()
        df['Volume_ratio'] = df['Volume'] / df['Volume_SMA']
        
        # Price-based indicators
        df['High_Low_Pct'] = (df['High'] - df['Low']) / df['Close'] * 100
        df['Price_Change'] = df['Close'] - df['Open']
        df['Price_Change_Pct'] = (df['Close'] - df['Open']) / df['Open'] * 100
        
        # Volatility (Average True Range approximation)
        df['High_Low'] = df['High'] - df['Low']
        df['High_Close'] = np.abs(df['High'] - df['Close'].shift())
        df['Low_Close'] = np.abs(df['Low'] - df['Close'].shift())
        df['True_Range'] = df[['High_Low', 'High_Close', 'Low_Close']].max(axis=1)
        df['ATR'] = df['True_Range'].rolling(window=14).mean()
        
        # Stochastic Oscillator
        low_14 = df['Low'].rolling(window=14).min()
        high_14 = df['High'].rolling(window=14).max()
        df['Stoch_K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
        df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()
        
        return df
    
    def prepare_ml_features(self, data):
        """Prepare features for machine learning"""
        df = data.copy()
        
        # Returns and momentum
        df['Returns'] = df['Close'].pct_change()
        df['Returns_5d'] = df['Close'].pct_change(5)
        df['Returns_10d'] = df['Close'].pct_change(10)
        
        # Lag features
        for lag in [1, 2, 3, 5, 10]:
            df[f'Close_lag_{lag}'] = df['Close'].shift(lag)
            df[f'Volume_lag_{lag}'] = df['Volume'].shift(lag)
            df[f'Returns_lag_{lag}'] = df['Returns'].shift(lag)
        
        # Rolling statistics
        for window in [5, 10, 20, 50]:
            df[f'Close_mean_{window}'] = df['Close'].rolling(window).mean()
            df[f'Close_std_{window}'] = df['Close'].rolling(window).std()
            df[f'Volume_mean_{window}'] = df['Volume'].rolling(window).mean()
            df[f'High_mean_{window}'] = df['High'].rolling(window).mean()
            df[f'Low_mean_{window}'] = df['Low'].rolling(window).mean()
        
        # Price position relative to moving averages
        df['Price_vs_SMA20'] = (df['Close'] - df['SMA_20']) / df['SMA_20'] * 100
        df['Price_vs_SMA50'] = (df['Close'] - df['SMA_50']) / df['SMA_50'] * 100
        
        # Volatility features
        df['Price_volatility_10d'] = df['Returns'].rolling(10).std()
        df['Price_volatility_20d'] = df['Returns'].rolling(20).std()
        
        return df
    
    def train_prediction_model(self, data):
        """Train ML model for price prediction"""
        df = self.prepare_ml_features(data)
        df = df.dropna()
        
        if len(df) < 100:  # Need minimum data
            return None
        
        # Features for prediction (excluding target-related columns)
        exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits', 
                       'Returns', 'Returns_5d', 'Returns_10d']
        feature_cols = [col for col in df.columns if not any(exc in col for exc in exclude_cols)]
        feature_cols = [col for col in feature_cols if 'lag' in col or 'mean' in col or 
                       'std' in col or col in ['RSI', 'MACD', 'Price_vs_SMA20', 'Price_vs_SMA50', 
                                              'Price_volatility_10d', 'Price_volatility_20d', 'ATR']]
        
        if len(feature_cols) < 5:
            return None
        
        # pandas 3 removed fillna(method=...), so use the direct forward/back
        # fill helpers for compatibility with the current dependency set.
        X = df[feature_cols].ffill().bfill()
        y = df['Close'].shift(-1)  # Predict next day's close
        
        # Remove last row (no target) and any remaining NaN
        X = X[:-1]
        y = y[:-1]
        
        # Remove any remaining NaN values
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]
        
        if len(X) < 50:
            return None
        
        # Split and train
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.model.fit(X_train_scaled, y_train)
        
        # Calculate accuracy
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        return {
            'train_score': train_score,
            'test_score': test_score,
            'feature_importance': dict(zip(feature_cols, self.model.feature_importances_)),
            'last_features': X.iloc[-1:],
            'feature_cols': feature_cols
        }
    
    def predict_next_price(self, model_info):
        """Predict next trading day price"""
        if model_info is None:
            return None
        
        last_features_scaled = self.scaler.transform(model_info['last_features'])
        prediction = self.model.predict(last_features_scaled)[0]
        
        return prediction
    
    def generate_market_analysis(self, data, info, symbol):
        """Generate AI-powered market analysis"""
        latest = data.iloc[-1]
        prev = data.iloc[-2]
        
        # Price movement
        price_change = latest['Close'] - prev['Close']
        price_change_pct = (price_change / prev['Close']) * 100
        
        # Technical analysis
        rsi = latest.get('RSI', 50)
        sma_20 = latest.get('SMA_20', latest['Close'])
        sma_50 = latest.get('SMA_50', latest['Close'])
        bb_upper = latest.get('BB_upper', latest['Close'])
        bb_lower = latest.get('BB_lower', latest['Close'])
        
        # Volume analysis
        avg_volume = data['Volume'].rolling(20).mean().iloc[-1]
        volume_ratio = latest['Volume'] / avg_volume if avg_volume > 0 else 1
        
        # MACD analysis
        macd = latest.get('MACD', 0)
        macd_signal = latest.get('MACD_signal', 0)
        
        # Generate analysis
        analysis = []
        
        # Price trend
        if price_change_pct > 3:
            analysis.append(f"🚀 {symbol} shows exceptional bullish momentum with a {price_change_pct:.2f}% surge")
        elif price_change_pct > 1:
            analysis.append(f"🟢 {symbol} demonstrates strong upward movement (+{price_change_pct:.2f}%)")
        elif price_change_pct > 0:
            analysis.append(f"🟡 {symbol} shows modest gains (+{price_change_pct:.2f}%)")
        elif price_change_pct > -1:
            analysis.append(f"🟡 {symbol} experiences slight decline ({price_change_pct:.2f}%)")
        elif price_change_pct > -3:
            analysis.append(f"🔴 {symbol} shows moderate bearish pressure ({price_change_pct:.2f}%)")
        else:
            analysis.append(f"🔻 {symbol} faces significant selling pressure ({price_change_pct:.2f}%)")
        
        # RSI analysis
        if rsi > 80:
            analysis.append(f"🚨 RSI at {rsi:.1f} indicates severely overbought conditions - potential reversal ahead")
        elif rsi > 70:
            analysis.append(f"⚠️ RSI at {rsi:.1f} shows overbought territory - exercise caution")
        elif rsi < 20:
            analysis.append(f"🛒 RSI at {rsi:.1f} signals severely oversold - strong buying opportunity")
        elif rsi < 30:
            analysis.append(f"💡 RSI at {rsi:.1f} suggests oversold conditions - potential buying opportunity")
        elif 40 <= rsi <= 60:
            analysis.append(f"⚖️ RSI at {rsi:.1f} indicates balanced momentum")
        else:
            analysis.append(f"📊 RSI at {rsi:.1f} shows {('bullish' if rsi > 50 else 'bearish')} bias")
        
        # Moving average analysis
        if latest['Close'] > sma_20 > sma_50:
            analysis.append("📈 Strong bullish alignment - price above both 20 and 50-day MAs")
        elif latest['Close'] < sma_20 < sma_50:
            analysis.append("📉 Bearish trend confirmed - price below key moving averages")
        elif latest['Close'] > sma_20 and sma_20 < sma_50:
            analysis.append("🔄 Mixed signals - short-term bullish but longer-term bearish")
        else:
            analysis.append("➡️ Consolidation phase - awaiting directional breakout")
        
        # Bollinger Bands analysis
        if latest['Close'] > bb_upper:
            analysis.append("📊 Price trading above upper Bollinger Band - potential overbought")
        elif latest['Close'] < bb_lower:
            analysis.append("📊 Price near lower Bollinger Band - potential oversold bounce")
        
        # MACD analysis
        if macd > macd_signal and macd > 0:
            analysis.append("⚡ MACD shows strong bullish momentum")
        elif macd < macd_signal and macd < 0:
            analysis.append("⚡ MACD indicates bearish momentum")
        elif macd > macd_signal:
            analysis.append("⚡ MACD bullish crossover - momentum improving")
        else:
            analysis.append("⚡ MACD bearish crossover - momentum weakening")
        
        # Volume analysis
        if volume_ratio > 2:
            analysis.append("🔥 Exceptional volume surge confirms strong conviction")
        elif volume_ratio > 1.5:
            analysis.append("📊 High volume validates price movement")
        elif volume_ratio < 0.5:
            analysis.append("📊 Below-average volume suggests weak conviction")
        else:
            analysis.append("📊 Normal volume levels")
        
        # Market cap context
        market_cap = info.get('marketCap', 0)
        if market_cap:
            if market_cap > 200e9:  # > 200B
                analysis.append("🏢 Large-cap stability with lower volatility expected")
            elif market_cap > 10e9:  # > 10B
                analysis.append("🏢 Mid-cap stock with balanced growth-stability profile")
            else:
                analysis.append("🏢 Small-cap stock with higher growth potential and volatility")
        
        return analysis

def create_advanced_chart(data, symbol):
    """Create advanced candlestick chart with technical indicators"""
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=(f'{symbol} Price Action & Moving Averages', 'Volume', 'MACD', 'RSI & Stochastic'),
        row_heights=[0.5, 0.15, 0.2, 0.15]
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Price',
            increasing_line_color='#00ff88',
            decreasing_line_color='#ff4444'
        ),
        row=1, col=1
    )
    
    # Moving averages
    colors = ['#ff9500', '#007aff', '#5856d6']
    mas = [('SMA_20', 'SMA 20'), ('SMA_50', 'SMA 50'), ('SMA_200', 'SMA 200')]
    
    for i, (ma_col, ma_name) in enumerate(mas):
        if ma_col in data.columns and not data[ma_col].isna().all():
            fig.add_trace(
                go.Scatter(x=data.index, y=data[ma_col], 
                          line=dict(color=colors[i], width=1.5), name=ma_name),
                row=1, col=1
            )
    
    # Bollinger Bands
    if all(col in data.columns for col in ['BB_upper', 'BB_lower']):
        fig.add_trace(
            go.Scatter(x=data.index, y=data['BB_upper'], 
                      line=dict(color='rgba(128,128,128,0.5)', width=1), name='BB Upper',
                      showlegend=False),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=data.index, y=data['BB_lower'], 
                      line=dict(color='rgba(128,128,128,0.5)', width=1), name='BB Lower',
                      fill='tonexty', fillcolor='rgba(128,128,128,0.1)',
                      showlegend=False),
            row=1, col=1
        )
    
    # Volume
    volume_colors = ['#00ff88' if data['Close'].iloc[i] >= data['Open'].iloc[i] else '#ff4444' 
                    for i in range(len(data))]
    fig.add_trace(
        go.Bar(x=data.index, y=data['Volume'], 
              marker_color=volume_colors, name='Volume', opacity=0.7),
        row=2, col=1
    )
    
    if 'Volume_SMA' in data.columns:
        fig.add_trace(
            go.Scatter(x=data.index, y=data['Volume_SMA'], 
                      line=dict(color='white', width=1), name='Vol SMA'),
            row=2, col=1
        )
    
    # MACD
    if all(col in data.columns for col in ['MACD', 'MACD_signal', 'MACD_histogram']):
        fig.add_trace(
            go.Scatter(x=data.index, y=data['MACD'], 
                      line=dict(color='#007aff', width=2), name='MACD'),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(x=data.index, y=data['MACD_signal'], 
                      line=dict(color='#ff9500', width=2), name='Signal'),
            row=3, col=1
        )
        
        histogram_colors = ['#00ff88' if val >= 0 else '#ff4444' for val in data['MACD_histogram']]
        fig.add_trace(
            go.Bar(x=data.index, y=data['MACD_histogram'], 
                  marker_color=histogram_colors, name='Histogram', opacity=0.6),
            row=3, col=1
        )
    
    # RSI and Stochastic
    if 'RSI' in data.columns:
        fig.add_trace(
            go.Scatter(x=data.index, y=data['RSI'], 
                      line=dict(color='#af52de', width=2), name='RSI'),
            row=4, col=1
        )
        # RSI levels
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.7, row=4, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.7, row=4, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.5, row=4, col=1)
    
    if 'Stoch_K' in data.columns:
        fig.add_trace(
            go.Scatter(x=data.index, y=data['Stoch_K'], 
                      line=dict(color='#ffcc00', width=1.5), name='Stoch %K'),
            row=4, col=1
        )
        fig.add_trace(
            go.Scatter(x=data.index, y=data['Stoch_D'], 
                      line=dict(color='#ff6600', width=1.5), name='Stoch %D'),
            row=4, col=1
        )
    
    fig.update_layout(
        title=f'{symbol} - Complete Technical Analysis Dashboard',
        xaxis_rangeslider_visible=False,
        height=900,
        showlegend=True,
        template='plotly_dark',
        font=dict(size=10)
    )
    
    # Remove x-axis labels from all but bottom subplot
    for i in range(1, 4):
        fig.update_xaxes(showticklabels=False, row=i, col=1)
    
    return fig

def create_performance_metrics(data, symbol):
    """Create performance metrics visualization"""
    # Calculate returns
    data['Daily_Returns'] = data['Close'].pct_change()
    data['Cumulative_Returns'] = (1 + data['Daily_Returns']).cumprod() - 1
    
    # Performance metrics
    total_return = data['Cumulative_Returns'].iloc[-1] * 100
    volatility = data['Daily_Returns'].std() * np.sqrt(252) * 100  # Annualized
    sharpe_ratio = (data['Daily_Returns'].mean() * 252) / (data['Daily_Returns'].std() * np.sqrt(252))
    
    max_drawdown = ((data['Close'] / data['Close'].expanding().max()) - 1).min() * 100
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Return", f"{total_return:.1f}%")
    with col2:
        st.metric("Volatility (Ann.)", f"{volatility:.1f}%")
    with col3:
        st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
    with col4:
        st.metric("Max Drawdown", f"{max_drawdown:.1f}%")
    
    # Cumulative returns chart
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['Cumulative_Returns'] * 100,
            mode='lines',
            name='Cumulative Returns',
            line=dict(color='#00ff88', width=2)
        )
    )
    
    fig.update_layout(
        title=f'{symbol} Cumulative Returns (%)',
        xaxis_title='Date',
        yaxis_title='Cumulative Return (%)',
        template='plotly_dark',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)


def load_yaml_config(path, fallback):
    """Load a small YAML config file with a safe fallback."""
    try:
        with open(path, "r", encoding="utf-8") as file:
            return yaml.safe_load(file) or fallback
    except FileNotFoundError:
        st.warning(f"Config file not found: {path}")
        return fallback
    except yaml.YAMLError as exc:
        st.warning(f"Could not parse {path.name}: {exc}")
        return fallback


def load_watchlist_config():
    # Fallback keeps the app usable if the YAML file is missing or malformed.
    fallback = {
        "benchmark": "XIC.TO",
        "sector_etfs": {},
        "watchlist": [
            {"ticker": "RY.TO", "name": "Royal Bank of Canada", "sector": "Financials"},
            {"ticker": "TD.TO", "name": "Toronto-Dominion Bank", "sector": "Financials"},
            {"ticker": "SHOP.TO", "name": "Shopify", "sector": "Technology"},
        ],
    }
    return load_yaml_config(WATCHLIST_PATH, fallback)


def load_scoring_weights():
    # Weights are normalized below so the score formula remains stable if the
    # config values do not add up to exactly 1.0.
    fallback = {
        "momentum": 0.25,
        "trend": 0.20,
        "volume": 0.20,
        "rsi": 0.15,
        "sector_strength": 0.10,
        "benchmark_relative": 0.10,
    }
    weights = load_yaml_config(WEIGHTS_PATH, fallback)
    total = sum(float(value) for value in weights.values()) or 1
    return {key: float(value) / total for key, value in weights.items()}


@st.cache_data(ttl=900)
def fetch_history(symbol, period):
    stock = yf.Ticker(symbol)
    return stock.history(period=period)


def clip_score(value):
    if pd.isna(value) or np.isinf(value):
        return 0.0
    return float(np.clip(value, -1, 1))


def normalize_return(value, strong_move=0.08):
    # Convert raw returns into the internal -1..+1 scale. The strong_move
    # threshold controls what counts as a full-strength bullish/bearish move.
    return clip_score(value / strong_move)


def calculate_return(data, window):
    if data is None or len(data) <= window:
        return np.nan
    previous_close = data["Close"].iloc[-window - 1]
    if previous_close == 0 or pd.isna(previous_close):
        return np.nan
    return (data["Close"].iloc[-1] - previous_close) / previous_close


def calculate_rsi_score(rsi):
    # Reward the healthy bullish zone, but penalize overheated RSI readings so
    # the ranking does not blindly chase extended moves.
    if pd.isna(rsi):
        return 0.0
    if 50 <= rsi <= 70:
        return 0.8
    if 40 <= rsi < 50:
        return 0.2
    if 30 <= rsi < 40:
        return -0.2
    if 70 < rsi <= 80:
        return 0.3
    if rsi > 80:
        return -0.4
    return -0.5


def calculate_trend_score(latest):
    # Short-term ranking leans most on price vs SMA20 and SMA20 vs SMA50, while
    # SMA200 acts as broader trend confirmation when enough history exists.
    score = 0
    close = latest.get("Close")
    sma_20 = latest.get("SMA_20")
    sma_50 = latest.get("SMA_50")
    sma_200 = latest.get("SMA_200")

    if pd.notna(close) and pd.notna(sma_20):
        score += 0.35 if close > sma_20 else -0.35
    if pd.notna(sma_20) and pd.notna(sma_50):
        score += 0.35 if sma_20 > sma_50 else -0.35
    if pd.notna(sma_50) and pd.notna(sma_200):
        score += 0.30 if sma_50 > sma_200 else -0.30
    return clip_score(score)


def calculate_signal_tier(score):
    if score >= 80:
        return "Strong Buy Candidate"
    if score >= 60:
        return "Watch Closely"
    if score >= 40:
        return "Neutral"
    if score >= 20:
        return "Weak"
    return "Bearish / Ignore"


def generate_signal_explanation(row, window_label):
    reasons = []
    if row["Benchmark Relative Strength"] > 0:
        reasons.append("outperforming XIC.TO")
    if row["Volume Ratio"] >= 1.5:
        reasons.append("above-average volume")
    if 50 <= row["RSI"] <= 70:
        reasons.append("healthy RSI")
    if row["MA Trend Score"] > 0.3:
        reasons.append("positive moving-average trend")
    if row["Sector Strength"] > 0:
        reasons.append("sector support")

    if not reasons:
        return f"{row['Ticker']} has mixed or weak {window_label.lower()} signals versus the watchlist."

    return f"{row['Ticker']} ranks well for the {window_label.lower()} outlook due to " + ", ".join(reasons[:3]) + "."


def score_watchlist(analyzer, watchlist_config, weights, period, selected_window):
    benchmark = watchlist_config.get("benchmark", "XIC.TO")
    sector_etfs = watchlist_config.get("sector_etfs", {})
    watchlist = watchlist_config.get("watchlist", [])

    # Fetch each symbol once per run, including benchmark and sector ETFs used
    # for relative-strength comparisons.
    needed_symbols = {benchmark}
    needed_symbols.update(item["ticker"] for item in watchlist)
    needed_symbols.update(sector_etfs.get(item.get("sector")) for item in watchlist if sector_etfs.get(item.get("sector")))

    histories = {}
    for symbol in sorted(needed_symbols):
        try:
            data = fetch_history(symbol, period)
            if data is not None and not data.empty:
                histories[symbol] = analyzer.calculate_technical_indicators(data)
        except Exception:
            histories[symbol] = pd.DataFrame()

    benchmark_data = histories.get(benchmark)
    rows = []

    for item in watchlist:
        ticker = item["ticker"]
        data = histories.get(ticker)
        if data is None or data.empty or len(data) <= 50:
            continue

        latest = data.iloc[-1]
        sector = item.get("sector", "Unknown")
        sector_symbol = sector_etfs.get(sector, benchmark)
        sector_data = histories.get(sector_symbol)
        volume_ratio = latest.get("Volume_ratio", 1)
        rsi = latest.get("RSI", np.nan)
        trend_score = calculate_trend_score(latest)
        volume_score = clip_score((volume_ratio - 1) / 2)
        rsi_score = calculate_rsi_score(rsi)

        row = {
            "Ticker": ticker,
            "Company": item.get("name", ticker),
            "Sector": sector,
            "Price": latest["Close"],
            "RSI": rsi,
            "Volume Ratio": volume_ratio,
            "MA Trend Score": trend_score,
        }

        for label, window in OUTLOOK_WINDOWS.items():
            stock_return = calculate_return(data, window)
            benchmark_return = calculate_return(benchmark_data, window)
            sector_return = calculate_return(sector_data, window)
            benchmark_relative = stock_return - benchmark_return if pd.notna(benchmark_return) else 0
            sector_strength = sector_return - benchmark_return if pd.notna(sector_return) and pd.notna(benchmark_return) else 0

            # Quantitative inputs decide the ranking. Fundamentals and ML
            # prediction stay in the detail view as context, not score drivers.
            raw_score = (
                weights["momentum"] * normalize_return(stock_return)
                + weights["trend"] * trend_score
                + weights["volume"] * volume_score
                + weights["rsi"] * rsi_score
                + weights["sector_strength"] * normalize_return(sector_strength, 0.05)
                + weights["benchmark_relative"] * normalize_return(benchmark_relative, 0.05)
            )
            final_score = round((raw_score + 1) * 50, 1)
            prefix = f"{window}D"
            row[f"{prefix} Momentum"] = stock_return
            row[f"{prefix} Score"] = final_score
            row[f"{prefix} Benchmark Relative"] = benchmark_relative
            row[f"{prefix} Sector Strength"] = sector_strength

        # Keep all window scores, but rank by the user-selected outlook.
        score_col = f"{selected_window}D Score"
        row["Selected Score"] = row[score_col]
        row["Signal Tier"] = calculate_signal_tier(row["Selected Score"])
        row["Benchmark Relative Strength"] = row[f"{selected_window}D Benchmark Relative"]
        row["Sector Strength"] = row[f"{selected_window}D Sector Strength"]
        rows.append(row)

    ranking = pd.DataFrame(rows)
    if ranking.empty:
        return ranking

    ranking = ranking.sort_values("Selected Score", ascending=False).reset_index(drop=True)
    ranking.insert(0, "Rank", ranking.index + 1)
    return ranking


def display_ranked_dashboard(ranking, selected_window, window_label):
    if ranking.empty:
        st.warning("No watchlist data available yet. Try a longer period or refresh data.")
        return

    st.subheader("Canadian Watchlist Signal Ranking")
    st.caption("Scores rank short-term signal strength against the watchlist and XIC.TO. Research only, not financial advice.")

    display_columns = [
        "Rank", "Ticker", "Company", "Sector", "Selected Score", "Signal Tier",
        "5D Momentum", "10D Momentum", "20D Momentum", "RSI", "Volume Ratio",
        "Benchmark Relative Strength", "Sector Strength",
    ]
    # Format only the displayed copy so the underlying ranking DataFrame keeps
    # numeric values for explanations and future backtesting.
    formatted = ranking[display_columns].copy()
    percent_columns = [
        "5D Momentum", "10D Momentum", "20D Momentum",
        "Benchmark Relative Strength", "Sector Strength",
    ]
    for column in percent_columns:
        formatted[column] = formatted[column].map(lambda value: f"{value:+.2%}" if pd.notna(value) else "N/A")
    formatted["RSI"] = formatted["RSI"].map(lambda value: f"{value:.1f}" if pd.notna(value) else "N/A")
    formatted["Volume Ratio"] = formatted["Volume Ratio"].map(lambda value: f"{value:.2f}x" if pd.notna(value) else "N/A")
    formatted["Selected Score"] = formatted["Selected Score"].map(lambda value: f"{value:.1f}")

    st.dataframe(formatted, use_container_width=True, hide_index=True)

    top_rows = ranking.head(3).copy()
    if not top_rows.empty:
        st.write("### Top Signal Notes")
        for _, row in top_rows.iterrows():
            st.info(generate_signal_explanation(row, window_label))

# Streamlit App
def main():
    st.title("🚀 Canadian Stock Signal Intelligence")
    st.markdown("*Rank TSX watchlist opportunities by 1-week, 2-week, and 4-week signal strength, with technical detail and retained ML price prediction.*")
    
    # Sidebar
    st.sidebar.header("📊 Signal Controls")
    st.sidebar.markdown("---")

    watchlist_config = load_watchlist_config()
    weights = load_scoring_weights()
    watchlist = watchlist_config.get("watchlist", [])
    
    outlook_label = st.sidebar.selectbox(
        "🎯 Outlook Window:",
        options=list(OUTLOOK_WINDOWS.keys()),
        index=1
    )
    selected_window = OUTLOOK_WINDOWS[outlook_label]
    
    period = st.sidebar.selectbox(
        "📅 Data Period:",
        options=['6mo', '1y', '2y', '5y'],
        index=1
    )

    ticker_options = {f"{item.get('ticker')} - {item.get('name', item.get('ticker'))}": item.get("ticker") for item in watchlist}
    selected_stock_label = st.sidebar.selectbox(
        "🏢 Detail Stock:",
        options=list(ticker_options.keys()) + ["Custom"],
        index=0
    )

    if selected_stock_label == "Custom":
        symbol = st.sidebar.text_input("Enter Stock Symbol:", value="RY.TO", max_chars=12).upper()
    else:
        symbol = ticker_options[selected_stock_label]
    
    st.sidebar.markdown("---")
    
    # Analysis options
    st.sidebar.subheader("🔧 Analysis Options")
    show_ranking = st.sidebar.checkbox("🏆 Watchlist Ranking", value=True)
    show_technical = st.sidebar.checkbox("📈 Technical Charts", value=True)
    show_performance = st.sidebar.checkbox("📊 Performance Metrics", value=True)
    show_analysis = st.sidebar.checkbox("🧠 AI Market Analysis", value=True)
    show_prediction = st.sidebar.checkbox("🔮 ML Price Prediction", value=True)
    
    st.sidebar.markdown("---")
    
    if st.sidebar.button("🔄 Refresh Data", type="primary"):
        st.cache_data.clear()
        st.rerun()
    
    # Initialize analyzer
    analyzer = StockAnalyzer()

    # The ranking is the MVP's primary workflow. The selected-stock detail below
    # retains the original chart, ML prediction, and fundamentals experience.
    if show_ranking:
        with st.spinner("📡 Scoring Canadian watchlist against XIC.TO..."):
            ranking = score_watchlist(analyzer, watchlist_config, weights, period, selected_window)

        display_ranked_dashboard(ranking, selected_window, outlook_label)
        st.markdown("---")
    
    # Fetch and display data
    with st.spinner(f"📡 Fetching live data for {symbol}..."):
        data, info = analyzer.fetch_stock_data(symbol, period)
    
    if data is None or data.empty:
        st.error(f"❌ Could not fetch data for {symbol}. Please verify the symbol and try again.")
        st.info("💡 Try Canadian symbols like RY.TO, TD.TO, SHOP.TO, SU.TO, or CNQ.TO.")
        return
    
    # Calculate technical indicators
    with st.spinner("⚙️ Calculating technical indicators..."):
        data = analyzer.calculate_technical_indicators(data)
    
    # Main dashboard header
    st.subheader(f"{symbol} Detail View")
    st.markdown("---")
    
    # Key metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    latest_price = data['Close'].iloc[-1]
    prev_price = data['Close'].iloc[-2]
    price_change = latest_price - prev_price
    price_change_pct = (price_change / prev_price) * 100
    
    with col1:
        st.metric(
            label="💰 Current Price",
            value=f"${latest_price:.2f}",
            delta=f"{price_change:.2f} ({price_change_pct:+.2f}%)"
        )
    
    with col2:
        volume = data['Volume'].iloc[-1]
        avg_volume = data['Volume'].rolling(30).mean().iloc[-1]
        volume_change = ((volume - avg_volume) / avg_volume) * 100 if avg_volume > 0 else 0
        st.metric(
            label="📊 Volume",
            value=f"{volume:,.0f}",
            delta=f"{volume_change:+.1f}% vs 30d avg"
        )
    
    with col3:
        if 'RSI' in data.columns and not pd.isna(data['RSI'].iloc[-1]):
            rsi = data['RSI'].iloc[-1]
            rsi_status = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
            st.metric(
                label="⚡ RSI (14)",
                value=f"{rsi:.1f}",
                delta=rsi_status
            )
        else:
            st.metric(label="⚡ RSI (14)", value="N/A")
    
    with col4:
        if 'SMA_20' in data.columns and not pd.isna(data['SMA_20'].iloc[-1]):
            sma_20 = data['SMA_20'].iloc[-1]
            sma_distance = ((latest_price - sma_20) / sma_20) * 100
            st.metric(
                label="📈 vs SMA 20",
                value=f"{sma_distance:+.1f}%",
                delta="Above" if sma_distance > 0 else "Below"
            )
        else:
            st.metric(label="📈 vs SMA 20", value="N/A")
    
    with col5:
        market_cap = info.get('marketCap', 0)
        if market_cap:
            if market_cap > 1e12:
                cap_display = f"${market_cap/1e12:.2f}T"
            elif market_cap > 1e9:
                cap_display = f"${market_cap/1e9:.1f}B"
            else:
                cap_display = f"${market_cap/1e6:.0f}M"
            st.metric(label="🏢 Market Cap", value=cap_display)
        else:
            st.metric(label="🏢 Market Cap", value="N/A")
    
    st.markdown("---")
    
    # Advanced Chart
    if show_technical:
        st.subheader("📈 Advanced Technical Analysis")
        with st.spinner("Creating advanced charts..."):
            chart = create_advanced_chart(data, symbol)
            st.plotly_chart(chart, use_container_width=True)
    
    # Performance Metrics
    if show_performance:
        st.subheader("📊 Performance Analysis")
        create_performance_metrics(data, symbol)
    
    # ML Prediction
    if show_prediction:
        st.subheader("🔮 Machine Learning Price Prediction")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            with st.spinner("🤖 Training AI prediction model..."):
                model_info = analyzer.train_prediction_model(data)
            
            if model_info:
                prediction = analyzer.predict_next_price(model_info)
                current_price = data['Close'].iloc[-1]
                predicted_change = ((prediction - current_price) / current_price) * 100
                
                st.success("✅ Model trained successfully!")
                
                pred_col1, pred_col2 = st.columns(2)
                with pred_col1:
                    st.metric(
                        label="🎯 Next Day Prediction",
                        value=f"${prediction:.2f}",
                        delta=f"{predicted_change:+.2f}%"
                    )
                
                with pred_col2:
                    confidence = model_info['test_score']
                    confidence_level = "High" if confidence > 0.8 else "Medium" if confidence > 0.6 else "Low"
                    st.metric(
                        label="🎲 Model Confidence",
                        value=f"{confidence:.1%}",
                        delta=confidence_level
                    )
                
                # Model performance
                st.info(f"📈 **Training Accuracy:** {model_info['train_score']:.1%} | **Test Accuracy:** {model_info['test_score']:.1%}")
            else:
                st.warning("⚠️ Insufficient data for reliable ML prediction. Need more historical data.")
        
        with col2:
            if model_info:
                # Feature importance
                importance_df = pd.DataFrame(
                    list(model_info['feature_importance'].items()),
                    columns=['Feature', 'Importance']
                ).sort_values('Importance', ascending=False).head(10)
                
                fig_importance = px.bar(
                    importance_df, 
                    x='Importance', 
                    y='Feature',
                    orientation='h',
                    title="🔍 Top 10 Most Important Features",
                    template='plotly_dark'
                )
                fig_importance.update_layout(height=400)
                st.plotly_chart(fig_importance, use_container_width=True)
    
    # AI Market Analysis
    if show_analysis:
        st.subheader("🧠 AI-Powered Market Analysis")
        
        with st.spinner("🤖 Generating intelligent market insights..."):
            analysis = analyzer.generate_market_analysis(data, info, symbol)
        
        # Display analysis in an attractive format
        for i, insight in enumerate(analysis):
            if i == 0:  # First insight (price movement) gets special treatment
                if "🚀" in insight or "🟢" in insight:
                    st.success(insight)
                elif "🔴" in insight or "🔻" in insight:
                    st.error(insight)
                else:
                    st.warning(insight)
            else:
                st.info(insight)
    
    # Additional Analysis Tabs
    st.markdown("---")
    
    tab1, tab2, tab3 = st.tabs(["📋 Company Info", "📊 Raw Data", "🔧 Technical Indicators"])
    
    with tab1:
        if info:
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("### 🏢 Company Details")
                company_info = {
                    "Company Name": info.get('longName', 'N/A'),
                    "Sector": info.get('sector', 'N/A'),
                    "Industry": info.get('industry', 'N/A'),
                    "Country": info.get('country', 'N/A'),
                    "Website": info.get('website', 'N/A'),
                    "Employees": f"{info.get('fullTimeEmployees', 'N/A'):,}" if info.get('fullTimeEmployees') else 'N/A'
                }
                
                for key, value in company_info.items():
                    st.write(f"**{key}:** {value}")
            
            with col2:
                st.write("### 📈 Financial Metrics")
                financial_info = {
                    "P/E Ratio": f"{info.get('trailingPE', 'N/A'):.2f}" if info.get('trailingPE') else 'N/A',
                    "Forward P/E": f"{info.get('forwardPE', 'N/A'):.2f}" if info.get('forwardPE') else 'N/A',
                    "PEG Ratio": f"{info.get('pegRatio', 'N/A'):.2f}" if info.get('pegRatio') else 'N/A',
                    "Price to Book": f"{info.get('priceToBook', 'N/A'):.2f}" if info.get('priceToBook') else 'N/A',
                    "Dividend Yield": f"{info.get('dividendYield', 0)*100:.2f}%" if info.get('dividendYield') else 'N/A',
                    "Beta": f"{info.get('beta', 'N/A'):.2f}" if info.get('beta') else 'N/A',
                    "52W High": f"${info.get('fiftyTwoWeekHigh', 'N/A'):.2f}" if info.get('fiftyTwoWeekHigh') else 'N/A',
                    "52W Low": f"${info.get('fiftyTwoWeekLow', 'N/A'):.2f}" if info.get('fiftyTwoWeekLow') else 'N/A'
                }
                
                for key, value in financial_info.items():
                    st.write(f"**{key}:** {value}")
        else:
            st.warning("Company information not available")
    
    with tab2:
        st.write("### 📊 Recent Price Data")
        display_data = data[['Open', 'High', 'Low', 'Close', 'Volume']].tail(20)
        display_data.index = display_data.index.strftime('%Y-%m-%d')
        st.dataframe(display_data, use_container_width=True)
        
        # Download option
        csv = display_data.to_csv()
        st.download_button(
            label="📥 Download Data as CSV",
            data=csv,
            file_name=f'{symbol}_stock_data.csv',
            mime='text/csv'
        )
    
    with tab3:
        st.write("### 🔧 Technical Indicators (Last 10 Days)")
        
        tech_columns = ['Close', 'SMA_20', 'SMA_50', 'RSI', 'MACD', 'MACD_signal', 'BB_upper', 'BB_lower', 'ATR']
        available_columns = [col for col in tech_columns if col in data.columns]
        
        if available_columns:
            tech_data = data[available_columns].tail(10)
            tech_data.index = tech_data.index.strftime('%Y-%m-%d')
            st.dataframe(tech_data.round(3), use_container_width=True)
        else:
            st.warning("Technical indicators not available")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; padding: 20px;'>
            <p>🚀 <strong>AI Stock Dashboard</strong> - Professional technical analysis with machine learning</p>
            <p><em>⚠️ This is for educational purposes only. Not financial advice.</em></p>
            <p>Built with ❤️ by <a href='https://erikthiart.com' target='_blank'>Erik Thiart</a></p>
            <p>📊 Powered by <a href='https://plotly.com' target='_blank'>Plotly</a> and <a href='https://streamlit.io' target='_blank'>Streamlit</a></p>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
