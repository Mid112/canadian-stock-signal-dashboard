# 🚀 Canadian Stock Signal Intelligence

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> **A Canadian stock signal ranking dashboard for short-term watchlist research**

This app ranks a focused Canadian stock watchlist by 1-week, 2-week, and 4-week outlook. It combines momentum, 30-day volume surge, RSI, moving-average trend, sector strength, and benchmark-relative strength against `XIC.TO`.

The retained machine learning price prediction and company fundamentals are secondary detail tools for a selected ticker. The core MVP ranking does not depend on exact-price prediction or fundamentals.

![Main Dashboard](screenshots/main_dashboard.jpg)

## ✨ Features

### 🤖 **Artificial Intelligence**
- **Watchlist Signal Ranking** - Canadian stocks ranked by short-term technical strength
- **Machine Learning Price Prediction** - Retained experimental detail view for selected tickers
- **AI Market Analysis** - Natural language insights based on technical indicators
- **Feature Importance Analysis** - Understand what drives price movements
- **Model Performance Metrics** - Train/test accuracy with confidence levels

### 🇨🇦 **Canadian Signal Engine**
- **1W, 2W, and 4W Outlooks** - 5, 10, and 20 trading-day ranking windows
- **Benchmark Comparison** - Measures relative strength versus `XIC.TO`
- **Sector Strength** - Compares mapped sector ETFs against the Canadian benchmark
- **Signal Tiers** - Strong Buy Candidate, Watch Closely, Neutral, Weak, Bearish / Ignore
- **Historical Backtesting** - Tests whether high-scoring signals beat `XIC.TO` after the selected holding window
- **Paper Trading Simulator** - Single-user virtual portfolio with market buy/sell orders and local transaction history

### 📈 **Advanced Technical Analysis**
- **Professional Charts** - Multi-panel candlestick charts with technical overlays
- **20+ Technical Indicators** - RSI, MACD, Bollinger Bands, Moving Averages, Stochastic
- **Volume Analysis** - Volume trends and confirmation signals
- **Performance Metrics** - Sharpe ratio, volatility, maximum drawdown

### 🎯 **Real-Time Data**
- **Latest Available Stock Data** - Yahoo Finance data through `yfinance`, usually delayed rather than a paid real-time exchange feed
- **Multiple Timeframes** - 6M to 5Y analysis periods
- **Canadian Watchlist Config** - Edit `config/watchlist.yaml` to change tracked stocks
- **Custom Symbol Input** - Analyze any publicly traded stock

### 🎨 **Professional Interface**
- **Dark Theme** - Easy on the eyes for extended analysis
- **Responsive Design** - Works perfectly on desktop and mobile
- **Interactive Charts** - Zoom, pan, and explore data
- **Organized Tabs** - Clean separation of different analysis types

![Technical Analysis](screenshots/technical_analysis.jpg)

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.8 or higher
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/erikthiart/ai-stock-dashboard.git
cd ai-stock-dashboard
```

2. **Create and activate a virtual environment**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the application**
```bash
streamlit run stock_dashboard.py
```

5. **Open your browser**
```
Navigate to http://localhost:8501
```

![ML Predictions](screenshots/ml_predictions.jpg)

## 📦 Dependencies

```
streamlit>=1.28.0
yfinance>=0.2.18
pandas>=1.5.0
numpy>=1.24.0
plotly>=5.15.0
scikit-learn>=1.3.0
PyYAML>=6.0.0
```

## 🎮 How to Use

### 1. **Rank the Canadian Watchlist**
- Choose an outlook window: 1 week, 2 weeks, or 4 weeks
- Review the ranked table of Canadian stocks
- Use the signal tier and explanation as research context

### 2. **Open a Stock Detail View**
- Select a ticker from the configured watchlist
- **Technical Charts**: Advanced multi-panel analysis
- **Performance**: Risk metrics and cumulative returns
- **ML Prediction**: Retained experimental next-day price forecast
- **Market Analysis**: AI-generated insights
- **Company Info**: Fundamentals such as sector, P/E, dividend yield, beta, and 52-week range
- **Backtest Signals**: Select dates, score threshold, and holding window to test historical outperformance

### 3. **Use Paper Trading**
- Start with `$10,000` virtual cash
- Place simple market buy or sell orders for whole shares
- Review the portfolio tab for cash, positions, average cost, market value, unrealized P/L, realized P/L, and account value
- Review transaction history for each filled or rejected order
- Reset the local paper account when you want a fresh practice portfolio

### 4. **Understand the Insights**
- **Ranking score**: Uses technical and benchmark-relative signals only
- **Company fundamentals**: Secondary context, not part of the MVP ranking score
- **ML prediction**: Experimental selected-stock detail, not the core ranking engine
- **Paper trading fills**: Educational simulation only; fills use the latest available `yfinance` price, not a real broker quote or live order book

![Performance Metrics](screenshots/performance_metrics.jpg)

## 💼 Paper Trading Scope

The simulator is intentionally small enough to live inside this Streamlit dashboard:

- Single user only, with no login or hosted accounts
- Local SQLite storage at `data/paper_trading.db`
- `$10,000` starting virtual cash
- Market buy and sell orders only
- Whole-share long positions only
- No limit orders, stop orders, margin, shorting, FX, commissions, slippage, partial fills, or market-hours checks

### Expected Data Delay

The app uses Yahoo Finance data through `yfinance`, not a paid exchange feed. Expect prices to be delayed, commonly around 15-20 minutes where intraday prices are available. When markets are closed, or when Yahoo only returns daily candles for a symbol, paper trades may fill at the most recent daily close.

### Portfolio Tab Components

The portfolio view tracks virtual cash, positions value, total account value, total return, realized P/L, ticker, quantity, average cost, latest available price, cost basis, market value, unrealized P/L, unrealized return percentage, and price timestamp.

### Transaction Details Stored

Each paper order records timestamp, ticker, buy/sell side, quantity, fill price, gross amount, realized P/L for sells, cash balance after the order, status, rejection reason when applicable, and price source.

## 🧠 Machine Learning Model

Our AI uses a **Random Forest Regressor** trained on 30+ features including:

- **Price-based features**: Returns, volatility, price changes
- **Technical indicators**: RSI, MACD, moving averages
- **Volume features**: Volume ratios and trends  
- **Lag features**: Historical price and volume data
- **Statistical features**: Rolling means and standard deviations

**Model Performance:**
- Real-time training on historical data
- Cross-validation with train/test splits
- Feature importance analysis
- Confidence metrics displayed

![AI Analysis](screenshots/ai_analysis.jpg)

## 📊 Technical Indicators

| Indicator | Purpose | Interpretation |
|-----------|---------|----------------|
| **RSI** | Momentum | >70 Overbought, <30 Oversold |
| **MACD** | Trend | Signal line crossovers |
| **Bollinger Bands** | Volatility | Price vs. bands position |
| **Moving Averages** | Trend | Price vs. MA relationships |
| **Stochastic** | Momentum | %K and %D oscillator |
| **Volume** | Confirmation | Volume vs. average ratios |

## 🎯 Use Cases

### 📈 **For Traders**
- Quick technical analysis of any stock
- AI-powered price predictions for next trading day
- Volume confirmation signals
- Multiple timeframe analysis

### 💼 **For Investors**
- Long-term performance metrics
- Risk assessment (volatility, drawdown)
- Company fundamental information
- Market trend analysis

### 🎓 **For Learning**
- Understanding technical indicators
- Machine learning in finance
- Market behavior patterns
- Professional chart analysis

![Company Info](screenshots/company_info.jpg)

## ⚠️ Disclaimer

**This tool is for educational and informational purposes only.**

- Not financial advice or investment recommendations
- Past performance doesn't guarantee future results
- Always do your own research before investing
- Consider consulting with financial professionals
- Markets involve risk and potential loss of capital

## 🛠️ Technical Architecture

```
├── stock_dashboard.py      # Main application
├── requirements.txt        # Dependencies
├── README.md              # Documentation
└── screenshots/           # UI screenshots
    ├── main_dashboard.jpg
    ├── technical_analysis.jpg
    ├── ml_predictions.jpg
    ├── performance_metrics.jpg
    ├── ai_analysis.jpg
    └── company_info.jpg
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🌟 Acknowledgments

- **Yahoo Finance** for providing free stock data
- **Streamlit** for the amazing web framework
- **Plotly** for interactive visualizations
- **scikit-learn** for machine learning capabilities

## 📞 Support

If you find this project helpful, please give it a ⭐ on GitHub!

For questions or issues:
- Open an [Issue](https://github.com/erikthiart/ai-stock-dashboard/issues)

---

<div align="center">

**Built with ❤️ and Python**

[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/erikthiart)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/erikthiart)

</div>
