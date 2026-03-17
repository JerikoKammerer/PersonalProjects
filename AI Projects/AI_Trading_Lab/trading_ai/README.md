# AI Trading Lab

A full-featured Windows desktop application for AI-powered trading strategy development, backtesting, and paper trading simulation.

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue)

---

## Features

### 1. Market Data Explorer
- **Stocks**: 25+ popular tickers (AAPL, MSFT, NVDA, etc.) plus any custom symbol
- **Crypto**: BTC, ETH, SOL, DOGE, and more via Yahoo Finance
- **Futures**: S&P 500, Nasdaq, Crude Oil, Gold, VIX, and other futures contracts
- **Options**: View nearest-expiration options chains for any stock
- Configurable period (1 day → max) and interval (1 minute → 1 month)
- Local file caching for fast reloads

### 2. Technical Indicators (20+)
| Category   | Indicators                                                  |
|------------|-------------------------------------------------------------|
| Trend      | SMA, EMA, DEMA, MACD, ADX, Ichimoku Cloud, Parabolic SAR   |
| Volatility | Bollinger Bands, ATR, Keltner Channel                       |
| Momentum   | RSI, Stochastic, Williams %R, CCI, ROC, MFI                |
| Volume     | OBV, VWAP, Accumulation/Distribution Line                   |

All indicators are configurable with custom parameters and displayed as chart overlays.

### 3. AI Model Training
- **Random Forest** — Robust ensemble classifier
- **Gradient Boosting** — High-accuracy boosted trees
- **Support Vector Machine** — Kernel-based classification
- **LSTM Neural Network** — Deep learning for sequence data (requires TensorFlow)

Each model uses your selected indicators as features and predicts whether price will go up or down over a configurable horizon. Models include full evaluation metrics: accuracy, precision, recall, F1, confusion matrix.

### 4. Backtesting Engine
- Walk-forward historical simulation
- Configurable initial capital, commission, slippage
- Stop-loss and take-profit support
- Short selling option
- Full trade log with entry/exit prices and P&L
- Equity curve and drawdown visualization
- Key metrics: Sharpe ratio, Sortino ratio, max drawdown, win rate, profit factor
- Buy-and-hold benchmark comparison
- Side-by-side model comparison

### 5. Paper Trading Simulator
- Virtual portfolio with real-time market data
- AI-powered trade signals from any trained model
- Manual or auto-trade mode
- Position tracking with unrealized P&L
- Full trade history and equity curve
- Session save/load to persist across restarts
- Auto-refresh every 60 seconds

---

## Installation

### Prerequisites
- **Python 3.10 or later** — [Download from python.org](https://www.python.org/downloads/)
- **Windows 10/11** (also works on macOS/Linux)

### Setup

```bash
# 1. Clone or download this project
cd trading_ai

# 2. (Recommended) Create a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the application
python main.py
```

### Dependency Notes
- **TensorFlow** is optional — LSTM models require it, but the app works fine without it using the other 3 model types.
- If you have issues installing TensorFlow, you can remove it from `requirements.txt` and still use Random Forest, Gradient Boosting, and SVM.
- On Apple Silicon Macs, use `pip install tensorflow-macos` instead.

---

## Quick Start Guide

### Step 1: Load Data
1. Open the **Market Data** tab
2. Type a symbol (e.g., `AAPL`) or click a quick-pick button
3. Select period and interval, then click **Fetch Data**
4. Use **Batch Fetch** to load multiple symbols at once

### Step 2: Select Indicators
1. Switch to the **Indicators** tab
2. Check the indicators you want to use as features (e.g., RSI, MACD, Bollinger Bands)
3. Click **Apply to Chart** to preview the indicators on the price chart

### Step 3: Train a Model
1. Go to the **Train Models** tab
2. Select a symbol, model type, and configure hyperparameters
3. Click **Train Model**
4. Review accuracy, F1 score, and the classification report
5. Optionally **Save Model** for later use

### Step 4: Backtest
1. Open the **Backtest** tab
2. Select a trained model from the dropdown
3. Configure capital, commission, stop-loss, etc.
4. Click **Run Backtest**
5. Analyze the equity curve, drawdown, trade log, and performance metrics

### Step 5: Paper Trade
1. Go to **Paper Trading** and click **New Session**
2. Select a symbol and model, then click **Get AI Signal**
3. Use **BUY** / **SELL** buttons or enable **Auto-Trade**
4. Monitor your virtual portfolio in real time

---

## Project Structure

```
trading_ai/
├── main.py              # Application entry point
├── gui.py               # PyQt5 GUI (all 5 tabs)
├── data_manager.py      # Market data fetching & caching
├── indicators.py        # 20+ technical indicators
├── models.py            # ML model training & management
├── backtester.py        # Historical backtesting engine
├── paper_trader.py      # Paper trading simulator
├── requirements.txt     # Python dependencies
├── cache/               # Cached market data (auto-created)
├── saved_models/        # Persisted trained models (auto-created)
└── paper_sessions/      # Saved paper trading sessions (auto-created)
```

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│                    gui.py                       │
│  ┌──────────┬──────────┬──────────┬───────────┐ │
│  │  Data    │Indicators│ Training │ Backtest  │ │
│  │  Tab     │  Tab     │   Tab    │   Tab     │ │
│  └────┬─────┴────┬─────┴────┬─────┴─────┬─────┘ │
│       │          │          │           │       │
│  ┌────▼──────┐ ┌─▼────────┐│     ┌─────▼─────┐ │
│  │data_      │ │indicators││     │backtester │ │
│  │manager.py │ │.py       ││     │.py        │ │
│  └───────────┘ └──────────┘│     └───────────┘ │
│                       ┌────▼────┐               │
│                       │models.py│               │
│                       └────┬────┘               │
│                       ┌────▼────────────┐       │
│                       │paper_trader.py  │       │
│                       └─────────────────┘       │
└─────────────────────────────────────────────────┘
```

---

## Disclaimer

**This application is for educational and research purposes only.**

- It does not constitute financial advice.
- Past performance does not guarantee future results.
- AI model predictions are probabilistic, not certain.
- Paper trading results may differ significantly from real trading due to factors like liquidity, market impact, and execution speed.
- Always consult a qualified financial advisor before making real investment decisions.

---

## License

MIT License — free for personal and commercial use.
