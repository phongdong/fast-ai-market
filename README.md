# Stock Price Prediction App

A Streamlit web app that predicts short-term stock movements using machine learning (XGBoost). Enter a ticker symbol to download historical data, train a model, and view interactive predictions and charts.

**This is for educational and research purposes only — not financial advice.**

## Features

- Download historical stock data for any ticker via yfinance
- Engineered features: returns, moving averages, volatility, volume signals
- XGBoost classifier predicting next-day up/down movement
- Time-based train/test split (no data leakage)
- Interactive Plotly charts: price history, predictions, feature importance
- Simple backtest comparing model strategy vs buy-and-hold
- Latest trading signal with confidence score

## Project Structure

```
stock-prediction-app/
├── app.py              # Streamlit web app (main entry point)
├── train_model.py      # Standalone CLI training script
├── requirements.txt    # Python dependencies
├── README.md           # This file
└── src/
    ├── __init__.py
    ├── data.py          # Data download and caching
    ├── features.py      # Feature engineering
    ├── model.py         # Model training and evaluation
    ├── backtest.py      # Strategy backtest
    └── plots.py         # Plotly chart functions
```

## Run Locally

### Prerequisites

- Python 3.10 or higher

### Setup

```bash
# Clone / navigate to the project
cd stock-prediction-app

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run the App

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

### Run the CLI Training Script

```bash
python train_model.py --ticker AAPL --period 5y
```

## Deploy on Streamlit Community Cloud (Free)

### Step 1: Push to GitHub

```bash
# Initialize git repo (if not already)
cd stock-prediction-app
git init
git add .
git commit -m "Initial commit: stock prediction app"

# Create a repo on GitHub, then:
git remote add origin https://github.com/YOUR_USERNAME/stock-prediction-app.git
git branch -M main
git push -u origin main
```

### Step 2: Deploy

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click **"New app"**
4. Select your repository: `YOUR_USERNAME/stock-prediction-app`
5. Set **Main file path**: `app.py`
6. Click **"Deploy"**

That's it! Your app will be live at `https://YOUR_USERNAME-stock-prediction-app.streamlit.app`

### No Environment Variables Required

The app has no secrets or API keys. yfinance uses public Yahoo Finance data.

## Alternative: Deploy on Hugging Face Spaces

1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space)
2. Select **Streamlit** as the SDK
3. Push your code to the Space's git repo
4. The app will auto-deploy

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| "No data found for ticker" | Check the ticker symbol is valid on Yahoo Finance |
| App is slow on first run | yfinance download takes a few seconds; subsequent runs use Streamlit's cache |
| Deployment fails | Ensure `requirements.txt` is in the repo root and `app.py` is the main file |
| XGBoost install fails | Try `pip install xgboost --no-cache-dir`, or use Python 3.10+ |

## How It Works

1. **Data**: Downloads daily OHLCV data from Yahoo Finance via yfinance
2. **Features**: Computes technical indicators (returns, moving averages, volatility, volume ratios) using only past data
3. **Model**: Trains an XGBoost binary classifier (up vs down) on the training period
4. **Evaluation**: Tests on the held-out chronological test set with accuracy, precision, recall, F1
5. **Backtest**: Compares a simple long-only strategy (hold when predicting UP, cash when predicting DOWN) against buy-and-hold
6. **Signal**: Shows the model's prediction for the next trading day

## Disclaimer

This application is for **educational and research purposes only**. It is **not financial advice**. Stock market predictions are inherently uncertain. Past performance does not guarantee future results. Always do your own research before making investment decisions.
