# Predictive Price Forecasting

> Automated daily market forecasts powered by [Kronos](https://github.com/shiyu-coder/Kronos), the first open-source foundation model for financial candlestick data.

---

## What This Does

Every night after the U.S. market close, this system:

1. **Downloads the latest price data** for your tracked assets (via Yahoo Finance)
2. **Runs the Kronos-base model** (102.3M parameters) to generate 20 sampled futures per asset
3. **Calculates a forecast range** (5th-95th percentile) and the likely price path (median)
4. **Sends you a newsletter email** with a market overview, per-asset insights, and signal table
5. **Saves three-month EUR charts** in `forecasts/charts/` (one per asset, auto-replaced daily), while the email reports both USD and EUR

**Runs 100% in the cloud via GitHub Actions** -- no need to keep your PC on.

---

## How the Model Works

Kronos is **not** a trading bot. It doesn't read news, scan social media, or check order books. It's a pure **pattern-matching model** trained on 12B+ candlestick records from 45+ global exchanges.

Think of it like autocomplete for price charts: you show it recent candles (Open, High, Low, Close, Volume), and it predicts what comes next based on statistical patterns it learned during training. By sampling 20 separate futures and taking percentiles, you get a **range** rather than a single guess -- the width of the band is the uncertainty.

| What it does                    | What it does NOT do            |
|---------------------------------|--------------------------------|
| Recognizes chart patterns       | Read news or sentiment         |
| Predicts next candles from OHLCV| Search the internet            |
| Gives a probabilistic range     | Understand fundamentals        |
| Run on CPU or GPU               | Guarantee any outcome          |

> **Paper:** [arxiv.org/abs/2508.02739](https://arxiv.org/abs/2508.02739) (AAAI 2026)  
> **Original repo:** [github.com/shiyu-coder/Kronos](https://github.com/shiyu-coder/Kronos)  
> **Models:** [huggingface.co/NeoQuasar](https://huggingface.co/NeoQuasar)

---

## Available Models

| Model        | Params  | Context (candles) | Notes                         |
|--------------|---------|-------------------|-------------------------------|
| Kronos-mini  | 4.1M    | 2048              | Fastest, good for experiments |
| Kronos-small | 24.7M   | 512               | Good balance                  |
| **Kronos-base** | **102.3M** | **512**       | **Used in this project**      |

---

## Quick Start

### Prerequisites

- **Python 3.12** (not 3.14 -- pandas 2.2.2 won't build on it)
- **git**
- ~2 GB disk space (PyTorch + model weights)
- GPU optional (RTX 5070 runs 10 tickers in ~65s; CPU takes ~10min)

### Local Setup

```bash
# 1. Clone and enter the repo
git clone https://github.com/Toro1401/Predictive-Price-Forecasting.git
cd Predictive-Price-Forecasting

# 2. Create a Python 3.12 virtual environment
# Using uv (recommended):
uv venv --python 3.12 .venv
# Or using venv:
python3.12 -m venv .venv

# 3. Activate
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt yfinance
# For GPU (NVIDIA):
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

### Run a Forecast

```bash
# Single ticker (original simple script)
python forecast.py BTC-USD              # Crypto: next 24 hours
python forecast.py AAPL daily           # Stocks: next 5 trading days

# Multi-ticker system with newsletter
python forecast_all.py --all            # All active tickers
python forecast_all.py --crypto         # Just crypto
python forecast_all.py --stocks         # Just stocks
python forecast_all.py --all --email    # Run + send email newsletter
```

---

## Email Newsletter Setup (Gmail)

1. Enable **2-Step Verification** on your Google account
2. Go to [myaccount.google.com/apppasswords](https://myaccount.google.com/apppasswords)
3. Generate an App Password (name it "Kronos")
4. Edit `forecast_config.json`:

```json
{
  "email": {
    "sender": "your.email@gmail.com",
    "password": "your_app_password_here",
    "recipient": "your.email@gmail.com"
  }
}
```

---

## Cloud Deployment (GitHub Actions)

The forecast runs automatically every day at **23:30 Europe/Rome** via GitHub Actions -- **even when your PC is off**. The workflow uses GitHub's native timezone-aware schedule, so daylight-saving changes do not move the local run time.

### Setup

1. Push this repo to GitHub (private recommended)
2. Go to **Settings > Secrets and variables > Actions**
3. Add these 3 secrets:
   - `KRONOS_EMAIL_SENDER` -- your Gmail address
   - `KRONOS_EMAIL_PASSWORD` -- your Gmail App Password
   - `KRONOS_EMAIL_RECIPIENT` -- recipient email address (supports multiple comma-separated emails, e.g. `user1@gmail.com, user2@gmail.com`)
4. The workflow will run daily automatically. You can also trigger it manually from the **Actions** tab.

The newsletter contains an **Open Charts & Artifacts** link to the exact workflow run. After the run completes, download the named `forecasts-*` artifact to access `forecasts/charts/`. A private repository requires GitHub sign-in.

---

## Ticker Configuration

Edit `forecast_config.json` to customize which assets to track. Tickers are organized by category and can be toggled active/inactive.

### Currently Active

| Category       | Tickers                                      |
|----------------|----------------------------------------------|
| **Crypto**     | BTC-USD, ETH-USD                             |
| **Magnificent 7** | AAPL, MSFT, AMZN, NVDA, GOOGL, META, TSLA |
| **Commodities**| GC=F (Gold)                                  |

### Available (Inactive) Categories

The config includes pre-built clusters of stocks by sector that you can activate by moving them from `inactive_clusters` to the active section. See `forecast_config.json` for the full list:

- **AI & Semiconductors** -- AMD, INTC, AVGO, QCOM, ARM, MRVL, MU, TSM, ASML, AMAT, LRCX, KLAC, SNPS, CDNS
- **Energy** -- XOM, CVX, COP, SLB, EOG, OXY, MPC, VLO, PSX, LNG
- **Power & Renewables** -- NEE, CEG, VST, ENPH, FSLR
- **Telecommunications** -- T, VZ, TMUS, CMCSA, CHTR, AMX
- **People Transportation** -- DAL, UAL, AAL, LUV, JBLU, UBER, LYFT
- **Logistics Transportation** -- UPS, FDX, FDXF, UNP, CSX, JBHT, ODFL, XPO, RXO, CHRW, EXPD, GXO, MATX
- **Industrials & Aerospace** -- CAT, DE, HON, GE, MMM, RTX, LMT, BA, NOC
- **Gaming & Entertainment** -- TTWO, RBLX, U, NFLX, DIS, WBD, PSKY, SONY
- **Finance & Banking** -- JPM, GS, MS, BAC, WFC, C, BLK, SCHW, AXP, V, MA, BRK-B
- **Healthcare & Pharma** -- JNJ, UNH, PFE, ABBV, LLY, MRK, TMO, ABT, AMGN, GILD
- **Retail & Consumer** -- WMT, COST, TGT, HD, LOW, NKE, SBUX, MCD, PG, KO

All categories in this section contain stocks only and remain inactive until explicitly moved into the top-level `tickers` object.

---

## Tuning Parameters

| Setting     | Default | What to try                                      |
|-------------|---------|--------------------------------------------------|
| `paths`     | 20      | 50+ for smoother bands (slower)                  |
| `lookback`  | 400     | Keep <= 512 for small/base models                |
| `pred_len`  | 24h crypto / 5d stocks / 20d commodities | Shorter = more reliable; band widens fast |
| `T`         | 1.0     | Lower = tighter, more conservative paths         |
| `top_p`     | 0.9     | Lower = less diverse sampling                    |
| Model       | base    | `Kronos-mini` for speed, `Kronos-base` for accuracy |

---

## Project Structure

```
.
├── model/                  # Kronos model code (from original repo)
│   ├── __init__.py
│   ├── kronos.py
│   └── module.py
├── forecasts/
│   └── charts/             # Three-month EUR charts; older same-ticker chart deleted
├── .github/
│   └── workflows/
│       └── daily_forecast.yml   # GitHub Actions daily cron job
├── forecast.py             # Simple single-ticker forecast script
├── forecast_all.py         # Multi-ticker system with newsletter email
├── forecast_config.json    # Ticker lists, email settings, model params
├── requirements.txt        # Python dependencies
├── LICENSE                 # MIT License
└── README.md               # This file
```

---

## Important Disclaimers

- **This is a research model, not financial advice.** The authors state their pipeline is "not a production-ready quantitative trading system."
- **A forecast range is a probability, not a promise.** Treat the band as one input alongside your own analysis.
- **Paper-trade first.** Run it daily for a few weeks and compare forecasts to actual outcomes before risking real money.
- The model is blind to news, earnings, regulation, and all real-world events. It only knows historical price patterns.

---

## Credits

- **Kronos Model:** [shiyu-coder/Kronos](https://github.com/shiyu-coder/Kronos) (AAAI 2026)
- **Paper:** [arxiv.org/abs/2508.02739](https://arxiv.org/abs/2508.02739)
- **Models:** [huggingface.co/NeoQuasar](https://huggingface.co/NeoQuasar)
- **Live Demo:** [shiyu-coder.github.io/Kronos-demo](https://shiyu-coder.github.io/Kronos-demo/)

Licensed under MIT.
