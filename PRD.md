# Product Requirements Document (PRD)
# Predictive Price Forecasting System

---

## 1. Overview

**Product Name:** Predictive Price Forecasting  
**Version:** 1.0  
**Last Updated:** September 29, 2026  

### 1.1 Purpose

An automated daily forecasting system that uses the Kronos foundation model to generate probabilistic price forecasts for stocks, crypto, commodities, and forex. Forecasts are delivered as a styled newsletter email every morning before market open.

### 1.2 Problem Statement

Retail investors lack access to institutional-grade quantitative forecasting tools. Existing solutions require expensive data subscriptions, specialized hardware, or deep ML expertise. This project democratizes AI-powered price forecasting by:

- Using a free, open-source model (Kronos, AAAI 2026)
- Running entirely in the cloud via GitHub Actions (no PC required)
- Delivering insights via email -- zero UI to maintain
- Requiring zero ongoing cost (free GitHub Actions + free Yahoo Finance data)

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────┐
│                  GitHub Actions                      │
│  (Daily cron: 2:00 AM UTC / 4:00 AM CEST)          │
│                                                      │
│  ┌─────────┐    ┌──────────┐    ┌────────────────┐  │
│  │ Yahoo   │───>│ Kronos   │───>│ Newsletter     │  │
│  │ Finance │    │ Model    │    │ Generator      │  │
│  │ (data)  │    │ (predict)│    │ (HTML email)   │  │
│  └─────────┘    └──────────┘    └───────┬────────┘  │
│                                          │           │
│                                    ┌─────▼─────┐    │
│                                    │  Gmail     │    │
│                                    │  SMTP      │    │
│                                    └─────┬─────┘    │
└──────────────────────────────────────────┼──────────┘
                                           │
                                    ┌──────▼──────┐
                                    │  User Inbox  │
                                    └─────────────┘
```

### 2.1 Components

| Component           | Technology           | Description                              |
|---------------------|----------------------|------------------------------------------|
| Data Source          | Yahoo Finance (yfinance) | Free OHLCV candle data, no API key   |
| Forecasting Model   | Kronos-base (102.3M)  | Pre-trained candlestick foundation model |
| Scheduler           | GitHub Actions         | Daily cron, runs on free cloud runners   |
| Email Delivery      | Gmail SMTP             | HTML newsletter via App Password         |
| Configuration       | JSON file              | Ticker lists, model params, email settings|
| Chart Storage       | Local `forecasts/charts/` | Auto-managed, old charts replaced     |

---

## 3. Functional Requirements

### 3.1 Forecasting Engine (P0 -- Critical)

| ID    | Requirement                                                    | Status |
|-------|----------------------------------------------------------------|--------|
| FR-01 | Download latest OHLCV data from Yahoo Finance for each ticker  | Done   |
| FR-02 | Load Kronos-base model and tokenizer from Hugging Face         | Done   |
| FR-03 | Generate 20 sampled futures per ticker (configurable)          | Done   |
| FR-04 | Calculate 5th/50th/95th percentile forecast bands              | Done   |
| FR-05 | Detect signal direction (Bullish/Bearish/Neutral)              | Done   |
| FR-06 | Measure forecast confidence from band width                    | Done   |
| FR-07 | Support both hourly (crypto) and daily (stocks) intervals      | Done   |

### 3.2 Newsletter Email (P0 -- Critical)

| ID    | Requirement                                                    | Status |
|-------|----------------------------------------------------------------|--------|
| FR-08 | Generate HTML newsletter with market overview paragraph        | Done   |
| FR-09 | Include per-asset insight sentences (narrative style)           | Done   |
| FR-10 | Show scoreboard (bullish/neutral/bearish counts)               | Done   |
| FR-11 | Color-coded signal badges and percentage changes               | Done   |
| FR-12 | Include "How to read this report" section                      | Done   |
| FR-13 | Include disclaimer footer                                      | Done   |
| FR-14 | No chart attachments -- charts stored locally only             | Done   |

### 3.3 Scheduling & Cloud (P0 -- Critical)

| ID    | Requirement                                                    | Status |
|-------|----------------------------------------------------------------|--------|
| FR-15 | Run daily at 2:00 AM UTC via GitHub Actions cron               | Done   |
| FR-16 | Support manual trigger from GitHub Actions UI                  | Done   |
| FR-17 | Read email credentials from GitHub Secrets (cloud mode)        | Done   |
| FR-18 | Cache model weights and pip packages between runs              | Done   |

### 3.4 Chart Management (P0 -- Critical)

| ID    | Requirement                                                    | Status |
|-------|----------------------------------------------------------------|--------|
| FR-19 | Save charts to `forecasts/charts/` directory                   | Done   |
| FR-20 | Auto-delete previous chart for same ticker when new one created| Done   |
| FR-21 | Chart filename format: `{TICKER}_{DATE}.png`                  | Done   |

### 3.5 Ticker Configuration (P1 -- Important)

| ID    | Requirement                                                    | Status |
|-------|----------------------------------------------------------------|--------|
| FR-22 | Active tickers configurable via JSON                           | Done   |
| FR-23 | Inactive ticker clusters organized by sector                   | Done   |
| FR-24 | Support categories: crypto, stocks, commodities, forex         | Done   |
| FR-25 | Easy activation: move cluster from inactive to active          | Done   |

---

## 4. Non-Functional Requirements

| ID     | Requirement                              | Target                      |
|--------|------------------------------------------|-----------------------------|
| NFR-01 | Total run time (10 tickers, CPU)         | < 15 minutes                |
| NFR-02 | Total run time (10 tickers, GPU)         | < 2 minutes                 |
| NFR-03 | GitHub Actions free tier compatibility   | < 60 min/run, < 2000 min/mo |
| NFR-04 | Email delivery reliability               | Gmail SMTP with TLS         |
| NFR-05 | No API keys required for data            | Yahoo Finance (free)        |
| NFR-06 | Secrets security                         | GitHub Secrets (cloud), local JSON (dev) |

---

## 5. Data Flow

1. **Trigger:** GitHub Actions cron fires at 2:00 AM UTC
2. **Data Pull:** yfinance downloads latest candles for each active ticker
3. **Inference:** Kronos-base runs 20 forward passes per ticker on CPU
4. **Aggregation:** Percentile bands + direction signals calculated
5. **Newsletter:** HTML email generated with overview, insights, tables
6. **Delivery:** Email sent via Gmail SMTP
7. **Artifacts:** Charts saved locally, JSON summary saved, old charts cleaned up

---

## 6. Tracked Assets (v1.0)

### Active (10 tickers)

| Category         | Tickers                                       | Interval |
|------------------|-----------------------------------------------|----------|
| Crypto           | BTC-USD, ETH-USD                              | Hourly   |
| Magnificent 7    | AAPL, MSFT, AMZN, NVDA, GOOGL, META, TSLA    | Daily    |
| Commodities      | GC=F (Gold)                                   | Daily    |

### Inactive Clusters (available for activation)

11 sector clusters with 100+ tickers ready to activate. See `forecast_config.json`.

---

## 7. Future Enhancements (Backlog)

| Priority | Feature                                                |
|----------|--------------------------------------------------------|
| P2       | Track forecast accuracy over time (predicted vs actual)|
| P2       | Web dashboard for historical forecast visualization    |
| P2       | Telegram/Discord bot delivery option                   |
| P3       | Fine-tune Kronos on specific sectors for better accuracy|
| P3       | Multi-model ensemble (mini + small + base)             |
| P3       | Intraday alerts when price breaks forecast band        |

---

## 8. Risks & Mitigations

| Risk                                    | Mitigation                                    |
|-----------------------------------------|-----------------------------------------------|
| Yahoo Finance API changes/rate limits   | Graceful skip per ticker; retry logic          |
| GitHub Actions downtime                 | Manual trigger available; local fallback       |
| Model generates unrealistic forecasts   | Band width = confidence; disclaimer in email   |
| Gmail blocks sending                    | App Password auth; monitor delivery            |
| User trades on forecasts alone          | Prominent disclaimers in every email           |

---

## 9. Success Metrics

- Daily email delivered before market open (4:00 AM CEST)
- 100% of active tickers forecasted (skip gracefully on data errors)
- Newsletter opens (track via Gmail read receipts if desired)
- Forecast accuracy tracking (future enhancement)
