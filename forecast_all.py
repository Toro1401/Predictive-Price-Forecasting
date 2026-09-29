"""
Kronos Multi-Ticker Forecast System
Runs forecasts for a configurable list of tickers, generates charts locally,
and emails a newsletter-style report with market insights.

Usage:
  python forecast_all.py --all                   # Run all categories
  python forecast_all.py --crypto --stocks       # Run specific categories
  python forecast_all.py --all --email           # Run + send newsletter email
  python forecast_all.py --all --email --cloud   # Cloud mode (reads secrets from env vars)
"""

import sys
import os
import json
import html as html_lib
import re
import time
import datetime
import smtplib
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import torch
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from zoneinfo import ZoneInfo
from model import Kronos, KronosTokenizer, KronosPredictor

# ─── CONFIGURATION ───────────────────────────────────────────────────────────

CONFIG_FILE = Path(__file__).parent / "forecast_config.json"
REPORT_TIMEZONE = ZoneInfo("Europe/Rome")

DEFAULT_CONFIG = {
    "email": {
        "sender": "your.email@gmail.com",
        "password": "your_app_password_here",
        "recipient": "your.email@gmail.com",
        "smtp_server": "smtp.gmail.com",
        "smtp_port": 587
    },
    "model": {
        "name": "NeoQuasar/Kronos-base",
        "tokenizer": "NeoQuasar/Kronos-Tokenizer-base",
        "max_context": 512
    },
    "forecast": {
        "lookback": 400,
        "paths": 20,
        "pred_len_hourly": 24,
        "pred_len_stock": 5,
        "pred_len_commodity": 20,
        "temperature": 1.0,
        "top_p": 0.9
    },
    "currency": {
        "fx_ticker": "EURUSD=X",
        "chart": "EUR",
        "email": ["USD", "EUR"]
    },
    "tickers": {
        "crypto": ["BTC-USD", "ETH-USD"],
        "commodities": ["GC=F"],
        "sp500_top": ["AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA"]
    },
    "inactive_clusters": {
        "ai_semiconductors": ["AMD", "INTC", "AVGO", "QCOM", "ARM", "MRVL", "MU", "TSM", "ASML", "AMAT", "LRCX", "KLAC", "SNPS", "CDNS"],
        "energy": ["XOM", "CVX", "COP", "SLB", "EOG", "OXY", "MPC", "VLO", "PSX", "LNG"],
        "power_renewables": ["NEE", "CEG", "VST", "ENPH", "FSLR"],
        "telecommunications": ["T", "VZ", "TMUS", "CMCSA", "CHTR", "AMX"],
        "people_transportation": ["DAL", "UAL", "AAL", "LUV", "JBLU", "UBER", "LYFT"],
        "logistics_transportation": ["UPS", "FDX", "FDXF", "UNP", "CSX", "JBHT", "ODFL", "XPO", "RXO", "CHRW", "EXPD", "GXO", "MATX"],
        "industrials_aerospace": ["CAT", "DE", "HON", "GE", "MMM", "RTX", "LMT", "BA", "NOC"],
        "videogames_entertainment": ["TTWO", "RBLX", "U", "NFLX", "DIS", "WBD", "PSKY", "SONY"],
        "finance": ["JPM", "GS", "MS", "BAC", "WFC", "C", "BLK", "SCHW", "AXP", "V", "MA", "BRK-B"],
        "healthcare": ["JNJ", "UNH", "PFE", "ABBV", "LLY", "MRK", "TMO", "ABT", "AMGN", "GILD"],
        "retail_consumer": ["WMT", "COST", "TGT", "HD", "LOW", "NKE", "SBUX", "MCD", "PG", "KO"]
    }
}

# ─── FRIENDLY NAMES ─────────────────────────────────────────────────────────

TICKER_NAMES = {
    # Top Crypto
    "BTC-USD": "Bitcoin", "ETH-USD": "Ethereum", "SOL-USD": "Solana",
    "BNB-USD": "BNB", "XRP-USD": "XRP", "ADA-USD": "Cardano",
    "AVAX-USD": "Avalanche", "DOT-USD": "Polkadot", "DOGE-USD": "Dogecoin",
    "LINK-USD": "Chainlink",
    # Commodities
    "GC=F": "Gold", "SI=F": "Silver", "CL=F": "Crude Oil",
    "NG=F": "Natural Gas", "HG=F": "Copper",
    # Magnificent 7
    "AAPL": "Apple", "MSFT": "Microsoft", "AMZN": "Amazon",
    "NVDA": "Nvidia", "GOOGL": "Alphabet", "META": "Meta", "TSLA": "Tesla",
    # AI & Semiconductors
    "AMD": "AMD", "INTC": "Intel", "AVGO": "Broadcom", "QCOM": "Qualcomm",
    "ARM": "ARM Holdings", "MRVL": "Marvell", "MU": "Micron", "TSM": "TSMC",
    "ASML": "ASML", "AMAT": "Applied Materials", "LRCX": "Lam Research",
    "KLAC": "KLA", "SNPS": "Synopsys", "CDNS": "Cadence",
    # Energy
    "XOM": "ExxonMobil", "CVX": "Chevron", "COP": "ConocoPhillips",
    "SLB": "Schlumberger", "EOG": "EOG Resources", "OXY": "Occidental",
    "MPC": "Marathon Petroleum", "VLO": "Valero", "PSX": "Phillips 66",
    "LNG": "Cheniere Energy", "NEE": "NextEra Energy", "CEG": "Constellation Energy",
    "VST": "Vistra", "ENPH": "Enphase", "FSLR": "First Solar",
    # Telco
    "T": "AT&T", "VZ": "Verizon", "TMUS": "T-Mobile", "CMCSA": "Comcast",
    "CHTR": "Charter", "AMX": "America Movil",
    # People transportation
    "DAL": "Delta Air Lines", "UAL": "United Airlines", "LUV": "Southwest Airlines",
    "AAL": "American Airlines", "JBLU": "JetBlue", "UBER": "Uber", "LYFT": "Lyft",
    # Logistics transportation
    "UPS": "UPS", "FDX": "FedEx", "FDXF": "FedEx Freight",
    "UNP": "Union Pacific", "CSX": "CSX Corp", "JBHT": "J.B. Hunt",
    "ODFL": "Old Dominion Freight Line", "XPO": "XPO", "RXO": "RXO",
    "CHRW": "C.H. Robinson", "EXPD": "Expeditors", "GXO": "GXO Logistics",
    "MATX": "Matson",
    # Industrials & Aerospace
    "CAT": "Caterpillar", "DE": "Deere & Co", "HON": "Honeywell",
    "GE": "GE Aerospace", "MMM": "3M", "RTX": "RTX Corp",
    "LMT": "Lockheed Martin", "BA": "Boeing", "NOC": "Northrop Grumman",
    # Videogames & Entertainment
    "TTWO": "Take-Two", "RBLX": "Roblox", "U": "Unity Software",
    "NFLX": "Netflix", "DIS": "Disney", "WBD": "Warner Bros Discovery",
    "PSKY": "Paramount Skydance", "SONY": "Sony",
    # Finance
    "JPM": "JPMorgan Chase", "GS": "Goldman Sachs", "MS": "Morgan Stanley",
    "BAC": "Bank of America", "WFC": "Wells Fargo", "C": "Citigroup",
    "BLK": "BlackRock", "SCHW": "Charles Schwab", "AXP": "American Express",
    "V": "Visa", "MA": "Mastercard", "BRK-B": "Berkshire Hathaway",
    # Healthcare
    "JNJ": "Johnson & Johnson", "UNH": "UnitedHealth", "PFE": "Pfizer",
    "ABBV": "AbbVie", "LLY": "Eli Lilly", "MRK": "Merck",
    "TMO": "Thermo Fisher", "ABT": "Abbott", "AMGN": "Amgen", "GILD": "Gilead",
    # Retail & Consumer
    "WMT": "Walmart", "COST": "Costco", "TGT": "Target", "HD": "Home Depot",
    "LOW": "Lowe's", "NKE": "Nike", "SBUX": "Starbucks", "MCD": "McDonald's",
    "PG": "Procter & Gamble", "KO": "Coca-Cola",
    # Forex
    "EURUSD=X": "EUR/USD", "GBPUSD=X": "GBP/USD", "USDJPY=X": "USD/JPY",
    "AUDUSD=X": "AUD/USD", "USDCAD=X": "USD/CAD", "USDCHF=X": "USD/CHF",
}

def friendly(ticker):
    return TICKER_NAMES.get(ticker, ticker)


def report_now():
    """Return the current timestamp in the report's canonical timezone."""
    return datetime.datetime.now(REPORT_TIMEZONE)


def safe_filename_component(value):
    """Convert a display name or ticker into a readable, portable filename part."""
    return re.sub(r"[^A-Za-z0-9.-]+", "-", value).strip("-.")


def chart_filename(ticker, mode, date_str=None, pred_len=None):
    """Build a descriptive filename such as Bitcoin_BTC-USD_24-hour_forecast_2026-09-29.png."""
    date_str = date_str or report_now().strftime("%Y-%m-%d")
    pred_len = pred_len or (5 if mode == "daily" else 24)
    horizon = f"{pred_len}-trading-day" if mode == "daily" else f"{pred_len}-hour"
    name = safe_filename_component(friendly(ticker))
    ticker_part = safe_filename_component(ticker.replace("=", "-"))
    asset = ticker_part if name.casefold() == ticker_part.casefold() else f"{name}_{ticker_part}"
    return f"{asset}_{horizon}_forecast_{date_str}.png"


def remove_old_charts(charts_dir, ticker, current_chart):
    """Keep only the newest chart for a ticker, including legacy filename formats."""
    ticker_part = safe_filename_component(ticker.replace("=", "-"))
    legacy_ticker = ticker.replace("=", "_")

    for old_chart in charts_dir.glob("*.png"):
        belongs_to_ticker = (
            f"_{ticker_part}_" in old_chart.name
            or old_chart.name.startswith(f"{ticker_part}_")
            or old_chart.name == f"{legacy_ticker}.png"
            or old_chart.name.startswith(f"{legacy_ticker}_")
        )
        if belongs_to_ticker and old_chart != current_chart:
            old_chart.unlink(missing_ok=True)

# ─── HELPERS ─────────────────────────────────────────────────────────────────

def load_config():
    """Load or create configuration file."""
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    else:
        with open(CONFIG_FILE, "w") as f:
            json.dump(DEFAULT_CONFIG, f, indent=2)
        print(f"[!] Created default config at: {CONFIG_FILE}")
        print("    Edit it with your Gmail App Password before running with --email")
        return DEFAULT_CONFIG


def get_email_config(config, cloud_mode=False):
    """Get email config, using env vars in cloud mode."""
    if cloud_mode:
        return {
            "sender": os.environ.get("KRONOS_EMAIL_SENDER", config["email"]["sender"]),
            "password": os.environ.get("KRONOS_EMAIL_PASSWORD", config["email"]["password"]),
            "recipient": os.environ.get("KRONOS_EMAIL_RECIPIENT", config["email"]["recipient"]),
            "smtp_server": config["email"]["smtp_server"],
            "smtp_port": config["email"]["smtp_port"]
        }
    return config["email"]


def get_forecast_length(config, category, mode):
    """Return the configured horizon without conflating stocks and commodities."""
    fc = config["forecast"]
    if mode != "daily":
        return fc["pred_len_hourly"]
    if category == "commodities":
        return fc["pred_len_commodity"]
    return fc["pred_len_stock"]


def fetch_usd_to_eur_rate(config, attempts=3):
    """Fetch the latest EUR/USD close and return the value of one USD in EUR."""
    fx_ticker = config.get("currency", {}).get("fx_ticker", "EURUSD=X")
    last_error = None

    for attempt in range(1, attempts + 1):
        try:
            raw = yf.download(fx_ticker, period="5d", interval="1d", progress=False, auto_adjust=False)
            if raw.empty:
                raise ValueError("no FX candles returned")
            close = raw["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            eur_usd = float(close.dropna().iloc[-1])
            if not np.isfinite(eur_usd) or eur_usd <= 0:
                raise ValueError(f"invalid EUR/USD close: {eur_usd}")
            return 1.0 / eur_usd
        except Exception as exc:
            last_error = exc
            if attempt < attempts:
                time.sleep(2)

    raise RuntimeError(f"Unable to fetch USD/EUR conversion from {fx_ticker}: {last_error}")


def euro_axis_label(value, _position=None):
    """Format chart-axis values in euros without unnecessary decimal noise."""
    absolute = abs(value)
    if absolute >= 1000:
        return f"€{value:,.0f}"
    if absolute >= 1:
        return f"€{value:,.2f}"
    return f"€{value:,.4f}"


def render_forecast_chart(plot_df, y_ts, band, result, chart_path, usd_to_eur):
    """Render a dark, analyst-focused three-month forecast chart in euros."""
    history_x = plot_df["timestamps"]
    history_eur = plot_df["close"] * usd_to_eur
    band_eur = band * usd_to_eur
    start_eur = result["current_price"] * usd_to_eur
    end_eur = result["median_end"] * usd_to_eur
    low_eur = result["low_5"] * usd_to_eur
    high_eur = result["high_95"] * usd_to_eur
    history_change = ((history_eur.iloc[-1] / history_eur.iloc[0]) - 1) * 100 if len(history_eur) > 1 else 0

    palette = {
        "Bullish": "#22c55e",
        "Bearish": "#ef4444",
        "Neutral": "#f59e0b",
    }
    signal_color = palette[result["direction"]]
    background = "#0e1117"
    panel = "#141821"
    text_primary = "#f8fafc"
    text_secondary = "#94a3b8"

    fig, ax = plt.subplots(figsize=(13, 7), facecolor=background)
    ax.set_facecolor(panel)
    ax.axvspan(y_ts.iloc[0], y_ts.iloc[-1], color="#1d4ed8", alpha=0.08, zorder=0)
    ax.plot(history_x, history_eur, color="#cbd5e1", linewidth=1.6, label="3-month history", zorder=3)
    ax.fill_between(
        y_ts,
        band_eur["low_5%"],
        band_eur["high_95%"],
        color="#2563eb",
        alpha=0.24,
        label="90% forecast range",
        zorder=1,
    )
    ax.plot(
        y_ts,
        band_eur["likely_50%"],
        color=signal_color,
        linewidth=2.4,
        label="Median forecast",
        zorder=4,
    )
    ax.axvline(history_x.iloc[-1], color="#64748b", linestyle=(0, (4, 4)), linewidth=1.1, alpha=0.9)
    ax.axhline(start_eur, color="#64748b", linestyle=(0, (3, 5)), linewidth=0.9, alpha=0.7)
    ax.scatter(history_x.iloc[-1], start_eur, s=42, color="#f8fafc", edgecolor=panel, linewidth=1, zorder=5)
    ax.scatter(y_ts.iloc[-1], end_eur, s=54, color=signal_color, edgecolor=panel, linewidth=1, zorder=5)

    ax.annotate(
        f"Latest  {euro_axis_label(start_eur)}",
        (history_x.iloc[-1], start_eur),
        xytext=(-10, 15),
        textcoords="offset points",
        ha="right",
        color=text_primary,
        fontsize=9,
    )
    ax.annotate(
        f"Median  {euro_axis_label(end_eur)}",
        (y_ts.iloc[-1], end_eur),
        xytext=(-8, 15),
        textcoords="offset points",
        ha="right",
        color=signal_color,
        fontsize=9,
        fontweight="bold",
    )

    horizon_unit = "trading days" if result["mode"] == "daily" else "hours"
    fig.suptitle(
        f"{result['name']}  ·  {result['ticker']}",
        x=0.075,
        y=0.965,
        ha="left",
        color=text_primary,
        fontsize=20,
        fontweight="bold",
    )
    ax.set_title(
        f"Three-month price context  |  {result['pred_len']} {horizon_unit} probabilistic outlook",
        loc="left",
        color=text_secondary,
        fontsize=10.5,
        pad=18,
    )
    metrics = (
        f"3M TREND  {history_change:+.2f}%     "
        f"FORECAST  {result['pct_change']:+.2f}%     "
        f"SIGNAL  {result['direction'].upper()}     "
        f"CONFIDENCE  {result['confidence'].upper()}\n"
        f"TERMINAL 90% RANGE  {euro_axis_label(low_eur)} — {euro_axis_label(high_eur)}"
    )
    ax.text(
        0.015,
        0.965,
        metrics,
        transform=ax.transAxes,
        va="top",
        ha="left",
        color=text_primary,
        fontsize=9.5,
        linespacing=1.7,
        bbox={"boxstyle": "round,pad=0.75", "facecolor": "#1e293b", "edgecolor": "#334155", "alpha": 0.95},
    )

    ax.set_ylabel("Price (EUR)", color=text_secondary, labelpad=12)
    ax.yaxis.set_major_formatter(FuncFormatter(euro_axis_label))
    locator = mdates.AutoDateLocator(minticks=5, maxticks=9)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.tick_params(colors=text_secondary, labelsize=9)
    ax.grid(axis="y", color="#334155", linewidth=0.7, alpha=0.45)
    ax.grid(axis="x", visible=False)
    for spine in ax.spines.values():
        spine.set_color("#263244")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        loc="lower left",
        frameon=True,
        facecolor="#1e293b",
        edgecolor="#334155",
        labelcolor=text_primary,
        fontsize=9,
        ncol=3,
    )
    ax.margins(x=0.01)
    fig.text(
        0.075,
        0.025,
        f"Model inference uses native USD prices. Display converted at $1 = €{usd_to_eur:.4f}. Forecasts are probabilistic, not financial advice.",
        color="#64748b",
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0.04, 0.06, 0.98, 0.91))
    fig.savefig(chart_path, dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)


def run_forecast(ticker, mode, category, predictor, config, usd_to_eur):
    """Run forecast for a single ticker. Returns (result_dict, chart_path) or (None, None) on failure."""
    fc = config["forecast"]
    LOOKBACK = fc["lookback"]
    PATHS = fc["paths"]
    DAILY = mode == "daily"
    PRED_LEN = get_forecast_length(config, category, mode)
    INTERVAL, PERIOD, STEP = ("1d", "3y", pd.offsets.BDay(1)) if DAILY else ("1h", "3mo", pd.Timedelta(hours=1))

    # 1. Pull candles
    try:
        raw = yf.download(ticker, period=PERIOD, interval=INTERVAL, progress=False, auto_adjust=False)
        if raw.empty:
            print(f"  [SKIP] {ticker}: no data returned from Yahoo Finance")
            return None, None
        raw.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in raw.columns]
        price_history = raw[["open", "high", "low", "close", "volume"]].dropna().reset_index()
        price_history = price_history.rename(columns={price_history.columns[0]: "timestamps"})
        price_history["timestamps"] = pd.to_datetime(price_history["timestamps"]).dt.tz_localize(None)
        df = price_history.tail(LOOKBACK).reset_index(drop=True)
        chart_cutoff = price_history["timestamps"].iloc[-1] - pd.DateOffset(months=3)
        plot_df = price_history[price_history["timestamps"] >= chart_cutoff].copy()
        if len(df) < 50:
            print(f"  [SKIP] {ticker}: only {len(df)} candles available (need at least 50)")
            return None, None
    except Exception as e:
        print(f"  [SKIP] {ticker}: data error: {e}")
        return None, None

    # 2. Build timestamps
    x_ts = df["timestamps"]
    y_ts = pd.Series([x_ts.iloc[-1] + STEP * (i + 1) for i in range(PRED_LEN)])

    # 3. Sample multiple futures
    paths = []
    for i in range(PATHS):
        try:
            pred = predictor.predict(
                df=df[["open", "high", "low", "close", "volume"]],
                x_timestamp=x_ts,
                y_timestamp=y_ts,
                pred_len=PRED_LEN,
                T=fc["temperature"],
                top_p=fc["top_p"],
                sample_count=1,
                verbose=False
            )
            paths.append(pred["close"].values)
        except Exception as e:
            print(f"  [WARN] {ticker}: path {i} failed: {e}")
            continue

    if len(paths) < 3:
        print(f"  [SKIP] {ticker}: only {len(paths)} paths succeeded, need at least 3")
        return None, None

    closes = pd.DataFrame(paths, columns=y_ts)
    band = pd.DataFrame({
        "low_5%": closes.quantile(0.05),
        "likely_50%": closes.median(),
        "high_95%": closes.quantile(0.95)
    })

    # 4. Determine direction and confidence
    start_price = df["close"].iloc[-1]
    end_median = band["likely_50%"].iloc[-1]
    pct_change = ((end_median - start_price) / start_price) * 100
    band_width = ((band["high_95%"].iloc[-1] - band["low_5%"].iloc[-1]) / start_price) * 100

    # Recent momentum (last 24 candles)
    recent = df["close"].tail(24)
    recent_change = ((recent.iloc[-1] - recent.iloc[0]) / recent.iloc[0]) * 100 if len(recent) >= 2 else 0

    result = {
        "ticker": ticker,
        "name": friendly(ticker),
        "mode": mode,
        "category": category,
        "pred_len": PRED_LEN,
        "current_price": float(start_price),
        "current_price_eur": float(start_price * usd_to_eur),
        "median_end": float(end_median),
        "median_end_eur": float(end_median * usd_to_eur),
        "pct_change": float(pct_change),
        "low_5": float(band["low_5%"].iloc[-1]),
        "high_95": float(band["high_95%"].iloc[-1]),
        "low_5_eur": float(band["low_5%"].iloc[-1] * usd_to_eur),
        "high_95_eur": float(band["high_95%"].iloc[-1] * usd_to_eur),
        "usd_to_eur": float(usd_to_eur),
        "band_width_pct": float(band_width),
        "recent_momentum_pct": float(recent_change),
        "direction": "Bullish" if pct_change > 0.3 else ("Bearish" if pct_change < -0.3 else "Neutral"),
        "confidence": "High" if band_width < 3 else ("Medium" if band_width < 7 else "Low"),
        "horizon": f"{PRED_LEN} {'trading days' if DAILY else 'hours'}",
    }

    # 5. Plot (saved locally, not sent in email)
    charts_dir = Path(__file__).parent / "forecasts" / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    date_str = report_now().strftime("%Y-%m-%d")
    chart_path = charts_dir / chart_filename(ticker, mode, date_str, PRED_LEN)

    # Auto-delete old charts for the same ticker
    remove_old_charts(charts_dir, ticker, chart_path)

    render_forecast_chart(plot_df, y_ts, band, result, chart_path, usd_to_eur)

    return result, chart_path


# ─── NEWSLETTER EMAIL ───────────────────────────────────────────────────────

def format_money(price, symbol):
    """Format a monetary value with appropriate decimals."""
    if price >= 1000:
        return f"{symbol}{price:,.0f}"
    elif price >= 1:
        return f"{symbol}{price:,.2f}"
    else:
        return f"{symbol}{price:,.4f}"


def format_price(price):
    """Format a USD price."""
    return format_money(price, "$")


def format_euro(price):
    """Format a EUR price."""
    return format_money(price, "€")


def generate_insight(result):
    """Generate a one-line insight for a ticker."""
    name = result["name"]
    pct = result["pct_change"]
    conf = result["confidence"]
    momentum = result["recent_momentum_pct"]
    band = result["band_width_pct"]

    # Build a narrative sentence
    if abs(pct) < 0.3:
        trend = "is expected to trade sideways"
    elif pct > 0:
        trend = f"is showing upside potential of {pct:+.1f}%"
    else:
        trend = f"may pull back by {abs(pct):.1f}%"

    if conf == "High":
        conf_text = "The model is fairly confident in this call"
    elif conf == "Medium":
        conf_text = "The model shows moderate uncertainty"
    else:
        conf_text = "The model is quite uncertain here -- the range is wide"

    if abs(momentum) > 3:
        if momentum > 0:
            mom_text = f" Recent momentum has been strong (+{momentum:.1f}% in the last session)."
        else:
            mom_text = f" Recent momentum has been negative ({momentum:.1f}% in the last session)."
    else:
        mom_text = ""

    return f"<b>{name}</b> {trend} over the next {result['horizon']}. {conf_text}, with a 90% range spanning {band:.1f}% of the current price.{mom_text}"


def generate_market_overview(results):
    """Generate a market overview paragraph."""
    bullish = [r for r in results if r["direction"] == "Bullish"]
    bearish = [r for r in results if r["direction"] == "Bearish"]
    neutral = [r for r in results if r["direction"] == "Neutral"]

    total = len(results)
    crypto = [r for r in results if r["ticker"].endswith("-USD") and "=" not in r["ticker"]]
    stocks = [r for r in results if not r["ticker"].endswith("-USD") and "=" not in r["ticker"]]

    # Overall sentiment
    if len(bullish) > len(bearish) * 2:
        sentiment = "decidedly optimistic"
        tone = "The model sees broad strength across the board."
    elif len(bearish) > len(bullish) * 2:
        sentiment = "cautious"
        tone = "The model is flagging weakness across most assets."
    elif len(bullish) > len(bearish):
        sentiment = "mildly optimistic"
        tone = "A slight bullish lean, but it's a mixed picture."
    elif len(bearish) > len(bullish):
        sentiment = "slightly defensive"
        tone = "More assets are leaning bearish than bullish."
    else:
        sentiment = "neutral"
        tone = "No clear directional bias -- the market looks undecided."

    overview = f"Today's outlook across {total} assets is <b>{sentiment}</b>. {tone}"

    # Biggest mover
    if results:
        sorted_abs = sorted(results, key=lambda x: abs(x["pct_change"]), reverse=True)
        top = sorted_abs[0]
        overview += f" The biggest expected move is <b>{top['name']}</b> at {top['pct_change']:+.1f}%."

    # Crypto vs stocks divergence
    if crypto and stocks:
        avg_crypto = sum(r["pct_change"] for r in crypto) / len(crypto)
        avg_stocks = sum(r["pct_change"] for r in stocks) / len(stocks)
        if abs(avg_crypto - avg_stocks) > 2:
            if avg_crypto > avg_stocks:
                overview += " Interestingly, crypto is showing more strength than equities today."
            else:
                overview += " Equities are holding up better than crypto in today's forecast."

    return overview


def build_newsletter(results, config):
    """Build a newsletter-style HTML email."""
    now = report_now()
    date_str = now.strftime("%A, %B %d, %Y")
    time_str = now.strftime("%H:%M %Z")

    # Separate by category
    crypto = [r for r in results if r["ticker"].endswith("-USD") and "=" not in r["ticker"]]
    stocks = [r for r in results if not r["ticker"].endswith("-USD") and "=" not in r["ticker"]]
    commodities = [r for r in results if r["ticker"] in ["GC=F", "SI=F", "CL=F", "NG=F", "HG=F"]]
    forex = [r for r in results if r["ticker"].endswith("=X")]

    overview = generate_market_overview(results)
    usd_to_eur = results[0].get("usd_to_eur", 1.0) if results else 1.0
    charts_url = os.environ.get("KRONOS_CHARTS_URL", "").strip()
    artifact_name = os.environ.get("KRONOS_ARTIFACT_NAME", "forecasts").strip()

    if charts_url:
        safe_charts_url = html_lib.escape(charts_url, quote=True)
        safe_artifact_name = html_lib.escape(artifact_name)
        charts_access = f"""
            <div style="background:#eff6ff;border:1px solid #bfdbfe;border-radius:12px;padding:18px 22px;margin:16px 0;text-align:center">
                <div style="font-weight:700;color:#1e3a8a;font-size:15px;margin-bottom:6px">Forecast Charts</div>
                <div style="color:#475569;font-size:13px;line-height:1.5;margin-bottom:12px">
                    Open this GitHub Actions run and download <b>{safe_artifact_name}</b> from the Artifacts section.
                    The files become available when the workflow finishes.
                </div>
                <a href="{safe_charts_url}" style="display:inline-block;background:#2563eb;color:white;text-decoration:none;padding:10px 18px;border-radius:8px;font-size:13px;font-weight:700">Open Charts &amp; Artifacts</a>
            </div>"""
    else:
        charts_access = """
            <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;padding:16px 20px;margin:16px 0;text-align:center;color:#64748b;font-size:13px">
                Forecast charts are saved in <code>forecasts/charts/</code>.
            </div>"""

    def signal_badge(direction):
        colors = {
            "Bullish": ("#dcfce7", "#16a34a"),
            "Bearish": ("#fee2e2", "#dc2626"),
            "Neutral": ("#f3f4f6", "#6b7280"),
        }
        bg, fg = colors.get(direction, ("#f3f4f6", "#6b7280"))
        return f'<span style="background:{bg};color:{fg};padding:3px 10px;border-radius:12px;font-size:12px;font-weight:600">{direction}</span>'

    def pct_color(pct):
        if pct > 0.3: return "#16a34a"
        if pct < -0.3: return "#dc2626"
        return "#6b7280"

    def build_section(items, title, icon_color):
        if not items:
            return ""
        rows = ""
        for r in items:
            rate = r.get("usd_to_eur", usd_to_eur)
            current_eur = r.get("current_price_eur", r["current_price"] * rate)
            median_eur = r.get("median_end_eur", r["median_end"] * rate)
            low_eur = r.get("low_5_eur", r["low_5"] * rate)
            high_eur = r.get("high_95_eur", r["high_95"] * rate)
            rows += f"""
            <tr style="border-bottom:1px solid #f1f5f9">
                <td style="padding:14px 16px">
                    <div style="font-weight:700;color:#1e293b;font-size:15px">{r['name']}</div>
                    <div style="color:#94a3b8;font-size:12px;margin-top:2px">{r['ticker']}</div>
                </td>
                <td style="padding:14px 16px;text-align:right">
                    <div style="font-weight:600;color:#1e293b">{format_price(r['current_price'])}</div>
                    <div style="color:#64748b;font-size:11px;margin-top:2px">{format_euro(current_eur)}</div>
                </td>
                <td style="padding:14px 16px;text-align:right">
                    <div style="font-weight:600;color:{pct_color(r['pct_change'])}">{r['pct_change']:+.2f}%</div>
                    <div style="color:#64748b;font-size:11px">{format_price(r['median_end'])} / {format_euro(median_eur)}</div>
                </td>
                <td style="padding:14px 16px;text-align:center">
                    {signal_badge(r['direction'])}
                </td>
                <td style="padding:14px 16px;text-align:right">
                    <div style="color:#64748b;font-size:13px">{format_price(r['low_5'])} - {format_price(r['high_95'])}</div>
                    <div style="color:#94a3b8;font-size:11px">{format_euro(low_eur)} - {format_euro(high_eur)}</div>
                    <div style="color:#94a3b8;font-size:11px">{r['confidence']} confidence</div>
                </td>
            </tr>"""

        # Individual insights
        insights = ""
        for r in items:
            insight = generate_insight(r)
            insights += f'<li style="margin-bottom:10px;line-height:1.6;color:#475569">{insight}</li>'

        return f"""
        <div style="margin-top:32px">
            <div style="display:flex;align-items:center;margin-bottom:16px">
                <div style="width:4px;height:24px;background:{icon_color};border-radius:2px;margin-right:12px"></div>
                <h2 style="margin:0;color:#1e293b;font-size:20px;font-weight:700">{title}</h2>
            </div>
            <table style="width:100%;border-collapse:collapse;background:white;border-radius:12px;overflow:hidden;box-shadow:0 1px 3px rgba(0,0,0,0.08)">
                <tr style="background:#f8fafc">
                    <th style="padding:10px 16px;text-align:left;color:#64748b;font-size:12px;font-weight:600;text-transform:uppercase;letter-spacing:0.5px">Asset</th>
                    <th style="padding:10px 16px;text-align:right;color:#64748b;font-size:12px;font-weight:600;text-transform:uppercase;letter-spacing:0.5px">Price<br>USD / EUR</th>
                    <th style="padding:10px 16px;text-align:right;color:#64748b;font-size:12px;font-weight:600;text-transform:uppercase;letter-spacing:0.5px">Forecast<br>USD / EUR</th>
                    <th style="padding:10px 16px;text-align:center;color:#64748b;font-size:12px;font-weight:600;text-transform:uppercase;letter-spacing:0.5px">Signal</th>
                    <th style="padding:10px 16px;text-align:right;color:#64748b;font-size:12px;font-weight:600;text-transform:uppercase;letter-spacing:0.5px">90% Range</th>
                </tr>
                {rows}
            </table>
            <div style="margin-top:16px;padding:16px 20px;background:#f8fafc;border-radius:10px;border-left:3px solid {icon_color}">
                <div style="font-weight:600;color:#1e293b;margin-bottom:8px;font-size:14px">Key Insights</div>
                <ul style="margin:0;padding-left:18px;font-size:14px">{insights}</ul>
            </div>
        </div>"""

    # Build the newsletter
    html = f"""
    <!DOCTYPE html>
    <html>
    <head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"></head>
    <body style="margin:0;padding:0;background:#f1f5f9;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,'Helvetica Neue',Arial,sans-serif">
        <div style="max-width:680px;margin:0 auto;padding:20px">

            <!-- Header -->
            <div style="background:linear-gradient(135deg,#0f172a 0%,#1e3a5f 100%);border-radius:16px;padding:36px 32px;margin-bottom:24px;text-align:center">
                <h1 style="margin:0;color:white;font-size:28px;font-weight:800;letter-spacing:-0.5px">KRONOS</h1>
                <div style="color:#94a3b8;font-size:14px;margin-top:6px;letter-spacing:1px">DAILY MARKET FORECAST</div>
                <div style="color:#64748b;font-size:13px;margin-top:12px">{date_str} &bull; Generated at {time_str}</div>
            </div>

            <!-- Market Overview -->
            <div style="background:white;border-radius:12px;padding:24px 28px;margin-bottom:8px;box-shadow:0 1px 3px rgba(0,0,0,0.08)">
                <h2 style="margin:0 0 12px 0;color:#1e293b;font-size:18px;font-weight:700">Today's Outlook</h2>
                <p style="margin:0;color:#475569;font-size:15px;line-height:1.7">{overview}</p>
                <div style="margin-top:12px;color:#94a3b8;font-size:12px">Currency snapshot: $1 = €{usd_to_eur:.4f}. Email values show USD and EUR; downloadable charts use EUR.</div>
            </div>

            <!-- Scoreboard -->
            <div style="display:flex;gap:8px;margin:16px 0">
                <div style="flex:1;background:white;border-radius:10px;padding:16px;text-align:center;box-shadow:0 1px 3px rgba(0,0,0,0.08)">
                    <div style="font-size:28px;font-weight:800;color:#16a34a">{len([r for r in results if r['direction']=='Bullish'])}</div>
                    <div style="font-size:12px;color:#64748b;margin-top:4px;font-weight:600">BULLISH</div>
                </div>
                <div style="flex:1;background:white;border-radius:10px;padding:16px;text-align:center;box-shadow:0 1px 3px rgba(0,0,0,0.08)">
                    <div style="font-size:28px;font-weight:800;color:#6b7280">{len([r for r in results if r['direction']=='Neutral'])}</div>
                    <div style="font-size:12px;color:#64748b;margin-top:4px;font-weight:600">NEUTRAL</div>
                </div>
                <div style="flex:1;background:white;border-radius:10px;padding:16px;text-align:center;box-shadow:0 1px 3px rgba(0,0,0,0.08)">
                    <div style="font-size:28px;font-weight:800;color:#dc2626">{len([r for r in results if r['direction']=='Bearish'])}</div>
                    <div style="font-size:12px;color:#64748b;margin-top:4px;font-weight:600">BEARISH</div>
                </div>
            </div>

            <!-- Chart access -->
            {charts_access}

            <!-- Sections -->
            {build_section(crypto, "Crypto -- Next 24 Hours", "#f59e0b")}
            {build_section(stocks, f"Equities -- Next {config['forecast']['pred_len_stock']} Trading Days", "#3b82f6")}
            {build_section(commodities, f"Commodities -- Next {config['forecast']['pred_len_commodity']} Trading Days", "#a855f7")}
            {build_section(forex, "Forex -- Next 24 Hours", "#14b8a6")}

            <!-- How to read -->
            <div style="margin-top:32px;background:#f8fafc;border-radius:12px;padding:20px 24px;border:1px solid #e2e8f0">
                <div style="font-weight:700;color:#1e293b;font-size:14px;margin-bottom:8px">How to read this report</div>
                <div style="color:#64748b;font-size:13px;line-height:1.7">
                    Each forecast is generated by running the Kronos-base model (102.3M parameters) {config['forecast']['paths']} separate times
                    to sample different possible futures. The <b>Forecast %</b> shows where the median path ends.
                    The <b>90% Range</b> shows the spread between the 5th and 95th percentile -- a wider range means
                    more uncertainty. Charts are stored in <code>forecasts/charts/</code> and included in each cloud
                    run's downloadable artifact. Charts use EUR for visual consistency; email tables retain both the
                    source USD values and their EUR conversion.
                </div>
            </div>

            <!-- Footer -->
            <div style="margin-top:24px;padding:20px;text-align:center">
                <div style="color:#94a3b8;font-size:12px;line-height:1.6">
                    This is a research model, not financial advice. The Kronos model is trained on historical
                    price patterns only -- it has no knowledge of news, fundamentals, or real-world events.
                    Always do your own research before making any investment decisions.
                </div>
                <div style="color:#cbd5e1;font-size:11px;margin-top:12px">
                    Powered by Kronos (AAAI 2026) &bull; {len(results)} assets analyzed
                </div>
            </div>

        </div>
    </body>
    </html>"""

    return html


def send_email(html, email_config):
    """Send newsletter email via SMTP."""
    msg = MIMEMultipart("alternative")
    msg["From"] = email_config["sender"]
    msg["To"] = email_config["recipient"]
    msg["Subject"] = f"Kronos Daily Forecast - {report_now().strftime('%b %d, %Y')}"

    # Plain text fallback
    plain = "Your Kronos Daily Forecast is ready. View this email in HTML for the full report."
    charts_url = os.environ.get("KRONOS_CHARTS_URL", "").strip()
    if charts_url:
        plain += f"\n\nForecast charts and artifacts: {charts_url}"
    msg.attach(MIMEText(plain, "plain"))
    msg.attach(MIMEText(html, "html"))

    with smtplib.SMTP(email_config["smtp_server"], email_config["smtp_port"]) as server:
        server.starttls()
        server.login(email_config["sender"], email_config["password"])
        server.sendmail(email_config["sender"], email_config["recipient"], msg.as_string())

    print(f"\n  [OK] Newsletter sent to {email_config['recipient']}")


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    config = load_config()

    # Parse arguments
    send_email_flag = "--email" in sys.argv
    cloud_mode = "--cloud" in sys.argv

    categories = set()
    if "--crypto" in sys.argv: categories.add("crypto")
    if "--stocks" in sys.argv: categories.update([k for k in config.get("tickers", {}) if k not in ("crypto", "commodities", "forex")])
    if "--commodities" in sys.argv: categories.add("commodities")
    if "--forex" in sys.argv: categories.add("forex")
    if "--all" in sys.argv or not categories:
        categories = set(config.get("tickers", {}).keys())

    # Build ticker list
    tickers = []
    for cat in categories:
        for t in config.get("tickers", {}).get(cat, []):
            is_hourly = cat in ("crypto", "forex") or t.endswith("-USD") or t.endswith("=X")
            mode = "hourly" if is_hourly else "daily"
            tickers.append((t, mode, cat))

    print(f"{'='*60}")
    print(f"  KRONOS DAILY FORECAST")
    print(f"  Model: {config['model']['name']}")
    print(f"  Device: {'CUDA (GPU)' if torch.cuda.is_available() else 'CPU'}")
    print(f"  Tickers: {len(tickers)}")
    print(f"  Paths per ticker: {config['forecast']['paths']}")
    print(f"  Email: {'Yes' if send_email_flag else 'No (use --email to send)'}")
    print(f"  Mode: {'Cloud' if cloud_mode else 'Local'}")
    print(f"{'='*60}\n")

    # Fetch a single FX snapshot for consistent charts and email values.
    print("[1/4] Fetching USD/EUR conversion...")
    usd_to_eur = fetch_usd_to_eur_rate(config)
    print(f"  Currency snapshot: $1 = €{usd_to_eur:.4f}\n")

    # Load model once
    print("[2/4] Loading Kronos model...")
    mc = config["model"]
    tokenizer = KronosTokenizer.from_pretrained(mc["tokenizer"])
    model = Kronos.from_pretrained(mc["name"])
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    predictor = KronosPredictor(model, tokenizer, device=device, max_context=mc["max_context"])
    print(f"  Model loaded on {device}\n")

    # Run forecasts
    print(f"[3/4] Running forecasts for {len(tickers)} tickers...")
    results = []
    chart_paths = []
    total = len(tickers)

    for i, (ticker, mode, category) in enumerate(tickers, 1):
        t0 = time.time()
        print(f"  [{i}/{total}] {friendly(ticker)} ({ticker}, {mode})...", end=" ", flush=True)
        result, chart_path = run_forecast(ticker, mode, category, predictor, config, usd_to_eur)
        elapsed = time.time() - t0

        if result:
            results.append(result)
            chart_paths.append(chart_path)
            print(f"[{result['direction'].upper()}] {result['pct_change']:+.2f}% ({elapsed:.1f}s)")
        else:
            print(f"skipped ({elapsed:.1f}s)")

    # Summary
    print(f"\n[4/4] Summary: {len(results)}/{total} tickers forecasted successfully")

    bullish = [r for r in results if r["direction"] == "Bullish"]
    bearish = [r for r in results if r["direction"] == "Bearish"]
    neutral = [r for r in results if r["direction"] == "Neutral"]
    print(f"  Bullish: {len(bullish)}  |  Bearish: {len(bearish)}  |  Neutral: {len(neutral)}")

    # Top movers
    if results:
        sorted_results = sorted(results, key=lambda x: x["pct_change"], reverse=True)
        print(f"\n  Top movers:")
        for r in sorted_results[:3]:
            print(f"    {r['name']:12s} {r['pct_change']:+.2f}%  ({format_price(r['current_price'])} -> {format_price(r['median_end'])})")
        print(f"  ---")
        for r in sorted_results[-3:]:
            print(f"    {r['name']:12s} {r['pct_change']:+.2f}%  ({format_price(r['current_price'])} -> {format_price(r['median_end'])})")

    # Save JSON summary
    output_dir = Path(__file__).parent / "forecasts"
    output_dir.mkdir(exist_ok=True)
    date_str = report_now().strftime("%Y-%m-%d")
    summary_path = output_dir / f"summary_{date_str}.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {summary_path}")

    # Email
    if send_email_flag and results:
        print("\n  Building newsletter...")
        try:
            email_config = get_email_config(config, cloud_mode)
            html = build_newsletter(results, config)

            # Also save newsletter locally
            newsletter_path = output_dir / f"newsletter_{date_str}.html"
            with open(newsletter_path, "w", encoding="utf-8") as f:
                f.write(html)
            print(f"  Newsletter saved to: {newsletter_path}")

            print("  Sending email...")
            send_email(html, email_config)
        except Exception as e:
            print(f"  [ERROR] Failed to send email: {e}")
            print("  Make sure forecast_config.json has the correct Gmail App Password.")
    elif send_email_flag and not results:
        print("\n  No results to email.")

    print(f"\n{'='*60}")
    print("  DONE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
