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
import time
import datetime
import smtplib
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import torch
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from model import Kronos, KronosTokenizer, KronosPredictor

# ─── CONFIGURATION ───────────────────────────────────────────────────────────

CONFIG_FILE = Path(__file__).parent / "forecast_config.json"

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
        "pred_len_daily": 20,
        "temperature": 1.0,
        "top_p": 0.9
    },
    "tickers": {
        "crypto": ["BTC-USD", "ETH-USD"],
        "commodities": ["GC=F"],
        "sp500_top": ["AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA"],
        "nasdaq100_extra": [],
        "forex": []
    }
}

# ─── FRIENDLY NAMES ─────────────────────────────────────────────────────────

TICKER_NAMES = {
    "BTC-USD": "Bitcoin", "ETH-USD": "Ethereum", "SOL-USD": "Solana",
    "BNB-USD": "BNB", "XRP-USD": "XRP", "ADA-USD": "Cardano",
    "AVAX-USD": "Avalanche", "DOT-USD": "Polkadot", "DOGE-USD": "Dogecoin",
    "LINK-USD": "Chainlink",
    "GC=F": "Gold", "SI=F": "Silver", "CL=F": "Crude Oil",
    "AAPL": "Apple", "MSFT": "Microsoft", "AMZN": "Amazon",
    "NVDA": "Nvidia", "GOOGL": "Alphabet", "META": "Meta", "TSLA": "Tesla",
    "BRK-B": "Berkshire", "UNH": "UnitedHealth", "XOM": "Exxon",
    "JNJ": "Johnson & Johnson", "JPM": "JPMorgan", "V": "Visa",
    "EURUSD=X": "EUR/USD", "GBPUSD=X": "GBP/USD", "USDJPY=X": "USD/JPY",
}

def friendly(ticker):
    return TICKER_NAMES.get(ticker, ticker)

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


def run_forecast(ticker, mode, predictor, config):
    """Run forecast for a single ticker. Returns (result_dict, chart_path) or (None, None) on failure."""
    fc = config["forecast"]
    LOOKBACK = fc["lookback"]
    PATHS = fc["paths"]
    DAILY = mode == "daily"
    PRED_LEN = fc["pred_len_daily"] if DAILY else fc["pred_len_hourly"]
    INTERVAL, PERIOD, STEP = ("1d", "3y", pd.offsets.BDay(1)) if DAILY else ("1h", "60d", pd.Timedelta(hours=1))

    # 1. Pull candles
    try:
        raw = yf.download(ticker, period=PERIOD, interval=INTERVAL, progress=False, auto_adjust=False)
        if raw.empty:
            print(f"  [SKIP] {ticker}: no data returned from Yahoo Finance")
            return None, None
        raw.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in raw.columns]
        df = raw[["open", "high", "low", "close", "volume"]].dropna().tail(LOOKBACK).reset_index()
        df = df.rename(columns={df.columns[0]: "timestamps"})
        df["timestamps"] = pd.to_datetime(df["timestamps"]).dt.tz_localize(None)
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
        "current_price": float(start_price),
        "median_end": float(end_median),
        "pct_change": float(pct_change),
        "low_5": float(band["low_5%"].iloc[-1]),
        "high_95": float(band["high_95%"].iloc[-1]),
        "band_width_pct": float(band_width),
        "recent_momentum_pct": float(recent_change),
        "direction": "Bullish" if pct_change > 0.3 else ("Bearish" if pct_change < -0.3 else "Neutral"),
        "confidence": "High" if band_width < 3 else ("Medium" if band_width < 7 else "Low"),
        "horizon": f"{'20 trading days' if DAILY else '24 hours'}",
    }

    # 5. Plot (saved locally, not sent in email)
    output_dir = Path(__file__).parent / "forecasts"
    output_dir.mkdir(exist_ok=True)
    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    chart_path = output_dir / f"{ticker.replace('=', '_')}_{date_str}.png"

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(x_ts.tail(120), df["close"].tail(120), color="black", linewidth=1.2, label="History")
    ax.fill_between(y_ts, band["low_5%"], band["high_95%"], alpha=0.25, color="#4A90D9", label="90% range")
    ax.plot(y_ts, band["likely_50%"], color="#2563EB", linewidth=1.5, label="Likely path (median)")
    ax.axhline(y=start_price, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)

    horizon_label = "trading days" if DAILY else "hours"
    signal_text = f"[{result['direction'].upper()}]"
    ax.set_title(f"{friendly(ticker)} ({ticker}) -- next {PRED_LEN} {horizon_label}  |  {signal_text}  ({pct_change:+.2f}%)", fontsize=13)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.15)
    fig.tight_layout()
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)

    return result, chart_path


# ─── NEWSLETTER EMAIL ───────────────────────────────────────────────────────

def format_price(price):
    """Format price with appropriate decimals."""
    if price >= 1000:
        return f"${price:,.0f}"
    elif price >= 1:
        return f"${price:,.2f}"
    else:
        return f"${price:,.4f}"


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
    now = datetime.datetime.now()
    date_str = now.strftime("%A, %B %d, %Y")
    time_str = now.strftime("%H:%M")

    # Separate by category
    crypto = [r for r in results if r["ticker"].endswith("-USD") and "=" not in r["ticker"]]
    stocks = [r for r in results if not r["ticker"].endswith("-USD") and "=" not in r["ticker"]]
    commodities = [r for r in results if r["ticker"] in ["GC=F", "SI=F", "CL=F"]]

    overview = generate_market_overview(results)

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
            rows += f"""
            <tr style="border-bottom:1px solid #f1f5f9">
                <td style="padding:14px 16px">
                    <div style="font-weight:700;color:#1e293b;font-size:15px">{r['name']}</div>
                    <div style="color:#94a3b8;font-size:12px;margin-top:2px">{r['ticker']}</div>
                </td>
                <td style="padding:14px 16px;text-align:right">
                    <div style="font-weight:600;color:#1e293b">{format_price(r['current_price'])}</div>
                </td>
                <td style="padding:14px 16px;text-align:right">
                    <div style="font-weight:600;color:{pct_color(r['pct_change'])}">{r['pct_change']:+.2f}%</div>
                    <div style="color:#94a3b8;font-size:11px">{format_price(r['median_end'])}</div>
                </td>
                <td style="padding:14px 16px;text-align:center">
                    {signal_badge(r['direction'])}
                </td>
                <td style="padding:14px 16px;text-align:right">
                    <div style="color:#64748b;font-size:13px">{format_price(r['low_5'])} - {format_price(r['high_95'])}</div>
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
                    <th style="padding:10px 16px;text-align:right;color:#64748b;font-size:12px;font-weight:600;text-transform:uppercase;letter-spacing:0.5px">Price</th>
                    <th style="padding:10px 16px;text-align:right;color:#64748b;font-size:12px;font-weight:600;text-transform:uppercase;letter-spacing:0.5px">Forecast</th>
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

            <!-- Sections -->
            {build_section(crypto, "Crypto -- Next 24 Hours", "#f59e0b")}
            {build_section(stocks, "Magnificent 7 -- Next 20 Trading Days", "#3b82f6")}
            {build_section(commodities, "Commodities -- Next 20 Trading Days", "#a855f7")}

            <!-- How to read -->
            <div style="margin-top:32px;background:#f8fafc;border-radius:12px;padding:20px 24px;border:1px solid #e2e8f0">
                <div style="font-weight:700;color:#1e293b;font-size:14px;margin-bottom:8px">How to read this report</div>
                <div style="color:#64748b;font-size:13px;line-height:1.7">
                    Each forecast is generated by running the Kronos-base model (102.3M parameters) 20 separate times
                    to sample different possible futures. The <b>Forecast %</b> shows where the median path ends.
                    The <b>90% Range</b> shows the spread between the 5th and 95th percentile -- a wider range means
                    more uncertainty. Charts are stored locally on your machine in the <code>forecasts/</code> folder.
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
    msg["Subject"] = f"Kronos Daily Forecast - {datetime.datetime.now().strftime('%b %d, %Y')}"

    # Plain text fallback
    plain = "Your Kronos Daily Forecast is ready. View this email in HTML for the full report."
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
    if "--stocks" in sys.argv: categories.update(["sp500_top", "nasdaq100_extra"])
    if "--commodities" in sys.argv: categories.add("commodities")
    if "--forex" in sys.argv: categories.add("forex")
    if "--all" in sys.argv or not categories:
        categories = {"crypto", "sp500_top", "nasdaq100_extra", "commodities", "forex"}

    # Build ticker list
    tickers = []
    for cat in categories:
        for t in config["tickers"].get(cat, []):
            mode = "hourly" if cat in ("crypto", "forex") else "daily"
            tickers.append((t, mode))

    print(f"{'='*60}")
    print(f"  KRONOS DAILY FORECAST")
    print(f"  Model: {config['model']['name']}")
    print(f"  Device: {'CUDA (GPU)' if torch.cuda.is_available() else 'CPU'}")
    print(f"  Tickers: {len(tickers)}")
    print(f"  Paths per ticker: {config['forecast']['paths']}")
    print(f"  Email: {'Yes' if send_email_flag else 'No (use --email to send)'}")
    print(f"  Mode: {'Cloud' if cloud_mode else 'Local'}")
    print(f"{'='*60}\n")

    # Load model once
    print("[1/3] Loading Kronos model...")
    mc = config["model"]
    tokenizer = KronosTokenizer.from_pretrained(mc["tokenizer"])
    model = Kronos.from_pretrained(mc["name"])
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    predictor = KronosPredictor(model, tokenizer, device=device, max_context=mc["max_context"])
    print(f"  Model loaded on {device}\n")

    # Run forecasts
    print(f"[2/3] Running forecasts for {len(tickers)} tickers...")
    results = []
    chart_paths = []
    total = len(tickers)

    for i, (ticker, mode) in enumerate(tickers, 1):
        t0 = time.time()
        print(f"  [{i}/{total}] {friendly(ticker)} ({ticker}, {mode})...", end=" ", flush=True)
        result, chart_path = run_forecast(ticker, mode, predictor, config)
        elapsed = time.time() - t0

        if result:
            results.append(result)
            chart_paths.append(chart_path)
            print(f"[{result['direction'].upper()}] {result['pct_change']:+.2f}% ({elapsed:.1f}s)")
        else:
            print(f"skipped ({elapsed:.1f}s)")

    # Summary
    print(f"\n[3/3] Summary: {len(results)}/{total} tickers forecasted successfully")

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
    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
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
