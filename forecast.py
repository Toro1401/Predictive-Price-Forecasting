import sys
from pathlib import Path

import pandas as pd
import torch
import yfinance as yf

from model import Kronos, KronosPredictor, KronosTokenizer
from forecast_all import (
    DEFAULT_CONFIG,
    chart_filename,
    fetch_usd_to_eur_rate,
    friendly,
    remove_old_charts,
    render_forecast_chart,
)


def main():
    ticker = sys.argv[1] if len(sys.argv) > 1 else "BTC-USD"
    daily = len(sys.argv) > 2 and sys.argv[2].lower() == "daily"
    lookback, paths = 400, 20
    commodity_tickers = {"GC=F", "SI=F", "CL=F", "NG=F", "HG=F"}
    pred_len = 20 if daily and ticker in commodity_tickers else (5 if daily else 24)
    interval, period, step = (
        ("1d", "3y", pd.offsets.BDay(1))
        if daily
        else ("1h", "3mo", pd.Timedelta(hours=1))
    )

    # 1. Pull enough candles for model context and a three-month chart.
    print(f"Downloading candles for {ticker}...")
    raw = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False)
    if raw.empty:
        raise RuntimeError(f"No data returned for {ticker}")
    raw.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in raw.columns]
    price_history = raw[["open", "high", "low", "close", "volume"]].dropna().reset_index()
    price_history = price_history.rename(columns={price_history.columns[0]: "timestamps"})
    price_history["timestamps"] = pd.to_datetime(price_history["timestamps"]).dt.tz_localize(None)
    df = price_history.tail(lookback).reset_index(drop=True)
    chart_cutoff = price_history["timestamps"].iloc[-1] - pd.DateOffset(months=3)
    plot_df = price_history[price_history["timestamps"] >= chart_cutoff].copy()

    # 2. Load the model.
    print("Loading Kronos model...")
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-base")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    predictor = KronosPredictor(model, tokenizer, device=device, max_context=512)

    x_ts = df["timestamps"]
    y_ts = pd.Series([x_ts.iloc[-1] + step * (i + 1) for i in range(pred_len)])

    # 3. Sample paths.
    print(f"Sampling {paths} forecast paths on {device}...")
    sampled_paths = []
    for _ in range(paths):
        pred = predictor.predict(
            df=df[["open", "high", "low", "close", "volume"]],
            x_timestamp=x_ts,
            y_timestamp=y_ts,
            pred_len=pred_len,
            T=1.0,
            top_p=0.9,
            sample_count=1,
            verbose=False,
        )
        sampled_paths.append(pred["close"].values)

    closes = pd.DataFrame(sampled_paths, columns=y_ts)
    band = pd.DataFrame({
        "low_5%": closes.quantile(0.05),
        "likely_50%": closes.median(),
        "high_95%": closes.quantile(0.95),
    })
    print("\nForecast Band:")
    print(band.round(2).to_string())

    # 4. Build the shared analyst-style EUR chart.
    usd_to_eur = fetch_usd_to_eur_rate(DEFAULT_CONFIG)
    start_price = float(df["close"].iloc[-1])
    end_median = float(band["likely_50%"].iloc[-1])
    pct_change = ((end_median - start_price) / start_price) * 100
    band_width = ((band["high_95%"].iloc[-1] - band["low_5%"].iloc[-1]) / start_price) * 100
    result = {
        "ticker": ticker,
        "name": friendly(ticker),
        "mode": "daily" if daily else "hourly",
        "pred_len": pred_len,
        "current_price": start_price,
        "median_end": end_median,
        "pct_change": float(pct_change),
        "low_5": float(band["low_5%"].iloc[-1]),
        "high_95": float(band["high_95%"].iloc[-1]),
        "direction": "Bullish" if pct_change > 0.3 else ("Bearish" if pct_change < -0.3 else "Neutral"),
        "confidence": "High" if band_width < 3 else ("Medium" if band_width < 7 else "Low"),
    }

    charts_dir = Path(__file__).parent / "forecasts" / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    chart_path = charts_dir / chart_filename(ticker, result["mode"], pred_len=pred_len)
    remove_old_charts(charts_dir, ticker, chart_path)
    render_forecast_chart(plot_df, y_ts, band, result, chart_path, usd_to_eur)
    print(f"\nSaved chart to: {chart_path}")


if __name__ == "__main__":
    main()
