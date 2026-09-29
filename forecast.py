import sys
from pathlib import Path
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import torch
from model import Kronos, KronosTokenizer, KronosPredictor


def main():
    ticker = sys.argv[1] if len(sys.argv) > 1 else "BTC-USD"
    daily = len(sys.argv) > 2 and sys.argv[2].lower() == "daily"
    lookback, paths = 400, 20
    pred_len = 20 if daily else 24
    interval, period, step = ("1d", "3y", pd.offsets.BDay(1)) if daily else ("1h", "60d", pd.Timedelta(hours=1))

    # 1. Pull candles
    print(f"Downloading candles for {ticker}...")
    raw = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False)
    raw.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in raw.columns]
    df = raw[["open", "high", "low", "close", "volume"]].dropna().tail(lookback).reset_index()
    df = df.rename(columns={df.columns[0]: "timestamps"})
    df["timestamps"] = pd.to_datetime(df["timestamps"]).dt.tz_localize(None)

    # 2. Load the model
    print("Loading Kronos model...")
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-base")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    predictor = KronosPredictor(model, tokenizer, device=device, max_context=512)

    x_ts = df["timestamps"]
    y_ts = pd.Series([x_ts.iloc[-1] + step * (i + 1) for i in range(pred_len)])

    # 3. Sample paths
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

    # 4. Plot history + range + likely path
    charts_dir = Path(__file__).parent / "forecasts" / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    safe_ticker = ticker.replace("=", "_")
    chart_path = charts_dir / f"{safe_ticker}.png"

    plt.figure(figsize=(11, 5))
    plt.plot(x_ts.tail(120), df["close"].tail(120), color="black", label="history")
    plt.fill_between(y_ts, band["low_5%"], band["high_95%"], alpha=0.25, label="90% range")
    plt.plot(y_ts, band["likely_50%"], label="likely path")
    plt.title(f"{ticker} — next {pred_len} {'trading days' if daily else 'hours'} ({paths} sampled paths)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"\nSaved chart to: {chart_path}")


if __name__ == "__main__":
    main()
