import sys
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import torch
from model import Kronos, KronosTokenizer, KronosPredictor

TICKER = sys.argv[1] if len(sys.argv) > 1 else "BTC-USD"
DAILY = len(sys.argv) > 2 and sys.argv[2] == "daily"      # use "daily" for stocks (hourly stock data has overnight gaps)
LOOKBACK, PATHS = 400, 20                                   # 400 candles of history in, 20 sampled futures out
PRED_LEN = 20 if DAILY else 24                              # next 20 trading days, or next 24 hours
INTERVAL, PERIOD, STEP = ("1d", "3y", pd.offsets.BDay(1)) if DAILY else ("1h", "60d", pd.Timedelta(hours=1))

# 1. Pull candles
raw = yf.download(TICKER, period=PERIOD, interval=INTERVAL, progress=False, auto_adjust=False)
raw.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in raw.columns]
df = raw[["open", "high", "low", "close", "volume"]].dropna().tail(LOOKBACK).reset_index()
df = df.rename(columns={df.columns[0]: "timestamps"})
df["timestamps"] = pd.to_datetime(df["timestamps"]).dt.tz_localize(None)

# 2. Load the model (downloads from Hugging Face on first run)
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("NeoQuasar/Kronos-base")
# NVIDIA GPU if you have one, otherwise CPU (Apple "mps" currently crashes in this model)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
predictor = KronosPredictor(model, tokenizer, device=device, max_context=512)

x_ts = df["timestamps"]
y_ts = pd.Series([x_ts.iloc[-1] + STEP * (i + 1) for i in range(PRED_LEN)])

# 3. Sample many possible futures -> that spread IS the uncertainty
paths = []
for _ in range(PATHS):
    pred = predictor.predict(df=df[["open", "high", "low", "close", "volume"]], x_timestamp=x_ts,
                             y_timestamp=y_ts, pred_len=PRED_LEN, T=1.0, top_p=0.9, sample_count=1, verbose=False)
    paths.append(pred["close"].values)
closes = pd.DataFrame(paths, columns=y_ts)
band = pd.DataFrame({"low_5%": closes.quantile(0.05), "likely_50%": closes.median(), "high_95%": closes.quantile(0.95)})
print(band.round(2).to_string())

# 4. Plot history + range + likely path
plt.figure(figsize=(11, 5))
plt.plot(x_ts.tail(120), df["close"].tail(120), color="black", label="history")
plt.fill_between(y_ts, band["low_5%"], band["high_95%"], alpha=0.25, label="90% range")
plt.plot(y_ts, band["likely_50%"], label="likely path")
plt.title(f"{TICKER} — next {PRED_LEN} {'trading days' if DAILY else 'hours'} ({PATHS} sampled paths)")
plt.legend(); plt.tight_layout(); plt.savefig("forecast.png", dpi=150)
print("saved forecast.png")
