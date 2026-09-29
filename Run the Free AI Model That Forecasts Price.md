# Run the Free AI Model That Forecasts Price — On Your Laptop

**This is the exact setup to get a forecast range and the likely price path for any stock or crypto, from a free, open-source model. No terminal subscription and no GPU.** I tested every step below on a normal MacBook.

**Repo:** [github.com/shiyu-coder/Kronos](https://github.com/shiyu-coder/Kronos)

---

## What It Is

**Kronos** is the first open-source *foundation model* for candlestick (K-line) data, built by academic researchers. It isn't a trading bot. It's a model that learned the "language" of price action the way an LLM learns text.

- **Trained on:** 12B+ candlestick records from 45+ global exchanges
- **Output:** you give it a price history and it predicts the next candles. Sample it many times and you get a **range** plus the **likely path**, not just one guess.
- **Credibility:** accepted at AAAI 2026 ([paper](https://arxiv.org/abs/2508.02739)), ~39k GitHub stars, MIT license
- **See it first:** [live BTC/USDT forecast demo](https://shiyu-coder.github.io/Kronos-demo/)

| Model | Size | Context (candles) | Runs on a laptop? |
| --- | --- | --- | --- |
| Kronos-mini | 4.1M params | 2048 | ✅ fastest |
| **Kronos-small** | **24.7M params** | **512** | ✅ **used in this guide** |
| Kronos-base | 102.3M params | 512 | ✅ slower |

---

## What You Need

- **Python 3.12** (or 3.11). ⚠️ Not 3.14: the repo pins an older pandas that fails to install on 3.14 (a new Mac's default).
- **git**
- About 2 GB of free disk space (PyTorch plus the model download)
- **No GPU needed.** A 24-hour forecast with 20 sampled paths took about 70 seconds on my laptop CPU.

---

## Option A — Let Claude Code Set It Up (Fastest)

Open Claude Code in an empty folder and paste:

> "Clone https://github.com/shiyu-coder/Kronos, create a Python 3.12 virtual environment (use uv if it's installed), install requirements.txt plus yfinance, and create a forecast.py that downloads the last 400 hourly candles for a ticker with yfinance, runs KronosPredictor with Kronos-small 20 separate times (sample_count=1 each) on device='cpu' unless CUDA is available, and plots the 5–95% range and the median path. Then run it for BTC-USD."
> 

Then skip to **Reading Your Forecast** below.

---

## Option B — Manual Setup (5 Minutes)

**1. Clone the repo and make a Python 3.12 environment:**

```
git clone https://github.com/shiyu-coder/Kronos
cd Kronos
uv venv --python 3.12 .venv        # or: python3.12 -m venv .venv
source .venv/bin/activate
```

**2. Install dependencies, plus yfinance for free market data:**

```
uv pip install -r requirements.txt yfinance   # or: pip install -r requirements.txt yfinance
```

**3. Save this as `forecast.py` inside the `Kronos` folder:**

```python
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
model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
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
closes = pd.DataFrame(paths, columns=y
```

```python
_ts)
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
```

**4. Run it:**

```
python forecast.py BTC-USD          # crypto: next 24 hours from hourly candles
python forecast.py AAPL daily       # stocks: next 20 trading days from daily candles
```

The first run downloads the model, so give it a minute. After that each forecast takes about 1 minute on CPU.

---

## Reading Your Forecast

You get a table in the terminal and a `forecast.png` chart:

| Column | What it means |
| --- | --- |
| `low_5%` | 5% of sampled futures ended below this |
| `likely_50%` | the median path, the model's "most likely" line |
| `high_95%` | 95% of sampled futures ended below this |

**The width of the band is the point.** A narrow band means the model sees a calm, predictable stretch. A wide band means it's uncertain. That's what separates this from a single-line guess.

**Why the script samples 20 separate times:** the repo's quickstart uses `sample_count`, which *averages* its paths into one line. Averaging erases the uncertainty. Sampling separately and taking percentiles keeps it.

---

## Knobs Worth Turning

| Setting | Default | What to try |
| --- | --- | --- |
| `PATHS` | 20 | 50+ for a smoother, more reliable band (slower) |
| `LOOKBACK` | 400 | Keep ≤ 512 for small/base. Kronos-mini takes up to 2048. |
| `PRED_LEN` | 24 h / 20 days | Shorter horizons are more meaningful; the band widens fast |
| `T` / `top_p` | 1.0 / 0.9 | Lower `T` gives tighter, more conservative paths |
| Model | `Kronos-small` | `NeoQuasar/Kronos-mini` (with `Kronos-Tokenizer-2k`) for speed, `Kronos-base` for size |

---

## Two Errors You'll Probably Hit (Already Fixed Above)

- **`pip install` fails building pandas:** you're on Python 3.13/3.14. Recreate the environment with Python 3.12.
- **`scaled_dot_product_attention for MPS does not support dropout`:** Apple Silicon auto-picks the Mac GPU backend, which crashes. The script forces `device="cpu"`, which is why it works.

---

## Read This Before You Trade On It

- **This is a research model, not financial advice.** The authors say their own pipeline is "not a production-ready quantitative trading system."
- **A forecast range is a probability, not a promise.** Treat the band as one input alongside your own analysis, never as a signal to act on alone.
- **Paper-trade first.** Run it daily for a few weeks and compare the band to what actually happened before you trust it with money.

---

## One Safety Note Before You Install

You're cloning third-party code and letting it download model weights, so:

- **Install from the official repo only:** `github.com/shiyu-coder/Kronos`. There are many forks, and a fork can change the code.
- **Skim what you run.** `model/` and `requirements.txt` are short. Know what you're executing.
- **Keep it away from live credentials.** Don't run it inside a folder or environment with exchange/broker API keys until you've read the code.
- **If Claude Code does the setup (Option A),** review the commands it proposes before approving them.

---

## Resources

- **Repo:** [github.com/shiyu-coder/Kronos](https://github.com/shiyu-coder/Kronos)
- **Live demo:** [shiyu-coder.github.io/Kronos-demo](https://shiyu-coder.github.io/Kronos-demo/)
- **Models:** [huggingface.co/NeoQuasar](https://huggingface.co/NeoQuasar)
- **Paper:** [arxiv.org/abs/2508.02739](https://arxiv.org/abs/2508.02739)

Built a forecast you're proud of? Show me — Jesse / [@jesserurka](https://www.instagram.com/jesserurka)