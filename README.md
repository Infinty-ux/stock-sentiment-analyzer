# Stock Sentiment Analyzer

NLP-powered sentiment analysis on Reddit financial communities, correlated with real-time stock price data using **FinBERT** — a BERT model fine-tuned on financial text.

## Architecture

```
Reddit (PRAW)                     yfinance
  └── r/wallstreetbets               └── OHLCV daily
  └── r/stocks                       └── log-returns
  └── r/investing
        │                                  │
        ▼                                  │
  Text cleaning & preprocessing           │
  (HTML unescape, URL strip, etc.)        │
        │                                  │
        ▼                                  │
  FinBERT inference (batch)               │
  → negative / neutral / positive         │
  → compound score [−1, 1]               │
        │                                  │
        ▼                                  │
  Engagement-weighted daily aggregate     │
        │                                  │
        └──────────────────────────────────┘
                        │
                        ▼
              Pearson / Spearman correlation
              Granger causality test
              Rolling 14-day window
                        │
                        ▼
              Streamlit dashboard (Plotly)
              + APScheduler (every 4h)
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Fill in Reddit API credentials (free at reddit.com/prefs/apps)
```

### Get Reddit API credentials

1. Go to [reddit.com/prefs/apps](https://www.reddit.com/prefs/apps)
2. Create a "script" type application
3. Copy `client_id` and `client_secret` to `.env`

## Usage

### Live dashboard

```bash
streamlit run src/dashboard.py
```

Open `http://localhost:8501`, enter a ticker (e.g. `NVDA`), click **Analyze**.

### Scheduled snapshots

```bash
python src/scheduler.py
```

Runs every 4 hours; saves Parquet files + JSON summary to `data/snapshots/`.

## Modules

| File | Description |
|---|---|
| `scraper.py` | Reddit PRAW scraper + yfinance OHLCV fetcher |
| `preprocessor.py` | Text cleaning, engagement scoring, daily aggregation |
| `sentiment.py` | FinBERT batch inference, engagement-weighted sentiment |
| `correlator.py` | Pearson, Spearman, Granger causality, rolling correlation |
| `dashboard.py` | Streamlit + Plotly interactive dashboard |
| `scheduler.py` | APScheduler cron — periodic data snapshots |

## Statistical Methods

- **Pearson correlation** — linear relationship between sentiment compound and next-day returns
- **Spearman rank correlation** — monotonic relationship, robust to outliers
- **Granger causality** — does sentiment time-series "cause" price movement? (up to 5-day lag)
- **Rolling 14-day window** — how correlation evolves over time

## Tech Stack

- **FinBERT** (`ProsusAI/finbert`) — BERT fine-tuned on financial news
- **PRAW** — Reddit API wrapper
- **yfinance** — Yahoo Finance OHLCV data
- **Streamlit + Plotly** — interactive dashboard
- **scipy / statsmodels** — statistical correlation tests
- **APScheduler** — periodic background job scheduling
