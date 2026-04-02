import os
import logging
import json
from datetime import datetime, timezone
from pathlib import Path

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
import pandas as pd

from scraper import RedditScraper, PriceScraper
from preprocessor import preprocess_posts, aggregate_daily
from sentiment import SentimentAnalyzer
from correlator import merge_sentiment_price, analyze_correlations

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "data/snapshots"))
TICKERS = os.getenv("TICKERS", "TSLA,AAPL,NVDA,MSFT").split(",")

_analyzer = None


def get_analyzer() -> SentimentAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = SentimentAnalyzer(device="cpu", batch_size=32)
    return _analyzer


def run_snapshot(tickers=None):
    tickers = tickers or TICKERS
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    logger.info(f"Snapshot {ts} — tickers: {tickers}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    reddit, price, analyzer = RedditScraper(), PriceScraper(), get_analyzer()

    summary = {}
    for ticker in tickers:
        try:
            posts = reddit.fetch_posts(ticker, limit=150)
            if posts.empty:
                logger.warning(f"{ticker}: no posts")
                continue

            posts = preprocess_posts(posts)
            posts = analyzer.analyze_dataframe(posts)
            posts["date"] = posts["created_utc"].dt.normalize()
            daily_sent = analyzer.aggregate_sentiment(posts)

            prices = price.fetch_ohlcv(ticker)
            merged = merge_sentiment_price(daily_sent, prices)
            corr = analyze_correlations(merged, [ticker])

            out = OUTPUT_DIR / ticker
            out.mkdir(exist_ok=True)
            posts.to_parquet(out / f"{ts}_posts.parquet", index=False)
            daily_sent.to_parquet(out / f"{ts}_sentiment.parquet", index=False)

            latest = daily_sent.sort_values("date").iloc[-1]
            summary[ticker] = {
                "compound": round(float(latest["weighted_compound"]), 4),
                "bullish_ratio": round(float(latest["bullish_ratio"]), 4),
                "post_count": int(latest["post_count"]),
            }
            if not corr.empty:
                summary[ticker]["pearson_r"] = round(float(corr.iloc[0]["pearson_r"]), 4)

            logger.info(f"{ticker}: compound={summary[ticker]['compound']:.4f}")
        except Exception as e:
            logger.error(f"{ticker} failed: {e}", exc_info=True)

    with open(OUTPUT_DIR / f"{ts}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Snapshot complete: {len(summary)} tickers processed")
    return summary


def main():
    scheduler = BlockingScheduler(timezone="UTC")
    scheduler.add_job(run_snapshot, CronTrigger(hour="*/4", minute="0"), id="snapshot")
    logger.info("Scheduler started — snapshot every 4 hours")
    try:
        scheduler.start()
    except KeyboardInterrupt:
        scheduler.shutdown()


if __name__ == "__main__":
    main()
