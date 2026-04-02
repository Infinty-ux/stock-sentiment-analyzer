import os
import time
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional

import praw
import yfinance as yf
import pandas as pd

logger = logging.getLogger(__name__)


class RedditScraper:
    SUBREDDITS = ["wallstreetbets", "stocks", "investing", "SecurityAnalysis"]

    def __init__(self):
        self.reddit = praw.Reddit(
            client_id=os.environ["REDDIT_CLIENT_ID"],
            client_secret=os.environ["REDDIT_CLIENT_SECRET"],
            user_agent=os.environ.get("REDDIT_USER_AGENT", "stock-sentiment-bot/1.0"),
        )

    def fetch_posts(
        self,
        ticker: str,
        subreddits: Optional[List[str]] = None,
        limit: int = 200,
        time_filter: str = "week",
    ) -> pd.DataFrame:
        subreddits = subreddits or self.SUBREDDITS
        rows = []
        for sub_name in subreddits:
            try:
                sub = self.reddit.subreddit(sub_name)
                for post in sub.search(ticker, limit=limit, time_filter=time_filter, sort="new"):
                    rows.append({
                        "id": post.id,
                        "subreddit": sub_name,
                        "title": post.title,
                        "selftext": post.selftext,
                        "score": post.score,
                        "upvote_ratio": post.upvote_ratio,
                        "num_comments": post.num_comments,
                        "created_utc": datetime.fromtimestamp(post.created_utc, tz=timezone.utc),
                        "ticker": ticker.upper(),
                    })
                    time.sleep(0.05)
            except Exception as e:
                logger.warning(f"Subreddit {sub_name}: {e}")
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.drop_duplicates(subset="id")
        return df

    def fetch_comments(self, post_id: str, limit: int = 50) -> List[str]:
        submission = self.reddit.submission(id=post_id)
        submission.comments.replace_more(limit=0)
        texts = []
        for comment in submission.comments.list()[:limit]:
            if hasattr(comment, "body") and len(comment.body) > 20:
                texts.append(comment.body)
        return texts


class PriceScraper:
    def fetch_ohlcv(
        self,
        ticker: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        period: str = "1mo",
        interval: str = "1d",
    ) -> pd.DataFrame:
        start = start or datetime.now(timezone.utc) - timedelta(days=30)
        end = end or datetime.now(timezone.utc)
        stock = yf.Ticker(ticker)
        df = stock.history(start=start, end=end, interval=interval)
        df.index = df.index.tz_localize("UTC") if df.index.tzinfo is None else df.index.tz_convert("UTC")
        df = df[["Open", "High", "Low", "Close", "Volume"]].rename(str.lower, axis=1)
        df["ticker"] = ticker.upper()
        df["returns"] = df["close"].pct_change()
        df["log_returns"] = (df["close"] / df["close"].shift(1)).apply(lambda x: x ** 0.5 - 1 if x > 0 else 0)
        return df

    def fetch_multiple(self, tickers: List[str], **kwargs) -> Dict[str, pd.DataFrame]:
        return {t: self.fetch_ohlcv(t, **kwargs) for t in tickers}
