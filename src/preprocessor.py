import re
import html
from typing import List, Optional

import pandas as pd
import numpy as np


_TICKER_RE = re.compile(r"\$[A-Z]{1,5}\b")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"u/\w+|r/\w+")
_SPECIAL_RE = re.compile(r"[^a-zA-Z0-9\s.,!?\'\"$%+\-]")
_WHITESPACE_RE = re.compile(r"\s{2,}")


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = _URL_RE.sub(" ", text)
    text = _MENTION_RE.sub(" ", text)
    text = _SPECIAL_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


def extract_tickers(text: str) -> List[str]:
    return list(set(_TICKER_RE.findall(text)))


def combine_post_text(row: pd.Series) -> str:
    parts = []
    if pd.notna(row.get("title")):
        parts.append(str(row["title"]))
    if pd.notna(row.get("selftext")) and str(row.get("selftext")).strip() not in ("", "[removed]", "[deleted]"):
        parts.append(str(row["selftext"])[:512])
    return " ".join(parts)


def preprocess_posts(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["combined_text"] = df.apply(combine_post_text, axis=1)
    df["clean_text"] = df["combined_text"].apply(clean_text)
    df = df[df["clean_text"].str.len() > 20].reset_index(drop=True)
    df["text_length"] = df["clean_text"].str.len()
    df["engagement"] = np.log1p(df["score"]) * df["upvote_ratio"] + np.log1p(df["num_comments"])
    return df


def aggregate_daily(df: pd.DataFrame, date_col: str = "created_utc") -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df[date_col]).dt.normalize()
    daily = df.groupby(["ticker", "date"]).agg(
        post_count=("clean_text", "count"),
        avg_engagement=("engagement", "mean"),
        total_score=("score", "sum"),
    ).reset_index()
    return daily
