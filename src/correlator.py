from typing import Tuple, Dict, List
import numpy as np
import pandas as pd
from scipy import stats


def merge_sentiment_price(
    sentiment_df: pd.DataFrame,
    price_df: pd.DataFrame,
    lag_days: int = 0,
) -> pd.DataFrame:
    sentiment_df = sentiment_df.copy()
    price_df = price_df.copy()

    price_df["date"] = pd.to_datetime(price_df.index).normalize()
    sentiment_df["date"] = pd.to_datetime(sentiment_df["date"]).dt.normalize()
    price_df = price_df.reset_index(drop=True)

    merged = pd.merge(sentiment_df, price_df, on=["ticker", "date"], how="inner")
    if lag_days != 0:
        merged["returns_lagged"] = merged.groupby("ticker")["returns"].shift(-lag_days)
        merged = merged.dropna(subset=["returns_lagged"])
        merged["returns"] = merged["returns_lagged"]

    return merged


def pearson_correlation(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    r, p = stats.pearsonr(x, y)
    return float(r), float(p)


def spearman_correlation(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    r, p = stats.spearmanr(x, y)
    return float(r), float(p)


def granger_causality_test(y: np.ndarray, x: np.ndarray, max_lag: int = 5) -> Dict:
    from statsmodels.tsa.stattools import grangercausalitytests
    data = np.column_stack([y, x])
    result = grangercausalitytests(data, maxlag=max_lag, verbose=False)
    best_lag, best_p = min(
        ((lag, res[0]["ssr_ftest"][1]) for lag, res in result.items()),
        key=lambda item: item[1],
    )
    return {"best_lag": best_lag, "p_value": float(best_p), "significant": best_p < 0.05}


def analyze_correlations(merged: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    records = []
    for ticker in tickers:
        sub = merged[merged["ticker"] == ticker].dropna(subset=["weighted_compound", "returns"])
        if len(sub) < 10:
            continue
        x = sub["weighted_compound"].values
        y = sub["returns"].values

        pr, pp = pearson_correlation(x, y)
        sr, sp = spearman_correlation(x, y)

        try:
            granger = granger_causality_test(y, x)
        except Exception:
            granger = {"best_lag": None, "p_value": None, "significant": False}

        records.append({
            "ticker": ticker,
            "n_observations": len(sub),
            "pearson_r": pr,
            "pearson_p": pp,
            "spearman_r": sr,
            "spearman_p": sp,
            "pearson_significant": pp < 0.05,
            "spearman_significant": sp < 0.05,
            "granger_best_lag": granger["best_lag"],
            "granger_p": granger["p_value"],
            "granger_significant": granger["significant"],
        })
    return pd.DataFrame(records)


def rolling_correlation(
    merged: pd.DataFrame,
    ticker: str,
    window: int = 14,
) -> pd.DataFrame:
    sub = merged[merged["ticker"] == ticker].sort_values("date").copy()
    sub["rolling_corr"] = sub["weighted_compound"].rolling(window).corr(sub["returns"])
    return sub[["date", "rolling_corr", "weighted_compound", "returns", "close"]].dropna()
