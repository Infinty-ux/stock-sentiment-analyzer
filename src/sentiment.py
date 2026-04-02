from typing import List, Dict, Union
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

FINBERT_MODEL = "ProsusAI/finbert"
LABELS = ["negative", "neutral", "positive"]
SCORE_MAP = {"negative": -1.0, "neutral": 0.0, "positive": 1.0}


class SentimentAnalyzer:
    def __init__(self, model_name: str = FINBERT_MODEL, device: str = "cpu", batch_size: int = 32):
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        self.model.to(self.device)

    @torch.no_grad()
    def _predict_batch(self, texts: List[str]) -> List[Dict]:
        encoding = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt",
        )
        encoding = {k: v.to(self.device) for k, v in encoding.items()}
        logits = self.model(**encoding).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        results = []
        for prob_row in probs:
            idx = int(np.argmax(prob_row))
            label = LABELS[idx]
            results.append({
                "label": label,
                "score": float(prob_row[idx]),
                "compound": float(
                    prob_row[2] * SCORE_MAP["positive"]
                    + prob_row[0] * SCORE_MAP["negative"]
                ),
                "probabilities": {LABELS[i]: float(prob_row[i]) for i in range(3)},
            })
        return results

    def analyze(self, texts: Union[str, List[str]]) -> List[Dict]:
        if isinstance(texts, str):
            texts = [texts]
        all_results = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i: i + self.batch_size]
            all_results.extend(self._predict_batch(batch))
        return all_results

    def analyze_dataframe(self, df: pd.DataFrame, text_col: str = "clean_text") -> pd.DataFrame:
        df = df.copy()
        results = self.analyze(df[text_col].tolist())
        df["sentiment_label"] = [r["label"] for r in results]
        df["sentiment_score"] = [r["score"] for r in results]
        df["sentiment_compound"] = [r["compound"] for r in results]
        df["sentiment_pos"] = [r["probabilities"]["positive"] for r in results]
        df["sentiment_neu"] = [r["probabilities"]["neutral"] for r in results]
        df["sentiment_neg"] = [r["probabilities"]["negative"] for r in results]
        return df

    def aggregate_sentiment(self, df: pd.DataFrame, weight_col: str = "engagement") -> pd.DataFrame:
        df = df.copy()
        weights = df[weight_col].fillna(1.0).clip(lower=0.01)
        daily = df.groupby(["ticker", "date"]).apply(
            lambda g: pd.Series({
                "weighted_compound": float(np.average(g["sentiment_compound"], weights=weights.loc[g.index])),
                "avg_positive": float(g["sentiment_pos"].mean()),
                "avg_negative": float(g["sentiment_neg"].mean()),
                "bullish_ratio": float((g["sentiment_label"] == "positive").mean()),
                "bearish_ratio": float((g["sentiment_label"] == "negative").mean()),
                "post_count": len(g),
            })
        ).reset_index()
        return daily
