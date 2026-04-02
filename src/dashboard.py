import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import timedelta

from scraper import RedditScraper, PriceScraper
from preprocessor import preprocess_posts, aggregate_daily
from sentiment import SentimentAnalyzer
from correlator import merge_sentiment_price, analyze_correlations, rolling_correlation

st.set_page_config(page_title="Stock Sentiment Analyzer", layout="wide", page_icon="📈")

@st.cache_resource
def load_models():
    return SentimentAnalyzer(device="cpu", batch_size=16)


@st.cache_data(ttl=3600, show_spinner="Fetching Reddit data...")
def fetch_and_analyze(ticker: str, days: int, limit: int):
    reddit = RedditScraper()
    price = PriceScraper()
    analyzer = load_models()

    posts_df = reddit.fetch_posts(ticker, limit=limit)
    if posts_df.empty:
        return None, None, None

    posts_df = preprocess_posts(posts_df)
    posts_df = analyzer.analyze_dataframe(posts_df)
    posts_df["date"] = posts_df["created_utc"].dt.normalize()

    sentiment_daily = analyzer.aggregate_sentiment(posts_df)
    price_df = price.fetch_ohlcv(ticker)
    merged = merge_sentiment_price(sentiment_daily, price_df)

    return posts_df, sentiment_daily, merged


st.title("📈 Stock Sentiment Analyzer")
st.caption("Reddit NLP (FinBERT) × Price correlation")

with st.sidebar:
    st.header("Configuration")
    ticker = st.text_input("Ticker symbol", value="TSLA").upper().strip()
    days = st.slider("Days lookback", 7, 90, 30)
    limit = st.slider("Posts per subreddit", 50, 500, 200)
    run = st.button("Analyze", type="primary")

if run and ticker:
    posts_df, sentiment_daily, merged = fetch_and_analyze(ticker, days, limit)

    if posts_df is None:
        st.error("No data returned for that ticker.")
        st.stop()

    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Sentiment Timeline", "Price Correlation", "Raw Posts"])

    with tab1:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Posts analyzed", len(posts_df))
        col2.metric("Avg compound score", f"{posts_df['sentiment_compound'].mean():.3f}")
        col3.metric("Bullish %", f"{(posts_df['sentiment_label']=='positive').mean():.1%}")
        col4.metric("Bearish %", f"{(posts_df['sentiment_label']=='negative').mean():.1%}")

        fig_pie = px.pie(
            values=posts_df["sentiment_label"].value_counts().values,
            names=posts_df["sentiment_label"].value_counts().index,
            color_discrete_map={"positive": "#22c55e", "neutral": "#94a3b8", "negative": "#ef4444"},
            title="Sentiment Distribution",
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with tab2:
        if sentiment_daily is not None and not sentiment_daily.empty:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4])
            fig.add_trace(go.Scatter(
                x=sentiment_daily["date"], y=sentiment_daily["weighted_compound"],
                mode="lines+markers", name="Compound score",
                line=dict(color="#6366f1", width=2),
            ), row=1, col=1)
            fig.add_trace(go.Bar(
                x=sentiment_daily["date"], y=sentiment_daily["post_count"],
                name="Post volume", marker_color="#0ea5e9",
            ), row=2, col=1)
            fig.update_layout(title=f"{ticker} Daily Sentiment", height=500)
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        if merged is not None and not merged.empty:
            rc = rolling_correlation(merged, ticker)
            if not rc.empty:
                fig2 = px.line(rc, x="date", y="rolling_corr",
                               title=f"{ticker} 14-day Rolling Correlation (Sentiment vs Returns)",
                               color_discrete_sequence=["#f59e0b"])
                fig2.add_hline(y=0, line_dash="dash", line_color="gray")
                st.plotly_chart(fig2, use_container_width=True)

            corr_df = analyze_correlations(merged, [ticker])
            if not corr_df.empty:
                st.subheader("Statistical Correlations")
                st.dataframe(corr_df.set_index("ticker").T, use_container_width=True)

    with tab4:
        st.dataframe(
            posts_df[["created_utc", "subreddit", "title", "score", "sentiment_label", "sentiment_compound"]].head(100),
            use_container_width=True,
        )
else:
    st.info("Enter a ticker and click **Analyze** to begin.")
