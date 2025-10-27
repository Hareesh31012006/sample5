# =========================================================
# 📊 Stock Sentiment & Price Prediction Dashboard (Stable)
# =========================================================

import os
import random
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from alpha_vantage.timeseries import TimeSeries
from gnews import GNews
from textblob import TextBlob
from transformers import pipeline
import torch
import torch.nn as nn
import time

# =========================================================
# 🔑 API KEY
# =========================================================
ALPHA_VANTAGE_API_KEY = "9EJ41V9XS6Q5ZN1Y"  # Replace with your key

# =========================================================
# 🧱 Streamlit Setup
# =========================================================
st.set_page_config(page_title="Stock Sentiment Predictor", layout="wide")
st.title("📈 Stock Sentiment + Price Prediction Dashboard")

# =========================================================
# 🧠 Cached Resources
# =========================================================
@st.cache_resource
def load_sentiment_model():
    """Load Hugging Face sentiment model (cached once)."""
    return pipeline("sentiment-analysis")

@st.cache_resource
def load_pytorch_model(input_size):
    """Build simple regression model."""
    model = nn.Linear(input_size, 1)
    return model

hf_pipeline = load_sentiment_model()

# =========================================================
# 🔎 Data Fetching Functions
# =========================================================
@st.cache_data(ttl=1800)
def fetch_stock_data(symbol):
    """Fetch stock data from Alpha Vantage."""
    ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format="pandas")
    for attempt in range(3):
        try:
            data, _ = ts.get_daily(symbol=symbol, outputsize="compact")
            data = data.rename(columns={
                '1. open': 'Open',
                '2. high': 'High',
                '3. low': 'Low',
                '4. close': 'Close',
                '5. volume': 'Volume'
            })
            data.index = pd.to_datetime(data.index)
            return data.sort_index()
        except Exception as e:
            time.sleep(2)
    raise RuntimeError("API error — please retry later.")

@st.cache_data(ttl=1800)
def fetch_news(symbol):
    """Fetch news using GNews."""
    try:
        gnews = GNews(language="en", max_results=10)
        return gnews.get_news(symbol)
    except Exception:
        return []

# =========================================================
# 💬 Sentiment Analysis
# =========================================================
def compute_sentiment(text):
    try:
        tb_score = TextBlob(text).sentiment.polarity
        hf_result = hf_pipeline(text[:512])[0]
        hf_score = hf_result['score'] if hf_result['label'] == 'POSITIVE' else -hf_result['score']
        return (tb_score + hf_score) / 2
    except Exception:
        return 0

# =========================================================
# 🧮 Model Training
# =========================================================
def train_model(X, y):
    model = load_pytorch_model(X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    for _ in range(100):
        opt.zero_grad()
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        loss.backward()
        opt.step()
    return model

# =========================================================
# 🔍 Main Analysis Function
# =========================================================
def analyze_stock(symbol):
    # Fetch data
    df = fetch_stock_data(symbol)
    df["Return"] = df["Close"].pct_change().fillna(0)

    # Fetch and score news
    articles = fetch_news(symbol)
    sentiments = []
    for a in articles:
        text = f"{a.get('title', '')} {a.get('description', '')}"
        sentiments.append(compute_sentiment(text))
    avg_sentiment = np.mean(sentiments) if sentiments else 0

    # Prepare ML data
    df = df.dropna()
    X = torch.tensor(df[["Open", "High", "Low", "Volume"]].values, dtype=torch.float32)
    y = torch.tensor(df["Close"].values.reshape(-1, 1), dtype=torch.float32)

    model = train_model(X, y)
    last = torch.tensor(df.iloc[-1][["Open", "High", "Low", "Volume"]].values, dtype=torch.float32)
    next_pred = model(last.unsqueeze(0)).item()

    # Suggestion logic
    last_close = df["Close"].iloc[-1]
    suggestion = "📈 BUY" if next_pred > last_close and avg_sentiment > 0 else "📉 SELL"

    return df, sentiments, avg_sentiment, next_pred, suggestion

# =========================================================
# 🖥️ UI
# =========================================================
symbol = st.text_input("Enter Stock Symbol (e.g., AAPL, TSLA, MSFT):", "AAPL")

if st.button("Run Analysis"):
    with st.spinner("Analyzing data..."):
        try:
            df, sentiments, avg_sent, pred, sug = analyze_stock(symbol)
            st.success("✅ Analysis complete!")

            col1, col2 = st.columns(2)
            col1.metric("Predicted Next Close", f"${pred:.2f}")
            col2.metric("Recommendation", sug)

            st.line_chart(df["Close"], use_container_width=True)

            st.subheader("Sentiment Distribution")
            fig, ax = plt.subplots(figsize=(8, 3))
            sns.histplot(sentiments, bins=10, kde=True, ax=ax)
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Error: {e}")

st.caption("Stable version with caching and retry logic.")
