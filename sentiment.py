"""
PSX Sentiment Analysis Module
- Alpha Vantage NEWS_SENTIMENT for topic-based global signal
- OpenAI GPT-4 for PSX-contextualised interpretation and forecast adjustment
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd
import requests
import streamlit as st

# ─── Alpha Vantage constants ──────────────────────────────────────────────────

AV_BASE = "https://www.alphavantage.co/query"

# Sentiment score thresholds (Alpha Vantage definition)
#  ≥ 0.35        = Bullish
#  0.15 – 0.35   = Somewhat-Bullish
# -0.15 – 0.15   = Neutral
# -0.35 – -0.15  = Somewhat-Bearish
#  ≤ -0.35       = Bearish
LABEL_MAP = {
    "Bullish":           1.0,
    "Somewhat-Bullish":  0.5,
    "Neutral":           0.0,
    "Somewhat-Bearish": -0.5,
    "Bearish":          -1.0,
}

# PSX sector → Alpha Vantage topic(s)
_SECTOR_TOPICS: dict[str, list[str]] = {
    "energy":     ["energy_transportation", "economy_macro"],
    "banking":    ["finance", "economy_monetary", "economy_fiscal"],
    "fertilizer": ["manufacturing", "economy_macro"],
    "cement":     ["manufacturing", "real_estate"],
    "technology": ["technology"],
    "pharma":     ["life_sciences"],
    "food":       ["retail_wholesale", "manufacturing"],
    "textile":    ["manufacturing", "retail_wholesale"],
    "auto":       ["manufacturing", "energy_transportation"],
    "default":    ["economy_fiscal", "economy_monetary", "financial_markets"],
}

# PSX ticker → sector
_TICKER_SECTOR: dict[str, str] = {
    # Energy
    "OGDC": "energy", "PPL": "energy", "PSO": "energy",
    "HUBC": "energy", "KAPCO": "energy", "MARI": "energy",
    # Banking / Finance
    "HBL": "banking", "UBL": "banking", "MCB": "banking",
    "BAFL": "banking", "MEBL": "banking", "BAHL": "banking",
    "NBP": "banking", "AKBL": "banking",
    # Fertilizer
    "ENGRO": "fertilizer", "EFERT": "fertilizer", "FFC": "fertilizer",
    "FFBL": "fertilizer", "FATIMA": "fertilizer",
    # Cement
    "LUCK": "cement", "CHCC": "cement", "DGKC": "cement",
    "FCCL": "cement", "PIOC": "cement", "MLCF": "cement",
    # Technology
    "SYS": "technology", "TRG": "technology", "NETSOL": "technology",
    "AVN": "technology",
    # Pharma
    "SEARL": "pharma", "ABOT": "pharma", "GLAXO": "pharma",
    # Food / FMCG
    "NESTLE": "food", "COLG": "food", "UNITY": "food",
    # Textile
    "NML": "textile", "NCL": "textile",
    # Auto
    "INDU": "auto", "HCAR": "auto", "PSMC": "auto",
}


def sector_for(symbol: str) -> str:
    return _TICKER_SECTOR.get(symbol.upper(), "default")


def topics_for(symbol: str) -> list[str]:
    return _SECTOR_TOPICS.get(sector_for(symbol), _SECTOR_TOPICS["default"])


# ─── Alpha Vantage helpers ────────────────────────────────────────────────────

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_av_sentiment(
    av_key: str,
    topics: str,
    time_from: str | None = None,
    limit: int = 50,
) -> list[dict]:
    """
    Fetch news articles from Alpha Vantage NEWS_SENTIMENT.
    Returns a list of article dicts (empty list on error / rate limit).
    """
    params: dict[str, Any] = {
        "function": "NEWS_SENTIMENT",
        "topics":   topics,
        "sort":     "LATEST",
        "limit":    limit,
        "apikey":   av_key,
    }
    if time_from:
        params["time_from"] = time_from

    try:
        resp = requests.get(AV_BASE, params=params, timeout=15)
        data = resp.json()

        if "feed" not in data:
            # Quota hit or bad key
            note = data.get("Note") or data.get("Information") or str(data)
            if "premium" in note.lower() or "rate limit" in note.lower():
                st.warning("⚠️ Alpha Vantage rate limit reached. Sentiment data unavailable.")
            elif "invalid" in note.lower():
                st.error("❌ Invalid Alpha Vantage API key.")
            return []

        return data["feed"]

    except Exception as e:
        st.warning(f"Alpha Vantage fetch error: {e}")
        return []


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_av_sentiment_by_ticker(
    av_key: str,
    tickers: str,
    limit: int = 30,
) -> list[dict]:
    """Fetch news by ticker symbol (best-effort; PSX tickers may return empty)."""
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers":  tickers,
        "sort":     "LATEST",
        "limit":    limit,
        "apikey":   av_key,
    }
    try:
        resp = requests.get(AV_BASE, params=params, timeout=15)
        data = resp.json()
        return data.get("feed", [])
    except Exception:
        return []


# ─── Sentiment aggregation ────────────────────────────────────────────────────

def aggregate_sentiment(articles: list[dict], days_back: int = 7) -> dict:
    """
    Collapse a list of AV articles into a single sentiment summary dict.
    Weights recent articles more heavily (exponential decay).
    """
    if not articles:
        return {
            "score": 0.0,
            "label": "Neutral",
            "article_count": 0,
            "bullish_pct": 0.0,
            "bearish_pct": 0.0,
            "top_headlines": [],
        }

    cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
    scores: list[float] = []
    weights: list[float] = []
    label_counts: dict[str, int] = {}
    headlines: list[dict] = []

    for art in articles:
        ts_str = art.get("time_published", "")
        try:
            # Format: 20240315T120000
            ts = datetime.strptime(ts_str, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
        except ValueError:
            ts = datetime.now(timezone.utc)

        if ts < cutoff:
            continue

        raw_score = float(art.get("overall_sentiment_score", 0.0))
        label     = art.get("overall_sentiment_label", "Neutral")
        age_days  = max((datetime.now(timezone.utc) - ts).total_seconds() / 86400, 0.001)
        w         = np.exp(-0.4 * age_days)   # half-life ≈ 1.7 days

        scores.append(raw_score)
        weights.append(w)
        label_counts[label] = label_counts.get(label, 0) + 1

        headlines.append({
            "title":     art.get("title", ""),
            "source":    art.get("source", ""),
            "score":     raw_score,
            "label":     label,
            "published": ts.strftime("%Y-%m-%d %H:%M"),
            "summary":   art.get("summary", "")[:300],
        })

    if not scores:
        return {
            "score": 0.0,
            "label": "Neutral",
            "article_count": 0,
            "bullish_pct": 0.0,
            "bearish_pct": 0.0,
            "top_headlines": [],
        }

    weighted_score = float(np.average(scores, weights=weights))
    n = len(scores)
    bullish_count = sum(v for k, v in label_counts.items() if "Bullish" in k)
    bearish_count = sum(v for k, v in label_counts.items() if "Bearish" in k)

    if weighted_score >= 0.35:
        agg_label = "Bullish"
    elif weighted_score >= 0.15:
        agg_label = "Somewhat-Bullish"
    elif weighted_score <= -0.35:
        agg_label = "Bearish"
    elif weighted_score <= -0.15:
        agg_label = "Somewhat-Bearish"
    else:
        agg_label = "Neutral"

    # Sort headlines by recency
    headlines.sort(key=lambda x: x["published"], reverse=True)

    return {
        "score":        round(weighted_score, 4),
        "label":        agg_label,
        "article_count": n,
        "bullish_pct":  round(bullish_count / n * 100, 1),
        "bearish_pct":  round(bearish_count / n * 100, 1),
        "top_headlines": headlines[:15],
        "label_counts": label_counts,
    }


def build_sentiment_series(articles: list[dict], date_index: pd.DatetimeIndex) -> pd.Series:
    """
    Build a daily sentiment score series aligned to the price date index.
    Used as ML feature.
    """
    if not articles:
        return pd.Series(0.0, index=date_index, name="sentiment")

    daily: dict[str, list[float]] = {}
    for art in articles:
        ts_str = art.get("time_published", "")
        try:
            day = datetime.strptime(ts_str[:8], "%Y%m%d").strftime("%Y-%m-%d")
        except ValueError:
            continue
        score = float(art.get("overall_sentiment_score", 0.0))
        daily.setdefault(day, []).append(score)

    agg = {k: float(np.mean(v)) for k, v in daily.items()}
    s = pd.Series(agg, name="sentiment")
    s.index = pd.to_datetime(s.index)
    s = s.sort_index()

    # Reindex to price dates and forward-fill gaps (news doesn't publish on weekends)
    aligned = s.reindex(date_index).ffill().bfill().fillna(0.0)
    return aligned


# ─── GPT-4 Analysis ───────────────────────────────────────────────────────────

def _build_system_prompt() -> str:
    return (
        "You are a seasoned equity analyst specialising in the Pakistan Stock Exchange (PSX) "
        "and emerging-market investments. "
        "You combine quantitative signals (technical indicators, price action) with "
        "qualitative macro context (Pakistan's monetary policy, PKR/USD dynamics, CPEC, "
        "commodity exposure, political stability) to form balanced investment views.\n\n"
        "When responding:\n"
        "- Be concise but specific. Avoid generic disclaimers.\n"
        "- Reference the data provided — don't invent figures.\n"
        "- Express directional views with explicit confidence (High / Medium / Low).\n"
        "- Output a single valid JSON object exactly matching the schema requested.\n"
        "- Do NOT include markdown code fences or extra text outside the JSON.\n"
    )


def _build_user_prompt(
    symbol: str,
    sector: str,
    price_data: dict,
    technical_signals: dict,
    sentiment: dict,
) -> str:
    headlines_text = ""
    for i, h in enumerate(sentiment.get("top_headlines", [])[:10], 1):
        sign = "+" if h["score"] >= 0 else ""
        headlines_text += (
            f'{i}. [{h["label"]} {sign}{h["score"]:.2f}] "{h["title"]}" '
            f'— {h["source"]} ({h["published"]})\n'
            f'   Summary: {h["summary"]}\n\n'
        )

    if not headlines_text:
        headlines_text = "No recent news articles retrieved from Alpha Vantage for this sector.\n"

    prompt = f"""
## Stock Under Analysis
- **Ticker:** {symbol}  (Pakistan Stock Exchange)
- **Sector:** {sector}

## Recent Price Performance
- Current Close: PKR {price_data.get('close', 'N/A'):,.2f}
- 1-Day Change: {price_data.get('change_pct', 0):+.2f}%
- 5-Day Return: {price_data.get('ret_5d', 0):+.2f}%
- 20-Day Return: {price_data.get('ret_20d', 0):+.2f}%
- 52-Week High: PKR {price_data.get('high_52w', 0):,.2f}
- 52-Week Low: PKR {price_data.get('low_52w', 0):,.2f}
- Annualised Volatility: {price_data.get('ann_vol', 0):.1f}%

## Technical Signals
{json.dumps(technical_signals, indent=2)}

## Alpha Vantage Global Sentiment (topics: {sector})
- Aggregated Score: {sentiment.get('score', 0.0):+.4f}  ({sentiment.get('label', 'N/A')})
- Articles Analysed: {sentiment.get('article_count', 0)}
- Bullish: {sentiment.get('bullish_pct', 0):.1f}%  |  Bearish: {sentiment.get('bearish_pct', 0):.1f}%

### Recent Headlines
{headlines_text}

## Task
Analyse the above data from a PSX-specific perspective and respond with a JSON object
matching this exact schema (no extra keys, no markdown fences):

{{
  "overall_sentiment": "Bullish|Somewhat-Bullish|Neutral|Somewhat-Bearish|Bearish",
  "sentiment_score": <float -1.0 to 1.0>,
  "technical_outlook": "Bullish|Neutral|Bearish",
  "price_direction": "Up|Sideways|Down",
  "confidence": "High|Medium|Low",
  "short_term_target": <float PKR or null>,
  "key_bullish_factors": ["...", "...", "..."],
  "key_bearish_factors": ["...", "...", "..."],
  "psx_macro_context": "<2-3 sentence PSX-specific macro commentary>",
  "news_interpretation": "<2-3 sentence interpretation of the global news in PSX context>",
  "recommendation": "Strong Buy|Buy|Hold|Sell|Strong Sell",
  "recommendation_rationale": "<2-3 sentences explaining the recommendation>",
  "risk_factors": ["...", "...", "..."],
  "forecast_adjustment": <float: percentage adjustment to ML price forecast, e.g. 2.5 means +2.5%>
}}
"""
    return prompt.strip()


@st.cache_data(ttl=900, show_spinner=False)
def gpt4_stock_analysis(
    openai_key: str,
    symbol: str,
    price_data_json: str,      # JSON-serialised dict
    technical_signals_json: str,
    sentiment_json: str,
) -> dict:
    """
    Call GPT-4 with assembled context and return a structured analysis dict.
    Cached per (symbol, approximate price snapshot) for 15 min.
    """
    from openai import OpenAI

    price_data        = json.loads(price_data_json)
    technical_signals = json.loads(technical_signals_json)
    sentiment         = json.loads(sentiment_json)
    sector            = sector_for(symbol)

    client = OpenAI(api_key=openai_key)

    system_msg = _build_system_prompt()
    user_msg   = _build_user_prompt(symbol, sector, price_data, technical_signals, sentiment)

    try:
        resp = client.chat.completions.create(
            model="gpt-4o",         # gpt-4o is cheaper & faster; use gpt-4 if preferred
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0.3,
            max_tokens=1200,
        )
        raw = resp.choices[0].message.content.strip()

        # Strip accidental markdown code fences if model adds them
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        return json.loads(raw)

    except json.JSONDecodeError as e:
        return {"error": f"GPT-4 returned malformed JSON: {e}", "raw": raw}
    except Exception as e:
        return {"error": str(e)}


# ─── Helper: build price_data dict from DataFrame ─────────────────────────────

def build_price_data(df: pd.DataFrame) -> dict:
    c = df["Close"]
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else last

    ret_5d  = float((c.iloc[-1] / c.iloc[-5]  - 1) * 100) if len(c) >= 5  else 0.0
    ret_20d = float((c.iloc[-1] / c.iloc[-20] - 1) * 100) if len(c) >= 20 else 0.0
    ann_vol = float(c.pct_change().std() * np.sqrt(252) * 100)

    return {
        "close":      float(last["Close"]),
        "change_pct": float((last["Close"] - prev["Close"]) / prev["Close"] * 100),
        "ret_5d":     ret_5d,
        "ret_20d":    ret_20d,
        "high_52w":   float(df["High"].max()),
        "low_52w":    float(df["Low"].min()),
        "ann_vol":    ann_vol,
    }


def build_technical_signals(df: pd.DataFrame) -> dict:
    """Extract the latest values of all technical indicators as a flat dict."""
    signals: dict[str, Any] = {}
    last = df.iloc[-1]

    def _v(col: str) -> str:
        val = last.get(col)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "N/A"
        return round(float(val), 4)

    close = float(last["Close"])

    for col in ["SMA_20", "SMA_50", "SMA_200", "EMA_12", "EMA_26",
                "BB_Upper", "BB_Mid", "BB_Lower",
                "RSI", "MACD", "MACD_Signal", "MACD_Hist",
                "ATR", "Stoch_K", "Stoch_D", "BB_Pct"]:
        signals[col] = _v(col)

    # Derived signal labels
    def rel(col: str) -> str:
        v = last.get(col)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "N/A"
        return "Above" if close > float(v) else "Below"

    signals["price_vs_SMA20"]  = rel("SMA_20")
    signals["price_vs_SMA50"]  = rel("SMA_50")
    signals["price_vs_SMA200"] = rel("SMA_200")

    rsi = last.get("RSI")
    if rsi and not np.isnan(float(rsi)):
        r = float(rsi)
        signals["rsi_signal"] = "Overbought" if r > 70 else ("Oversold" if r < 30 else "Neutral")

    macd = last.get("MACD")
    sig  = last.get("MACD_Signal")
    if macd and sig and not np.isnan(float(macd)):
        signals["macd_signal"] = "Bullish crossover" if float(macd) > float(sig) else "Bearish crossover"

    return signals
