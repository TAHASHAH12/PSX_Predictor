"""
dawn_scraper.py — Dawn.com Pakistan business news scraper + BERT sentiment
No API key required. BERT model downloads from HuggingFace on first use (~680 MB).
Model: nlptown/bert-base-multilingual-uncased-sentiment (1–5 star scale)
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
from bs4 import BeautifulSoup

try:
    import streamlit as st
    _HAS_ST = True
except ImportError:
    _HAS_ST = False

# ── Constants ──────────────────────────────────────────────────────────────────
DAWN_BASE   = "https://www.dawn.com"
DAWN_BIZ_URL = "https://www.dawn.com/business/business-finance"
BERT_MODEL  = "nlptown/bert-base-multilingual-uncased-sentiment"
MAX_PAGES   = 3
PAGE_DELAY  = 1.0   # seconds between page requests (polite crawl)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Cache-Control": "no-cache",
}

# Keywords that indicate an article is relevant to Pakistan equities/finance
_PSX_KEYWORDS = {
    "psx", "kse", "karachi stock", "pakistan stock", "stock market",
    "shares", "equity", "listed company",
    "sbp", "state bank", "rupee", "pkr",
    "interest rate", "policy rate", "inflation", "cpi",
    "dividend", "earnings", "profit", "revenue", "eps", "net income",
    "ogdc", "ppl", "pso", "hbl", "mcb", "ubl", "engro", "luck",
    "hubc", "efert", "ffc", "nestle", "colgate", "ipo",
    "bonds", "sukuk", "t-bills", "treasury",
    "imf", "current account", "forex reserve", "trade deficit",
    "cpec", "investment", "gdp", "budget", "fiscal deficit",
    "monetary policy", "economic growth", "balance of payments",
    "privatisation", "privatization",
}

# ─────────────────────────────────────────────────────────────────────────────
# BERT sentiment model (cached so it's loaded only once per session)
# nlptown/bert-base-multilingual-uncased-sentiment: labels "1 star"…"5 stars"
# ─────────────────────────────────────────────────────────────────────────────

def _get_bert_pipeline():
    """Load the BERT text-classification pipeline (cached)."""
    from transformers import pipeline as hf_pipeline
    return hf_pipeline(
        "text-classification",
        model=BERT_MODEL,
        top_k=None,      # return all 5 class scores
        device=-1,       # CPU
        truncation=True,
        max_length=512,
    )


if _HAS_ST:
    @st.cache_resource(show_spinner=False)
    def load_bert_model():
        return _get_bert_pipeline()
else:
    _bert_cache: dict = {}
    def load_bert_model():
        if "pipe" not in _bert_cache:
            _bert_cache["pipe"] = _get_bert_pipeline()
        return _bert_cache["pipe"]


# ─────────────────────────────────────────────────────────────────────────────
# Dawn.com scraping
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_html(url: str, timeout: int = 15) -> str | None:
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=timeout)
        resp.raise_for_status()
        return resp.text
    except Exception:
        return None


def _parse_datetime(raw: str) -> datetime | None:
    """Try several formats used by Dawn.com timestamps."""
    if not raw:
        return None
    raw = raw.strip()
    fmts = [
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M%z",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
        "%B %d, %Y",
        "%b %d, %Y",
        "%d %B %Y",
    ]
    for fmt in fmts:
        try:
            dt = datetime.strptime(raw[:25], fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            pass
    # Try ISO 8601 with timezone offset like +05:00
    try:
        import re as _re
        cleaned = _re.sub(r'(\d{2}):(\d{2})$', r'\1\2', raw[:25])
        dt = datetime.strptime(cleaned, "%Y-%m-%dT%H:%M:%S%z")
        return dt
    except Exception:
        pass
    return None


def _articles_from_jsonld(soup: BeautifulSoup) -> list[dict]:
    """Extract articles from JSON-LD structured data (most reliable)."""
    articles = []
    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(tag.string or "{}")
        except (json.JSONDecodeError, TypeError):
            continue

        items = []
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            if data.get("@graph"):
                items = data["@graph"]
            else:
                items = [data]

        for item in items:
            typ = item.get("@type", "")
            if typ not in ("NewsArticle", "Article", "BlogPosting"):
                continue
            headline = item.get("headline") or item.get("name", "")
            if not headline:
                continue
            main = item.get("mainEntityOfPage", {})
            url  = item.get("url") or (main.get("@id") if isinstance(main, dict) else "") or ""
            articles.append({
                "title":    headline.strip(),
                "url":      url,
                "excerpt":  (item.get("description") or "")[:500],
                "time_str": item.get("datePublished") or item.get("dateModified") or "",
                "source":   "Dawn.com",
            })
    return articles


def _articles_from_css(soup: BeautifulSoup, base_url: str) -> list[dict]:
    """
    Extract articles using CSS selectors.
    Tries multiple selector strategies to handle Dawn.com layout variations.
    """
    # Container candidates (order = preference)
    container_selectors = [
        ("article", {}),
        ("div",     {"class": re.compile(r"\bstory\b", re.I)}),
        ("div",     {"class": re.compile(r"article-card|news-item|list-item", re.I)}),
        ("li",      {"class": re.compile(r"\bstory\b|\barticle\b", re.I)}),
    ]

    containers = []
    for tag, attrs in container_selectors:
        found = soup.find_all(tag, attrs) if attrs else soup.find_all(tag)
        if found:
            containers = found
            break

    articles = []
    seen_urls: set[str] = set()

    for cont in containers:
        # Title
        title_el = (
            cont.find(["h1", "h2", "h3", "h4"], class_=re.compile(r"title|headline", re.I)) or
            cont.find(["h1", "h2", "h3", "h4"]) or
            cont.find("a", class_=re.compile(r"title|headline", re.I))
        )
        if not title_el:
            continue
        title_text = title_el.get_text(strip=True)
        if len(title_text) < 10:
            continue

        # URL
        link_el = title_el.find("a") if title_el.name != "a" else title_el
        if not link_el:
            link_el = cont.find("a", href=re.compile(r"/\d{4}/"))
        href = (link_el.get("href") or "") if link_el else ""
        if href and not href.startswith("http"):
            href = base_url.rstrip("/") + "/" + href.lstrip("/")
        if href in seen_urls:
            continue
        seen_urls.add(href)

        # Excerpt
        excerpt_el = (
            cont.find(["p", "span"], class_=re.compile(r"excerpt|deck|summary|description|teaser", re.I)) or
            cont.find("p")
        )
        excerpt = excerpt_el.get_text(strip=True)[:500] if excerpt_el else ""

        # Published time
        time_el = cont.find("time") or cont.find(attrs={"datetime": True})
        if time_el:
            time_str = time_el.get("datetime") or time_el.get_text(strip=True)
        else:
            ts_el = cont.find(class_=re.compile(r"timestamp|time|date|published", re.I))
            time_str = ts_el.get_text(strip=True) if ts_el else ""

        articles.append({
            "title":    title_text,
            "url":      href,
            "excerpt":  excerpt,
            "time_str": time_str,
            "source":   "Dawn.com",
        })

    return articles


def _articles_fallback_links(soup: BeautifulSoup, base_url: str) -> list[dict]:
    """
    Last-resort: scrape all links that look like Dawn news article URLs.
    Pattern: /YYYY/MMM/DD/ in the path.
    """
    articles = []
    seen: set[str] = set()
    pattern = re.compile(r"/\d{4}/[a-z]{3}/\d{2}/")

    for a in soup.find_all("a", href=pattern):
        href = a.get("href", "")
        if not href.startswith("http"):
            href = base_url.rstrip("/") + "/" + href.lstrip("/")
        if href in seen or len(href) < 20:
            continue
        seen.add(href)
        title = a.get_text(strip=True)
        if len(title) < 10:
            continue
        articles.append({
            "title":    title,
            "url":      href,
            "excerpt":  "",
            "time_str": "",
            "source":   "Dawn.com",
        })
    return articles


def _parse_page(html: str, page_url: str) -> list[dict]:
    """Parse a single Dawn.com page HTML and return article dicts."""
    soup = BeautifulSoup(html, "lxml")

    # Strategy 1: JSON-LD (most structured)
    articles = _articles_from_jsonld(soup)
    if articles:
        return articles

    # Strategy 2: CSS selectors
    articles = _articles_from_css(soup, DAWN_BASE)
    if articles:
        return articles

    # Strategy 3: generic link extraction
    return _articles_fallback_links(soup, DAWN_BASE)


def _scrape_raw(max_pages: int = MAX_PAGES) -> list[dict]:
    """Scrape Dawn.com business/finance pages. Returns raw article dicts."""
    all_articles: list[dict] = []
    seen_urls: set[str] = set()

    for page_num in range(1, max_pages + 1):
        url = DAWN_BIZ_URL if page_num == 1 else f"{DAWN_BIZ_URL}?page={page_num}"
        html = _fetch_html(url)
        if html is None:
            break

        page_articles = _parse_page(html, url)
        added = 0
        for art in page_articles:
            u = art.get("url", "")
            if u and u in seen_urls:
                continue
            if u:
                seen_urls.add(u)
            all_articles.append(art)
            added += 1

        if added == 0:
            break  # no new articles → stop paginating

        if page_num < max_pages:
            time.sleep(PAGE_DELAY)

    return all_articles


# ── Cached public entry-point ──────────────────────────────────────────────────

if _HAS_ST:
    @st.cache_data(ttl=1800, show_spinner=False)
    def scrape_dawn_articles(max_pages: int = MAX_PAGES) -> list[dict]:
        return _scrape_raw(max_pages)
else:
    def scrape_dawn_articles(max_pages: int = MAX_PAGES) -> list[dict]:
        return _scrape_raw(max_pages)


# ─────────────────────────────────────────────────────────────────────────────
# BERT inference
# ─────────────────────────────────────────────────────────────────────────────

def _compound_from_stars(result_row: list[dict]) -> tuple[float, str, float, float, float]:
    """
    Convert nlptown 5-star top_k=None result into a scalar compound score.

    Labels: "1 star", "2 stars", "3 stars", "4 stars", "5 stars"
    weighted_stars = Σ(star_i * prob_i)  →  compound = (weighted_stars - 3) / 2

    This maps:
        1 star  →  -1.0  (very bearish)
        3 stars →   0.0  (neutral)
        5 stars →  +1.0  (very bullish)

    Returns (compound, label, pos_prob, neg_prob, neu_prob)
    where pos/neg/neu are probability masses above/below/at 3 stars.
    """
    # Build {star_int: probability}
    star_probs: dict[int, float] = {}
    for r in result_row:
        raw_label = r["label"]
        # Label format: "1 star" or "2 stars"
        try:
            star = int(raw_label.split()[0])
        except (ValueError, IndexError):
            star = 3  # fallback to neutral
        star_probs[star] = float(r["score"])

    # Weighted average star rating → compound score
    weighted = sum(star * prob for star, prob in star_probs.items())
    compound = float((weighted - 3.0) / 2.0)
    compound = max(-1.0, min(1.0, compound))  # clamp

    # Map 5 classes → 3 sentiment groups
    pos = star_probs.get(4, 0.0) + star_probs.get(5, 0.0)   # 4+5 stars = positive
    neg = star_probs.get(1, 0.0) + star_probs.get(2, 0.0)   # 1+2 stars = negative
    neu = star_probs.get(3, 0.0)                              # 3 stars   = neutral

    if compound >= 0.15:
        label = "Positive"
    elif compound <= -0.15:
        label = "Negative"
    else:
        label = "Neutral"

    return compound, label, pos, neg, neu


def run_bert_sentiment(texts: list[str]) -> list[dict]:
    """
    Run BERT sentiment on a list of texts.
    Returns list of dicts with keys: compound, label, pos, neg, neu.
    Empty / very short texts get neutral score 0.0.
    """
    pipe = load_bert_model()
    results = []

    for text in texts:
        text = (text or "").strip()
        if len(text) < 5:
            results.append({"compound": 0.0, "label": "Neutral", "pos": 0.0, "neg": 0.0, "neu": 1.0})
            continue
        try:
            raw = pipe(text[:1000])  # safety truncation before tokenizer
            # pipeline with top_k=None wraps in an extra list when batching=False
            row = raw[0] if isinstance(raw[0], list) else raw
            compound, label, pos, neg, neu = _compound_from_stars(row)
            results.append({
                "compound": compound,
                "label":    label,
                "pos":      pos,
                "neg":      neg,
                "neu":      neu,
            })
        except Exception:
            results.append({"compound": 0.0, "label": "Neutral", "pos": 0.0, "neg": 0.0, "neu": 1.0})

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Analysis pipeline: scrape → BERT → enrich
# ─────────────────────────────────────────────────────────────────────────────

def _is_psx_relevant(title: str, excerpt: str) -> bool:
    """Return True if the article is likely relevant to Pakistan finance/stocks."""
    combined = (title + " " + excerpt).lower()
    return any(kw in combined for kw in _PSX_KEYWORDS)


def analyze_dawn_articles(articles: list[dict], psx_only: bool = False) -> list[dict]:
    """
    Run BERT sentiment on scraped articles. Adds 'compound', 'label', 'pos', 'neg',
    'neu', 'published', 'psx_relevant' keys to each article dict.
    """
    if not articles:
        return []

    if psx_only:
        articles = [a for a in articles if _is_psx_relevant(a.get("title", ""), a.get("excerpt", ""))]

    # Build input texts: title + excerpt for richer context
    texts = [
        f"{a.get('title', '')}. {a.get('excerpt', '')}".strip()
        for a in articles
    ]

    sentiments = run_bert_sentiment(texts)

    enriched = []
    for art, sent in zip(articles, sentiments):
        dt = _parse_datetime(art.get("time_str", ""))
        art_out = dict(art)
        art_out.update(sent)
        art_out["published"] = dt.strftime("%Y-%m-%d %H:%M") if dt else "unknown"
        art_out["published_dt"] = dt
        art_out["psx_relevant"] = _is_psx_relevant(art.get("title", ""), art.get("excerpt", ""))
        enriched.append(art_out)

    # Sort newest first
    enriched.sort(
        key=lambda x: x["published_dt"] or datetime(1970, 1, 1, tzinfo=timezone.utc),
        reverse=True,
    )
    return enriched


# ─────────────────────────────────────────────────────────────────────────────
# Sentiment aggregation (compatible with sentiment.py aggregate_sentiment format)
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_dawn_sentiment(
    articles: list[dict],
    days_back: int = 7,
    psx_only: bool = True,
) -> dict:
    """
    Collapse enriched Dawn articles into a single sentiment summary dict.
    Applies exponential time decay (same half-life as Alpha Vantage module).
    """
    empty = {
        "score": 0.0, "label": "Neutral", "article_count": 0,
        "bullish_pct": 0.0, "bearish_pct": 0.0,
        "top_headlines": [], "label_counts": {},
        "source": "Dawn.com + BERT",
    }
    if not articles:
        return empty

    now    = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=days_back)

    scores:  list[float] = []
    weights: list[float] = []
    label_counts: dict[str, int] = {}
    headlines: list[dict] = []

    for art in articles:
        if psx_only and not art.get("psx_relevant", True):
            continue
        dt = art.get("published_dt")
        if dt is None:
            dt = now  # if unparseable, treat as current
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        if dt < cutoff:
            continue

        compound = float(art.get("compound", 0.0))
        label    = art.get("label", "Neutral")
        age_days = max((now - dt).total_seconds() / 86400, 0.001)
        w        = float(np.exp(-0.4 * age_days))

        scores.append(compound)
        weights.append(w)
        label_counts[label] = label_counts.get(label, 0) + 1

        headlines.append({
            "title":     art.get("title", ""),
            "source":    art.get("source", "Dawn.com"),
            "score":     compound,
            "label":     label,
            "published": art.get("published", ""),
            "url":       art.get("url", ""),
            "summary":   art.get("excerpt", "")[:250],
            "psx_relevant": art.get("psx_relevant", False),
        })

    if not scores:
        return empty

    weighted_score = float(np.average(scores, weights=weights))
    n              = len(scores)
    bullish        = label_counts.get("Positive", 0)
    bearish        = label_counts.get("Negative", 0)

    if weighted_score >= 0.25:
        agg_label = "Bullish"
    elif weighted_score >= 0.08:
        agg_label = "Somewhat-Bullish"
    elif weighted_score <= -0.25:
        agg_label = "Bearish"
    elif weighted_score <= -0.08:
        agg_label = "Somewhat-Bearish"
    else:
        agg_label = "Neutral"

    headlines.sort(key=lambda x: x["published"], reverse=True)

    return {
        "score":         round(weighted_score, 4),
        "label":         agg_label,
        "article_count": n,
        "bullish_pct":   round(bullish / n * 100, 1),
        "bearish_pct":   round(bearish / n * 100, 1),
        "top_headlines": headlines[:20],
        "label_counts":  label_counts,
        "source":        "Dawn.com + BERT",
    }


# ─────────────────────────────────────────────────────────────────────────────
# ML feature: daily sentiment series aligned to price dates
# ─────────────────────────────────────────────────────────────────────────────

def build_dawn_sentiment_series(
    articles: list[dict],
    date_index: pd.DatetimeIndex,
) -> pd.Series:
    """
    Build a daily BERT sentiment score series aligned to the price date index.
    Forward-fills gaps (weekends, no articles) so every trading day has a value.
    """
    if not articles:
        return pd.Series(0.0, index=date_index, name="dawn_sentiment")

    daily: dict[str, list[float]] = {}
    for art in articles:
        dt = art.get("published_dt")
        if dt is None:
            continue
        day = dt.strftime("%Y-%m-%d")
        daily.setdefault(day, []).append(float(art.get("compound", 0.0)))

    if not daily:
        return pd.Series(0.0, index=date_index, name="dawn_sentiment")

    agg = {k: float(np.mean(v)) for k, v in daily.items()}
    s   = pd.Series(agg, name="dawn_sentiment")
    s.index = pd.to_datetime(s.index)
    s = s.sort_index()

    aligned = s.reindex(date_index).ffill().bfill().fillna(0.0)
    return aligned


# ─────────────────────────────────────────────────────────────────────────────
# Combined sentiment: merge Alpha Vantage + Dawn/BERT
# ─────────────────────────────────────────────────────────────────────────────

def combine_sentiment_series(
    av_series: pd.Series | None,
    dawn_series: pd.Series | None,
    av_weight: float = 0.4,
    dawn_weight: float = 0.6,
) -> pd.Series:
    """
    Weighted blend of Alpha Vantage and Dawn/BERT sentiment series.
    Weights normalised automatically when only one series is available.
    """
    if av_series is None and dawn_series is None:
        return None
    if av_series is None:
        return dawn_series.rename("sentiment")
    if dawn_series is None:
        return av_series.rename("sentiment")

    idx = av_series.index.union(dawn_series.index)
    av_r    = av_series.reindex(idx).ffill().bfill().fillna(0.0)
    dawn_r  = dawn_series.reindex(idx).ffill().bfill().fillna(0.0)
    total_w = av_weight + dawn_weight
    combined = (av_r * av_weight + dawn_r * dawn_weight) / total_w
    return combined.rename("sentiment")
