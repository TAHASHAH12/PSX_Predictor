"""
fear_greed.py — PSX Fear & Greed Index
Data: https://www.psx-fear-greed.com/api/public/current
      https://www.psx-fear-greed.com/api/public/history
No API key required.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd
import requests

try:
    import streamlit as st
    _HAS_ST = True
except ImportError:
    _HAS_ST = False

# ── Constants ─────────────────────────────────────────────────────────────────
_BASE   = "https://www.psx-fear-greed.com/api/public"
_HDRS   = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.psx-fear-greed.com/",
    "Accept":  "application/json",
}

# Zone boundaries (score 0-100)
_ZONES = [
    (0,  25,  "Extreme Fear", "#c62828", "rgba(198,40,40,0.25)"),
    (25, 45,  "Fear",         "#ef5350", "rgba(239,83,80,0.18)"),
    (45, 55,  "Neutral",      "#ffd54f", "rgba(255,213,79,0.15)"),
    (55, 75,  "Greed",        "#66bb6a", "rgba(102,187,106,0.18)"),
    (75, 100, "Extreme Greed","#2e7d32", "rgba(46,125,50,0.25)"),
]

_COMPONENT_LABELS = {
    "momentum":         "Momentum",
    "momentumIntraday": "Intraday Momentum",
    "news":             "News Sentiment",
    "social":           "Social Media",
    "trends":           "Google Trends",
    "volatility":       "Volatility",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def score_to_label(score: float) -> str:
    for lo, hi, label, *_ in _ZONES:
        if lo <= score < hi:
            return label
    return "Extreme Greed"


def label_color(label: str) -> str:
    for *_, lbl, color, _ in _ZONES:
        if lbl == label:
            return color
    return "#aaa"


def label_emoji(label: str) -> str:
    return {
        "Extreme Fear":  "😱",
        "Fear":          "😨",
        "Neutral":       "😐",
        "Greed":         "😏",
        "Extreme Greed": "🤑",
    }.get(label, "❓")


def _age_str(age_minutes: int | None) -> str:
    if age_minutes is None:
        return "unknown"
    if age_minutes < 60:
        return f"{age_minutes} min ago"
    h = age_minutes // 60
    m = age_minutes % 60
    return f"{h}h {m}m ago" if m else f"{h}h ago"


# ── API fetchers ──────────────────────────────────────────────────────────────

def _get_current() -> dict | None:
    try:
        r = requests.get(f"{_BASE}/current", headers=_HDRS, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def _get_history() -> list[dict]:
    try:
        r = requests.get(f"{_BASE}/history", headers=_HDRS, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return []


# ── Cached public entry-points ────────────────────────────────────────────────

if _HAS_ST:
    @st.cache_data(ttl=900, show_spinner=False)   # 15-min cache matches update cycle
    def fetch_current() -> dict | None:
        return _get_current()

    @st.cache_data(ttl=900, show_spinner=False)
    def fetch_history() -> list[dict]:
        return _get_history()
else:
    def fetch_current() -> dict | None:
        return _get_current()

    def fetch_history() -> list[dict]:
        return _get_history()


# ── Data helpers ──────────────────────────────────────────────────────────────

def history_dataframe(records: list[dict]) -> pd.DataFrame:
    """Convert raw history list to a clean DataFrame."""
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    df["timestamp"]       = pd.to_datetime(df["timestamp"], utc=True)
    df["composite_score"] = pd.to_numeric(df["composite_score"], errors="coerce")
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def components_dataframe(current: dict) -> pd.DataFrame:
    """Build a DataFrame of indicator components from the /current response."""
    comps = current.get("components", [])
    rows = []
    for c in comps:
        name   = c.get("component", "")
        score  = float(c.get("score", 0))
        weight = float(c.get("weight", 0))
        rows.append({
            "Component": _COMPONENT_LABELS.get(name, name.title()),
            "Score":     score,
            "Weight":    weight,
            "Label":     score_to_label(score),
        })
    return pd.DataFrame(rows)
