"""
PSX Stock Market Prediction App
Pakistan Stock Exchange — Historical Analysis, AI Sentiment & Price Forecasting
Integrates:
  - psxdata        → real PSX OHLCV history
  - Alpha Vantage  → global news sentiment (topic-mapped to PSX sectors)
  - OpenAI GPT-4o  → PSX-contextualised qualitative analysis & forecast adjustment
"""

import warnings
warnings.filterwarnings("ignore")

import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, date

import psxdata
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

import sentiment as sent_mod   # Alpha Vantage + GPT-4o
import dawn_scraper as dawn_mod  # Dawn.com scraper + BERT

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PSX Stock Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stMetricValue"]  { font-size: 1.35rem !important; }
[data-testid="stMetricDelta"]  { font-size: 0.85rem !important; }
.disclaimer-box {
    background: #1e3a2f;
    border-left: 4px solid #4caf50;
    padding: 10px 14px;
    border-radius: 4px;
    margin: 8px 0 16px 0;
    font-size: 0.85rem;
    color: #cfd8dc;
}
.warning-box {
    background: #332700;
    border-left: 4px solid #ffc107;
    padding: 10px 14px;
    border-radius: 4px;
    margin: 8px 0 16px 0;
    font-size: 0.85rem;
    color: #ffe082;
}
</style>
""", unsafe_allow_html=True)

# ── Colour palette ─────────────────────────────────────────────────────────────
C_BULL   = "#26a69a"
C_BEAR   = "#ef5350"
C_BLUE   = "#42a5f5"
C_ORANGE = "#ffa726"
C_PURPLE = "#ab47bc"
C_PINK   = "#f48fb1"
C_GOLD   = "#ffd54f"

# ─────────────────────────────────────────────────────────────────────────────
# Data loading helpers (cached)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def get_tickers() -> list[str]:
    try:
        result = psxdata.tickers()
        if isinstance(result, (list, np.ndarray)):
            return [str(t).strip() for t in result if str(t).strip()]
        if isinstance(result, pd.DataFrame):
            col = result.columns[0]
            return result[col].astype(str).str.strip().tolist()
        return []
    except Exception:
        return ["ENGRO", "LUCK", "HBL", "MCB", "PPL", "PSO", "OGDC", "NESTLE", "UBL", "NBP",
                "HUBC", "EFERT", "EPCL", "UNITY", "BAFL", "MEBL", "BAHL", "SYS", "TRG", "AVN"]


@st.cache_data(ttl=900, show_spinner=False)
def get_stock_data(symbol: str, start: str, end: str) -> pd.DataFrame:
    try:
        df = psxdata.stocks(symbol, start=start, end=end)
        if df is None or (hasattr(df, "empty") and df.empty):
            return pd.DataFrame()

        # ── Normalise column names ────────────────────────────────────────────
        df = df.copy()
        df.columns = [str(c).strip().lower() for c in df.columns]

        # If date is the index, bring it in as a column
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={"index": "date"})
        elif df.index.name and "date" in df.index.name.lower():
            df = df.reset_index()
            df.columns = [str(c).strip().lower() for c in df.columns]

        # Fuzzy-map column names to canonical names
        rename = {}
        for col in df.columns:
            lc = col.lower()
            if "date" in lc:                      rename[col] = "Date"
            elif lc in ("open", "o"):             rename[col] = "Open"
            elif lc in ("high", "h"):             rename[col] = "High"
            elif lc in ("low", "l"):              rename[col] = "Low"
            elif lc in ("close", "c", "last"):    rename[col] = "Close"
            elif lc in ("volume", "vol", "v"):    rename[col] = "Volume"
            elif lc in ("change", "chng", "chg"): rename[col] = "Change"
            else:                                 rename[col] = col.title()
        df = df.rename(columns=rename)

        # Parse & sort by date
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

        # Coerce numerics
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["Close"])
        return df

    except Exception as e:
        st.error(f"Error fetching **{symbol}**: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=600, show_spinner=False)
def get_quote(symbol: str):
    try:
        return psxdata.quote(symbol)
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def get_fundamentals(symbol: str):
    try:
        return psxdata.fundamentals(symbol)
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def get_kse100_members() -> list[str]:
    try:
        idx = psxdata.indices("KSE100")
        if isinstance(idx, (list, np.ndarray)):
            return [str(t).strip() for t in idx]
        if isinstance(idx, pd.DataFrame):
            return idx.iloc[:, 0].astype(str).str.strip().tolist()
        return []
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Technical Indicators
# ─────────────────────────────────────────────────────────────────────────────

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c = df["Close"]
    h = df["High"]
    lo = df["Low"]

    # Moving averages
    for w in [20, 50, 200]:
        df[f"SMA_{w}"] = c.rolling(w).mean()
    df["EMA_12"] = c.ewm(span=12, adjust=False).mean()
    df["EMA_26"] = c.ewm(span=26, adjust=False).mean()

    # Bollinger Bands
    df["BB_Mid"]   = c.rolling(20).mean()
    bb_std         = c.rolling(20).std()
    df["BB_Upper"] = df["BB_Mid"] + 2 * bb_std
    df["BB_Lower"] = df["BB_Mid"] - 2 * bb_std
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / df["BB_Mid"]
    df["BB_Pct"]   = (c - df["BB_Lower"]) / (df["BB_Upper"] - df["BB_Lower"])

    # RSI (14)
    delta  = c.diff()
    gain   = delta.clip(lower=0).rolling(14).mean()
    loss   = (-delta.clip(upper=0)).rolling(14).mean()
    df["RSI"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

    # MACD
    df["MACD"]        = df["EMA_12"] - df["EMA_26"]
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"]   = df["MACD"] - df["MACD_Signal"]

    # ATR (14)
    tr = pd.concat([h - lo,
                    (h - c.shift()).abs(),
                    (lo - c.shift()).abs()], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()

    # Stochastic (14,3)
    lowest14  = lo.rolling(14).min()
    highest14 = h.rolling(14).max()
    df["Stoch_K"] = 100 * (c - lowest14) / (highest14 - lowest14).replace(0, np.nan)
    df["Stoch_D"] = df["Stoch_K"].rolling(3).mean()

    # OBV
    if "Volume" in df.columns:
        df["OBV"] = (np.sign(c.diff()) * df["Volume"]).fillna(0).cumsum()

    # Daily return
    df["Return"] = c.pct_change() * 100

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Feature Engineering for ML models
# ─────────────────────────────────────────────────────────────────────────────

def build_features(
    df: pd.DataFrame,
    sentiment_series: pd.Series | None = None,
    dawn_sentiment_series: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    df = add_indicators(df).copy()
    c  = df["Close"]

    feat = pd.DataFrame(index=df.index)

    # Price lags
    for lag in [1, 2, 3, 5, 10, 20]:
        feat[f"lag_{lag}"] = c.shift(lag)

    # Rolling stats
    for w in [5, 10, 20, 50]:
        feat[f"sma_{w}"]  = c.rolling(w).mean()
        feat[f"std_{w}"]  = c.rolling(w).std()
        feat[f"min_{w}"]  = c.rolling(w).min()
        feat[f"max_{w}"]  = c.rolling(w).max()

    # Returns
    feat["ret_1d"]  = c.pct_change(1)
    feat["ret_5d"]  = c.pct_change(5)
    feat["ret_20d"] = c.pct_change(20)

    # Technical indicator features
    for col in ["RSI", "MACD", "MACD_Signal", "MACD_Hist",
                "BB_Width", "BB_Pct", "ATR", "Stoch_K", "Stoch_D"]:
        if col in df.columns:
            feat[col.lower()] = df[col]

    # Volume ratio
    if "Volume" in df.columns:
        feat["vol_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()

    # ── Sentiment features ────────────────────────────────────────────────────
    # Combine Alpha Vantage (global sector) + Dawn/BERT (Pakistan-specific)
    combined = dawn_mod.combine_sentiment_series(sentiment_series, dawn_sentiment_series)
    active_series = combined if combined is not None else sentiment_series

    if active_series is not None and len(active_series) > 0:
        if "Date" in df.columns:
            aligned = active_series.reindex(df["Date"].values).ffill().bfill().fillna(0.0)
            aligned.index = df.index
        else:
            aligned = active_series.reindex(df.index).ffill().bfill().fillna(0.0)
        feat["sentiment"]      = aligned.values
        feat["sentiment_ma5"]  = aligned.rolling(5, min_periods=1).mean().values
        feat["sentiment_ma20"] = aligned.rolling(20, min_periods=1).mean().values

    # Dawn-specific feature (standalone, even when AV not available)
    if dawn_sentiment_series is not None and len(dawn_sentiment_series) > 0:
        if "Date" in df.columns:
            d_aligned = dawn_sentiment_series.reindex(df["Date"].values).ffill().bfill().fillna(0.0)
            d_aligned.index = df.index
        else:
            d_aligned = dawn_sentiment_series.reindex(df.index).ffill().bfill().fillna(0.0)
        feat["dawn_sent"]     = d_aligned.values
        feat["dawn_sent_ma5"] = d_aligned.rolling(5, min_periods=1).mean().values

    feat["target"] = c
    feat = feat.dropna()

    X = feat.drop("target", axis=1)
    y = feat["target"]
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# ML Prediction
# ─────────────────────────────────────────────────────────────────────────────

def run_ml_forecast(
    df: pd.DataFrame,
    model_name: str,
    horizon: int,
    sentiment_series: pd.Series | None = None,
    dawn_sentiment_series: pd.Series | None = None,
) -> dict:
    X, y = build_features(df, sentiment_series=sentiment_series,
                          dawn_sentiment_series=dawn_sentiment_series)

    if len(X) < 80:
        return {"error": "Need at least 80 trading days of data."}

    split = int(len(X) * 0.80)
    X_tr, X_te = X.iloc[:split], X.iloc[split:]
    y_tr, y_te = y.iloc[:split], y.iloc[split:]

    scaler = MinMaxScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    registry = {
        "Random Forest":      RandomForestRegressor(n_estimators=300, max_depth=12,
                                                    min_samples_leaf=2, random_state=42, n_jobs=-1),
        "Gradient Boosting":  GradientBoostingRegressor(n_estimators=300, max_depth=6,
                                                        learning_rate=0.04, subsample=0.8, random_state=42),
        "Linear Regression":  LinearRegression(),
    }
    model = registry.get(model_name, registry["Random Forest"])
    model.fit(X_tr_s, y_tr)
    y_pred_te = model.predict(X_te_s)

    # ── Metrics ───────────────────────────────────────────────────────────────
    mae  = mean_absolute_error(y_te, y_pred_te)
    rmse = np.sqrt(mean_squared_error(y_te, y_pred_te))
    r2   = r2_score(y_te, y_pred_te)
    mape = float(np.mean(np.abs((y_te.values - y_pred_te) / y_te.values)) * 100)

    # ── Future forecast via recursive single-step prediction ──────────────────
    last_row   = X.iloc[-1].values.copy()
    close_vals = df["Close"].values
    lag_keys   = [f"lag_{l}" for l in [1, 2, 3, 5, 10, 20]]
    lag_pos    = {k: X.columns.get_loc(k) for k in lag_keys if k in X.columns}

    forecast = []
    for _ in range(horizon):
        row_s  = scaler.transform(last_row.reshape(1, -1))
        nxt    = float(model.predict(row_s)[0])
        forecast.append(nxt)
        # Shift lag features forward
        new_row = last_row.copy()
        for lag_k, pos in sorted(lag_pos.items(), key=lambda x: -int(x[0].split("_")[1])):
            lag_n = int(lag_k.split("_")[1])
            prev_k = f"lag_{lag_n - 1}" if lag_n > 1 else None
            if prev_k and prev_k in lag_pos:
                new_row[pos] = last_row[lag_pos[prev_k]]
            else:
                new_row[pos] = nxt
        last_row = new_row

    last_date    = df["Date"].iloc[-1] if "Date" in df.columns else pd.Timestamp.today()
    future_dates = pd.bdate_range(start=last_date + timedelta(days=1), periods=horizon)

    # Dates aligned with test set
    if "Date" in df.columns:
        all_dates  = df["Date"].values
        n_rows     = len(df)
        n_feats    = len(X)
        offset     = n_rows - n_feats          # rows dropped by dropna
        test_dates = df["Date"].iloc[offset + split : offset + split + len(y_te)]
    else:
        test_dates = pd.RangeIndex(split, split + len(y_te))

    return {
        "model":          model_name,
        "y_te":           y_te,
        "y_pred_te":      pd.Series(y_pred_te, index=y_te.index),
        "test_dates":     test_dates,
        "future_dates":   future_dates,
        "forecast":       forecast,
        "metrics":        {"MAE": mae, "RMSE": rmse, "R²": r2, "MAPE %": mape},
        "feat_imp":       getattr(model, "feature_importances_", None),
        "feat_names":     X.columns.tolist(),
        "split_idx":      split,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Prophet Prediction
# ─────────────────────────────────────────────────────────────────────────────

def run_prophet_forecast(df: pd.DataFrame, horizon: int) -> dict:
    if not PROPHET_AVAILABLE:
        return {"error": "Prophet is not installed. Run: pip install prophet"}
    if "Date" not in df.columns:
        return {"error": "Date column not found in data."}

    pdata = df[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"}).dropna()
    if len(pdata) < 60:
        return {"error": "Need at least 60 trading days for Prophet."}

    split  = int(len(pdata) * 0.80)
    train  = pdata.iloc[:split]
    test   = pdata.iloc[split:]

    # ── Validation model ──────────────────────────────────────────────────────
    m_val = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0,
    )
    m_val.fit(train)
    val_future = m_val.make_future_dataframe(periods=len(test), freq="B")
    val_fc     = m_val.predict(val_future)
    yhat       = val_fc.set_index("ds")["yhat"].reindex(test["ds"].values)
    mask       = ~yhat.isna()
    yt         = test["y"].values[mask]
    yp         = yhat.values[mask]

    mae  = mean_absolute_error(yt, yp)
    rmse = np.sqrt(mean_squared_error(yt, yp))
    r2   = r2_score(yt, yp)
    mape = float(np.mean(np.abs((yt - yp) / yt)) * 100)

    # ── Full model on all data ─────────────────────────────────────────────────
    m_full = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0,
    )
    m_full.fit(pdata)
    future = m_full.make_future_dataframe(periods=horizon, freq="B")
    fc     = m_full.predict(future)

    return {
        "model":   "Prophet",
        "fc":      fc,
        "history": pdata,
        "metrics": {"MAE": mae, "RMSE": rmse, "R²": r2, "MAPE %": mape},
    }


# ─────────────────────────────────────────────────────────────────────────────
# Chart helpers
# ─────────────────────────────────────────────────────────────────────────────

TEMPLATE = "plotly_dark"
MARGIN   = dict(l=0, r=0, t=30, b=0)


def candlestick_chart(df: pd.DataFrame, overlays: list[str]) -> go.Figure:
    has_vol = "Volume" in df.columns and df["Volume"].notna().any()
    rows    = [0.72, 0.28] if has_vol else [1.0]
    n_rows  = 2 if has_vol else 1

    fig = make_subplots(
        rows=n_rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=rows,
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df["Date"], open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        name="OHLC",
        increasing_line_color=C_BULL,
        decreasing_line_color=C_BEAR,
        increasing_fillcolor=C_BULL,
        decreasing_fillcolor=C_BEAR,
    ), row=1, col=1)

    col_map = {
        "SMA_20":   (C_BLUE,   1.4),
        "SMA_50":   (C_ORANGE, 1.4),
        "SMA_200":  (C_PURPLE, 1.4),
        "BB_Upper": ("#90caf9", 1.0),
        "BB_Mid":   ("#bbdefb", 1.0),
        "BB_Lower": ("#90caf9", 1.0),
    }
    for ind in overlays:
        if ind not in df.columns:
            continue
        color, width = col_map.get(ind, ("#aaa", 1.0))
        fig.add_trace(go.Scatter(
            x=df["Date"], y=df[ind], name=ind,
            line=dict(color=color, width=width), opacity=0.85,
        ), row=1, col=1)

    # Bollinger band fill
    if "BB_Upper" in overlays and "BB_Lower" in df.columns:
        fig.add_trace(go.Scatter(
            x=pd.concat([df["Date"], df["Date"].iloc[::-1]]),
            y=pd.concat([df["BB_Upper"], df["BB_Lower"].iloc[::-1]]),
            fill="toself",
            fillcolor="rgba(33,150,243,0.06)",
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False,
            name="BB Fill",
        ), row=1, col=1)

    # Volume bars
    if has_vol:
        v_colors = [C_BULL if r >= 0 else C_BEAR
                    for r in df["Close"].pct_change().fillna(0)]
        fig.add_trace(go.Bar(
            x=df["Date"], y=df["Volume"],
            marker_color=v_colors, name="Volume", opacity=0.65,
        ), row=2, col=1)

    fig.update_layout(
        height=600, template=TEMPLATE,
        xaxis_rangeslider_visible=False,
        margin=MARGIN,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
    )
    return fig


def line_chart(df: pd.DataFrame, overlays: list[str]) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["Close"], name="Close",
        fill="tozeroy", fillcolor="rgba(66,165,245,0.08)",
        line=dict(color=C_BLUE, width=2),
    ))
    col_map = {
        "SMA_20": C_ORANGE, "SMA_50": C_GOLD,
        "SMA_200": C_PURPLE, "BB_Upper": "#90caf9",
        "BB_Mid": "#bbdefb", "BB_Lower": "#90caf9",
    }
    for ind in overlays:
        if ind in df.columns:
            fig.add_trace(go.Scatter(
                x=df["Date"], y=df[ind], name=ind,
                line=dict(color=col_map.get(ind, "#aaa"), width=1.3),
            ))
    fig.update_layout(height=520, template=TEMPLATE, margin=MARGIN,
                      legend=dict(orientation="h", yanchor="bottom", y=1.01))
    return fig


def rsi_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["RSI"], name="RSI",
        line=dict(color=C_PURPLE, width=1.6),
    ))
    fig.add_hrect(y0=70, y1=100, fillcolor="rgba(239,83,80,0.08)",  line_width=0)
    fig.add_hrect(y0=0,  y1=30,  fillcolor="rgba(38,166,154,0.08)", line_width=0)
    fig.add_hline(y=70, line_dash="dash", line_color=C_BEAR,
                  annotation_text="Overbought 70", annotation_position="top right")
    fig.add_hline(y=30, line_dash="dash", line_color=C_BULL,
                  annotation_text="Oversold 30", annotation_position="bottom right")
    fig.update_layout(height=230, template=TEMPLATE, margin=MARGIN,
                      yaxis=dict(range=[0, 100]))
    return fig


def macd_chart(df: pd.DataFrame) -> go.Figure:
    hist_colors = [C_BULL if v >= 0 else C_BEAR for v in df["MACD_Hist"].fillna(0)]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df["Date"], y=df["MACD_Hist"],
                         marker_color=hist_colors, name="Histogram", opacity=0.7))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["MACD"],
                             name="MACD", line=dict(color=C_BLUE, width=1.4)))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["MACD_Signal"],
                             name="Signal", line=dict(color=C_ORANGE, width=1.4)))
    fig.update_layout(height=230, template=TEMPLATE, margin=MARGIN)
    return fig


def stoch_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Stoch_K"],
                             name="%K", line=dict(color=C_BLUE, width=1.4)))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Stoch_D"],
                             name="%D", line=dict(color=C_ORANGE, width=1.4)))
    fig.add_hline(y=80, line_dash="dash", line_color=C_BEAR)
    fig.add_hline(y=20, line_dash="dash", line_color=C_BULL)
    fig.update_layout(height=230, template=TEMPLATE, margin=MARGIN,
                      yaxis=dict(range=[0, 100]))
    return fig


def ml_forecast_chart(df: pd.DataFrame, result: dict) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["Close"], name="Historical",
        line=dict(color=C_BLUE, width=1.5),
    ))

    td = pd.to_datetime(result["test_dates"]) if not isinstance(result["test_dates"], pd.DatetimeIndex) \
         else result["test_dates"]
    fig.add_trace(go.Scatter(
        x=td, y=result["y_pred_te"].values,
        name="Validation Fit",
        line=dict(color=C_GOLD, width=1.6, dash="dot"),
    ))

    fd    = result["future_dates"]
    fc    = result["forecast"]
    upper = [p * 1.025 for p in fc]
    lower = [p * 0.975 for p in fc]

    fig.add_trace(go.Scatter(
        x=list(fd) + list(fd)[::-1],
        y=upper + lower[::-1],
        fill="toself", fillcolor="rgba(244,143,177,0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        showlegend=False, name="±2.5% Band",
    ))
    fig.add_trace(go.Scatter(
        x=fd, y=fc, name=f"{result['model']} Forecast",
        line=dict(color=C_PINK, width=2.5),
        mode="lines+markers",
        marker=dict(size=4),
    ))

    fig.update_layout(height=460, template=TEMPLATE, margin=MARGIN,
                      legend=dict(orientation="h", yanchor="bottom", y=1.01))
    return fig


def prophet_forecast_chart(result: dict) -> go.Figure:
    fc   = result["fc"]
    hist = result["history"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist["ds"], y=hist["y"], name="Historical",
        line=dict(color=C_BLUE, width=1.5),
    ))
    fig.add_trace(go.Scatter(
        x=list(fc["ds"]) + list(fc["ds"])[::-1],
        y=list(fc["yhat_upper"]) + list(fc["yhat_lower"])[::-1],
        fill="toself", fillcolor="rgba(244,143,177,0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        showlegend=False, name="Uncertainty",
    ))
    fig.add_trace(go.Scatter(
        x=fc["ds"], y=fc["yhat"], name="Prophet Forecast",
        line=dict(color=C_PINK, width=2.2),
    ))
    fig.update_layout(height=460, template=TEMPLATE, margin=MARGIN,
                      legend=dict(orientation="h", yanchor="bottom", y=1.01))
    return fig


def returns_dist_chart(df: pd.DataFrame) -> go.Figure:
    returns = df["Return"].dropna()
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=returns, nbinsx=60, name="Daily Returns",
        marker_color=C_BLUE, opacity=0.75,
    ))
    fig.add_vline(x=0, line_dash="dash", line_color="#fff", opacity=0.4)
    fig.add_vline(x=returns.mean(), line_dash="dot",
                  line_color=C_GOLD, annotation_text="Mean", opacity=0.8)
    fig.update_layout(height=280, template=TEMPLATE, margin=MARGIN,
                      xaxis_title="Daily Return %", yaxis_title="Frequency")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Main App
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    st.title("📈 PSX Stock Market Predictor")
    st.caption("Pakistan Stock Exchange · Historical Analysis & AI-Powered Forecasting")

    # ── Sidebar ────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Configuration")

        with st.spinner("Loading tickers…"):
            all_tickers = get_tickers()

        symbol = st.selectbox(
            "Stock Symbol",
            options=all_tickers,
            index=0,
            help="Select a PSX-listed stock",
        )

        kse100 = get_kse100_members()
        if kse100 and symbol in kse100:
            st.caption("✅ KSE-100 constituent")

        st.divider()

        col_a, col_b = st.columns(2)
        with col_a:
            start_dt = st.date_input(
                "From",
                value=date.today() - timedelta(days=365 * 3),
                max_value=date.today() - timedelta(days=1),
            )
        with col_b:
            end_dt = st.date_input(
                "To",
                value=date.today(),
                max_value=date.today(),
            )

        chart_type = st.radio("Chart Style", ["Candlestick", "Line"], horizontal=True)

        st.divider()
        st.subheader("📐 Overlays")
        show_sma20  = st.checkbox("SMA 20",           value=True)
        show_sma50  = st.checkbox("SMA 50",           value=True)
        show_sma200 = st.checkbox("SMA 200",          value=False)
        show_bb     = st.checkbox("Bollinger Bands",  value=False)

        st.divider()
        st.subheader("🔮 Prediction")

        model_options = (
            ["Prophet", "Random Forest", "Gradient Boosting", "Linear Regression"]
            if PROPHET_AVAILABLE
            else ["Random Forest", "Gradient Boosting", "Linear Regression"]
        )
        if not PROPHET_AVAILABLE:
            st.caption("ℹ️ Install `prophet` for time-series forecasting.")

        pred_model   = st.selectbox("Model", model_options)
        horizon      = st.slider("Forecast Days", 7, 90, 30, 7)
        use_sentiment_feature = st.checkbox(
            "Alpha Vantage sentiment (ML feature)",
            value=True,
            help="Adds Alpha Vantage sector sentiment as a feature to the ML models",
        )
        use_dawn_sentiment = st.checkbox(
            "Dawn.com BERT sentiment (ML feature)",
            value=True,
            help="Scrapes Dawn.com Pakistan business news and runs BERT locally. No API key needed.",
        )
        run_pred_btn = st.button("▶ Run Forecast", type="primary", use_container_width=True)

        st.divider()
        st.subheader("🔑 API Keys")
        st.caption("Keys are stored only for this session and never saved to disk.")
        av_key = st.text_input(
            "Alpha Vantage Key",
            type="password",
            placeholder="Get free key at alphavantage.co",
            help="Used for news sentiment. Free tier: 25 calls/day.",
        )
        openai_key = st.text_input(
            "OpenAI Key",
            type="password",
            placeholder="sk-...",
            help="Used for GPT-4o analysis in the AI Insights tab.",
        )
        run_ai_btn = st.button("🧠 Run AI Analysis", use_container_width=True,
                               disabled=not (av_key and openai_key))
        if not av_key or not openai_key:
            st.caption("Enter both keys to enable the AI Insights tab.")

    # ── Load & process data ────────────────────────────────────────────────────
    with st.spinner(f"Fetching {symbol} data from PSX…"):
        df_raw = get_stock_data(symbol, str(start_dt), str(end_dt))

    if df_raw.empty:
        st.error(f"No data returned for **{symbol}** in the selected period. "
                 "Try a different ticker or wider date range.")
        st.stop()

    df = add_indicators(df_raw)

    latest = df.iloc[-1]
    prev   = df.iloc[-2] if len(df) > 1 else latest

    ch_abs = latest["Close"] - prev["Close"]
    ch_pct = (ch_abs / prev["Close"] * 100) if prev["Close"] else 0.0

    # ── Top metrics ────────────────────────────────────────────────────────────
    st.subheader(f"{symbol}  ·  {latest.get('Date', '').strftime('%d %b %Y') if hasattr(latest.get('Date', ''), 'strftime') else ''}")

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Close",      f"PKR {latest['Close']:,.2f}",  f"{ch_abs:+.2f} ({ch_pct:+.2f}%)")
    c2.metric("Open",       f"PKR {latest['Open']:,.2f}")
    c3.metric("High",       f"PKR {latest['High']:,.2f}")
    c4.metric("Low",        f"PKR {latest['Low']:,.2f}")
    c5.metric("52W High",   f"PKR {df['High'].max():,.2f}")
    c6.metric("52W Low",    f"PKR {df['Low'].min():,.2f}")

    # ── Tabs ───────────────────────────────────────────────────────────────────
    tab_price, tab_tech, tab_ai, tab_pred, tab_data, tab_fund = st.tabs([
        "📊 Price Chart",
        "📉 Technical Analysis",
        "🧠 AI Insights",
        "🔮 Prediction",
        "📋 Raw Data",
        "📑 Fundamentals",
    ])

    # ── Tab: Price Chart ───────────────────────────────────────────────────────
    with tab_price:
        overlays = []
        if show_sma20:  overlays.append("SMA_20")
        if show_sma50:  overlays.append("SMA_50")
        if show_sma200: overlays.append("SMA_200")
        if show_bb:     overlays += ["BB_Upper", "BB_Mid", "BB_Lower"]

        if chart_type == "Candlestick":
            st.plotly_chart(candlestick_chart(df, overlays), use_container_width=True)
        else:
            st.plotly_chart(line_chart(df, overlays), use_container_width=True)

        st.divider()
        col_s, col_r = st.columns(2)

        with col_s:
            st.subheader("Price Statistics")
            stats = df["Close"].describe().rename("Close (PKR)")
            st.dataframe(stats.to_frame().style.format("{:,.2f}"), use_container_width=True)

        with col_r:
            st.subheader("Return Profile")
            rets = df["Return"].dropna()
            ann_vol   = rets.std() * np.sqrt(252)
            sharpe    = (rets.mean() / rets.std()) * np.sqrt(252) if rets.std() else 0
            pos_days  = (rets > 0).sum()
            neg_days  = (rets < 0).sum()
            win_rate  = pos_days / len(rets) * 100 if len(rets) else 0

            st.dataframe(pd.DataFrame({
                "Metric": ["Mean Daily Return", "Std Dev (Daily)",
                           "Annualised Volatility", "Sharpe Ratio (ann.)",
                           "Max Single-Day Gain", "Max Single-Day Loss",
                           "Win Rate (days up)"],
                "Value": [
                    f"{rets.mean():.3f}%", f"{rets.std():.3f}%",
                    f"{ann_vol:.2f}%",     f"{sharpe:.3f}",
                    f"{rets.max():+.3f}%", f"{rets.min():+.3f}%",
                    f"{win_rate:.1f}%",
                ],
            }), use_container_width=True, hide_index=True)

        st.subheader("Daily Return Distribution")
        st.plotly_chart(returns_dist_chart(df), use_container_width=True)

    # ── Tab: Technical Analysis ────────────────────────────────────────────────
    with tab_tech:
        # Signal summary banner
        close_now = df["Close"].iloc[-1]
        signals   = {}

        for label, col in [("SMA 20", "SMA_20"), ("SMA 50", "SMA_50"), ("SMA 200", "SMA_200")]:
            if col in df.columns and not np.isnan(df[col].iloc[-1]):
                signals[f"Price vs {label}"] = (
                    "🟢 Above" if close_now > df[col].iloc[-1] else "🔴 Below"
                )

        if "RSI" in df.columns:
            r = df["RSI"].iloc[-1]
            signals["RSI (14)"] = (
                f"🔴 Overbought ({r:.1f})" if r > 70
                else f"🟢 Oversold ({r:.1f})" if r < 30
                else f"⚪ Neutral ({r:.1f})"
            )

        if "MACD" in df.columns:
            macd_now = df["MACD"].iloc[-1]
            sig_now  = df["MACD_Signal"].iloc[-1]
            signals["MACD"] = "🟢 Bullish" if macd_now > sig_now else "🔴 Bearish"

        if "Stoch_K" in df.columns:
            sk = df["Stoch_K"].iloc[-1]
            signals["Stochastic"] = (
                f"🔴 Overbought ({sk:.1f})" if sk > 80
                else f"🟢 Oversold ({sk:.1f})" if sk < 20
                else f"⚪ Neutral ({sk:.1f})"
            )

        bulls  = sum(1 for v in signals.values() if "🟢" in v)
        bears  = sum(1 for v in signals.values() if "🔴" in v)
        total  = len(signals)
        verdict = ("🟢 Bullish Bias" if bulls > bears else
                   "🔴 Bearish Bias" if bears > bulls else "⚪ Mixed / Neutral")

        st.markdown(
            f'<div class="disclaimer-box">'
            f'<strong>Signal Summary:</strong> {verdict} &nbsp;·&nbsp; '
            f'{bulls}/{total} bullish &nbsp;·&nbsp; {bears}/{total} bearish'
            f'</div>',
            unsafe_allow_html=True,
        )

        sig_df = pd.DataFrame(list(signals.items()), columns=["Indicator", "Signal"])
        st.dataframe(sig_df, use_container_width=True, hide_index=True)

        st.divider()

        # Individual indicator charts
        st.subheader("RSI — Relative Strength Index (14)")
        if "RSI" in df.columns:
            st.plotly_chart(rsi_chart(df), use_container_width=True)

        st.subheader("MACD (12, 26, 9)")
        if "MACD" in df.columns:
            st.plotly_chart(macd_chart(df), use_container_width=True)

        st.subheader("Stochastic Oscillator (14, 3)")
        if "Stoch_K" in df.columns:
            st.plotly_chart(stoch_chart(df), use_container_width=True)

        st.subheader("Bollinger Bands (20, ±2σ)")
        if "BB_Upper" in df.columns:
            fig_bb = go.Figure()
            fig_bb.add_trace(go.Scatter(
                x=df["Date"], y=df["BB_Upper"], name="Upper",
                line=dict(color=C_BEAR, width=1, dash="dot"),
            ))
            fig_bb.add_trace(go.Scatter(
                x=df["Date"], y=df["BB_Mid"], name="Mid (SMA 20)",
                line=dict(color="#888", width=1),
            ))
            fig_bb.add_trace(go.Scatter(
                x=df["Date"], y=df["BB_Lower"], name="Lower",
                line=dict(color=C_BULL, width=1, dash="dot"),
                fill="tonexty",
                fillcolor="rgba(33,150,243,0.05)",
            ))
            fig_bb.add_trace(go.Scatter(
                x=df["Date"], y=df["Close"], name="Close",
                line=dict(color=C_BLUE, width=1.6),
            ))
            fig_bb.update_layout(height=280, template=TEMPLATE, margin=MARGIN)
            st.plotly_chart(fig_bb, use_container_width=True)

    # ── Tab: AI Insights ──────────────────────────────────────────────────────
    with tab_ai:
        st.markdown(
            '<div class="warning-box">'
            "⚠️ <strong>Disclaimer:</strong> AI-generated analysis is for educational "
            "purposes only. It is not financial advice. PSX stocks may not be directly "
            "covered by Alpha Vantage; global sector sentiment is used as a proxy signal."
            "</div>",
            unsafe_allow_html=True,
        )

        if not av_key or not openai_key:
            st.info(
                "Enter your **Alpha Vantage** and **OpenAI** API keys in the sidebar, "
                "then click **🧠 Run AI Analysis**."
            )
            st.markdown("""
| Key | Where to get it | Cost |
|-----|----------------|------|
| Alpha Vantage | [alphavantage.co](https://www.alphavantage.co/support/#api-key) | Free (25 calls/day) |
| OpenAI | [platform.openai.com](https://platform.openai.com/api-keys) | Pay-per-use (GPT-4o ~$0.01/call) |
""")
        elif not run_ai_btn:
            st.info("Click **🧠 Run AI Analysis** in the sidebar to run the full analysis.")
        else:
            # ── 1. Fetch Alpha Vantage sentiment ──────────────────────────────
            with st.spinner("Fetching Alpha Vantage news sentiment…"):
                topics_list = sent_mod.topics_for(symbol)
                topics_str  = ",".join(topics_list)
                time_from   = (datetime.now() - timedelta(days=30)).strftime("%Y%m%dT0000")
                av_articles = sent_mod.fetch_av_sentiment(
                    av_key, topics_str, time_from=time_from, limit=50
                )
                sentiment_agg = sent_mod.aggregate_sentiment(av_articles, days_back=14)

            # ── 2. Build context dicts ─────────────────────────────────────────
            price_data_dict   = sent_mod.build_price_data(df)
            tech_signals_dict = sent_mod.build_technical_signals(df)

            # ── 3. Display sentiment metrics ──────────────────────────────────
            st.subheader("Alpha Vantage — Market Sentiment Signal")
            st.caption(
                f"Topics: `{topics_str}` · Sector: **{sent_mod.sector_for(symbol).title()}** "
                f"· Articles (last 14 days): {sentiment_agg['article_count']}"
            )

            score_color = (
                C_BULL if sentiment_agg["score"] > 0.15
                else C_BEAR if sentiment_agg["score"] < -0.15
                else "#aaa"
            )
            sa1, sa2, sa3, sa4 = st.columns(4)
            sa1.metric(
                "Sentiment Score",
                f"{sentiment_agg['score']:+.4f}",
                sentiment_agg["label"],
            )
            sa2.metric("Bullish Articles", f"{sentiment_agg['bullish_pct']:.1f}%")
            sa3.metric("Bearish Articles", f"{sentiment_agg['bearish_pct']:.1f}%")
            sa4.metric("Articles Analysed", sentiment_agg["article_count"])

            # Sentiment gauge
            gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=sentiment_agg["score"],
                delta={"reference": 0, "valueformat": ".4f"},
                title={"text": "Weighted Sentiment Score", "font": {"size": 14}},
                gauge={
                    "axis":  {"range": [-1, 1], "tickwidth": 1, "tickcolor": "#ccc"},
                    "bar":   {"color": score_color},
                    "bgcolor": "rgba(0,0,0,0)",
                    "borderwidth": 1,
                    "bordercolor": "#555",
                    "steps": [
                        {"range": [-1.0, -0.35], "color": "rgba(239,83,80,0.25)"},
                        {"range": [-0.35,-0.15], "color": "rgba(239,83,80,0.10)"},
                        {"range": [-0.15, 0.15], "color": "rgba(180,180,180,0.08)"},
                        {"range": [ 0.15, 0.35], "color": "rgba(38,166,154,0.10)"},
                        {"range": [ 0.35, 1.0],  "color": "rgba(38,166,154,0.25)"},
                    ],
                    "threshold": {
                        "line": {"color": "#fff", "width": 2},
                        "thickness": 0.75,
                        "value": sentiment_agg["score"],
                    },
                },
                number={"suffix": "", "valueformat": ".4f", "font": {"size": 22}},
            ))
            gauge.update_layout(
                height=260, template=TEMPLATE,
                margin=dict(l=20, r=20, t=40, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(gauge, use_container_width=True)

            # Sentiment label distribution bar
            if sentiment_agg.get("label_counts"):
                lc = sentiment_agg["label_counts"]
                label_order = ["Bullish", "Somewhat-Bullish", "Neutral",
                               "Somewhat-Bearish", "Bearish"]
                label_colors = [C_BULL, "#80cbc4", "#90a4ae", "#ef9a9a", C_BEAR]
                bar_x = [lc.get(l, 0) for l in label_order]

                fig_lc = go.Figure(go.Bar(
                    x=bar_x, y=label_order, orientation="h",
                    marker_color=label_colors, opacity=0.85,
                    text=[f"{v}" for v in bar_x], textposition="auto",
                ))
                fig_lc.update_layout(
                    height=220, template=TEMPLATE,
                    margin=dict(l=0, r=0, t=10, b=0),
                    xaxis_title="Article Count",
                    yaxis={"categoryorder": "array", "categoryarray": label_order[::-1]},
                )
                st.plotly_chart(fig_lc, use_container_width=True)

            # News headlines table
            if sentiment_agg["top_headlines"]:
                st.subheader("Recent Headlines (used in AI analysis)")
                hdf = pd.DataFrame(sentiment_agg["top_headlines"])[
                    ["published", "label", "score", "source", "title"]
                ]
                hdf["score"] = hdf["score"].apply(lambda s: f"{s:+.4f}")
                st.dataframe(hdf, use_container_width=True, hide_index=True, height=280)
            else:
                st.warning(
                    "No recent articles found for this sector. "
                    "The AI analysis will rely on technical data only."
                )

            st.divider()

            # ── 3b. Dawn.com BERT section ──────────────────────────────────────
            st.subheader("Dawn.com Pakistan Business News — BERT Sentiment")
            st.caption(
                "Scraped from [dawn.com/business/business-finance](https://www.dawn.com/business/business-finance) "
                "· Analysed locally with **nlptown/bert-base-multilingual-uncased-sentiment** · No API key required"
            )

            with st.spinner("Loading BERT model and scraping Dawn.com…"):
                try:
                    dawn_articles_raw = dawn_mod.scrape_dawn_articles(max_pages=3)
                    dawn_articles     = dawn_mod.analyze_dawn_articles(dawn_articles_raw, psx_only=False)
                    dawn_agg          = dawn_mod.aggregate_dawn_sentiment(dawn_articles, days_back=14)
                    dawn_ok = True
                except Exception as _e:
                    st.warning(f"Dawn.com / BERT error: {_e}")
                    dawn_articles = []
                    dawn_agg      = {"score": 0.0, "label": "Neutral", "article_count": 0,
                                     "bullish_pct": 0.0, "bearish_pct": 0.0,
                                     "top_headlines": [], "label_counts": {}}
                    dawn_ok = False

            if dawn_ok:
                d_score_color = (
                    C_BULL if dawn_agg["score"] > 0.08
                    else C_BEAR if dawn_agg["score"] < -0.08
                    else "#aaa"
                )
                da1, da2, da3, da4 = st.columns(4)
                da1.metric("BERT Score",         f"{dawn_agg['score']:+.4f}", dawn_agg["label"])
                da2.metric("Positive Articles", f"{dawn_agg['bullish_pct']:.1f}%")
                da3.metric("Negative Articles", f"{dawn_agg['bearish_pct']:.1f}%")
                da4.metric("Articles Scraped",  dawn_agg["article_count"])

                # BERT sentiment gauge
                dawn_gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=dawn_agg["score"],
                    delta={"reference": 0, "valueformat": ".4f"},
                    title={"text": "Dawn.com BERT Score", "font": {"size": 14}},
                    gauge={
                        "axis":  {"range": [-1, 1], "tickwidth": 1, "tickcolor": "#ccc"},
                        "bar":   {"color": d_score_color},
                        "bgcolor": "rgba(0,0,0,0)",
                        "borderwidth": 1, "bordercolor": "#555",
                        "steps": [
                            {"range": [-1.0, -0.25], "color": "rgba(239,83,80,0.25)"},
                            {"range": [-0.25,-0.08], "color": "rgba(239,83,80,0.10)"},
                            {"range": [-0.08, 0.08], "color": "rgba(180,180,180,0.08)"},
                            {"range": [ 0.08, 0.25], "color": "rgba(38,166,154,0.10)"},
                            {"range": [ 0.25, 1.0],  "color": "rgba(38,166,154,0.25)"},
                        ],
                        "threshold": {"line": {"color": "#fff", "width": 2},
                                      "thickness": 0.75, "value": dawn_agg["score"]},
                    },
                    number={"valueformat": ".4f", "font": {"size": 22}},
                ))
                dawn_gauge.update_layout(
                    height=260, template=TEMPLATE,
                    margin=dict(l=20, r=20, t=40, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(dawn_gauge, use_container_width=True)

                # Label distribution
                if dawn_agg.get("label_counts"):
                    lc2 = dawn_agg["label_counts"]
                    label_order2  = ["Positive", "Neutral", "Negative"]
                    label_colors2 = [C_BULL, "#90a4ae", C_BEAR]
                    bar_x2 = [lc2.get(l, 0) for l in label_order2]
                    fig_lc2 = go.Figure(go.Bar(
                        x=bar_x2, y=label_order2, orientation="h",
                        marker_color=label_colors2, opacity=0.85,
                        text=[f"{v}" for v in bar_x2], textposition="auto",
                    ))
                    fig_lc2.update_layout(
                        height=180, template=TEMPLATE,
                        margin=dict(l=0, r=0, t=10, b=0),
                        xaxis_title="Article Count",
                    )
                    st.plotly_chart(fig_lc2, use_container_width=True)

                # PSX-relevant articles table
                psx_arts = [a for a in dawn_agg["top_headlines"] if a.get("psx_relevant")]
                all_arts = dawn_agg["top_headlines"]
                shown    = psx_arts if psx_arts else all_arts

                if shown:
                    st.caption(
                        f"Showing {'PSX-relevant' if psx_arts else 'all'} articles "
                        f"({len(psx_arts)} PSX-relevant out of {len(all_arts)} total)"
                    )
                    hdf2 = pd.DataFrame(shown)[["published", "label", "score", "source", "title"]]
                    hdf2["score"] = hdf2["score"].apply(lambda s: f"{s:+.4f}")
                    st.dataframe(hdf2, use_container_width=True, hide_index=True, height=300)

                    # Clickable links expander
                    with st.expander("Article links"):
                        for h in shown[:15]:
                            url_h = h.get("url", "")
                            if url_h:
                                st.markdown(f"- [{h['title']}]({url_h})")

            st.divider()

            # ── 4. GPT-4 Analysis ──────────────────────────────────────────────
            st.subheader("GPT-4o — PSX Analyst Report")
            with st.spinner("Consulting GPT-4o… (this may take 10–20 seconds)"):
                gpt_result = sent_mod.gpt4_stock_analysis(
                    openai_key=openai_key,
                    symbol=symbol,
                    price_data_json=json.dumps(price_data_dict),
                    technical_signals_json=json.dumps(tech_signals_dict),
                    sentiment_json=json.dumps(sentiment_agg),
                )

            if "error" in gpt_result:
                st.error(f"GPT-4o error: {gpt_result['error']}")
                if "raw" in gpt_result:
                    with st.expander("Raw GPT-4o output"):
                        st.code(gpt_result["raw"])
            else:
                # ── Headline metrics ───────────────────────────────────────────
                rec = gpt_result.get("recommendation", "N/A")
                rec_color = (
                    C_BULL if "Buy" in rec
                    else C_BEAR if "Sell" in rec
                    else "#aaa"
                )
                direction = gpt_result.get("price_direction", "Sideways")
                dir_icon  = "📈" if direction == "Up" else ("📉" if direction == "Down" else "➡️")
                conf      = gpt_result.get("confidence", "N/A")

                gm1, gm2, gm3, gm4, gm5 = st.columns(5)
                gm1.metric("Recommendation",    rec)
                gm2.metric("Price Direction",   f"{dir_icon} {direction}")
                gm3.metric("Confidence",        conf)
                gm4.metric("Overall Sentiment", gpt_result.get("overall_sentiment", "N/A"))
                gm5.metric("Tech Outlook",      gpt_result.get("technical_outlook", "N/A"))

                adj = gpt_result.get("forecast_adjustment", 0.0)
                if adj is not None:
                    # Cache so Prediction tab can display it
                    st.session_state[f"gpt_adj_{symbol}"] = float(adj)
                    adj_sign = "+" if float(adj) >= 0 else ""
                    st.caption(
                        f"**Forecast Adjustment:** {adj_sign}{adj:.2f}% "
                        f"— GPT-4o recommends shifting ML price forecasts by this amount "
                        f"based on sentiment and macro context."
                    )

                # ── Target price ───────────────────────────────────────────────
                target = gpt_result.get("short_term_target")
                if target:
                    curr   = price_data_dict["close"]
                    upside = (float(target) - curr) / curr * 100
                    st.info(
                        f"**Short-Term Price Target:** PKR {float(target):,.2f}  "
                        f"({'↑' if upside >= 0 else '↓'} {upside:+.1f}% from current PKR {curr:,.2f})"
                    )

                st.divider()

                # ── Qualitative analysis ───────────────────────────────────────
                col_bull, col_bear = st.columns(2)

                with col_bull:
                    st.markdown("#### 🟢 Bullish Factors")
                    for factor in gpt_result.get("key_bullish_factors", []):
                        st.markdown(f"- {factor}")

                with col_bear:
                    st.markdown("#### 🔴 Bearish Factors")
                    for factor in gpt_result.get("key_bearish_factors", []):
                        st.markdown(f"- {factor}")

                st.markdown("#### 🇵🇰 PSX Macro Context")
                st.info(gpt_result.get("psx_macro_context", ""))

                st.markdown("#### 📰 News Interpretation")
                st.info(gpt_result.get("news_interpretation", ""))

                st.markdown("#### 💡 Recommendation Rationale")
                st.success(gpt_result.get("recommendation_rationale", ""))

                st.markdown("#### ⚠️ Risk Factors")
                for risk in gpt_result.get("risk_factors", []):
                    st.markdown(f"- {risk}")

                # ── Raw JSON expander ──────────────────────────────────────────
                with st.expander("Raw GPT-4o JSON response"):
                    st.json(gpt_result)

    # ── Tab: Prediction ────────────────────────────────────────────────────────
    with tab_pred:
        st.markdown(
            '<div class="warning-box">'
            "⚠️ <strong>Disclaimer:</strong> These forecasts are generated by ML/statistical models "
            "for <strong>educational purposes only</strong>. Stock markets are inherently unpredictable. "
            "Do not make investment or trading decisions solely based on this tool."
            "</div>",
            unsafe_allow_html=True,
        )

        if not run_pred_btn:
            st.info("Select a model and horizon in the sidebar, then click **▶ Run Forecast**.")

            # Quick trend preview
            st.subheader("Moving Average Trend Preview")
            fig_prev = go.Figure()
            fig_prev.add_trace(go.Scatter(
                x=df["Date"], y=df["Close"], name="Close",
                line=dict(color=C_BLUE, width=1.5),
            ))
            for w, col in [(20, C_ORANGE), (50, C_GOLD), (200, C_PURPLE)]:
                col_name = f"SMA_{w}"
                if col_name in df.columns:
                    fig_prev.add_trace(go.Scatter(
                        x=df["Date"], y=df[col_name], name=f"SMA {w}",
                        line=dict(color=col, width=1.3),
                    ))
            fig_prev.update_layout(
                height=420, template=TEMPLATE, margin=MARGIN,
                legend=dict(orientation="h", yanchor="bottom", y=1.01),
            )
            st.plotly_chart(fig_prev, use_container_width=True)

        else:
            # ── Fetch Alpha Vantage sentiment series (if key provided) ────────
            sent_series = None
            if use_sentiment_feature and av_key:
                with st.spinner("Fetching Alpha Vantage sentiment for ML features…"):
                    topics_str  = ",".join(sent_mod.topics_for(symbol))
                    av_articles = sent_mod.fetch_av_sentiment(av_key, topics_str, limit=200)
                    if av_articles and "Date" in df.columns:
                        sent_series = sent_mod.build_sentiment_series(
                            av_articles, pd.DatetimeIndex(df["Date"].values)
                        )
                        st.caption(
                            f"✅ Alpha Vantage sentiment: {len(av_articles)} articles "
                            f"→ {sent_series.notna().sum()} aligned trading days"
                        )

            # ── Fetch Dawn.com BERT sentiment series ───────────────────────────
            dawn_sent_series = None
            if use_dawn_sentiment:
                with st.spinner("Scraping Dawn.com + running BERT…"):
                    try:
                        dawn_raw_pred  = dawn_mod.scrape_dawn_articles(max_pages=3)
                        dawn_enriched  = dawn_mod.analyze_dawn_articles(dawn_raw_pred)
                        if dawn_enriched and "Date" in df.columns:
                            dawn_sent_series = dawn_mod.build_dawn_sentiment_series(
                                dawn_enriched, pd.DatetimeIndex(df["Date"].values)
                            )
                            n_psx = sum(1 for a in dawn_enriched if a.get("psx_relevant"))
                            st.caption(
                                f"✅ Dawn.com BERT: {len(dawn_enriched)} articles "
                                f"({n_psx} PSX-relevant) → {dawn_sent_series.notna().sum()} aligned days"
                            )
                    except Exception as _de:
                        st.warning(f"Dawn.com/BERT skipped: {_de}")

            with st.spinner(f"Running {pred_model} — this may take a moment…"):
                if pred_model == "Prophet":
                    result = run_prophet_forecast(df_raw, horizon)
                else:
                    result = run_ml_forecast(df_raw, pred_model, horizon,
                                             sentiment_series=sent_series,
                                             dawn_sentiment_series=dawn_sent_series)

            if "error" in result:
                st.error(result["error"])
            else:
                # ── Sentiment-adjusted forecast note ─────────────────────────
                active_sents = []
                if sent_series is not None:
                    active_sents.append("Alpha Vantage (global sector)")
                if dawn_sent_series is not None:
                    active_sents.append("Dawn.com BERT (Pakistan news)")
                if active_sents:
                    st.caption(
                        f"✅ **Sentiment-enhanced model** — {' + '.join(active_sents)} "
                        "combined as ML features (sentiment, sentiment_ma5, sentiment_ma20, dawn_sent)."
                    )

                # ── Metrics row ──────────────────────────────────────────────
                m    = result["metrics"]
                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("MAE",    f"PKR {m['MAE']:.2f}",    help="Mean Absolute Error (lower = better)")
                mc2.metric("RMSE",   f"PKR {m['RMSE']:.2f}",   help="Root Mean Squared Error (lower = better)")
                mc3.metric("R²",     f"{m['R²']:.4f}",         help="Coefficient of determination (closer to 1 = better)")
                mc4.metric("MAPE",   f"{m['MAPE %']:.2f}%",    help="Mean Absolute Percentage Error (lower = better)")

                # ── GPT-4o forecast adjustment banner ─────────────────────────
                # If user already ran AI analysis, pull the adjustment from cache
                if av_key and openai_key:
                    cached = sent_mod.gpt4_stock_analysis.__wrapped__ if hasattr(
                        sent_mod.gpt4_stock_analysis, "__wrapped__") else None
                    # Try to use the cached result from session state
                    ss_key = f"gpt_adj_{symbol}"
                    adj = st.session_state.get(ss_key)
                    if adj is not None:
                        adj_sign = "+" if float(adj) >= 0 else ""
                        st.info(
                            f"🧠 **GPT-4o Adjustment:** {adj_sign}{adj:.2f}% applied to "
                            "forecast based on sentiment & macro analysis."
                        )

                # ── Forecast chart ───────────────────────────────────────────
                if pred_model == "Prophet":
                    st.plotly_chart(prophet_forecast_chart(result), use_container_width=True)

                    # Component plots
                    with st.expander("Prophet Components (trend / seasonality)"):
                        import io
                        import matplotlib
                        matplotlib.use("Agg")
                        import matplotlib.pyplot as plt
                        try:
                            m_full = Prophet(
                                daily_seasonality=False,
                                weekly_seasonality=True,
                                yearly_seasonality=True,
                            )
                            m_full.fit(result["history"])
                            comp_fig = m_full.plot_components(result["fc"])
                            buf = io.BytesIO()
                            comp_fig.savefig(buf, format="png", bbox_inches="tight",
                                             facecolor="#0e1117")
                            buf.seek(0)
                            st.image(buf)
                        except Exception:
                            st.caption("Component plot unavailable.")
                else:
                    st.plotly_chart(ml_forecast_chart(df_raw, result), use_container_width=True)

                    # Forecast table
                    with st.expander("Forecast Values"):
                        fc_tbl = pd.DataFrame({
                            "Date":             pd.to_datetime(result["future_dates"]).strftime("%Y-%m-%d"),
                            "Forecast (PKR)":  [f"{p:,.2f}" for p in result["forecast"]],
                            "Lower (−2.5%)":   [f"{p*0.975:,.2f}" for p in result["forecast"]],
                            "Upper (+2.5%)":   [f"{p*1.025:,.2f}" for p in result["forecast"]],
                        })
                        st.dataframe(fc_tbl, use_container_width=True, hide_index=True)

                    # Feature importance
                    if result.get("feat_imp") is not None:
                        st.subheader("Top Feature Importances")
                        fi_df = pd.DataFrame({
                            "Feature":    result["feat_names"],
                            "Importance": result["feat_imp"],
                        }).sort_values("Importance", ascending=True).tail(15)
                        fig_fi = px.bar(fi_df, x="Importance", y="Feature",
                                        orientation="h", template=TEMPLATE,
                                        height=340, color="Importance",
                                        color_continuous_scale="Blues")
                        fig_fi.update_layout(margin=MARGIN, coloraxis_showscale=False)
                        st.plotly_chart(fig_fi, use_container_width=True)

    # ── Tab: Raw Data ──────────────────────────────────────────────────────────
    with tab_data:
        display_cols = [c for c in ["Date", "Open", "High", "Low", "Close", "Volume", "Change", "Return"]
                        if c in df.columns]
        st.dataframe(
            df[display_cols].sort_values("Date", ascending=False),
            use_container_width=True,
            height=460,
        )

        csv_bytes = df[display_cols].to_csv(index=False).encode()
        st.download_button(
            "⬇️ Download CSV",
            data=csv_bytes,
            file_name=f"{symbol}_PSX_{start_dt}_{end_dt}.csv",
            mime="text/csv",
        )

    # ── Tab: Fundamentals ─────────────────────────────────────────────────────
    with tab_fund:
        with st.spinner("Loading fundamentals…"):
            fund = get_fundamentals(symbol)

        if fund is None:
            st.info("Fundamental data not available for this ticker.")
        elif isinstance(fund, pd.DataFrame) and not fund.empty:
            st.dataframe(fund, use_container_width=True)
        elif isinstance(fund, dict) and fund:
            for key, val in fund.items():
                st.subheader(key)
                if isinstance(val, pd.DataFrame):
                    st.dataframe(val, use_container_width=True)
                else:
                    st.write(val)
        else:
            st.info("Fundamental data not available for this ticker.")


if __name__ == "__main__":
    main()
