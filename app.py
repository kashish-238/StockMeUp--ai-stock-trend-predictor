import os
import warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import time
import yfinance as yf
import difflib

from model_utils import add_indicators, train_lstm

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(page_title="StockMeUp — Educational", layout="wide")

# --------------------------------------------------
# Premium Cyber UI (subtle, cute)
# --------------------------------------------------
CYBER_CSS = """
<style>
.stApp {
  background:
    radial-gradient(1100px 700px at 15% 10%, rgba(28, 140, 170, 0.14), transparent 60%),
    radial-gradient(900px 600px at 80% 20%, rgba(115, 85, 255, 0.10), transparent 55%),
    linear-gradient(180deg, #070a0f 0%, #0b0f14 100%);
  color: #e6eef7;
}
.stApp:before {
  content: "";
  position: fixed;
  inset: 0;
  pointer-events: none;
  background-image:
    linear-gradient(rgba(255,255,255,0.04) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255,255,255,0.04) 1px, transparent 1px);
  background-size: 64px 64px;
  opacity: 0.10;
  mix-blend-mode: screen;
}
.block-container { padding-top: 1.2rem; padding-bottom: 2.2rem; }
h1, h2, h3 { letter-spacing: 0.3px; }
h1 { font-weight: 900; }

section[data-testid="stSidebar"] {
  background: rgba(10, 14, 20, 0.72);
  border-right: 1px solid rgba(255,255,255,0.08);
}
section[data-testid="stSidebar"] * { color: #dbe8f7; }

.stButton > button {
  border-radius: 14px;
  border: 1px solid rgba(120, 220, 255, 0.22);
  background: linear-gradient(135deg, rgba(40, 180, 220, 0.16), rgba(120, 80, 255, 0.09));
  box-shadow: 0 0 28px rgba(40, 180, 220, 0.12);
  color: #eaf6ff;
  font-weight: 650;
  padding: 0.55rem 0.9rem;
}
.stButton > button:hover {
  border: 1px solid rgba(120, 220, 255, 0.40);
  box-shadow: 0 0 38px rgba(40, 180, 220, 0.18);
}

.hero {
  border: 1px solid rgba(255,255,255,0.10);
  background: rgba(255,255,255,0.03);
  border-radius: 22px;
  padding: 18px 18px;
  margin-bottom: 14px;
  box-shadow: 0 10px 40px rgba(0,0,0,0.25);
}
.pill {
  display: inline-block;
  border: 1px solid rgba(120,220,255,0.22);
  background: rgba(40,180,220,0.08);
  padding: 5px 10px;
  border-radius: 999px;
  font-size: 12px;
  margin-right: 8px;
}
.muted { color: rgba(230,238,247,0.74); font-size: 13px; }

.kpi {
  border: 1px solid rgba(255,255,255,0.10);
  background: rgba(255,255,255,0.03);
  border-radius: 18px;
  padding: 14px 14px;
  box-shadow: 0 0 0 1px rgba(115,85,255,0.05), 0 10px 30px rgba(0,0,0,0.22);
}
.kpi .label { font-size: 12px; color: rgba(230,238,247,0.70); }
.kpi .value { font-size: 22px; font-weight: 900; margin-top: 2px; }
.kpi .sub { font-size: 12px; color: rgba(230,238,247,0.65); margin-top: 4px; }

.badge-up {
  display:inline-block;
  border:1px solid rgba(120,220,255,0.35);
  background: rgba(40,180,220,0.12);
  padding: 6px 10px;
  border-radius: 999px;
  font-weight: 850;
}
.badge-down {
  display:inline-block;
  border:1px solid rgba(255,120,140,0.28);
  background: rgba(255,120,140,0.10);
  padding: 6px 10px;
  border-radius: 999px;
  font-weight: 850;
}

.footer {
  margin-top: 26px;
  padding-top: 14px;
  border-top: 1px solid rgba(255,255,255,0.08);
  color: rgba(230,238,247,0.62);
  font-size: 12px;
}
.footer a { color: rgba(120,220,255,0.70); text-decoration: none; }
.footer a:hover { text-decoration: underline; }
</style>
"""
st.markdown(CYBER_CSS, unsafe_allow_html=True)

# --------------------------------------------------
# Company name -> ticker resolver (with fuzzy suggestions)
# --------------------------------------------------
COMPANY_TICKER_MAP = {
    # US
    "walmart": "WMT",
    "apple": "AAPL",
    "microsoft": "MSFT",
    "tesla": "TSLA",
    "amazon": "AMZN",
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "meta": "META",
    "facebook": "META",
    "netflix": "NFLX",
    "nvidia": "NVDA",

    # Canada
    "rbc": "RY.TO",
    "royal bank": "RY.TO",
    "royal bank of canada": "RY.TO",
    "td": "TD.TO",
    "td bank": "TD.TO",
    "toronto dominion": "TD.TO",
    "bmo": "BMO.TO",
    "bank of montreal": "BMO.TO",
    "scotiabank": "BNS.TO",
    "bank of nova scotia": "BNS.TO",
    "cibc": "CM.TO",
    "canadian imperial bank": "CM.TO",
    "shopify": "SHOP.TO",

    # Global / Asia (common asks)
    "samsung": "005930.KS",
    "samsung electronics": "005930.KS",

    # India examples (common “brand” queries)
    "tcs": "TCS.NS",
    "tata consultancy services": "TCS.NS",
    "tata motors": "TATAMOTORS.NS",
    "tata steel": "TATASTEEL.NS",
    "reliance": "RELIANCE.NS",
    "infosys": "INFY.NS",
}

AMBIGUOUS_HINTS = {
    "tata": [
        "TCS.NS (Tata Consultancy Services)",
        "TATAMOTORS.NS (Tata Motors)",
        "TATASTEEL.NS (Tata Steel)",
    ]
}

def resolve_ticker(user_input: str):
    if not user_input or not user_input.strip():
        return "", "empty"
    raw = user_input.strip()
    key = raw.lower()

    if key in COMPANY_TICKER_MAP:
        return COMPANY_TICKER_MAP[key], "mapped"

    for name, ticker in COMPANY_TICKER_MAP.items():
        if name in key:
            return ticker, "mapped"

    return raw.upper(), "assumed_ticker"

def suggest_companies(user_input: str, k: int = 5):
    if not user_input:
        return []
    names = list(COMPANY_TICKER_MAP.keys())
    return difflib.get_close_matches(user_input.lower().strip(), names, n=k, cutoff=0.55)

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
with st.sidebar:
    st.header("Controls")
    symbol_input = st.text_input("Stock ticker or company name", value="Walmart")
    period = st.selectbox("Data period", ["1y", "2y", "5y", "10y"], index=2)
    fast = st.slider("Fast MA", 5, 30, 10)
    slow = st.slider("Slow MA", 20, 120, 30)
    fast_mode = st.toggle("⚡ Fast Mode", value=True)
    lookback = st.slider("LSTM Lookback", 10, 90, 20 if fast_mode else 30)
    epochs = st.slider("Training Epochs", 3, 15, 6 if fast_mode else 10)
    run = st.button("Run Analysis")

resolved_symbol, mode = resolve_ticker(symbol_input)

# --------------------------------------------------
# Data loading (Yahoo -> Stooq) with safe dates
# --------------------------------------------------
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0 Safari/537.36"
)

def standardize_date_close(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if "Date" not in df.columns and "Datetime" in df.columns:
        df = df.rename(columns={"Datetime": "Date"})
    if "Date" not in df.columns or "Close" not in df.columns:
        return pd.DataFrame()
    out = df[["Date", "Close"]].copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out = out.dropna(subset=["Date", "Close"]).copy()
    try:
        if getattr(out["Date"].dt, "tz", None) is not None:
            out["Date"] = out["Date"].dt.tz_localize(None)
    except Exception:
        pass
    return out.sort_values("Date")

def yahoo_download(ticker: str, period: str) -> pd.DataFrame:
    try:
        sess = requests.Session()
        sess.headers.update({"User-Agent": UA})
        df = yf.download(
            ticker, period=period, interval="1d",
            auto_adjust=True, progress=False, threads=False, session=sess
        )
        if df is None or df.empty:
            return pd.DataFrame()
        return standardize_date_close(df.reset_index())
    except Exception:
        return pd.DataFrame()

def stooq_candidates(yahoo_ticker: str):
    t = yahoo_ticker.strip().lower()
    if t.endswith(".to"):
        base = t[:-3]
        return [f"{base}.ca", base, f"{base}.us"]
    if "." not in t:
        return [f"{t}.us", t]
    return [t]

def stooq_download(symbol: str) -> pd.DataFrame:
    url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"
    try:
        r = requests.get(url, headers={"User-Agent": UA}, timeout=12)
        if r.status_code != 200:
            return pd.DataFrame()
        from io import StringIO
        df = pd.read_csv(StringIO(r.text))
        return standardize_date_close(df)
    except Exception:
        return pd.DataFrame()

def filter_by_period(df: pd.DataFrame, period: str) -> pd.DataFrame:
    if df.empty:
        return df
    years = {"1y": 1, "2y": 2, "5y": 5, "10y": 10}.get(period, 5)
    start = (pd.Timestamp.now() - pd.DateOffset(years=years)).normalize()
    return df[df["Date"] >= start].copy()

@st.cache_data(show_spinner=False)
def load_data_robust(ticker: str, period: str):
    for _ in range(2):
        df = yahoo_download(ticker, period)
        if not df.empty:
            return df, "Yahoo Finance (yfinance)", ticker
        time.sleep(0.5)

    for cand in stooq_candidates(ticker):
        df2 = stooq_download(cand)
        if not df2.empty:
            df2 = filter_by_period(df2, period)
            if not df2.empty:
                return df2, "Stooq (CSV fallback)", cand

    return pd.DataFrame(), "None", ticker

# --------------------------------------------------
# Plot helpers
# --------------------------------------------------
def accuracy(y_true, y_pred) -> float:
    y_true = np.array(y_true).astype(int)
    y_pred = np.array(y_pred).astype(int)
    return float((y_true == y_pred).mean()) if len(y_true) else 0.0

def plot_price(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["Date"], df["Close"], label="Close", linewidth=1.2)
    ax.plot(df["Date"], df["MA_FAST"], label="MA Fast", linewidth=1.0)
    ax.plot(df["Date"], df["MA_SLOW"], label="MA Slow", linewidth=1.0)
    ax.set_title("Price + Moving Averages")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    plt.tight_layout()
    return fig

def plot_equity_curve(dates, equity):
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(dates, equity, linewidth=1.4)
    ax.set_title("Baseline Paper Strategy — Equity Curve (Educational)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity (normalized)")
    plt.tight_layout()
    return fig

# --------------------------------------------------
# Hero / header + tabs
# --------------------------------------------------
st.markdown(
    f"""
<div class="hero">
  <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:12px;">
    <div>
      <div style="font-size:30px; font-weight:950; line-height:1.1;">StockMeUp</div>
      <div class="muted">Your favourite stock predictor buddy</div>
      <div style="margin-top:10px;">
        <span class="pill">Name → Ticker</span>
        <span class="pill">Baseline MA</span>
        <span class="pill">LSTM</span>
        <span class="pill">Fallback Data</span>
      </div>
    </div>
    <div style="text-align:right;">
      <div class="muted">Input</div>
      <div style="font-weight:900; font-size:18px;">{symbol_input}</div>
      <div class="muted">Resolved</div>
      <div style="font-weight:900; font-size:18px;">{resolved_symbol if resolved_symbol else "—"}</div>
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

tabs = st.tabs(["📊 Dashboard", "🧠 Models", "🗃️ Data", "ℹ️ About"])

# --------------------------------------------------
# Main flow
# --------------------------------------------------
if not run:
    with tabs[0]:
        st.info("Type a company name (e.g., **Walmart**, **Samsung**) or ticker (e.g., **WMT**, **AAPL**) and click **Run Analysis**.")
    with tabs[3]:
        st.markdown(
            """
**StockMeUp** is an educational demo showcasing:
- A simple **Moving Average baseline**
- A lightweight **LSTM** direction classifier
- A friendly **company name → ticker** resolver for real users
- Data fallback to **Stooq** when Yahoo is blocked

**Disclaimer:** Not financial advice.
"""
        )
    # Credits footer
    st.markdown(
        """
<div class="footer">
  <b>Credits</b> — <b> Built by Kashish Dhanani </b>
</div>
""",
        unsafe_allow_html=True,
    )
    st.stop()

if not resolved_symbol:
    st.error("Please enter a company name or ticker.")
    st.stop()

with st.spinner(f"Fetching market data for {resolved_symbol}..."):
    raw, source, used_symbol = load_data_robust(resolved_symbol, period)

with tabs[2]:
    st.caption(f"**Data source:** {source}  |  **Used symbol:** `{used_symbol}`")

if raw.empty:
    key = symbol_input.strip().lower()
    with tabs[0]:
        st.error("No data found for that input.")
        if key in AMBIGUOUS_HINTS:
            st.info(
                "That looks like a **brand/group name** (not one single stock). Try:\n\n"
                + "\n".join([f"- `{x}`" for x in AMBIGUOUS_HINTS[key]])
            )
        else:
            suggestions = suggest_companies(symbol_input)
            if suggestions:
                st.info("Did you mean:")
                for s in suggestions:
                    st.write(f"- **{s.title()}** → `{COMPANY_TICKER_MAP[s]}`")
            st.info("Tip: try a ticker like **AAPL**, **MSFT**, **WMT**, or a company name we support (Walmart, Samsung, Apple...).")
    # Credits footer
    st.markdown(
        """
<div class="footer">
  <b>Credits</b> — Built with Streamlit, NumPy, Pandas, Matplotlib, scikit-learn, TensorFlow/Keras, yfinance, and Stooq CSV fallback.<br/>
  Market data is provided by third-party sources (Yahoo Finance / Stooq) and may be delayed or incomplete. Educational use only.
</div>
""",
        unsafe_allow_html=True,
    )
    st.stop()

# Indicators
df = add_indicators(raw, fast=fast, slow=slow)
if df.empty:
    st.error("Not enough data after indicator calculation. Try a longer period or bigger MA windows.")
    st.stop()

# Speed trim for training
train_days = 700 if fast_mode else 1600
df = df.tail(train_days).copy()

# Baseline
baseline_pred = df["BASELINE_SIGNAL"].values
y_true = df["TARGET_UP"].values
baseline_acc = accuracy(y_true, baseline_pred)
baseline_up_rate = float(df["BASELINE_SIGNAL"].mean())

# LSTM
with st.spinner("Training LSTM (fast settings)..."):
    results = train_lstm(
        df["Close"],
        lookback=lookback,
        epochs=epochs,
        batch_size=64,
        max_points=1200 if fast_mode else 2200,
        fast_mode=fast_mode,
    )

lstm_acc = accuracy(results["y_test"], results["preds"])
last_prob = float(results["probs"][-1]) if len(results["probs"]) else 0.0
last_pred = "UP" if last_prob >= 0.5 else "DOWN"
badge = f'<span class="badge-up">SIGNAL: {last_pred}</span>' if last_pred == "UP" else f'<span class="badge-down">SIGNAL: {last_pred}</span>'

# Baseline paper equity curve
ret = df["Close"].pct_change().fillna(0.0).values
pos = df["BASELINE_SIGNAL"].shift(1).fillna(0).values
equity = np.cumprod(1 + ret * pos)

# --------------------------------------------------
# Dashboard tab
# --------------------------------------------------
with tabs[0]:
    st.markdown(badge, unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""
<div class="kpi">
  <div class="label">Ticker</div>
  <div class="value">{resolved_symbol}</div>
  <div class="sub">Period: {period}</div>
</div>
""", unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
<div class="kpi">
  <div class="label">Baseline Accuracy</div>
  <div class="value">{baseline_acc*100:.2f}%</div>
  <div class="sub">UP rate: {baseline_up_rate*100:.1f}%</div>
</div>
""", unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
<div class="kpi">
  <div class="label">LSTM Accuracy (Test)</div>
  <div class="value">{lstm_acc*100:.2f}%</div>
  <div class="sub">Lookback: {lookback} days</div>
</div>
""", unsafe_allow_html=True)
    with k4:
        st.markdown(f"""
<div class="kpi">
  <div class="label">Confidence (UP)</div>
  <div class="value">{last_prob:.2f}</div>
  <div class="sub">Fast Mode: {"On" if fast_mode else "Off"}</div>
</div>
""", unsafe_allow_html=True)

    left, right = st.columns([1.25, 1.0])
    with left:
        st.subheader("Market Overview")
        st.pyplot(plot_price(df), clear_figure=True)
        st.caption("Educational demo only — no trading advice.")
    with right:
        st.subheader("Baseline Strategy (Educational)")
        st.pyplot(plot_equity_curve(df["Date"].values, equity), clear_figure=True)
        st.caption("Paper strategy: hold asset when baseline signal is UP; otherwise cash.")

    st.subheader("Recent Signals (Last 30 Days)")
    recent = df.tail(30).copy()
    view = recent[["Date", "Close", "MA_FAST", "MA_SLOW"]].copy()
    view["Baseline_State"] = np.where(recent["BASELINE_SIGNAL"] == 1, "UP", "DOWN")
    view["True_Next_Day"] = np.where(recent["TARGET_UP"] == 1, "UP", "DOWN")
    st.dataframe(view, use_container_width=True)

# --------------------------------------------------
# Models tab
# --------------------------------------------------
with tabs[1]:
    st.subheader("Model Details")
    st.write("**Baseline:** Moving Average state rule (MA_FAST > MA_SLOW).")
    st.write("**LSTM:** Lightweight direction classifier (fast CPU-friendly).")

    st.divider()
    st.subheader("LSTM Training Curve")
    hist = results["history"]
    fig2, ax2 = plt.subplots(figsize=(10, 3.5))
    ax2.plot(hist.get("loss", []), label="loss", linewidth=1.2)
    ax2.plot(hist.get("val_loss", []), label="val_loss", linewidth=1.2)
    ax2.set_title("Loss vs Val Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    plt.tight_layout()
    st.pyplot(fig2, clear_figure=True)

# --------------------------------------------------
# Data tab
# --------------------------------------------------
with tabs[2]:
    st.subheader("Data Preview")
    st.write("Tries Yahoo Finance first; if blocked, falls back to Stooq CSV.")
    st.dataframe(raw.tail(30), use_container_width=True)

# --------------------------------------------------
# About tab
# --------------------------------------------------
with tabs[3]:
    st.subheader("About StockMeUp")
    st.markdown(
        """
**StockMeUp** is an educational demo project.

**What it demonstrates**
- Input normalization (company name → ticker)
- Baseline strategy + evaluation
- Lightweight LSTM direction prediction
- Clean, cute, resume-ready dashboard UI

**Disclaimer**
Not financial advice. For learning only.
"""
    )

# --------------------------------------------------
# Credits footer (always)
# --------------------------------------------------
st.markdown(
    """
<div class="footer">
  <b>Credits</b> — Built with Streamlit, NumPy, Pandas, Matplotlib, scikit-learn, TensorFlow/Keras, yfinance, and Stooq CSV fallback.<br/>
  Market data is provided by third-party sources (Yahoo Finance / Stooq) and may be delayed or incomplete. Educational use only.
</div>
""",
    unsafe_allow_html=True,
)
