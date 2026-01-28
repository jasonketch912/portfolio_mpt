import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from scipy.optimize import minimize

st.set_page_config(page_title="Portfolio Optimizer (MPT)", layout="wide")

st.title("📊 Portfolio Optimizer (MPT)")

# -------------------------
# Sidebar inputs
# -------------------------
st.sidebar.header("Inputs")

tickers_str = st.sidebar.text_input(
    "Tickers (comma-separated)",
    "AAPL,NVDA,META,NFLX,UBER"
)

start_date = st.sidebar.date_input("Start", value=pd.to_datetime("2020-01-01"))
risk_free = st.sidebar.number_input("Risk-free rate (annual)", value=0.02, step=0.005, format="%.3f")

tickers = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]

if len(tickers) < 2:
    st.warning("Ajoute au moins 2 tickers.")
    st.stop()

# -------------------------
# Download prices
# -------------------------
@st.cache_data
def load_prices(tickers_list, start):
    data = yf.download(tickers_list, start=start, auto_adjust=False, progress=False)
    # yfinance returns MultiIndex columns when multiple tickers
    if isinstance(data.columns, pd.MultiIndex):
        # Prefer Adj Close if available, otherwise Close
        if ("Adj Close" in data.columns.get_level_values(0)):
            prices_ = data["Adj Close"].copy()
        else:
            prices_ = data["Close"].copy()
    else:
        # single ticker case - but we already require 2; still handle
        prices_ = data["Adj Close"] if "Adj Close" in data.columns else data["Close"]

    prices_ = prices_.dropna(how="all")
    return prices_

prices = load_prices(tickers, start_date)

if prices.empty or prices.dropna().shape[0] < 30:
    st.error("Pas assez de données téléchargées. Change la date de début ou vérifie les tickers.")
    st.stop()

prices = prices.dropna()

# -------------------------
# Returns / covariance
# -------------------------
returns = prices.pct_change().dropna()
mu = returns.mean() * 252
cov = returns.cov() * 252

# -------------------------
# Optimization (Max Sharpe)
# -------------------------
n = len(tickers)

def portfolio_perf(w):
    port_ret = float(np.dot(w, mu))
    port_vol = float(np.sqrt(np.dot(w.T, np.dot(cov, w))))
    return port_ret, port_vol

def neg_sharpe(w):
    r, v = portfolio_perf(w)
    if v == 0:
        return 1e9
    return -((r - risk_free) / v)

constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
bounds = [(0.0, 1.0)] * n
w0 = np.array([1.0 / n] * n)

res = minimize(neg_sharpe, w0, method="SLSQP", bounds=bounds, constraints=constraints)

if not res.success:
    st.error(f"Optimisation échouée: {res.message}")
    st.stop()

w_opt = res.x
opt_ret, opt_vol = portfolio_perf(w_opt)
opt_sharpe = (opt_ret - risk_free) / opt_vol

# -------------------------
# Layout
# -------------------------
left, right = st.columns([1, 2], gap="large")

with left:
    st.subheader("✅ Max Sharpe Portfolio")
    st.metric("Return (annual)", f"{opt_ret*100:.2f}%")
    st.metric("Volatility (annual)", f"{opt_vol*100:.2f}%")
    st.metric("Sharpe", f"{opt_sharpe:.2f}")

    st.subheader("Allocation")
    alloc_df = pd.DataFrame({"Ticker": tickers, "Weight": w_opt})
    alloc_df["Weight"] = alloc_df["Weight"].round(4)
    st.dataframe(alloc_df.sort_values("Weight", ascending=False), use_container_width=True)

    fig_alloc = go.Figure()
    fig_alloc.add_bar(x=alloc_df["Ticker"], y=alloc_df["Weight"])
    fig_alloc.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_alloc, use_container_width=True)

with right:
    st.subheader("Prices (Close)")
    fig_prices = go.Figure()
    for t in tickers:
        fig_prices.add_trace(go.Scatter(x=prices.index, y=prices[t], mode="lines", name=t))
    fig_prices.update_layout(height=520, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_prices, use_container_width=True)

st.caption("Données: yfinance • Optimisation: Modern Portfolio Theory (Max Sharpe, contraintes 0-100%, somme=1)")
