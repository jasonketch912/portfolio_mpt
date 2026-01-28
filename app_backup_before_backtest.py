import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from scipy.optimize import minimize

st.set_page_config(page_title="Portfolio Optimizer (MPT)", layout="wide")
st.title("📊 Portfolio Optimizer (MPT) — Efficient Frontier + CML")

# -------------------------
# Sidebar
# -------------------------
st.sidebar.header("Assets")
tickers_str = st.sidebar.text_input("Tickers (comma-separated)", "AAPL,NVDA,META,NFLX,UBER")
tickers = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]

st.sidebar.header("Period")
start_date = st.sidebar.date_input("Start", value=pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End", value=pd.to_datetime("today"))

st.sidebar.header("Risk-Free / Objective")
risk_free = st.sidebar.number_input("Risk-free rate (annual)", value=0.02, step=0.005, format="%.3f")
objective = st.sidebar.selectbox("Objective", ["Max Sharpe (Tangency)", "Min Variance", "Target Return"])

target_return = None
if objective == "Target Return":
    target_return = st.sidebar.number_input("Target return (annual)", value=0.15, step=0.01, format="%.2f")

st.sidebar.header("CML (Risk-free asset)")
include_rf = st.sidebar.checkbox("Include Risk-free asset (CML)", value=True)

alpha = st.sidebar.number_input(
    "Max risky exposure alpha (leverage)",
    value=1.0,
    min_value=0.0,
    step=0.1,
    format="%.1f",
    help="alpha=1 => 100% risky. alpha<1 => mix avec risk-free. alpha>1 => leverage."
)

st.sidebar.header("Constraints")
min_w = st.sidebar.number_input("Min weight per asset", value=0.0, min_value=0.0, max_value=1.0, step=0.01, format="%.2f")
max_w = st.sidebar.number_input("Max weight per asset", value=1.0, min_value=0.0, max_value=1.0, step=0.01, format="%.2f")

simulate_n = st.sidebar.slider("Portfolios for frontier (simulation)", min_value=1000, max_value=20000, value=6000, step=1000)

if len(tickers) < 2:
    st.warning("Ajoute au moins 2 tickers.")
    st.stop()

if min_w * len(tickers) > 1.0:
    st.error("Min weight trop élevé: min_w * N > 1 (impossible).")
    st.stop()

if min_w > max_w:
    st.error("min_w ne peut pas être > max_w.")
    st.stop()

# -------------------------
# Data
# -------------------------
@st.cache_data
def load_prices(tickers_list, start, end):
    data = yf.download(tickers_list, start=start, end=end, auto_adjust=False, progress=False)

    if data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        # Prefer Adj Close, else Close
        level0 = data.columns.get_level_values(0)
        if "Adj Close" in level0:
            prices_ = data["Adj Close"].copy()
        else:
            prices_ = data["Close"].copy()
    else:
        prices_ = data["Adj Close"] if "Adj Close" in data.columns else data["Close"]

    prices_ = prices_.dropna(how="all").dropna()
    return prices_

prices = load_prices(tickers, start_date, end_date)

if prices.empty or prices.shape[0] < 60:
    st.error("Pas assez de données. Change la période ou vérifie les tickers.")
    st.stop()

returns = prices.pct_change().dropna()
mu = returns.mean() * 252
cov = returns.cov() * 252

n = len(tickers)
bounds = [(min_w, max_w)] * n
constraints_sum = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

def perf(w):
    r = float(np.dot(w, mu))
    v = float(np.sqrt(w.T @ cov @ w))
    return r, v

def neg_sharpe(w):
    r, v = perf(w)
    if v == 0:
        return 1e9
    return -((r - risk_free) / v)

def variance(w):
    return float(w.T @ cov @ w)

# Target return constraint if needed
constraints_target = constraints_sum.copy()
if objective == "Target Return":
    constraints_target = constraints_sum + [{"type": "eq", "fun": lambda w: float(np.dot(w, mu)) - float(target_return)}]

w0 = np.array([1.0 / n] * n)

# -------------------------
# Optimization
# -------------------------
if objective == "Max Sharpe (Tangency)":
    res = minimize(neg_sharpe, w0, method="SLSQP", bounds=bounds, constraints=constraints_sum)
elif objective == "Min Variance":
    res = minimize(variance, w0, method="SLSQP", bounds=bounds, constraints=constraints_sum)
else:
    res = minimize(variance, w0, method="SLSQP", bounds=bounds, constraints=constraints_target)

if not res.success:
    st.error(f"Optimisation échouée: {res.message}")
    st.stop()

w_opt = res.x
opt_ret, opt_vol = perf(w_opt)
opt_sharpe = (opt_ret - risk_free) / opt_vol if opt_vol > 0 else np.nan

# Also compute Min Variance portfolio (always)
res_mv = minimize(variance, w0, method="SLSQP", bounds=bounds, constraints=constraints_sum)
w_mv = res_mv.x if res_mv.success else w0
mv_ret, mv_vol = perf(w_mv)

# Tangency (Max Sharpe) always (for CML / reference)
res_tan = minimize(neg_sharpe, w0, method="SLSQP", bounds=bounds, constraints=constraints_sum)
w_tan = res_tan.x if res_tan.success else w0
tan_ret, tan_vol = perf(w_tan)
tan_sharpe = (tan_ret - risk_free) / tan_vol if tan_vol > 0 else np.nan

# Risk-free + risky mix (CML) with alpha
# Portfolio with risky weights = w_tan (standard). Total allocation:
# risky portion = alpha, risk-free portion = 1-alpha
rf_weight = 1.0 - alpha
risky_weight = alpha

# If alpha < 0, clamp
if risky_weight < 0:
    risky_weight = 0.0
    rf_weight = 1.0

# Effective portfolio return/vol under CML mix
cml_ret = risk_free + risky_weight * (tan_ret - risk_free)
cml_vol = abs(risky_weight) * tan_vol

# -------------------------
# Efficient Frontier simulation
# -------------------------
rng = np.random.default_rng(42)

def random_weights():
    # generate feasible weights within bounds and sum=1
    # simple rejection sampling
    for _ in range(5000):
        w = rng.random(n)
        w = w / w.sum()
        if np.all(w >= min_w) and np.all(w <= max_w):
            return w
    # fallback: project by clipping then renormalize
    w = rng.random(n)
    w = np.clip(w, min_w, max_w)
    if w.sum() == 0:
        w = np.array([1.0/n]*n)
    return w / w.sum()

sim_rets = np.zeros(simulate_n)
sim_vols = np.zeros(simulate_n)
sim_sharpes = np.zeros(simulate_n)

for i in range(simulate_n):
    w = random_weights()
    r, v = perf(w)
    sim_rets[i] = r
    sim_vols[i] = v
    sim_sharpes[i] = (r - risk_free) / v if v > 0 else np.nan

# -------------------------
# Layout
# -------------------------
col1, col2 = st.columns([1, 2], gap="large")

with col1:
    st.subheader("✅ Portfolio (selected objective)")
    st.metric("Return (annual)", f"{opt_ret*100:.2f}%")
    st.metric("Volatility (annual)", f"{opt_vol*100:.2f}%")
    st.metric("Sharpe", f"{opt_sharpe:.2f}" if np.isfinite(opt_sharpe) else "—")

    alloc_df = pd.DataFrame({"Ticker": tickers, "Weight": w_opt})
    alloc_df["Weight"] = alloc_df["Weight"].round(4)
    st.write("**Allocation**")
    st.dataframe(alloc_df.sort_values("Weight", ascending=False), use_container_width=True)

    fig_alloc = go.Figure()
    fig_alloc.add_bar(x=alloc_df["Ticker"], y=alloc_df["Weight"], name="Weights")
    fig_alloc.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_alloc, use_container_width=True)

    st.subheader("Reference (risky-only)")
    st.write(f"**Tangency Return:** {tan_ret*100:.2f}%")
    st.write(f"**Tangency Vol:** {tan_vol*100:.2f}%")
    st.write(f"**Min Var Vol:** {mv_vol*100:.2f}%")

    if include_rf:
        st.subheader("CML mix (with alpha)")
        st.write(f"**alpha (risky exposure):** {alpha:.1f}")
        st.write(f"**Risk-free weight:** {rf_weight:.2f}")
        st.write(f"**CML Return:** {cml_ret*100:.2f}%")
        st.write(f"**CML Vol:** {cml_vol*100:.2f}%")

with col2:
    st.subheader("Prices (Close/Adj Close)")
    fig_prices = go.Figure()
    for t in tickers:
        fig_prices.add_trace(go.Scatter(x=prices.index, y=prices[t], mode="lines", name=t))
    fig_prices.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_prices, use_container_width=True)

    st.subheader("Frontier (risky-only) + CML (if enabled)")
    fig_front = go.Figure()

    # Cloud of random portfolios
    fig_front.add_trace(
        go.Scatter(
            x=sim_vols,
            y=sim_rets,
            mode="markers",
            name="Risky portfolios",
            marker=dict(size=5, opacity=0.6, color=sim_sharpes, colorscale="Viridis", showscale=True),
        )
    )

    # Min variance point
    fig_front.add_trace(
        go.Scatter(
            x=[mv_vol],
            y=[mv_ret],
            mode="markers",
            name="Min Variance",
            marker=dict(size=12),
        )
    )

    # Tangency point
    fig_front.add_trace(
        go.Scatter(
            x=[tan_vol],
            y=[tan_ret],
            mode="markers",
            name="Max Sharpe (Tangency)",
            marker=dict(size=12),
        )
    )

    # Selected objective point
    fig_front.add_trace(
        go.Scatter(
            x=[opt_vol],
            y=[opt_ret],
            mode="markers",
            name="Selected",
            marker=dict(size=14, symbol="x"),
        )
    )

    # CML line
    if include_rf:
        sigma_max = max(sim_vols.max(), tan_vol) * 1.2
        sigma_line = np.linspace(0, sigma_max, 100)
        cml_line = risk_free + ((tan_ret - risk_free) / tan_vol) * sigma_line if tan_vol > 0 else np.nan

        fig_front.add_trace(
            go.Scatter(
                x=sigma_line,
                y=cml_line,
                mode="lines",
                name="CML (Rf → Tangency)",
            )
        )

    fig_front.update_layout(
        xaxis_title="Annualized Volatility",
        yaxis_title="Annualized Return",
        height=420,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig_front, use_container_width=True)

st.caption("Données: yfinance • MPT (Markowitz) + CML (CAPM) • Contraintes: somme=1, bounds=min/max")
