import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from scipy.optimize import minimize
from io import StringIO

st.set_page_config(page_title="Portfolio Optimizer (MPT)", layout="wide")
st.title("📊 Portfolio Optimizer (MPT) — Frontier + CML + Backtest + Benchmark + Export")

# -------------------------
# Helpers
# -------------------------
def annualized_return(equity: pd.Series, periods_per_year: int = 252) -> float:
    equity = equity.dropna()
    if equity.empty or len(equity) < 2:
        return np.nan
    total = equity.iloc[-1] / equity.iloc[0]
    years = (len(equity) - 1) / periods_per_year
    if years <= 0:
        return np.nan
    return total ** (1 / years) - 1

def annualized_vol(returns: pd.Series, periods_per_year: int = 252) -> float:
    returns = returns.dropna()
    if returns.empty:
        return np.nan
    return returns.std() * np.sqrt(periods_per_year)

def sharpe_ratio(returns: pd.Series, rf_annual: float, periods_per_year: int = 252) -> float:
    returns = returns.dropna()
    if returns.empty:
        return np.nan
    rf_daily = (1 + rf_annual) ** (1 / periods_per_year) - 1
    excess = returns - rf_daily
    vol = excess.std()
    if vol == 0 or np.isnan(vol):
        return np.nan
    return (excess.mean() / vol) * np.sqrt(periods_per_year)

def max_drawdown(equity: pd.Series) -> float:
    equity = equity.dropna()
    if equity.empty:
        return np.nan
    peak = equity.cummax()
    dd = (equity / peak) - 1
    return dd.min()

def make_metrics(name: str, equity: pd.Series, rf_annual: float) -> dict:
    rets = equity.pct_change().dropna()
    return {
        "Name": name,
        "CAGR": annualized_return(equity),
        "Volatility": annualized_vol(rets),
        "Sharpe": sharpe_ratio(rets, rf_annual),
        "Max Drawdown": max_drawdown(equity),
        "Final Value (1.0 start)": float(equity.dropna().iloc[-1]) if not equity.dropna().empty else np.nan,
    }

def rebalance_dates(index: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    # freq: "None", "Monthly", "Quarterly", "Yearly"
    if freq == "None":
        return pd.DatetimeIndex([index[0]])
    if freq == "Monthly":
        return index.to_period("M").drop_duplicates().to_timestamp()
    if freq == "Quarterly":
        return index.to_period("Q").drop_duplicates().to_timestamp()
    if freq == "Yearly":
        return index.to_period("Y").drop_duplicates().to_timestamp()
    return pd.DatetimeIndex([index[0]])

def backtest_portfolio(
    prices: pd.DataFrame,
    weights: np.ndarray,
    rf_annual: float = 0.0,
    include_rf: bool = False,
    alpha: float = 1.0,
    rebalance: str = "None",
    start_value: float = 1.0,
) -> pd.Series:
    """
    prices: dataframe of risky assets prices
    weights: target risky weights sum to 1
    include_rf: if True, mix risk-free with risky via alpha (risky exposure)
               rf weight = 1 - alpha
    rebalance: None/Monthly/Quarterly/Yearly
    """
    px = prices.dropna().copy()
    if px.empty or px.shape[0] < 2:
        return pd.Series(dtype=float)

    # daily returns of risky assets
    r = px.pct_change().dropna()
    idx = r.index

    # risk-free daily return
    rf_daily = (1 + rf_annual) ** (1 / 252) - 1

    # rebalance schedule
    rebal_idx = rebalance_dates(idx, rebalance)
    rebal_idx = rebal_idx.intersection(idx)
    if len(rebal_idx) == 0:
        rebal_idx = pd.DatetimeIndex([idx[0]])

    # holdings-based simulation
    value = start_value
    equity = []
    current_w = weights.copy().astype(float)

    # apply CML alpha
    risky_weight_total = alpha if include_rf else 1.0
    rf_weight = 1.0 - risky_weight_total if include_rf else 0.0

    # normalize risky weights to sum=1 (just in case)
    if current_w.sum() != 0:
        current_w = current_w / current_w.sum()

    # start with positions in risky assets + cash
    # positions in "value terms"
    risky_value = value * risky_weight_total
    cash_value = value * rf_weight

    positions = risky_value * current_w  # vector by asset

    for t in idx:
        # rebalance at date t (if t is a rebalance date)
        if t in rebal_idx:
            # recompute portfolio value = risky positions + cash
            value = float(positions.sum() + cash_value)
            risky_value = value * risky_weight_total
            cash_value = value * rf_weight
            positions = risky_value * current_w

        # apply returns for day t
        day_ret = r.loc[t].values
        positions = positions * (1.0 + day_ret)
        cash_value = cash_value * (1.0 + rf_daily)

        value = float(positions.sum() + cash_value)
        equity.append(value)

    return pd.Series(equity, index=idx, name="Portfolio")

def safe_download_csv(df: pd.DataFrame) -> str:
    return df.to_csv(index=True)

# -------------------------
# Sidebar inputs
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
    help="alpha=1 => 100% risky. alpha<1 => mix risk-free. alpha>1 => leverage."
)

st.sidebar.header("Constraints")
min_w = st.sidebar.number_input("Min weight per asset", value=0.0, min_value=0.0, max_value=1.0, step=0.01, format="%.2f")
max_w = st.sidebar.number_input("Max weight per asset", value=1.0, min_value=0.0, max_value=1.0, step=0.01, format="%.2f")
simulate_n = st.sidebar.slider("Portfolios for frontier (simulation)", min_value=1000, max_value=20000, value=6000, step=1000)

st.sidebar.header("Backtest & Benchmark")
rebalance = st.sidebar.selectbox("Rebalance frequency", ["None", "Monthly", "Quarterly", "Yearly"], index=3)
benchmarks_str = st.sidebar.text_input("Benchmarks (comma-separated)", "SPY,QQQ")

if len(tickers) < 2:
    st.warning("Ajoute au moins 2 tickers.")
    st.stop()

if min_w * len(tickers) > 1.0:
    st.error("Min weight trop élevé: min_w * N > 1 (impossible).")
    st.stop()

if min_w > max_w:
    st.error("min_w ne peut pas être > max_w.")
    st.stop()

benchmarks = [b.strip().upper() for b in benchmarks_str.split(",") if b.strip()]

# -------------------------
# Data
# -------------------------
@st.cache_data
def load_prices(tickers_list, start, end):
    data = yf.download(tickers_list, start=start, end=end, auto_adjust=False, progress=False)
    if data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
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

# Min Variance always
res_mv = minimize(variance, w0, method="SLSQP", bounds=bounds, constraints=constraints_sum)
w_mv = res_mv.x if res_mv.success else w0
mv_ret, mv_vol = perf(w_mv)

# Tangency always (for CML)
res_tan = minimize(neg_sharpe, w0, method="SLSQP", bounds=bounds, constraints=constraints_sum)
w_tan = res_tan.x if res_tan.success else w0
tan_ret, tan_vol = perf(w_tan)

# -------------------------
# Efficient Frontier simulation
# -------------------------
rng = np.random.default_rng(42)

def random_weights():
    for _ in range(5000):
        w = rng.random(n)
        w = w / w.sum()
        if np.all(w >= min_w) and np.all(w <= max_w):
            return w
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
    r_, v_ = perf(w)
    sim_rets[i] = r_
    sim_vols[i] = v_
    sim_sharpes[i] = (r_ - risk_free) / v_ if v_ > 0 else np.nan

# -------------------------
# Backtest + Benchmarks
# -------------------------
equity_port = backtest_portfolio(
    prices=prices,
    weights=w_opt,
    rf_annual=risk_free,
    include_rf=include_rf,
    alpha=alpha,
    rebalance=rebalance,
    start_value=1.0,
)

equities = pd.DataFrame({"Portfolio": equity_port}).dropna()

# Benchmarks
bench_prices = None
if len(benchmarks) > 0:
    bench_prices = load_prices(benchmarks, start_date, end_date)
    if not bench_prices.empty:
        bench_prices = bench_prices.reindex(equities.index).dropna()
        for b in bench_prices.columns:
            eq = bench_prices[b] / bench_prices[b].iloc[0]
            equities[b] = eq

equities = equities.dropna()

# Metrics table
metrics = []
metrics.append(make_metrics("Portfolio", equities["Portfolio"], risk_free))
for b in benchmarks:
    if b in equities.columns:
        metrics.append(make_metrics(b, equities[b], risk_free))

metrics_df = pd.DataFrame(metrics)
# nicer formatting
metrics_view = metrics_df.copy()
for c in ["CAGR", "Volatility", "Max Drawdown"]:
    metrics_view[c] = (metrics_view[c] * 100).round(2)
metrics_view["Sharpe"] = metrics_view["Sharpe"].round(2)
metrics_view["Final Value (1.0 start)"] = metrics_view["Final Value (1.0 start)"].round(3)

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
    st.dataframe(alloc_df.sort_values("Weight", ascending=False), width='stretch')

    fig_alloc = go.Figure()
    fig_alloc.add_bar(x=alloc_df["Ticker"], y=alloc_df["Weight"], name="Weights")
    fig_alloc.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_alloc, width='stretch')

    st.subheader("Reference (risky-only)")
    st.write(f"**Tangency Return:** {tan_ret*100:.2f}%")
    st.write(f"**Tangency Vol:** {tan_vol*100:.2f}%")
    st.write(f"**Min Var Vol:** {mv_vol*100:.2f}%")

    if include_rf:
        rf_weight = 1.0 - alpha
        risky_weight_total = alpha
        cml_ret = risk_free + risky_weight_total * (tan_ret - risk_free)
        cml_vol = abs(risky_weight_total) * tan_vol
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
    fig_prices.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_prices, width='stretch')

    st.subheader("Frontier (risky-only) + CML (if enabled)")
    fig_front = go.Figure()
    fig_front.add_trace(
        go.Scatter(
            x=sim_vols,
            y=sim_rets,
            mode="markers",
            name="Risky portfolios",
            marker=dict(size=5, opacity=0.6, color=sim_sharpes, colorscale="Viridis", showscale=True),
        )
    )
    fig_front.add_trace(go.Scatter(x=[mv_vol], y=[mv_ret], mode="markers", name="Min Variance", marker=dict(size=12)))
    fig_front.add_trace(go.Scatter(x=[tan_vol], y=[tan_ret], mode="markers", name="Max Sharpe (Tangency)", marker=dict(size=12)))
    fig_front.add_trace(go.Scatter(x=[opt_vol], y=[opt_ret], mode="markers", name="Selected", marker=dict(size=14, symbol="x")))

    if include_rf and tan_vol > 0:
        sigma_max = float(max(sim_vols.max(), tan_vol) * 1.2)
        sigma_line = np.linspace(0, sigma_max, 100)
        cml_line = risk_free + ((tan_ret - risk_free) / tan_vol) * sigma_line
        fig_front.add_trace(go.Scatter(x=sigma_line, y=cml_line, mode="lines", name="CML (Rf → Tangency)"))

    fig_front.update_layout(
        xaxis_title="Annualized Volatility",
        yaxis_title="Annualized Return",
        height=380,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig_front, width='stretch')

# -------------------------
# Backtest section
# -------------------------
st.divider()
st.subheader("📉 Backtest (Portfolio vs Benchmarks)")

if equities.empty or equities.shape[0] < 20:
    st.warning("Backtest indisponible: pas assez de points ou benchmarks vides.")
else:
    fig_bt = go.Figure()
    for col in equities.columns:
        fig_bt.add_trace(go.Scatter(x=equities.index, y=equities[col], mode="lines", name=col))
    fig_bt.update_layout(
        yaxis_title="Equity (start = 1.0)",
        height=420,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig_bt, width='stretch')

    st.write("**Performance stats** (en % sauf Sharpe & Final Value)")
    st.dataframe(metrics_view, width='stretch')

# -------------------------
# Export section
# -------------------------
st.divider()
st.subheader("📦 Export")

# 1) Weights
weights_export = alloc_df.sort_values("Weight", ascending=False).reset_index(drop=True)
st.download_button(
    "⬇️ Download Weights (CSV)",
    data=weights_export.to_csv(index=False),
    file_name="weights.csv",
    mime="text/csv",
)

# 2) Equity curves
if not equities.empty:
    st.download_button(
        "⬇️ Download Equity Curves (CSV)",
        data=equities.to_csv(index=True),
        file_name="equity_curves.csv",
        mime="text/csv",
    )

# 3) Metrics
st.download_button(
    "⬇️ Download Metrics (CSV)",
    data=metrics_df.to_csv(index=False),
    file_name="metrics.csv",
    mime="text/csv",
)

# 4) Simple HTML report
report_html = f"""
<html>
<head><meta charset="utf-8"><title>Portfolio Report</title></head>
<body>
<h2>Portfolio Optimizer Report</h2>
<h3>Inputs</h3>
<ul>
  <li>Tickers: {", ".join(tickers)}</li>
  <li>Period: {start_date} → {end_date}</li>
  <li>Risk-free (annual): {risk_free:.3f}</li>
  <li>Objective: {objective}</li>
  <li>Constraints: min_w={min_w:.2f}, max_w={max_w:.2f}</li>
  <li>Include risk-free (CML): {include_rf}</li>
  <li>Alpha (risky exposure): {alpha:.1f}</li>
  <li>Rebalance: {rebalance}</li>
  <li>Benchmarks: {", ".join(benchmarks)}</li>
</ul>

<h3>Optimized Allocation</h3>
{weights_export.to_html(index=False)}

<h3>Performance Metrics</h3>
{metrics_view.to_html(index=False)}

<p style="margin-top:20px; font-size:12px;">
Data: yfinance • Model: Markowitz (MPT) + CML (CAPM). Past performance is not indicative of future results.
</p>
</body>
</html>
"""

st.download_button(
    "⬇️ Download Report (HTML)",
    data=report_html,
    file_name="report.html",
    mime="text/html",
)

st.caption("Données: yfinance • MPT (Markowitz) + CML (CAPM) • Backtest simple (valeur initiale=1.0) • Exports CSV/HTML")
