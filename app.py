# app.py
import os
import io
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

import streamlit as st

# import your core BSM functions (must exist)
from src.bsm.core import bsm_price, bsm_greeks, implied_vol

# ========= Global page config / theme =========
st.set_page_config(
    page_title="BSM Explorer",
    page_icon="📈",
    layout="wide",         # makes app use more horizontal space
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    /* make headers a bit bigger & add spacing */
    .css-1d391kg h1 {font-size:38px;}
    .stSidebar .css-1d391kg { padding-top: 8px; }
    .stButton>button { border-radius: 10px; padding: .5rem 1rem; }
    /* card-like panels for dataframes / charts */
    .stDataFrame, .stTable { border-radius: 12px; padding: 6px; box-shadow: 0 6px 18px rgba(0,0,0,0.45); }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------- LIVE DATA HELPERS & UI (PASTE HERE) ----------------------
import pandas as pd
import numpy as np
# ---------- realtime / yahoo helper (robust) ----------
import datetime as dt
try:
    import yfinance as yf
except Exception:
    yf = None

def fetch_yahoo_option_chain(ticker: str):
    """
    Fetch nearest option expiry strikes and mid prices from Yahoo via yfinance.
    Returns:
      dict with keys: 'S' (spot), 'Ks_arr' (np.array of strikes),
                     'market_prices' (np.array), 'T' (time to expiry in years)
    Raises RuntimeError on failure with a helpful message.
    """
    if yf is None:
        raise RuntimeError("yfinance package not installed. Run: pip install yfinance")

    tk = yf.Ticker(ticker)
    if tk is None:
        raise RuntimeError(f"yfinance could not create ticker for {ticker}")

    # get today's date and list of expirations
    today = dt.datetime.utcnow().date()
    exps = tk.options  # list of expiry strings like '2025-12-19'
    if not exps:
        raise RuntimeError("No option expiries returned by Yahoo for this ticker")

    # pick the nearest expiry (first in list)
    nearest_exp = exps[0]
    opt_chain = tk.option_chain(nearest_exp)

    calls = opt_chain.calls
    puts  = opt_chain.puts

    # prefer calls if present; use calls' strike and mid price
    if not calls.empty:
        df = calls
    elif not puts.empty:
        df = puts
    else:
        raise RuntimeError("No calls or puts data returned for nearest expiry")

    # ensure columns exist
    if ("strike" not in df.columns) or ("lastPrice" not in df.columns):
        raise RuntimeError("Unexpected option CSV from yfinance (missing strike/lastPrice)")

    # compute mid price: use lastPrice if available; else use mid of bid/ask
    import numpy as np
    strikes = np.asarray(df["strike"].values, dtype=float)

    # try best available price
    if "lastPrice" in df.columns and df["lastPrice"].notna().any():
        prices = np.asarray(df["lastPrice"].fillna(0.0).values, dtype=float)
        # fallback: if lastPrice is zero, try mid of bid/ask
        zero_mask = prices == 0
        if zero_mask.any() and ("bid" in df.columns and "ask" in df.columns):
            mid = (np.asarray(df["bid"].fillna(0.0)) + np.asarray(df["ask"].fillna(0.0))) / 2.0
            prices[zero_mask] = mid[zero_mask]
    else:
        # compute mid from bid/ask
        if ("bid" in df.columns and "ask" in df.columns):
            prices = (np.asarray(df["bid"].fillna(0.0)) + np.asarray(df["ask"].fillna(0.0))) / 2.0
        else:
            raise RuntimeError("No usable price columns (lastPrice or bid/ask) in options data")

    # Spot price: take recent close from history
    hist = tk.history(period="1d")
    if hist is not None and not hist.empty and "Close" in hist.columns:
        spot = float(hist["Close"].iloc[-1])
    else:
        # fallback to info
        spot = float(tk.info.get("regularMarketPrice", 0.0) or 0.0)

    # compute T (years) from today until expiry
    exp_dt = dt.datetime.strptime(nearest_exp, "%Y-%m-%d").date()
    days = (exp_dt - today).days
    if days <= 0:
        T = 0.0
    else:
        T = days / 365.25

    return {"S": spot, "Ks_arr": strikes, "market_prices": prices, "T": T}


# UI wiring: add inputs in the Settings area of your app
def realtime_ui_block():
    """
    Call this from your settings page — it renders realtime controls and returns a dict
    {'enabled': bool, 'source': 'yahoo'|'nse'|'broker', 'ticker': str, 'api_key': str}
    """
    st.subheader("Real-time / API settings")
    real_enabled = st.checkbox("Enable real-time option chain fetch (requires API setup)", value=False)
    col1, col2 = st.columns([2, 3])
    with col1:
        source = st.selectbox("Source", options=["yahoo", "nsepy (todo)", "broker (todo)"], index=0)
        ticker = st.text_input("Ticker / Symbol (for Yahoo/Broker)", value="AAPL" if source == "yahoo" else "")
    with col2:
        api_key = st.text_input("API key (store here for local testing)", value="", type="password")
        expiry_pick = st.text_input("Expiry (YYYY-MM-DD) optional", value="")  # optional expiry if you want to lock

    # manual fetch button (and auto-fetch if toggle is on)
    fetch_button = st.button("Fetch live option chain")
    return {
        'enabled': real_enabled,
        'source': source,
        'ticker': ticker.strip(),
        'api_key': api_key.strip(),
        'expiry': expiry_pick.strip(),
        'fetch': fetch_button
    }

# Integration helper: call this from main flow (Explorer page) after you have st.session_state defaults set
def try_fetch_and_apply(settings, r_defaults):
    """
    settings: dict returned by realtime_ui_block()
    r_defaults: fallback dict with existing S,Ks,prices,T to use if fetch fails
    This function will set st.session_state['S'], ['Ks_arr'], ['market_prices'], ['T'], ['expiry'] on success
    """
    if not settings['enabled'] and not settings['fetch']:
        return False, "not requested"

    try:
        if settings['source'] == 'yahoo':
            if not settings['ticker']:
                return False, "ticker empty"
            payload = fetch_yahoo_option_chain(settings['ticker'], expiry_date=settings['expiry'] or None)
            # apply to session_state so rest of app uses these values
            st.session_state['S'] = payload['S']
            st.session_state['Ks_arr'] = payload['Ks']
            st.session_state['market_prices'] = payload['prices']
            st.session_state['T'] = payload['T']
            st.session_state['expiry'] = payload['expiry']
            return True, f"fetched expiry {payload['expiry']}"
        else:
            # placeholder for NSEpy / broker
            return False, "source not implemented"
    except Exception as e:
        return False, str(e)

# NOTE: end of live helpers
# -------------------------------------------------------------------------------


# ---------------------------
# Utility helpers
# ---------------------------
def ensure_1d(a):
    a = np.asarray(a)
    return a if a.ndim > 0 else np.array([a])

def compute_greeks_grid(Ks, sigmas, S, T, r, q, option="call"):
    """
    Compute a grid of a chosen greek (e.g. 'delta_call' or 'vega') across strikes (Ks)
    and volatilities (sigmas). We return a DataFrame indexed by Ks with columns = sigmas.
    """
    Ks = np.asarray(Ks)
    sigmas = np.asarray(sigmas)
    grid = np.empty((len(Ks), len(sigmas)))
    for i, K in enumerate(Ks):
        for j, sigma in enumerate(sigmas):
            # call bsm_greeks returns a dict of greeks for scalars
            g = bsm_greeks(S, K, T, r, q, sigma)
            # pick a representative greek, but we'll pass greek key from UI
            # placeholder: leave value assignment to caller
            grid[i, j] = g['vega']  # default; caller may override
    df = pd.DataFrame(grid, index=Ks, columns=np.round(sigmas, 4))
    return df

def compute_selected_greek_grid(Ks, sigmas, S, T, r, q, greek_key, option="call"):
    Ks = np.asarray(Ks)
    sigmas = np.asarray(sigmas)
    grid = np.empty((len(Ks), len(sigmas)))
    for i, K in enumerate(Ks):
        for j, sigma in enumerate(sigmas):
            g = bsm_greeks(S, K, T, r, q, sigma)
            grid[i, j] = g.get(greek_key, np.nan)
    df = pd.DataFrame(grid, index=Ks, columns=np.round(sigmas, 4))
    return df

def plot_heatmap(df: pd.DataFrame, title: str):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df, ax=ax, cmap="viridis", cbar_kws={'label': 'Greek value'})
    ax.set_title(title)
    ax.set_xlabel("Volatility")
    ax.set_ylabel("Strike")
    plt.tight_layout()
    return fig

def plot_greek_vs_sigma(Ks, sigmas, S, T, r, q, greek_key, option="call"):
    """
    For each strike in Ks, compute the greek across sigmas and plot.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    for K in Ks:
        vals = [bsm_greeks(S, K, T, r, q, sigma).get(greek_key, np.nan) for sigma in sigmas]
        ax.plot(sigmas, vals, label=f"K={K:.0f}", alpha=0.6)
    ax.set_xlabel("Volatility (sigma)")
    ax.set_ylabel(greek_key)
    ax.set_title(f"{greek_key} vs sigma for strikes")
    ax.legend(fontsize='small', ncol=2)
    plt.tight_layout()
    return fig

def save_report_pdf(out_path: str, table_df: pd.DataFrame, figs: list):
    """
    Save table (as matplotlib table) + list of matplotlib figures into a multipage PDF
    """
    with PdfPages(out_path) as pdf:
        # first page: table as figure
        fig_table, ax = plt.subplots(figsize=(10, 6))
        ax.axis('off')
        tbl = ax.table(cellText=table_df.round(6).values,
                       colLabels=table_df.columns,
                       rowLabels=table_df.index,
                       loc='center')
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        plt.title("Recovered IV / Prices table")
        plt.tight_layout()
        pdf.savefig(fig_table)
        plt.close(fig_table)

        # save each figure
        for fig in figs:
            pdf.savefig(fig)
            plt.close(fig)

# ---------------------------
# Streamlit app structure
# ---------------------------
st.set_page_config(layout="wide", page_title="BSM Explorer (extended)")



# global model inputs (left panel in Explorer)
# -------------------------
# Inputs (put these near top, before any Greeks/sensitivity computations)
# -------------------------
# ---------------------------
# Sidebar + presets + safe defaults
# (Replace your existing sidebar/presets block with this)
# ---------------------------

import streamlit as st
import numpy as np
from pathlib import Path
import json   # used later by settings/presets code

# ensure presets folder exists (used by Settings page)
PRESETS_DIR = Path("presets")
PRESETS_DIR.mkdir(exist_ok=True)

# ---------------------------
# Sidebar inputs (these MUST be defined before any "default_" variables)
# Put these near the top (left-panel inputs used by Explorer)
# ---------------------------
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Go to", ["Explorer", "Greeks Sensitivity", "Upload Market CSV", "Settings"], index=0)

S = st.sidebar.number_input("Spot Price (S)", value=100.00, step=1.0, format="%.2f")
K = st.sidebar.number_input("Strike (K)", value=100.00, step=1.0, format="%.2f")
T = st.sidebar.number_input("Time to expiry (years)", value=0.5, min_value=1e-6, format="%.4f")
r = st.sidebar.number_input("Risk-free rate (r)", value=0.01, format="%.4f")
q = st.sidebar.number_input("Dividend yield (q)", value=0.0, format="%.4f")
sigma = st.sidebar.number_input("Volatility (σ)", value=0.20, min_value=1e-6, format="%.6f")

# strike grid used by sensitivity panels (keeps consistent across pages)
Ks = np.linspace(60, 140, 17)

# Greeks sensitivity controls
bump_pct = st.sidebar.number_input("Bump percent for sensitivity (as fraction)", value=0.001, format="%.6f")
option_type = st.sidebar.selectbox("Option Type", options=["call", "put"], index=0)

# ---------------------------
# Safe defaults (use saved session values if present, else use current sidebar inputs)
# Put these AFTER the sidebar definitions above so S, K, ... exist
# ---------------------------
default_S     = st.session_state.get("settings_S", S)
default_K     = st.session_state.get("settings_K", K)
default_T     = st.session_state.get("settings_T", T)
default_r     = st.session_state.get("settings_r", r)
default_q     = st.session_state.get("settings_q", q)
default_sigma = st.session_state.get("settings_sigma", sigma)
default_bump_pct = st.session_state.get("settings_bump_pct", bump_pct)

# (Optional) now write these defaults back to session_state if you want them available elsewhere:
# st.session_state.setdefault("settings_S", default_S)
# st.session_state.setdefault("settings_K", default_K)
# ...etc

# ---------------------------
# End of replaceable block
# ---------------------------



# Explorer Page
if page == "Explorer":
    st.title("BSM Explorer – Pricing • Greeks • Implied Volatility")

    price = bsm_price(S, K, T, r, q, sigma, option=option_type)
    greeks = bsm_greeks(S, K, T, r, q, sigma)

    st.metric("Option Price (call)" if option_type == "call" else "Option Price (put)", f"{price:.6f}")
    st.write("Greeks (model σ):")
    st.table(pd.Series(greeks).rename("value"))

    st.header("Implied Volatility")
    st.write("Recovered implied volatility for (model price):", implied_vol(price, S, K, T, r, q))

    st.markdown("---")
    st.subheader("IV Smile Demo (synthetic)")

    # quick synthetic example (similar to your notebook)
    Ks = np.linspace(60, 140, 17)
    true_sigma = 0.20 + 0.10 * ((Ks - S) / (S * 0.4)) ** 2
    true_sigma = np.clip(true_sigma, 0.01, 2.0)
    prices = bsm_price(S, Ks, T, r, q, true_sigma, option="call")
    recovered_iv = implied_vol(prices, S, Ks, T, r, q, option="call")

    df = pd.DataFrame({"Strike": Ks, "true_sigma": np.round(true_sigma, 6),
                       "price": np.round(prices, 6), "implied_vol": np.round(recovered_iv, 6)})
    st.dataframe(df.set_index("Strike"))

    # plot the IV smile
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(Ks, true_sigma, label="true IV", marker="o")
    ax.plot(Ks, recovered_iv, label="recovered IV", marker="x")
    ax.set_xlabel("Strike (K)")
    ax.set_ylabel("Implied Volatility")
    ax.legend()
    st.pyplot(fig)

# ---------------------------
# Greeks Sensitivity page
# ---------------------------
elif page == "Greeks Sensitivity":
    st.header("Greeks Sensitivity & Heatmaps")
    # controls
    S = st.number_input("Spot (S)", value=100.0, format="%.2f")
    T = st.number_input("T (yrs)", value=0.5, format="%.3f")
    r = st.number_input("r", value=0.01, format="%.4f")
    q = st.number_input("q", value=0.0, format="%.4f")
    greek_key = st.selectbox("Greek to visualize", ["delta_call", "delta_put", "gamma", "vega", "theta_call", "theta_put", "rho_call", "rho_put"])
    Ks = np.linspace(60, 140, 17)
    sigmas = np.linspace(0.05, 0.4, 20)

    st.write("Heatmap: greek across (strike x sigma)")
    df_heat = compute_selected_greek_grid(Ks, sigmas, S, T, r, q, greek_key)
    fig_h = plot_heatmap(df_heat, title=f"{greek_key} heatmap")
    st.pyplot(fig_h)

    st.write("Sensitivity lines (greek vs sigma for each strike)")
    fig_lines = plot_greek_vs_sigma([70, 90, 100, 110, 130], sigmas, S, T, r, q, greek_key)
    st.pyplot(fig_lines)

# ---------------------------
# Upload Market CSV page
# ---------------------------
elif page == "Upload Market CSV":
    st.header("Upload market CSV and auto-fit IV")
    st.write("CSV expected: columns -> Strike, MarketPrice  (or close). We'll recover IV for each Strike.")
    uploaded = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded:
        df_in = pd.read_csv(uploaded)
        st.write("Preview:", df_in.head())
        # try to find columns
        if "Strike" not in df_in.columns:
            st.error("CSV must contain 'Strike' column.")
        else:
            # assume market price in 'price' or 'MarketPrice' or second column
            price_col = None
            for c in ["price", "Price", "MarketPrice", "market_price", "mid"]:
                if c in df_in.columns:
                    price_col = c
                    break
            if price_col is None:
                # fallback pick second numeric column
                numeric_cols = df_in.select_dtypes(include=[np.number]).columns.tolist()
                numeric_cols = [c for c in numeric_cols if c != "Strike"]
                if numeric_cols:
                    price_col = numeric_cols[0]
                else:
                    st.error("No price column could be found. Please include a numeric column for market price.")
            if price_col:
                Ks = np.asarray(df_in["Strike"])
                prices = np.asarray(df_in[price_col])
                # recover iv
                ivs = implied_vol(prices, S=np.mean(Ks), K=Ks, T=T, r=r, q=q, option="call")
                df_in["implied_vol"] = ivs
                st.write(df_in)
                st.success("Recovered implied vol added as 'implied_vol' column.")
                # show IV smile
                fig, ax = plt.subplots(figsize=(7,4))
                ax.plot(Ks, ivs, marker="o")
                ax.set_xlabel("Strike")
                ax.set_ylabel("Implied vol")
                st.pyplot(fig)
                # allow download csv
                csv_bytes = df_in.to_csv(index=False).encode('utf-8')
                st.download_button("Download with IV", data=csv_bytes, file_name="market_with_iv.csv", mime="text/csv")

# ---------------------------
# Export Report page
# ---------------------------
elif page == "Export Report":
    st.header("Export CSV / PDF report")
    st.write("We will create a CSV and a multi-page PDF containing the IV table and the charts.")
    # default synthetic table (re-use previous example)
    Ks = np.linspace(60, 140, 17)
    S = 100.0; T = 0.5; r = 0.01; q = 0.0
    true_sigma = 0.20 + 0.10 * ((Ks - S) / (S * 0.4)) ** 2
    true_sigma = np.clip(true_sigma, 0.01, 2.0)
    prices = bsm_price(S, Ks, T, r, q, true_sigma, option="call")
    recovered_iv = implied_vol(prices, S, Ks, T, r, q, option="call")

    df_table = pd.DataFrame({"Strike": Ks, "true_sigma": np.round(true_sigma, 6),
                             "price": np.round(prices, 6), "implied_vol": np.round(recovered_iv, 6)}).set_index("Strike")

    st.write("Preview table:")
    st.dataframe(df_table)

    if st.button("Export CSV"):
        fn_csv = os.path.join("outputs", "recovered_iv.csv")
        os.makedirs("outputs", exist_ok=True)
        df_table.to_csv(fn_csv)
        with open(fn_csv, "rb") as f:
            st.download_button("Download CSV", data=f, file_name="recovered_iv.csv", mime="text/csv")
    if st.button("Export PDF (table + plots)"):
        os.makedirs("outputs", exist_ok=True)
        figs = []
        # create a plot
        fig_smile, ax = plt.subplots(figsize=(8,4))
        ax.plot(Ks, true_sigma, label="true_iv", marker="o")
        ax.plot(Ks, recovered_iv, label="recovered_iv", marker="x")
        ax.legend(); ax.set_xlabel("Strike"); ax.set_ylabel("Implied vol")
        figs.append(fig_smile)

        # optional heatmap
        sigmas = np.linspace(0.05, 0.4, 20)
        df_heat = compute_selected_greek_grid(Ks, sigmas, S, T, r, q, "vega")
        figs.append(plot_heatmap(df_heat, "vega heatmap"))

        out_pdf = os.path.join("outputs", "bsm_report.pdf")
        save_report_pdf(out_pdf, df_table, figs)
        with open(out_pdf, "rb") as f:
            st.download_button("Download PDF report", data=f, file_name="bsm_report.pdf", mime="application/pdf")

# ---------------------------
# Settings
# ---------------------------
# -----------------------------
# Settings / Presets / Upload (REPLACE THIS ENTIRE SECTION)
# -----------------------------
else:
    st.header("Settings / Notes")
    st.info("This BSM Explorer has settings, preset save/load, CSV upload and quick export.")
    st.write("Save a preset to quickly restore a particular market / model setup. You can also upload market CSVs or fetch option chain via yfinance (optional).")

    # --- Default model inputs (editable) ---
    st.subheader("Default model inputs (editable)")
    col1, col2, col3 = st.columns(3)
    with col1:
        default_S = st.number_input("Spot Price (S)", value=100.0, step=1.0, format="%.2f", key="settings_S")
        default_K = st.number_input("Strike (K)", value=100.0, step=1.0, format="%.2f", key="settings_K")
    with col2:
        default_T = st.number_input("Time to expiry (years)", value=0.5, min_value=1e-6, format="%.6f", key="settings_T")
        default_r = st.number_input("Risk-free rate (r)", value=0.01, format="%.4f", key="settings_r")
    with col3:
        default_q = st.number_input("Dividend yield (q)", value=0.0, format="%.4f", key="settings_q")
        default_sigma = st.number_input("Volatility (σ)", value=0.20, min_value=1e-6, format="%.6f", key="settings_sigma")

    bump_pct = st.number_input("Bump percent for sensitivity (fraction)", value=0.001, format="%.6f", key="settings_bump_pct")

    st.markdown("---")

    # --- Save / Load Presets ---
    st.subheader("Save / Load Presets")
    preset_name = st.text_input("Preset name", value="my_preset_01", key="preset_name")
    save_col, load_col = st.columns(2)

    # Save preset
    with save_col:
        if st.button("Save preset"):
            preset = {
                "S": float(st.session_state.get("settings_S", default_S)),
                "K": float(st.session_state.get("settings_K", default_K)),
                "T": float(st.session_state.get("settings_T", default_T)),
                "r": float(st.session_state.get("settings_r", default_r)),
                "q": float(st.session_state.get("settings_q", default_q)),
                "sigma": float(st.session_state.get("settings_sigma", default_sigma)),
                "bump_pct": float(st.session_state.get("settings_bump_pct", bump_pct)),
            }
            fn = PRESETS_DIR / f"{preset_name}.json"
            with open(fn, "w") as f:
                json.dump(preset, f, indent=2)
            st.success(f"Preset saved: {fn.name}")

    # Load preset
    with load_col:
        preset_files = sorted([p.name for p in PRESETS_DIR.glob("*.json")])
        chosen = st.selectbox("Load preset", ["-- pick --"] + preset_files, key="load_preset_select")
        if st.button("Load selected preset"):
            if chosen and chosen != "-- pick --":
                with open(PRESETS_DIR / chosen, "r") as f:
                    d = json.load(f)
                # apply to session state (and to the sidebar inputs next rerun)
                st.session_state["settings_S"] = d.get("S", st.session_state.get("settings_S", default_S))
                st.session_state["settings_K"] = d.get("K", st.session_state.get("settings_K", default_K))
                st.session_state["settings_T"] = d.get("T", st.session_state.get("settings_T", default_T))
                st.session_state["settings_r"] = d.get("r", st.session_state.get("settings_r", default_r))
                st.session_state["settings_q"] = d.get("q", st.session_state.get("settings_q", default_q))
                st.session_state["settings_sigma"] = d.get("sigma", st.session_state.get("settings_sigma", default_sigma))
                st.session_state["settings_bump_pct"] = d.get("bump_pct", st.session_state.get("settings_bump_pct", bump_pct))
                st.success(f"Loaded preset: {chosen}")
                st.experimental_rerun()  # apply inputs immediately

    # small exports (download JSON or CSV for current values)
    st.markdown("**Export / Download**")
    export_col1, export_col2 = st.columns(2)
    current_preset = {
        "S": st.session_state.get("settings_S", default_S),
        "K": st.session_state.get("settings_K", default_K),
        "T": st.session_state.get("settings_T", default_T),
        "r": st.session_state.get("settings_r", default_r),
        "q": st.session_state.get("settings_q", default_q),
        "sigma": st.session_state.get("settings_sigma", default_sigma),
        "bump_pct": st.session_state.get("settings_bump_pct", bump_pct),
    }
    json_bytes = json.dumps(current_preset, indent=2).encode("utf-8")
    with export_col1:
        st.download_button("Download JSON preset", data=json_bytes, file_name="bsm_preset.json", mime="application/json")
    with export_col2:
        # produce a tiny CSV for convenience
        csv_text = "param,value\n" + "\n".join([f"{k},{v}" for k, v in current_preset.items()])
        st.download_button("Download CSV", data=csv_text, file_name="bsm_preset.csv", mime="text/csv")

    st.markdown("---")

    # --- Realtime / API settings (optional) ---
    st.subheader("Real-time / API settings")
    enable_live = st.checkbox("Enable real-time option chain fetch (requires internet & yfinance)", value=False, key="enable_live_fetch")
    if enable_live:
        ticker = st.text_input("Ticker (Yahoo format, e.g. AAPL)", value="", key="live_ticker")
        if st.button("Fetch live chain"):
            try:
                # the helper fetch_yahoo_option_chain should be defined earlier in your file
                res = fetch_yahoo_option_chain(ticker)
                # store some results for later use
                st.session_state["live_chain"] = res
                st.success("Live chain fetched — data saved to session (live_chain).")
                # show small preview
                df_preview = pd.DataFrame({"strike": res["Ks_arr"], "price": res["market_prices"]})
                st.dataframe(df_preview.head(50))
            except Exception as e:
                st.error(f"Live fetch failed: {e}")

    st.markdown("---")

    # --- Upload market data CSV ---
    st.subheader("Upload market data (CSV)")
    st.write("CSV should contain at least columns: strike, price (names not case-sensitive).")
    uploaded = st.file_uploader("Upload option/market CSV (columns: strike, price, ...)", type=["csv"], key="market_csv_upload")
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            lower_cols = [c.lower() for c in df.columns]
            if "strike" not in lower_cols or "price" not in lower_cols:
                st.error("CSV must include columns named 'strike' and 'price' (case-insensitive).")
            else:
                # normalize column names to 'strike' and 'price'
                rename_map = {}
                for orig in df.columns:
                    if orig.lower() == "strike":
                        rename_map[orig] = "strike"
                    if orig.lower() == "price":
                        rename_map[orig] = "price"
                df = df.rename(columns=rename_map)
                st.session_state["uploaded_market_df"] = df  # store for later use
                st.success("CSV uploaded and saved to session (uploaded_market_df).")
                st.dataframe(df.head(50))
        except Exception as e:
            st.error(f"Failed to parse CSV: {e}")

    st.markdown("---")

    # --- Reset button (clear saved settings in session_state) ---
    if st.button("Reset saved inputs & outputs"):
        keys_to_clear = [k for k in st.session_state.keys() if k.startswith("settings_") or k in ("live_chain", "uploaded_market_df")]
        for k in keys_to_clear:
            del st.session_state[k]
        st.success("Cleared session saved settings/outputs. Please refresh / rerun if required.")

# ------------------------ END Settings block ------------------------

# --------------------------------------------------------------------------------------


# ===== START: Greeks Sensitivity UI & logic =====
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from utils_sens import price_sensitivity_sigma, delta_profile, delta_sensitivity_wrt_S

st.header("Greeks Sensitivity")

# sensitivity controls
st.sidebar.markdown("### Sensitivity controls")
bump_pct = st.sidebar.number_input("Bump percent for sensitivity (as fraction)", value=0.0001, step=1e-5, format="%.6f")
# We'll use bump_pct for some eps values

# assume S, Ks, T, r, q, sigma, option are already defined in your app above this block
# usually you already have S, Ks, T, r, q, sigma variables in the app

# compute arrays
# --- FIXED: Compute arrays properly using user-input values ---

Ks_arr = np.asarray(Ks)     # strike array
option_type = option_type     # user-selected option type ("call" / "put")

# Vega sensitivity (price sensitivity to volatility bump)
price_vega_numeric = price_sensitivity_sigma(
    S, Ks_arr, T, r, q, sigma,
    option=option_type,
    eps=bump_pct
)

# Delta curve (Delta values across strikes)
delta_vals = delta_profile(
    S, Ks_arr, T, r, q, sigma,
    option=option_type
)

# Delta sensitivity to small stock move (cross-greek)
delta_sens_S = delta_sensitivity_wrt_S(
    S, Ks_arr, T, r, q, sigma,
    option=option_type,
    eps=max(1e-5, bump_pct)
)



# show table
df_sens = pd.DataFrame({
    "Strike": Ks_arr,
    "delta": np.round(delta_vals, 6),
    "dDelta/dS": np.round(delta_sens_S, 6),
    "dPrice/dSigma (num)": np.round(price_vega_numeric, 6)
})
st.dataframe(df_sens.set_index("Strike"))

# Plotly charts: Delta curve + Vega numeric
fig = go.Figure()
fig.add_trace(go.Scatter(x=Ks_arr, y=delta_vals, mode='lines+markers', name='Delta'))
fig.add_trace(go.Scatter(x=Ks_arr, y=price_vega_numeric, mode='lines+markers', name='dPrice/dSigma (numeric)', yaxis='y2'))
# add second y-axis
fig.update_layout(
    title="Delta & dPrice/dSigma across Strikes",
    xaxis_title="Strike",
    yaxis=dict(title="Delta"),
    yaxis2=dict(title="dPrice/dSigma", overlaying="y", side="right")
)
st.plotly_chart(fig, use_container_width=True)
# ===== END: Greeks Sensitivity UI & logic =====
