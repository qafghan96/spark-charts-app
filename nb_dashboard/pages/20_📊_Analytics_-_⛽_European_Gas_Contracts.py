import io
import os
import sys
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Ensure we can import sibling module "utils.py" when run as a Streamlit page
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
APP_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if APP_ROOT not in sys.path:
    sys.path.insert(0, APP_ROOT)

from utils import (
    get_credentials,
    get_access_token,
    api_get,
    add_axis_controls,
    add_color_controls,
    apply_axis_limits,
)

st.title("⛽ European Gas Contracts")
st.caption(
    "Plot SparkLEBA European gas contract prices (TTF, THE, THE-TTF spread) "
    "with daily high/low confidence bands. Fetches both front (-f) and forward (-fo) tickers."
)

# --- Credentials ---
client_id, client_secret = get_credentials()
if not client_id or not client_secret:
    st.error(
        "Missing Spark API credentials. Set streamlit secrets "
        "'spark.client_id' and 'spark.client_secret' (or env vars)."
    )
    st.stop()

token = get_access_token(client_id, client_secret)

# --- Ticker & unit mappings ---
GAS_TYPE_MAP = {
    "TTF":     ("sparkleba-ttf-f",     "sparkleba-ttf-fo"),
    "THE":     ("sparkleba-the-f",     "sparkleba-the-fo"),
    "THE-TTF": ("sparkleba-the-ttf-f", "sparkleba-the-ttf-fo"),
}

UNIT_MAP = {
    "EUR/MWh":   "eur-per-mwh",
    "USD/MMBtu": "usd-per-mmbtu",
}

# --- Fetch helper ---
def fetch_gas_contracts(
    access_token: str, ticker: str, unit: str, start: str, end: str
) -> pd.DataFrame:
    uri = f"/v1.0/gas/contracts/{ticker}/?unit={unit}&start={start}&end={end}"
    raw = api_get(uri, access_token, format="csv")
    if not raw:
        return pd.DataFrame()
    df = pd.read_csv(io.StringIO(raw.decode("utf-8")))
    df["ReleaseDate"] = pd.to_datetime(df["ReleaseDate"])
    return df


# --- Configuration controls ---
st.subheader("Configuration")

today = datetime.today().date()
one_year_ago = today - timedelta(days=365)

col1, col2, col3, col4 = st.columns(4)

with col1:
    gas_type = st.selectbox(
        "Gas Price",
        options=["TTF", "THE", "THE-TTF"],
        index=0,
        help="Selects the SparkLEBA gas price hub.",
    )

with col2:
    unit_label = st.selectbox(
        "Unit",
        options=["EUR/MWh", "USD/MMBtu"],
        index=0,
    )

with col3:
    start_date = st.date_input(
        "Start Date",
        value=one_year_ago,
        help="Start of data range (API allows a maximum span of 1 year).",
    )

with col4:
    end_date = st.date_input(
        "End Date",
        value=today,
    )

if (end_date - start_date).days > 365:
    st.warning("⚠️ Date range exceeds 1 year. The API allows a maximum span of 1 year.")

if st.button("Fetch Data", type="primary"):
    ticker_f, ticker_fo = GAS_TYPE_MAP[gas_type]
    unit = UNIT_MAP[unit_label]
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    with st.spinner("Fetching data..."):
        try:
            df_f = fetch_gas_contracts(token, ticker_f, unit, start_str, end_str)
            df_fo = fetch_gas_contracts(token, ticker_fo, unit, start_str, end_str)
            total_df = pd.concat([df_f, df_fo], ignore_index=True)

            if total_df.empty:
                st.error("No data returned. Check your date range or credentials.")
                st.stop()

            st.session_state["gas_df"] = total_df
            st.session_state["gas_unit_label"] = unit_label
            st.session_state["gas_type"] = gas_type
            st.success(f"✅ Fetched {len(total_df):,} rows for SparkLEBA-{gas_type}.")

        except Exception as e:
            st.error(f"Error fetching data: {e}")
            st.stop()


# --- Chart section (shown after data is fetched) ---
if "gas_df" in st.session_state:
    total_df: pd.DataFrame = st.session_state["gas_df"]
    stored_unit: str = st.session_state["gas_unit_label"]
    stored_gas_type: str = st.session_state["gas_type"]

    st.subheader("Chart Configuration")

    available_periods = sorted(total_df["PeriodName"].unique().tolist())

    col1, col2 = st.columns([2, 1])

    with col1:
        selected_period = st.selectbox(
            "Period Name",
            options=available_periods,
            help="Select the contract period to plot (e.g. Apr26, Cal27).",
        )

    with col2:
        ticker_filter = st.selectbox(
            "Ticker Filter",
            options=["Both", "F (Front)", "Fo (Forward)"],
            index=0,
            help=(
                "Filter to a specific ticker. 'Both' combines -f and -fo data, "
                "which is useful when a period transitions between tenors."
            ),
        )

    series_label = f"SparkLEBA-{stored_gas_type} ({selected_period})"
    color_controls = add_color_controls([series_label], ["#4caad2"], expanded=True)
    axis_controls = add_axis_controls(expanded=True)

    if st.button("Generate Chart", type="primary"):
        df_period = total_df[total_df["PeriodName"] == selected_period].copy()

        # Apply ticker filter
        ticker_f_name, ticker_fo_name = GAS_TYPE_MAP[stored_gas_type]
        if ticker_filter == "F (Front)":
            df_period = df_period[
                df_period["TickerName"].str.lower() == ticker_f_name
            ]
        elif ticker_filter == "Fo (Forward)":
            df_period = df_period[
                df_period["TickerName"].str.lower() == ticker_fo_name
            ]

        if df_period.empty:
            st.warning(
                f"No data for period '{selected_period}' with the selected ticker filter."
            )
            st.stop()

        df_period = df_period.sort_values("ReleaseDate").reset_index(drop=True)

        plot_color = color_controls[series_label]

        # Split by tenor
        ticker_f_name, ticker_fo_name = GAS_TYPE_MAP[stored_gas_type]
        df_f_plot  = df_period[df_period["TickerName"].str.lower() == ticker_f_name].copy()
        df_fo_plot = df_period[df_period["TickerName"].str.lower() == ticker_fo_name].copy()

        sns.set_theme(style="whitegrid")
        fig, ax = plt.subplots(figsize=(16, 7))

        # Plot -fo (Forward) series — dashed, small square markers
        if not df_fo_plot.empty:
            fo_label = f"SparkLEBA-{stored_gas_type}-Fo ({selected_period})"
            ax.plot(
                df_fo_plot["ReleaseDate"], df_fo_plot["Close"],
                linewidth=2.5, color=plot_color,
                linestyle="--", marker="s", markersize=4,
                label=fo_label,
            )
            ax.plot(df_fo_plot["ReleaseDate"], df_fo_plot["DailyHigh"], color=plot_color, alpha=0.1)
            ax.plot(df_fo_plot["ReleaseDate"], df_fo_plot["DailyLow"],  color=plot_color, alpha=0.1)
            ax.fill_between(
                df_fo_plot["ReleaseDate"], df_fo_plot["DailyLow"], df_fo_plot["DailyHigh"],
                alpha=0.2, color=plot_color,
            )
            

        # Plot -f (Front) series — solid, larger square markers
        if not df_f_plot.empty:
            f_label = f"SparkLEBA-{stored_gas_type}-F ({selected_period})"
            ax.plot(
                df_f_plot["ReleaseDate"], df_f_plot["Close"],
                linewidth=2.5, color=plot_color,
                linestyle="-", marker="s", markersize=6,
                label=f_label,
            )
            ax.plot(df_f_plot["ReleaseDate"], df_f_plot["DailyHigh"], color=plot_color, alpha=0.1)
            ax.plot(df_f_plot["ReleaseDate"], df_f_plot["DailyLow"],  color=plot_color, alpha=0.1)
            ax.fill_between(
                df_f_plot["ReleaseDate"], df_f_plot["DailyLow"], df_f_plot["DailyHigh"],
                alpha=0.2, color=plot_color,
            )
            ax.scatter(
                df_f_plot["ReleaseDate"].iloc[-1], df_f_plot["Close"].iloc[-1],
                color=plot_color, marker="o", s=120, zorder=5,
            )

        ax.set_xlabel("Release Date")
        ax.set_ylabel(stored_unit)
        ax.set_title(f"SparkLEBA-{stored_gas_type} – {selected_period}", fontsize=14)

        apply_axis_limits(
            ax, axis_controls,
            data_df=df_period,
            y_cols=["Close", "DailyHigh", "DailyLow"],
        )

        sns.despine(left=True, bottom=True)
        ax.legend()

        st.pyplot(fig)

        # --- Stats ---
        st.subheader("Statistics")

        latest_close = df_period["Close"].iloc[-1]
        latest_date = df_period["ReleaseDate"].iloc[-1]
        prev_close = df_period["Close"].iloc[-2] if len(df_period) > 1 else None

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Latest Close",
                f"{latest_close:.3f}",
                help=f"As of {latest_date.strftime('%Y-%m-%d')}",
            )

        with col2:
            if prev_close is not None:
                change = latest_close - prev_close
                pct = change / prev_close * 100
                st.metric(
                    "Change (prev. day)",
                    f"{change:+.3f}",
                    delta=f"{pct:+.2f}%",
                )
            else:
                st.metric("Change (prev. day)", "N/A")

        with col3:
            st.metric("Period High", f"{df_period['Close'].max():.3f}")

        with col4:
            st.metric("Period Low", f"{df_period['Close'].min():.3f}")

        # --- Data preview ---
        st.subheader("Data Preview")
        preview_cols = [
            "ReleaseDate", "TickerName", "PeriodName", "PeriodIndex",
            "DailyHigh", "DailyLow", "Close",
        ]
        st.dataframe(
            df_period[preview_cols].sort_values("ReleaseDate", ascending=False),
            use_container_width=True,
        )
