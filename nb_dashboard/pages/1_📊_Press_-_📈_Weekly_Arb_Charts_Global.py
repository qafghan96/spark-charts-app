import io
import os
import sys
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

# Ensure we can import sibling module "utils.py" when run as a Streamlit page
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
APP_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if APP_ROOT not in sys.path:
    sys.path.insert(0, APP_ROOT)

from utils import (
    get_credentials,
    get_access_token,
    api_get,
    list_netbacks_reference,
    add_axis_controls,
    add_color_controls,
    apply_axis_limits,
)

st.title("Press - Weekly Arb Charts - Global")
st.caption("Replicates Weekly Arb Charts Global using netbacks reference and history APIs.")

client_id, client_secret = get_credentials()
if not client_id or not client_secret:
    st.error("Missing Spark API credentials. Set streamlit secrets 'spark.client_id' and 'spark.client_secret' (or env vars).")
    st.stop()

scopes = "read:netbacks,read:routes"
token = get_access_token(client_id, client_secret, scopes=scopes)

tickers, names, available_via, release_dates, _raw = list_netbacks_reference(token)
port_options = {name: (uuid, vias) for uuid, name, vias in zip(tickers, names, available_via)}

# --- New CSV-based fetch helpers ---
def fetch_netbacks_csv(
    access_token: str,
    fob_uuid: str,
    start: str,
    end: str,
    via: str | None = None,
    laden: int | None = None,
    ballast: int | None = None,
) -> pd.DataFrame:
    uri = f"/v1.0/netbacks/?fob-port={fob_uuid}&start={start}&end={end}"
    if via:
        uri += f"&via-point={via}"
    if laden:
        uri += f"&laden-congestion-days={laden}"
    if ballast:
        uri += f"&ballast-congestion-days={ballast}"
    raw = api_get(uri, access_token, format="csv")
    if not raw:
        return pd.DataFrame()
    df = pd.read_csv(io.StringIO(raw.decode("utf-8")))
    if not df.empty:
        df["ReleaseDate"] = pd.to_datetime(df["ReleaseDate"])
        for col in df.columns[4:]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def fetch_netbacks_csv_chunked(
    access_token: str,
    fob_uuid: str,
    start: datetime,
    end: datetime,
    via: str | None = None,
    laden: int | None = None,
    ballast: int | None = None,
    status_label: str = "",
) -> pd.DataFrame:
    """Fetch in ≤365-day windows as required by the API."""
    chunks = []
    chunk_start = start
    total_days = max((end - start).days, 1)
    fetched_days = 0
    progress = st.progress(0)
    status = st.empty()

    while chunk_start <= end:
        chunk_end = min(chunk_start + timedelta(days=364), end)
        status.text(
            f"{status_label} {chunk_start.strftime('%Y-%m-%d')} → {chunk_end.strftime('%Y-%m-%d')} …"
        )
        chunk = fetch_netbacks_csv(
            access_token, fob_uuid,
            chunk_start.strftime("%Y-%m-%d"),
            chunk_end.strftime("%Y-%m-%d"),
            via=via, laden=laden, ballast=ballast,
        )
        if not chunk.empty:
            chunks.append(chunk)
        fetched_days += (chunk_end - chunk_start).days + 1
        progress.progress(min(fetched_days / total_days, 1.0))
        chunk_start = chunk_end + timedelta(days=1)

    progress.empty()
    status.empty()
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()

def get_chart_df(raw_df: pd.DataFrame, port_name: str) -> pd.DataFrame:
    """
    From the full CSV response, keep only the front month (LoadMonthIndex=1)
    per release date, rename columns to match chart expectations, and sort
    most-recent first so iloc[0] is the latest point.
    """
    if raw_df.empty:
        return pd.DataFrame()
    df = raw_df[raw_df["LoadMonthIndex"] == 1].copy()
    df = df.sort_values("ReleaseDate", ascending=False).reset_index(drop=True)
    df["Release Date"] = df["ReleaseDate"]
    df["Delta Outrights"] = df["DeltaNeaNwe"]
    df["FoB Port"] = port_name
    return df


# --- Port selection UI ---
left, right = st.columns(2)
with left:
    port_a = st.selectbox("Port A 🟡", options=list(port_options.keys()), index=names.index("Bonny LNG") if "Bonny LNG" in names else 0)
    via_a = st.selectbox("Port A via-point", options=port_options[port_a][1] or ["cogh"], index=0)
with right:
    port_b = st.selectbox("Port B 🟣", options=list(port_options.keys()), index=names.index("Sabine Pass") if "Sabine Pass" in names else 0)
    via_b_options = port_options[port_b][1] or ["cogh", "panama"]
    panama_index = via_b_options.index("panama") if "panama" in via_b_options else 0
    via_b = st.selectbox("Port B via-point", options=via_b_options, index=panama_index)

include_c = st.checkbox("Include Port C", value=True)
if include_c:
    c1, c2 = st.columns(2)
    with c1:
        default_c_idx = names.index("Sabine Pass") if "Sabine Pass" in names else 0
        port_c = st.selectbox("Port C 🟢", options=list(port_options.keys()), index=default_c_idx, key="port_c")
    with c2:
        via_c = st.selectbox(
            "Port C via-point",
            options=port_options[port_c][1] or ["panama", "cogh"],
            index=0,
            key="via_c",
        )

# --- Date range inputs (replaces release count slider) ---
today = datetime.today().date()
col_s, col_e = st.columns(2)
with col_s:
    start_date = st.date_input("Start Date", value=today - timedelta(days=90))
with col_e:
    end_date = st.date_input("End Date", value=today)

start_dt = datetime.combine(start_date, datetime.min.time())
end_dt = datetime.combine(end_date, datetime.min.time())
span_days = (end_dt - start_dt).days

# --- Color & axis controls ---
series_names = [f"{port_a} ({via_a})", f"{port_b} ({via_b})"]
if include_c:
    series_names.append(f"{port_c} ({via_c})")

default_colors = ["#FFC217", "#4F41F4", "#48C38D"]
color_controls = add_color_controls(series_names, default_colors[:len(series_names)], expanded=True)
axis_controls = add_axis_controls(expanded=True)

sns.set_theme(style="whitegrid")

if st.button("Generate Chart", type="primary"):
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axhline(0, color="grey")

    uuid_a = port_options[port_a][0]
    uuid_b = port_options[port_b][0]

    with st.spinner("Fetching netbacks data…"):
        fetch = fetch_netbacks_csv_chunked if span_days > 365 else None

        if span_days > 365:
            raw_a = fetch_netbacks_csv_chunked(token, uuid_a, start_dt, end_dt, via=via_a, status_label=f"Fetching {port_a}")
            raw_b = fetch_netbacks_csv_chunked(token, uuid_b, start_dt, end_dt, via=via_b, status_label=f"Fetching {port_b}")
        else:
            raw_a = fetch_netbacks_csv(token, uuid_a, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"), via=via_a)
            raw_b = fetch_netbacks_csv(token, uuid_b, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"), via=via_b)

        df_a = get_chart_df(raw_a, port_a)
        df_b = get_chart_df(raw_b, port_b)
        df_c = pd.DataFrame()

        if include_c:
            uuid_c = port_options[port_c][0]
            if span_days > 365:
                raw_c = fetch_netbacks_csv_chunked(token, uuid_c, start_dt, end_dt, via=via_c, status_label=f"Fetching {port_c}")
            else:
                raw_c = fetch_netbacks_csv(token, uuid_c, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"), via=via_c)
            df_c = get_chart_df(raw_c, port_c)

    if df_a.empty and df_b.empty and df_c.empty:
        st.warning("No data returned for the selected ports and date range.")
    else:
        port_prices = []

        if not df_a.empty:
            color_a = color_controls[f"{port_a} ({via_a})"]
            ax.plot(df_a["Release Date"], df_a["Delta Outrights"], color=color_a, label=f"{port_a} ({via_a})", linewidth=3.0)
            ax.scatter(df_a["Release Date"].iloc[0], df_a["Delta Outrights"].iloc[0], color=color_a, s=120)
            port_prices.append({"port": f"{port_a} ({via_a})", "price": df_a["Delta Outrights"].iloc[0], "date": df_a["Release Date"].iloc[0], "color": "🟡"})

        if not df_b.empty:
            color_b = color_controls[f"{port_b} ({via_b})"]
            ax.plot(df_b["Release Date"], df_b["Delta Outrights"], color=color_b, label=f"{port_b} ({via_b})", linewidth=3.0)
            ax.scatter(df_b["Release Date"].iloc[0], df_b["Delta Outrights"].iloc[0], color=color_b, s=120)
            port_prices.append({"port": f"{port_b} ({via_b})", "price": df_b["Delta Outrights"].iloc[0], "date": df_b["Release Date"].iloc[0], "color": "🟣"})

        if include_c and not df_c.empty:
            color_c = color_controls[f"{port_c} ({via_c})"]
            ax.plot(df_c["Release Date"], df_c["Delta Outrights"], color=color_c, label=f"{port_c} ({via_c})", linewidth=3.0)
            ax.scatter(df_c["Release Date"].iloc[0], df_c["Delta Outrights"].iloc[0], color=color_c, s=120)
            port_prices.append({"port": f"{port_c} ({via_c})", "price": df_c["Delta Outrights"].iloc[0], "date": df_c["Release Date"].iloc[0], "color": "🟢"})

        # Shaded band
        ref_df = next((d for d in [df_b, df_a, df_c] if not d.empty), None)
        if ref_df is not None:
            # iloc[0] = newest, iloc[-1] = oldest (sorted descending)
            negrange = [
                ref_df["Release Date"].iloc[-1] - pd.Timedelta(20, unit="day"),
                ref_df["Release Date"].iloc[0] + pd.Timedelta(20, unit="day"),
            ]
            ax.plot(negrange, [-3.0, -3.0], color="red", alpha=0.05)
            ax.plot(negrange, [0, 0], color="red", alpha=0.05)
            ax.fill_between(negrange, 0, -3.0, color="red", alpha=0.05)

        ax.set_ylabel("$/MMBtu")
        ax.set_xlabel("Release Date")
        sns.despine(left=True, bottom=True)

        all_data = pd.concat([d for d in [df_a, df_b, df_c] if not d.empty], ignore_index=True)
        apply_axis_limits(ax, axis_controls, data_df=all_data, y_cols=["Delta Outrights"])

        if not axis_controls["x_auto"]:
            ax.set_xlim(axis_controls["x_min"], axis_controls["x_max"])
        elif ref_df is not None:
            plt.xlim([
                ref_df["Release Date"].iloc[-1] - pd.Timedelta(1, unit="day"),
                ref_df["Release Date"].iloc[0] + pd.Timedelta(9, unit="day"),
            ])

        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)

        st.session_state.port_prices = port_prices

# Display latest prices if they exist
if "port_prices" in st.session_state and st.session_state.port_prices:
    port_prices = st.session_state.port_prices

    st.subheader("Latest Release Date Prices")
    cols = st.columns(len(port_prices))
    for i, price_data in enumerate(port_prices):
        with cols[i]:
            st.metric(
                label=f"{price_data['color']} {price_data['port']}",
                value=f"${price_data['price']:.3f}/MMBtu",
                help=f"Latest price as of {price_data['date'].strftime('%Y-%m-%d')}",
            )
