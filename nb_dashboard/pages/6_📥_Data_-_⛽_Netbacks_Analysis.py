import io
import os
import sys
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
APP_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if APP_ROOT not in sys.path:
    sys.path.insert(0, APP_ROOT)

from utils import get_credentials, get_access_token, api_get

st.title("⛽ Netbacks Analysis")
st.caption(
    "Download netback data for a selected FoB port and routing option. "
    "Fetches data from `/v1.0/netbacks/` using start/end date ranges for fast retrieval."
)

client_id, client_secret = get_credentials()
if not client_id or not client_secret:
    st.error("Missing Spark API credentials. Set streamlit secrets 'spark.client_id' and 'spark.client_secret' (or env vars).")
    st.stop()

scopes = "read:netbacks,read:access,read:prices,read:routes"
token = get_access_token(client_id, client_secret, scopes=scopes)

# --- Helpers ---
def fetch_reference_data(access_token: str):
    content = api_get("/v1.0/netbacks/reference-data/", access_token)
    ports = content["data"]["staticData"]["fobPorts"]
    tickers = [p["uuid"] for p in ports]
    names = [p["name"] for p in ports]
    via_options = [p["availableViaPoints"] for p in ports]
    return tickers, names, via_options

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
        status.text(f"Fetching {chunk_start.strftime('%Y-%m-%d')} → {chunk_end.strftime('%Y-%m-%d')} …")
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

def rename_columns(df: pd.DataFrame, port_name: str) -> pd.DataFrame:
    """Rename CSV columns to user-friendly display names."""
    df = df.rename(columns={
        "ReleaseDate":          "Release Date",
        "LoadMonth":            "Month",
        "LoadDate":             "Load Date",
        "LoadMonthIndex":       "Month Index",
        "NeaNetbackOutright":   "NEA Netback (Outright)",
        "NeaNetbackTtfBasis":   "NEA Netback (TTF Basis)",
        "NeaMetaTtfPrice":      "NEA TTF Price",
        "NeaMetaTtfBasis":      "NEA TTF Basis Meta",
        "NeaMetaDesLng":        "NEA DES LNG Price",
        "NeaMetaRouteCost":     "NEA Route Cost",
        "NeaMetaVolAdj":        "NEA Volume Adjustment",
        "NweNetbackOutright":   "NWE Netback (Outright)",
        "NweNetbackTtfBasis":   "NWE Netback (TTF Basis)",
        "NweMetaTtfPrice":      "NWE TTF Price",
        "NweMetaTtfBasis":      "NWE TTF Basis Meta",
        "NweMetaDesLng":        "NWE DES LNG Price",
        "NweMetaRouteCost":     "NWE Route Cost",
        "NweMetaVolAdj":        "NWE Volume Adjustment",
        "DeltaNeaNwe":          "NEA-NWE Arb",
    })
    df.insert(0, "FoB Port", port_name)
    df["Month"] = pd.to_datetime(df["Month"], errors="coerce")
    return df


# --- Load reference data ---
if "netbacks_ref" not in st.session_state:
    with st.spinner("Loading reference data…"):
        try:
            tickers, fobPort_names, availablevia = fetch_reference_data(token)
            st.session_state["netbacks_ref"] = (tickers, fobPort_names, availablevia)
        except Exception as e:
            st.error(f"Failed to load reference data: {e}")
            st.stop()

tickers, fobPort_names, availablevia = st.session_state["netbacks_ref"]

# --- Configuration ---
st.subheader("Configuration")

today = datetime.today().date()
one_year_ago = today - timedelta(days=365)

col1, col2 = st.columns(2)
with col1:
    selected_port = st.selectbox(
        "FoB Port",
        fobPort_names,
        index=fobPort_names.index("Sabine Pass") if "Sabine Pass" in fobPort_names else 0,
    )
    port_index = fobPort_names.index(selected_port)
    available_via_points = [v for v in availablevia[port_index] if v is not None] or ["cogh"]
    selected_via = st.selectbox("Via Point", available_via_points, index=0)

with col2:
    start_date = st.date_input("Start Date", value=one_year_ago)
    end_date = st.date_input("End Date", value=today)

    with st.expander("Advanced Options"):
        laden_days = st.number_input("Laden Congestion Days", min_value=0, max_value=30, value=0, step=1)
        ballast_days = st.number_input("Ballast Congestion Days", min_value=0, max_value=30, value=0, step=1)

start_dt = datetime.combine(start_date, datetime.min.time())
end_dt = datetime.combine(end_date, datetime.min.time())
span_days = (end_dt - start_dt).days

if st.button("Fetch Netbacks Data", type="primary"):
    fob_uuid = tickers[port_index]
    laden = laden_days if laden_days > 0 else None
    ballast = ballast_days if ballast_days > 0 else None

    with st.spinner("Fetching netbacks data…"):
        try:
            if span_days > 365:
                raw_df = fetch_netbacks_csv_chunked(token, fob_uuid, start_dt, end_dt, via=selected_via, laden=laden, ballast=ballast)
            else:
                raw_df = fetch_netbacks_csv(
                    token, fob_uuid,
                    start_date.strftime("%Y-%m-%d"),
                    end_date.strftime("%Y-%m-%d"),
                    via=selected_via, laden=laden, ballast=ballast,
                )

            if raw_df.empty:
                st.warning("No data returned. Check your parameters.")
                st.stop()

            df = rename_columns(raw_df, selected_port)
            df = df.drop_duplicates()
            df = df.sort_values("Release Date", ascending=False).reset_index(drop=True)

            st.session_state["netbacks_df"] = df
            st.session_state["netbacks_port"] = selected_port
            st.session_state["netbacks_via"] = selected_via
            st.success(f"✅ Fetched {len(df):,} rows of netbacks data!")

        except Exception as e:
            st.error(f"Error fetching data: {e}")
            st.stop()

if "netbacks_df" in st.session_state and not st.session_state["netbacks_df"].empty:
    df: pd.DataFrame = st.session_state["netbacks_df"]
    stored_port: str = st.session_state.get("netbacks_port", selected_port)
    stored_via: str = st.session_state.get("netbacks_via", selected_via)

    st.subheader(f"Netbacks Data — {stored_port} (via {stored_via})")

    # Data view toggle
    view_option = st.radio(
        "View",
        ["Summary", "Full Data", "Meta Components"],
        horizontal=True,
    )

    SUMMARY_COLS = [
        "Release Date", "FoB Port", "Month", "Month Index",
        "NEA Netback (Outright)", "NEA Netback (TTF Basis)",
        "NWE Netback (Outright)", "NWE Netback (TTF Basis)",
        "NEA-NWE Arb",
    ]
    META_COLS = [
        "Release Date", "Month",
        "NEA TTF Price", "NEA TTF Basis Meta", "NEA DES LNG Price", "NEA Route Cost", "NEA Volume Adjustment",
        "NWE TTF Price", "NWE TTF Basis Meta", "NWE DES LNG Price", "NWE Route Cost", "NWE Volume Adjustment",
    ]

    if view_option == "Summary":
        st.dataframe(df[[c for c in SUMMARY_COLS if c in df.columns]], use_container_width=True)
    elif view_option == "Full Data":
        st.dataframe(df, use_container_width=True)
    else:
        st.dataframe(df[[c for c in META_COLS if c in df.columns]], use_container_width=True)
        with st.expander("📖 Meta Field Explanations"):
            st.markdown("""
            **TTF Price** — TTF gas price used in the netback calculation
            **TTF Basis Meta** — TTF basis component
            **DES LNG Price** — Delivered Ex-Ship LNG price
            **Route Cost** — Shipping/transport cost for the route
            **Volume Adjustment** — Volume-based adjustment
            *Netback = DES LNG Price − Route Cost + Volume Adjustment*
            """)

    # Download
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download CSV",
        data=csv_bytes,
        file_name=f"netbacks_{stored_port.lower().replace(' ', '_')}_{stored_via}.csv",
        mime="text/csv",
        use_container_width=True,
    )

    # Summary stats
    st.subheader("Data Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", f"{len(df):,}")
    c2.metric("Unique Release Dates", df["Release Date"].nunique())
    c3.metric("Unique Months", df["Month"].nunique() if "Month" in df.columns else "—")
    if "NEA-NWE Arb" in df.columns:
        c4.metric("Avg NEA-NWE Arb", f"${df['NEA-NWE Arb'].mean():.3f}/MMBtu")

    st.subheader("Price Statistics ($/MMBtu)")
    show_meta = st.checkbox("Show meta breakdown", value=False)
    if show_meta:
        col1, col2 = st.columns(2)
        with col1:
            st.write("**NEA Components**")
            nea_cols = [c for c in ["NEA TTF Price", "NEA TTF Basis Meta", "NEA DES LNG Price", "NEA Route Cost", "NEA Volume Adjustment", "NEA Netback (Outright)"] if c in df.columns]
            st.dataframe(df[nea_cols].describe(), use_container_width=True)
        with col2:
            st.write("**NWE Components**")
            nwe_cols = [c for c in ["NWE TTF Price", "NWE TTF Basis Meta", "NWE DES LNG Price", "NWE Route Cost", "NWE Volume Adjustment", "NWE Netback (Outright)"] if c in df.columns]
            st.dataframe(df[nwe_cols].describe(), use_container_width=True)
    else:
        stat_cols = [c for c in ["NEA Netback (Outright)", "NWE Netback (Outright)", "NEA-NWE Arb"] if c in df.columns]
        st.dataframe(df[stat_cols].describe(), use_container_width=True)
