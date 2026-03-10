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

st.title("🏭 Access Terminal Costs")
st.caption(
    "Download LNG regasification terminal costs from the Spark API. "
    "Fetches data from `/v1.0/lng/access/regas-costs/`."
)

client_id, client_secret = get_credentials()
if not client_id or not client_secret:
    st.error("Missing Spark API credentials. Set streamlit secrets 'spark.client_id' and 'spark.client_secret' (or env vars).")
    st.stop()

scopes = "read:access"
token = get_access_token(client_id, client_secret, scopes=scopes)

# --- Helpers ---
def fetch_reference_data(access_token: str) -> pd.DataFrame:
    raw = api_get("/v1.0/lng/access/regas-costs/reference-data/", access_token, format="csv")
    return pd.read_csv(io.StringIO(raw.decode("utf-8")))

def fetch_regas_costs(
    access_token: str,
    vessel_size: str,
    unit: str,
    start: str,
    end: str,
    terminal_uuid: str | None = None,
) -> pd.DataFrame:
    uri = f"/v1.0/lng/access/regas-costs/?vessel-size={vessel_size}&unit={unit}&start={start}&end={end}"
    if terminal_uuid:
        uri += f"&terminal-uuid={terminal_uuid}"
    raw = api_get(uri, access_token, format="csv")
    if not raw:
        return pd.DataFrame()
    df = pd.read_csv(io.StringIO(raw.decode("utf-8")))
    if not df.empty:
        numeric_cols = df.columns[6:]
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
        df["ReleaseDate"] = pd.to_datetime(df["ReleaseDate"])
    return df

def fetch_regas_costs_chunked(
    access_token: str,
    vessel_size: str,
    unit: str,
    start: datetime,
    end: datetime,
    terminal_uuid: str | None = None,
) -> pd.DataFrame:
    """Fetch in ≤365-day chunks to respect the API's max date span."""
    chunks = []
    chunk_start = start
    progress = st.progress(0)
    status = st.empty()
    total_days = (end - start).days
    fetched_days = 0

    while chunk_start < end:
        chunk_end = min(chunk_start + timedelta(days=364), end)
        status.text(f"Fetching {chunk_start.strftime('%Y-%m-%d')} → {chunk_end.strftime('%Y-%m-%d')} …")
        df_chunk = fetch_regas_costs(
            access_token, vessel_size, unit,
            chunk_start.strftime("%Y-%m-%d"),
            chunk_end.strftime("%Y-%m-%d"),
            terminal_uuid,
        )
        if not df_chunk.empty:
            chunks.append(df_chunk)
        fetched_days += (chunk_end - chunk_start).days
        progress.progress(min(fetched_days / total_days, 1.0))
        chunk_start = chunk_end + timedelta(days=1)

    progress.empty()
    status.empty()
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


# --- Load reference data (terminal list) ---
if "regas_ref_df" not in st.session_state:
    with st.spinner("Loading terminal reference data…"):
        try:
            st.session_state["regas_ref_df"] = fetch_reference_data(token)
        except Exception as e:
            st.error(f"Failed to load reference data: {e}")
            st.stop()

ref_df: pd.DataFrame = st.session_state["regas_ref_df"]
terminal_options = ["All Terminals"] + sorted(ref_df["TerminalName"].dropna().tolist())

# --- Configuration ---
st.subheader("Configuration")

today = datetime.today().date()
one_year_ago = today - timedelta(days=365)

col1, col2, col3, col4 = st.columns(4)

with col1:
    vessel_size = st.selectbox(
        "Vessel Size (cbm)",
        options=["174000", "160000", "145000"],
        index=0,
    )

with col2:
    unit_label = st.selectbox("Unit", options=["USD/MMBtu", "EUR/MWh"], index=0)
    unit = "usd-per-mmbtu" if unit_label == "USD/MMBtu" else "eur-per-mwh"

with col3:
    start_date = st.date_input("Start Date", value=one_year_ago)

with col4:
    end_date = st.date_input("End Date", value=today)

terminal_filter = st.selectbox("Terminal Filter", options=terminal_options, index=0)
selected_uuid = None
if terminal_filter != "All Terminals":
    row = ref_df[ref_df["TerminalName"] == terminal_filter]
    if not row.empty:
        selected_uuid = row["TerminalUUID"].iloc[0]

with st.expander("📋 Available Terminals", expanded=False):
    st.dataframe(ref_df, use_container_width=True)

if st.button("Fetch Data", type="primary"):
    start_dt = datetime.combine(start_date, datetime.min.time())
    end_dt = datetime.combine(end_date, datetime.min.time())
    span_days = (end_dt - start_dt).days

    with st.spinner("Fetching regas cost data…"):
        try:
            if span_days > 365:
                df = fetch_regas_costs_chunked(token, vessel_size, unit, start_dt, end_dt, selected_uuid)
            else:
                df = fetch_regas_costs(
                    token, vessel_size, unit,
                    start_date.strftime("%Y-%m-%d"),
                    end_date.strftime("%Y-%m-%d"),
                    selected_uuid,
                )

            if df.empty:
                st.warning("No data returned. Check your parameters.")
                st.stop()

            st.session_state["regas_df"] = df
            st.success(f"✅ Fetched {len(df):,} rows.")

        except Exception as e:
            st.error(f"Error fetching data: {e}")
            st.stop()

if "regas_df" in st.session_state:
    df: pd.DataFrame = st.session_state["regas_df"]

    st.subheader("Data")
    st.dataframe(df, use_container_width=True)

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    fname = f"regas_costs_{vessel_size}cbm_{unit}"
    if terminal_filter != "All Terminals":
        fname += f"_{terminal_filter.replace(' ', '_')}"
    fname += ".csv"

    st.download_button("📥 Download CSV", data=csv_bytes, file_name=fname, mime="text/csv", use_container_width=True)

    st.subheader("Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Rows", f"{len(df):,}")
    c2.metric("Terminals", df["TerminalName"].nunique() if "TerminalName" in df.columns else "—")
    c3.metric("Release Dates", df["ReleaseDate"].nunique() if "ReleaseDate" in df.columns else "—")
    if "ReleaseDate" in df.columns and not df["ReleaseDate"].isna().all():
        c4.metric("Latest Release", df["ReleaseDate"].max().strftime("%Y-%m-%d"))
