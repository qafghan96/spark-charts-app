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

st.title("🏢 DES Hub Netbacks")
st.caption(
    "Download DES Hub Netback prices across European LNG terminals from the Spark API. "
    "Fetches data from `/v1.0/lng/access/des-hub-netbacks/`."
)

client_id, client_secret = get_credentials()
if not client_id or not client_secret:
    st.error("Missing Spark API credentials. Set streamlit secrets 'spark.client_id' and 'spark.client_secret' (or env vars).")
    st.stop()

scopes = "read:access"
token = get_access_token(client_id, client_secret, scopes=scopes)

# --- Helpers ---
def fetch_reference_data(access_token: str) -> pd.DataFrame:
    raw = api_get("/v1.0/lng/access/des-hub-netbacks/reference-data/", access_token, format="csv")
    return pd.read_csv(io.StringIO(raw.decode("utf-8")))

def fetch_des_hub_netbacks(
    access_token: str,
    unit: str,
    start: str,
    end: str,
    terminal_uuid: str | None = None,
) -> pd.DataFrame:
    uri = f"/v1.0/lng/access/des-hub-netbacks/?unit={unit}&start={start}&end={end}"
    if terminal_uuid:
        uri += f"&terminal-uuid={terminal_uuid}"
    raw = api_get(uri, access_token, format="csv")
    if not raw:
        return pd.DataFrame()
    df = pd.read_csv(io.StringIO(raw.decode("utf-8")))
    if not df.empty:
        numeric_cols = df.columns[8:]
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
        df["ReleaseDate"] = pd.to_datetime(df["ReleaseDate"])
    return df

def fetch_des_hub_netbacks_chunked(
    access_token: str,
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
        df_chunk = fetch_des_hub_netbacks(
            access_token, unit,
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
if "deshub_ref_df" not in st.session_state:
    with st.spinner("Loading terminal reference data…"):
        try:
            st.session_state["deshub_ref_df"] = fetch_reference_data(token)
        except Exception as e:
            st.error(f"Failed to load reference data: {e}")
            st.stop()

ref_df: pd.DataFrame = st.session_state["deshub_ref_df"]
terminal_options = ["All Terminals"] + sorted(ref_df["TerminalName"].dropna().tolist())

# --- Configuration ---
st.subheader("Configuration")

today = datetime.today().date()
one_year_ago = today - timedelta(days=365)

col1, col2, col3, col4 = st.columns(4)

with col1:
    unit_label = st.selectbox("Unit", options=["USD/MMBtu", "EUR/MWh"], index=0)
    unit = "usd-per-mmbtu" if unit_label == "USD/MMBtu" else "eur-per-mwh"

with col2:
    terminal_filter = st.selectbox("Terminal Filter", options=terminal_options, index=0)
    selected_uuid = None
    if terminal_filter != "All Terminals":
        row = ref_df[ref_df["TerminalName"] == terminal_filter]
        if not row.empty:
            selected_uuid = row["TerminalUUID"].iloc[0]

with col3:
    start_date = st.date_input("Start Date", value=one_year_ago)

with col4:
    end_date = st.date_input("End Date", value=today)

with st.expander("📋 Available Terminals", expanded=False):
    st.dataframe(ref_df, use_container_width=True)

if st.button("Fetch Data", type="primary"):
    start_dt = datetime.combine(start_date, datetime.min.time())
    end_dt = datetime.combine(end_date, datetime.min.time())
    span_days = (end_dt - start_dt).days

    with st.spinner("Fetching DES Hub Netbacks data…"):
        try:
            if span_days > 365:
                df = fetch_des_hub_netbacks_chunked(token, unit, start_dt, end_dt, selected_uuid)
            else:
                df = fetch_des_hub_netbacks(
                    token, unit,
                    start_date.strftime("%Y-%m-%d"),
                    end_date.strftime("%Y-%m-%d"),
                    selected_uuid,
                )

            if df.empty:
                st.warning("No data returned. Check your parameters.")
                st.stop()

            st.session_state["deshub_df"] = df
            st.session_state["deshub_unit"] = unit_label
            st.session_state["deshub_terminal"] = terminal_filter
            st.success(f"✅ Fetched {len(df):,} rows.")

        except Exception as e:
            st.error(f"Error fetching data: {e}")
            st.stop()

if "deshub_df" in st.session_state:
    df: pd.DataFrame = st.session_state["deshub_df"]
    stored_unit: str = st.session_state.get("deshub_unit", unit_label)
    stored_terminal: str = st.session_state.get("deshub_terminal", terminal_filter)

    st.subheader("Data")
    st.dataframe(df, use_container_width=True)

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    fname = f"des_hub_netbacks_{stored_unit.replace('/', '_')}"
    if stored_terminal != "All Terminals":
        fname += f"_{stored_terminal.replace(' ', '_')}"
    fname += ".csv"

    st.download_button("📥 Download CSV", data=csv_bytes, file_name=fname, mime="text/csv", use_container_width=True)

    st.subheader("Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Rows", f"{len(df):,}")
    c2.metric("Terminals", df["TerminalName"].nunique() if "TerminalName" in df.columns else "—")
    c3.metric("Release Dates", df["ReleaseDate"].nunique() if "ReleaseDate" in df.columns else "—")
    if "ReleaseDate" in df.columns and not df["ReleaseDate"].isna().all():
        c4.metric("Latest Release", df["ReleaseDate"].max().strftime("%Y-%m-%d"))

    # Terminal breakdown
    if "TerminalName" in df.columns and df["TerminalName"].nunique() > 1:
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        if numeric_cols:
            st.subheader("By Terminal")
            st.dataframe(
                df.groupby("TerminalName")[numeric_cols].mean().round(3),
                use_container_width=True,
            )
