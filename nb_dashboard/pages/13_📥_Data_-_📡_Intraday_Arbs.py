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

st.title("📡 Intraday Arbs")
st.caption(
    "Download live and historical intraday arbitrage values by FoB port, via point and percent-hire. "
    "Fetches from `/v1.0/intraday/arbs/`. "
    "Historical calls are capped at 20 days — longer ranges are chunked automatically."
)

client_id, client_secret = get_credentials()
if not client_id or not client_secret:
    st.error("Missing Spark API credentials.")
    st.stop()

token = get_access_token(client_id, client_secret)

# --- Helpers ---
def fetch_reference_data(access_token: str) -> pd.DataFrame:
    raw = api_get("/v1.0/intraday/arbs/reference-data/", access_token, format="csv")
    return pd.read_csv(io.StringIO(raw.decode("utf-8")))

def _build_uri(endpoint: str, fob_uuid: str, via: str, percent_hire: int, unit: str,
               start: str | None = None, end: str | None = None) -> str:
    uri = (
        f"/v1.0/intraday/arbs/{endpoint}/"
        f"?fob-port={fob_uuid}&via-point={via}&percent-hire={percent_hire}&unit={unit}"
    )
    if start:
        uri += f"&start={start}"
    if end:
        uri += f"&end={end}"
    return uri

def fetch_live(access_token: str, fob_uuid: str, via: str, percent_hire: int, unit: str) -> pd.DataFrame:
    uri = _build_uri("live", fob_uuid, via, percent_hire, unit)
    raw = api_get(uri, access_token, format="csv")
    if not raw:
        return pd.DataFrame()
    df = pd.read_csv(io.StringIO(raw.decode("utf-8")))
    if not df.empty:
        df["AsOf"] = pd.to_datetime(df["AsOf"], utc=True).dt.tz_localize(None)
        df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    return df

def fetch_historical_chunk(access_token: str, fob_uuid: str, via: str, percent_hire: int,
                           unit: str, start: str, end: str) -> pd.DataFrame:
    uri = _build_uri("historical", fob_uuid, via, percent_hire, unit, start, end)
    raw = api_get(uri, access_token, format="csv")
    if not raw:
        return pd.DataFrame()
    df = pd.read_csv(io.StringIO(raw.decode("utf-8")))
    if not df.empty:
        df["AsOf"] = pd.to_datetime(df["AsOf"], utc=True).dt.tz_localize(None)
        df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    return df

def fetch_historical(access_token: str, fob_uuid: str, via: str, percent_hire: int,
                     unit: str, start: datetime, end: datetime) -> pd.DataFrame:
    chunks = []
    chunk_start = start
    total_days = max((end - start).days, 1)
    fetched_days = 0
    progress = st.progress(0)
    status = st.empty()

    while chunk_start <= end:
        chunk_end = min(chunk_start + timedelta(days=19), end)
        status.text(f"Fetching {chunk_start.strftime('%Y-%m-%d')} → {chunk_end.strftime('%Y-%m-%d')} …")
        df_chunk = fetch_historical_chunk(
            access_token, fob_uuid, via, percent_hire, unit,
            chunk_start.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d"),
        )
        if not df_chunk.empty:
            chunks.append(df_chunk)
        fetched_days += (chunk_end - chunk_start).days + 1
        progress.progress(min(fetched_days / total_days, 1.0))
        chunk_start = chunk_end + timedelta(days=1)

    progress.empty()
    status.empty()
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


# --- Load reference data ---
if "idarbs_ref_df" not in st.session_state:
    with st.spinner("Loading reference data…"):
        try:
            st.session_state["idarbs_ref_df"] = fetch_reference_data(token)
        except Exception as e:
            st.error(f"Failed to load reference data: {e}")
            st.stop()

ref_df: pd.DataFrame = st.session_state["idarbs_ref_df"]

# --- Configuration ---
st.subheader("Configuration")

col1, col2, col3 = st.columns(3)

with col1:
    unit = st.selectbox("Unit", options=["usd-per-mmbtu"], index=0)
    percent_hire = st.selectbox("Percent Hire", options=[0, 50, 100], index=2)

with col2:
    port_names = sorted(ref_df["FobPortName"].dropna().unique().tolist())
    port_name = st.selectbox("FoB Port", options=port_names, index=0)
    port_row = ref_df[ref_df["FobPortName"] == port_name]
    fob_uuid = port_row["FobPortUuid"].iloc[0]

with col3:
    via_options = sorted(ref_df[ref_df["FobPortName"] == port_name]["ViaPoint"].dropna().unique().tolist())
    via = st.selectbox("Via Point", options=via_options, index=0)

data_mode = st.radio("Data Mode", options=["Live (current forward curve)", "Historical"], horizontal=True)

today = datetime.today().date()
if data_mode == "Historical":
    col_s, col_e = st.columns(2)
    with col_s:
        start_date = st.date_input("Start Date", value=today - timedelta(days=30))
    with col_e:
        end_date = st.date_input("End Date", value=today)

with st.expander("📋 Reference Data", expanded=False):
    st.dataframe(ref_df, use_container_width=True)

if st.button("Fetch Data", type="primary"):
    with st.spinner("Fetching…"):
        try:
            if data_mode == "Live (current forward curve)":
                df = fetch_live(token, fob_uuid, via, percent_hire, unit)
            else:
                start_dt = datetime.combine(start_date, datetime.min.time())
                end_dt = datetime.combine(end_date, datetime.min.time())
                df = fetch_historical(token, fob_uuid, via, percent_hire, unit, start_dt, end_dt)

            if df.empty:
                st.warning("No data returned.")
                st.stop()

            st.session_state["idarbs_df"] = df
            st.session_state["idarbs_mode"] = data_mode
            st.success(f"✅ Fetched {len(df):,} rows.")
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

if "idarbs_df" in st.session_state:
    df: pd.DataFrame = st.session_state["idarbs_df"]
    mode: str = st.session_state["idarbs_mode"]

    st.subheader("Data")
    st.dataframe(df, use_container_width=True)

    fname = f"intraday_arbs_{port_name.replace(' ', '_')}_{via}_{percent_hire}pct"
    if mode == "Historical":
        fname += f"_{start_date}_{end_date}"
    fname += ".csv"

    st.download_button(
        "📥 Download CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=fname,
        mime="text/csv",
        use_container_width=True,
    )

    st.subheader("Summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Rows", f"{len(df):,}")
    c2.metric("Periods", df["PeriodName"].nunique() if "PeriodName" in df.columns else "—")
    if "Value" in df.columns and not df["Value"].isna().all():
        c3.metric("Latest Value", f"{df['Value'].iloc[-1]:.3f}")
