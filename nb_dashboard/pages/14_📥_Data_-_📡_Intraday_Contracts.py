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

st.title("📡 Intraday Contracts")
st.caption(
    "Download live and historical intraday LNG and gas contract prices (JKM-TTF, TTF, etc.). "
    "Fetches from `/v1.0/intraday/contracts/`. "
    "Historical calls are capped at 20 days — longer ranges are chunked automatically."
)

client_id, client_secret = get_credentials()
if not client_id or not client_secret:
    st.error("Missing Spark API credentials.")
    st.stop()

token = get_access_token(client_id, client_secret)

# --- Available contracts & units ---
CONTRACT_OPTIONS = ["jkm-ttf", "ttf", "jkm", "nbp", "hen", "ztp", "peg", "gas"]
UNIT_OPTIONS = {
    "USD/MMBtu": "usd-per-mmbtu",
    "EUR/MWh":   "eur-per-mwh",
    "GBp/Therm": "gbp-per-therm",
}

# --- Helpers ---
def _build_uri(endpoint: str, contract: str, unit: str,
               start: str | None = None, end: str | None = None) -> str:
    uri = f"/v1.0/intraday/contracts/{endpoint}/?contract={contract}&unit={unit}"
    if start:
        uri += f"&start={start}"
    if end:
        uri += f"&end={end}"
    return uri

def parse_df(raw: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(raw.decode("utf-8")))
    if not df.empty:
        df["AsOf"] = pd.to_datetime(df["AsOf"], utc=True).dt.tz_localize(None)
        df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    return df

def fetch_live(access_token: str, contract: str, unit: str) -> pd.DataFrame:
    uri = _build_uri("live", contract, unit)
    raw = api_get(uri, access_token, format="csv")
    return parse_df(raw) if raw else pd.DataFrame()

def fetch_historical_chunk(access_token: str, contract: str, unit: str,
                           start: str, end: str) -> pd.DataFrame:
    uri = _build_uri("historical", contract, unit, start, end)
    raw = api_get(uri, access_token, format="csv")
    return parse_df(raw) if raw else pd.DataFrame()

def fetch_historical(access_token: str, contract: str, unit: str,
                     start: datetime, end: datetime) -> pd.DataFrame:
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
            access_token, contract, unit,
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


# --- Configuration ---
st.subheader("Configuration")

col1, col2 = st.columns(2)

with col1:
    contract = st.selectbox(
        "Contract",
        options=CONTRACT_OPTIONS,
        index=0,
        help="The intraday contract ticker.",
    )
    unit_label = st.selectbox("Unit", options=list(UNIT_OPTIONS.keys()), index=0)
    unit = UNIT_OPTIONS[unit_label]

with col2:
    data_mode = st.radio(
        "Data Mode",
        options=["Live (current forward curve)", "Historical"],
        index=1,
    )

today = datetime.today().date()
if data_mode == "Historical":
    col_s, col_e = st.columns(2)
    with col_s:
        start_date = st.date_input("Start Date", value=today - timedelta(days=30))
    with col_e:
        end_date = st.date_input("End Date", value=today)

if st.button("Fetch Data", type="primary"):
    with st.spinner("Fetching…"):
        try:
            if data_mode == "Live (current forward curve)":
                df = fetch_live(token, contract, unit)
            else:
                start_dt = datetime.combine(start_date, datetime.min.time())
                end_dt = datetime.combine(end_date, datetime.min.time())
                df = fetch_historical(token, contract, unit, start_dt, end_dt)

            if df.empty:
                st.warning("No data returned.")
                st.stop()

            st.session_state["idcontracts_df"] = df
            st.session_state["idcontracts_mode"] = data_mode
            st.session_state["idcontracts_contract"] = contract
            st.success(f"✅ Fetched {len(df):,} rows.")
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

if "idcontracts_df" in st.session_state:
    df: pd.DataFrame = st.session_state["idcontracts_df"]
    mode: str = st.session_state["idcontracts_mode"]
    stored_contract: str = st.session_state["idcontracts_contract"]

    st.subheader("Data")
    st.dataframe(df, use_container_width=True)

    fname = f"intraday_contracts_{stored_contract}_{unit_label.replace('/', '_')}"
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
    c2.metric("Period Types", df["PeriodType"].nunique() if "PeriodType" in df.columns else "—")
    c3.metric("Periods", df["PeriodName"].nunique() if "PeriodName" in df.columns else "—")
