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

st.title("📡 End-of-Day Contracts")
st.caption(
    "Download end-of-day (EoD) forward curve snapshots for LNG and gas contracts. "
    "Fetches from `/v1.0/intraday/contracts/eod/`."
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

# --- Helper ---
def fetch_eod(access_token: str, contract: str, unit: str,
              start: str, end: str) -> pd.DataFrame:
    uri = (
        f"/v1.0/intraday/contracts/eod/"
        f"?contract={contract}&unit={unit}&start={start}&end={end}"
    )
    raw = api_get(uri, access_token, format="csv")
    if not raw:
        return pd.DataFrame()
    df = pd.read_csv(io.StringIO(raw.decode("utf-8")))
    if not df.empty:
        df["AsOf"] = pd.to_datetime(df["AsOf"], utc=True).dt.tz_localize(None)
        df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    return df


# --- Configuration ---
st.subheader("Configuration")

col1, col2, col3, col4 = st.columns(4)

today = datetime.today().date()

with col1:
    contract = st.selectbox(
        "Contract",
        options=CONTRACT_OPTIONS,
        index=0,
        help="The intraday contract ticker.",
    )

with col2:
    unit_label = st.selectbox("Unit", options=list(UNIT_OPTIONS.keys()), index=0)
    unit = UNIT_OPTIONS[unit_label]

with col3:
    start_date = st.date_input("Start Date", value=today - timedelta(days=30))

with col4:
    end_date = st.date_input("End Date", value=today)

if st.button("Fetch EoD Data", type="primary"):
    with st.spinner("Fetching…"):
        try:
            df = fetch_eod(
                token, contract, unit,
                start_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d"),
            )

            if df.empty:
                st.warning("No data returned. Check your date range and contract.")
                st.stop()

            st.session_state["eod_df"] = df
            st.session_state["eod_contract"] = contract
            st.session_state["eod_unit"] = unit_label
            st.success(f"✅ Fetched {len(df):,} rows.")
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

if "eod_df" in st.session_state:
    df: pd.DataFrame = st.session_state["eod_df"]
    stored_contract: str = st.session_state["eod_contract"]
    stored_unit: str = st.session_state["eod_unit"]

    st.subheader("Data")
    st.dataframe(df, use_container_width=True)

    fname = f"eod_contracts_{stored_contract}_{stored_unit.replace('/', '_')}_{start_date}_{end_date}.csv"
    st.download_button(
        "📥 Download CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=fname,
        mime="text/csv",
        use_container_width=True,
    )

    st.subheader("Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Rows", f"{len(df):,}")
    c2.metric("EoD Snapshots", df["AsOf"].dt.date.nunique() if "AsOf" in df.columns else "—")
    c3.metric("Period Types", df["PeriodType"].nunique() if "PeriodType" in df.columns else "—")
    c4.metric("Periods", df["PeriodName"].nunique() if "PeriodName" in df.columns else "—")

    if "PeriodName" in df.columns and "Value" in df.columns:
        st.subheader("Latest EoD Snapshot by Period")
        if "AsOf" in df.columns:
            latest_snapshot = df[df["AsOf"] == df["AsOf"].max()][["PeriodName", "PeriodType", "Value"]].sort_values("PeriodName")
            st.dataframe(latest_snapshot, use_container_width=True)
