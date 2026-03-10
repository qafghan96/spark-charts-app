import io
import json
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

st.title("📊 Arb Breakevens")
st.caption(
    "Download JKM-TTF and freight arbitrage breakeven levels by FoB port and via point. "
    "Fetches data from `/v1.0/netbacks/arb-breakevens/`."
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
    tickers, names, via_options = [], [], []
    for port in content["data"]["staticData"]["fobPorts"]:
        tickers.append(port["uuid"])
        names.append(port["name"])
        via_options.append(port["availableViaPoints"])
    return tickers, names, via_options

def fetch_breakevens(
    access_token: str,
    fob_port_uuid: str,
    breakeven_type: str,
    start: str,
    end: str,
    via: str | None = None,
    fmt: str = "csv",
):
    uri = f"/v1.0/netbacks/arb-breakevens/{breakeven_type}/?fob-port={fob_port_uuid}&start={start}&end={end}"
    if via:
        uri += f"&via-point={via}"
    raw = api_get(uri, access_token, format=fmt)
    if fmt == "csv":
        if not raw:
            return pd.DataFrame()
        return pd.read_csv(io.StringIO(raw.decode("utf-8")))
    else:
        return raw.get("data") if isinstance(raw, dict) else raw


# --- Load reference data ---
if "arb_ref" not in st.session_state:
    with st.spinner("Loading reference data…"):
        try:
            tickers, names, via_options = fetch_reference_data(token)
            # Build ports table (only ports that have via options)
            rows = [
                {"Port": n, "UUID": t, "Available Via": v}
                for t, n, v in zip(tickers, names, via_options) if v
            ]
            st.session_state["arb_ref"] = rows
        except Exception as e:
            st.error(f"Failed to load reference data: {e}")
            st.stop()

ref_rows = st.session_state["arb_ref"]
port_names = sorted({r["Port"] for r in ref_rows})

# --- Configuration ---
st.subheader("Parameters")

today = datetime.today().date()
default_start = today - timedelta(days=180)

col1, col2 = st.columns(2)
with col1:
    port = st.selectbox(
        "FoB Port",
        options=port_names,
        index=port_names.index("Sabine Pass") if "Sabine Pass" in port_names else 0,
    )
    breakeven_type = st.selectbox("Breakeven Type", options=["jkm-ttf", "freight"], index=0)

with col2:
    port_row = next(r for r in ref_rows if r["Port"] == port)
    fob_uuid = port_row["UUID"]
    via_list = [v for v in port_row["Available Via"] if v is not None] or ["None"]
    via = st.selectbox("Via Point", options=via_list, index=0)

    output_format = st.radio("Output Format", options=["csv", "json"], horizontal=True)

col3, col4 = st.columns(2)
with col3:
    start_date = st.date_input("Start Date", value=default_start)
with col4:
    end_date = st.date_input("End Date", value=today)

st.write(f"**Port UUID:** `{fob_uuid}` &nbsp;|&nbsp; **Via:** `{via}`")

with st.expander("📋 All Available Ports", expanded=False):
    st.dataframe(pd.DataFrame(ref_rows), use_container_width=True)

if st.button("Fetch Arb Breakevens", type="primary"):
    via_param = None if via == "None" else via
    with st.spinner("Fetching arb breakevens…"):
        try:
            result = fetch_breakevens(
                token, fob_uuid, breakeven_type,
                start_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d"),
                via=via_param,
                fmt=output_format,
            )
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

    if output_format == "csv":
        if isinstance(result, pd.DataFrame) and not result.empty:
            st.success(f"✅ Fetched {len(result):,} rows.")
            st.subheader("Arb Breakevens Data")
            st.dataframe(result, use_container_width=True)

            fname = f"arb_breakevens_{port.replace(' ', '_')}_{breakeven_type}_{via}_{start_date}_{end_date}.csv"
            st.download_button(
                "📥 Download CSV",
                data=result.to_csv(index=False).encode("utf-8"),
                file_name=fname,
                mime="text/csv",
                use_container_width=True,
            )
            st.session_state["arb_df"] = result
        else:
            st.warning("No data returned.")
    else:
        if result:
            st.success("✅ Fetched JSON data.")
            st.subheader("Arb Breakevens Data (JSON)")
            st.json(result)
            fname = f"arb_breakevens_{port.replace(' ', '_')}_{breakeven_type}_{via}_{start_date}_{end_date}.json"
            st.download_button(
                "📥 Download JSON",
                data=json.dumps(result, indent=2),
                file_name=fname,
                mime="application/json",
                use_container_width=True,
            )
        else:
            st.warning("No data returned.")
