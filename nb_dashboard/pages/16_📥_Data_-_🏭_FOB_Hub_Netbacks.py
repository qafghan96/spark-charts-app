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

st.title("🏭 FOB Hub Netbacks")
st.caption(
    "Download FOB Hub Netback prices showing the netback value of shipping LNG from a FoB port "
    "to a specific European regas terminal. "
    "Fetches from `/v1.0/lng/access/fob-hub-netbacks/`."
)

client_id, client_secret = get_credentials()
if not client_id or not client_secret:
    st.error("Missing Spark API credentials. Set streamlit secrets 'spark.client_id' and 'spark.client_secret' (or env vars).")
    st.stop()

scopes = "read:access"
token = get_access_token(client_id, client_secret, scopes=scopes)

# --- Helpers ---
def fetch_reference_data(access_token: str) -> pd.DataFrame:
    raw = api_get("/v1.0/lng/access/fob-hub-netbacks/reference-data/", access_token, format="csv")
    return pd.read_csv(io.StringIO(raw.decode("utf-8")))

def fetch_fob_hub_netbacks(
    access_token: str,
    fob_port_uuid: str,
    unit: str,
    start: str,
    end: str,
    via_point: str | None = None,
    terminal_uuid: str | None = None,
) -> pd.DataFrame:
    uri = (
        f"/v1.0/lng/access/fob-hub-netbacks/"
        f"?unit={unit}&fob-port-uuid={fob_port_uuid}&start={start}&end={end}"
    )
    if via_point:
        uri += f"&via-point={via_point}"
    if terminal_uuid:
        uri += f"&regas-terminal-uuid={terminal_uuid}"
    raw = api_get(uri, access_token, format="csv")
    if not raw:
        return pd.DataFrame()
    df = pd.read_csv(io.StringIO(raw.decode("utf-8")))
    if not df.empty:
        df["ReleaseDate"] = pd.to_datetime(df["ReleaseDate"])
        numeric_cols = df.columns[13:]
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    return df

def fetch_fob_hub_netbacks_chunked(
    access_token: str,
    fob_port_uuid: str,
    unit: str,
    start: datetime,
    end: datetime,
    via_point: str | None = None,
    terminal_uuid: str | None = None,
) -> pd.DataFrame:
    """Fetch in ≤365-day windows to respect the API's max date span."""
    chunks = []
    chunk_start = start
    total_days = max((end - start).days, 1)
    fetched_days = 0
    progress = st.progress(0)
    status = st.empty()

    while chunk_start <= end:
        chunk_end = min(chunk_start + timedelta(days=364), end)
        status.text(f"Fetching {chunk_start.strftime('%Y-%m-%d')} → {chunk_end.strftime('%Y-%m-%d')} …")
        chunk = fetch_fob_hub_netbacks(
            access_token, fob_port_uuid, unit,
            chunk_start.strftime("%Y-%m-%d"),
            chunk_end.strftime("%Y-%m-%d"),
            via_point=via_point,
            terminal_uuid=terminal_uuid,
        )
        if not chunk.empty:
            chunks.append(chunk)
        fetched_days += (chunk_end - chunk_start).days + 1
        progress.progress(min(fetched_days / total_days, 1.0))
        chunk_start = chunk_end + timedelta(days=1)

    progress.empty()
    status.empty()
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


# --- Load reference data ---
if "fobhub_ref_df" not in st.session_state:
    with st.spinner("Loading reference data…"):
        try:
            st.session_state["fobhub_ref_df"] = fetch_reference_data(token)
        except Exception as e:
            st.error(f"Failed to load reference data: {e}")
            st.stop()

ref_df: pd.DataFrame = st.session_state["fobhub_ref_df"]

# --- Configuration ---
st.subheader("Configuration")

today = datetime.today().date()
one_year_ago = today - timedelta(days=365)

col1, col2 = st.columns(2)

with col1:
    # FoB Port
    port_names = sorted(ref_df["FOBPortName"].dropna().unique().tolist())
    selected_port = st.selectbox(
        "FoB Port",
        options=port_names,
        index=port_names.index("Sabine Pass") if "Sabine Pass" in port_names else 0,
    )
    fob_uuid = ref_df[ref_df["FOBPortName"] == selected_port]["FOBPortUUID"].iloc[0]

    # Via Point (filtered to selected port)
    port_vias = ref_df[ref_df["FOBPortName"] == selected_port]["ViaPoint"].dropna().unique().tolist()
    via_options = ["None"] + sorted(port_vias)
    selected_via_label = st.selectbox(
        "Via Point",
        options=via_options,
        index=0,
        help="Select a routing via point, or 'None' for direct routing.",
    )
    selected_via = None if selected_via_label == "None" else selected_via_label

    # Unit
    unit_label = st.selectbox("Unit", options=["USD/MMBtu", "EUR/MWh"], index=0)
    unit = "usd-per-mmbtu" if unit_label == "USD/MMBtu" else "eur-per-mwh"

with col2:
    # Regas Terminal filter
    terminal_options = ["All Terminals"] + sorted(ref_df["RegasTerminalName"].dropna().unique().tolist())
    selected_terminal = st.selectbox(
        "Regas Terminal Filter",
        options=terminal_options,
        index=0,
        help="Optionally filter to a single regas terminal.",
    )
    selected_terminal_uuid = None
    if selected_terminal != "All Terminals":
        selected_terminal_uuid = ref_df[
            ref_df["RegasTerminalName"] == selected_terminal
        ]["RegasTerminalUUID"].iloc[0]

    start_date = st.date_input("Start Date", value=one_year_ago)
    end_date = st.date_input("End Date", value=today)

with st.expander("📋 Reference Data", expanded=False):
    st.dataframe(ref_df, use_container_width=True)

start_dt = datetime.combine(start_date, datetime.min.time())
end_dt = datetime.combine(end_date, datetime.min.time())
span_days = (end_dt - start_dt).days

if st.button("Fetch Data", type="primary"):
    with st.spinner("Fetching FOB Hub Netbacks data…"):
        try:
            if span_days > 365:
                df = fetch_fob_hub_netbacks_chunked(
                    token, fob_uuid, unit, start_dt, end_dt,
                    via_point=selected_via,
                    terminal_uuid=selected_terminal_uuid,
                )
            else:
                df = fetch_fob_hub_netbacks(
                    token, fob_uuid, unit,
                    start_date.strftime("%Y-%m-%d"),
                    end_date.strftime("%Y-%m-%d"),
                    via_point=selected_via,
                    terminal_uuid=selected_terminal_uuid,
                )

            if df.empty:
                st.warning("No data returned. Check your parameters.")
                st.stop()

            df = df.drop_duplicates()
            st.session_state["fobhub_df"] = df
            st.session_state["fobhub_port"] = selected_port
            st.session_state["fobhub_via"] = selected_via_label
            st.session_state["fobhub_terminal"] = selected_terminal
            st.session_state["fobhub_unit"] = unit_label
            st.success(f"✅ Fetched {len(df):,} rows.")

        except Exception as e:
            st.error(f"Error fetching data: {e}")
            st.stop()

if "fobhub_df" in st.session_state:
    df: pd.DataFrame = st.session_state["fobhub_df"]
    stored_port: str = st.session_state.get("fobhub_port", selected_port)
    stored_via: str = st.session_state.get("fobhub_via", selected_via_label)
    stored_terminal: str = st.session_state.get("fobhub_terminal", selected_terminal)
    stored_unit: str = st.session_state.get("fobhub_unit", unit_label)

    st.subheader(f"Data — {stored_port} (via {stored_via})")

    # View toggle
    view = st.radio(
        "View",
        ["Summary", "Netback Components", "Regas Cost Breakdown", "Full Data"],
        horizontal=True,
    )

    SUMMARY_COLS = [
        "ReleaseDate", "FOBPortName", "ViaPoint", "RegasTerminalName",
        "LoadMonthName", "LoadMonthIndex",
        "NetbackOutright", "NetbackTTFBasis", "NetbackHHBasis",
        "FreightCost", "TotalRegasCost",
    ]
    NETBACK_COLS = [
        "ReleaseDate", "FOBPortName", "RegasTerminalName", "LoadMonthName",
        "NetbackOutright", "NetbackTTFBasis", "NetbackHHBasis", "NetbackHHBasis115",
        "FreightCost", "VolumeAdjustment",
        "GasHubPriceSourceFrontMonth", "GasHubPriceSourceForwardCurve",
    ]
    REGAS_COLS = [
        "ReleaseDate", "FOBPortName", "RegasTerminalName", "LoadMonthName",
        "TotalRegasCost", "SlotUnloadStorageRegas", "SlotBerth",
        "SlotBerthUnloadStorageRegas", "SlotLtCapacityEstimate",
        "AdditionalStorage", "AdditionalSendout",
        "FuelGasLossesGasInKind", "Power", "Emissions",
        "EntryCapacity", "EntryVariable",
    ]

    def safe_cols(cols):
        return [c for c in cols if c in df.columns]

    if view == "Summary":
        st.dataframe(df[safe_cols(SUMMARY_COLS)], use_container_width=True)
    elif view == "Netback Components":
        st.dataframe(df[safe_cols(NETBACK_COLS)], use_container_width=True)
    elif view == "Regas Cost Breakdown":
        st.dataframe(df[safe_cols(REGAS_COLS)], use_container_width=True)
    else:
        st.dataframe(df, use_container_width=True)

    # Download
    fname = f"fob_hub_netbacks_{stored_port.replace(' ', '_')}_{stored_via}_{stored_unit.replace('/', '_')}"
    if stored_terminal != "All Terminals":
        fname += f"_{stored_terminal.replace(' ', '_')}"
    fname += ".csv"

    st.download_button(
        "📥 Download CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=fname,
        mime="text/csv",
        use_container_width=True,
    )

    # Summary metrics
    st.subheader("Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Rows", f"{len(df):,}")
    c2.metric("Release Dates", df["ReleaseDate"].nunique() if "ReleaseDate" in df.columns else "—")
    c3.metric("Regas Terminals", df["RegasTerminalName"].nunique() if "RegasTerminalName" in df.columns else "—")
    if "ReleaseDate" in df.columns and not df["ReleaseDate"].isna().all():
        c4.metric("Latest Release", df["ReleaseDate"].max().strftime("%Y-%m-%d"))

    # Netback stats per terminal (if all terminals shown)
    if stored_terminal == "All Terminals" and "RegasTerminalName" in df.columns and "NetbackOutright" in df.columns:
        st.subheader("Average Netback Outright by Terminal")
        st.dataframe(
            df.groupby("RegasTerminalName")[["NetbackOutright", "NetbackTTFBasis", "TotalRegasCost"]]
            .mean()
            .round(3)
            .sort_values("NetbackOutright", ascending=False),
            use_container_width=True,
        )
