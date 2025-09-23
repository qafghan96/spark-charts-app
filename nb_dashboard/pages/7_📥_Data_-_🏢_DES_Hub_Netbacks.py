import os
import sys
import streamlit as st
import pandas as pd
import json
from base64 import b64encode
from urllib.parse import urljoin
from urllib import request
from urllib.error import HTTPError

# Ensure we can import sibling module "utils.py" when run as a Streamlit page
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
APP_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if APP_ROOT not in sys.path:
    sys.path.insert(0, APP_ROOT)

from utils import get_credentials, get_access_token

st.title("🏢 DES Hub Netbacks")
st.caption("Access DES Hub Netbacks data showing regasification costs and netback prices across European terminals.")

# Get credentials
client_id, client_secret = get_credentials()
if not client_id or not client_secret:
    st.error("Missing Spark API credentials. Set streamlit secrets 'spark.client_id' and 'spark.client_secret' (or env vars).")
    st.stop()

scopes = "read:access"
token = get_access_token(client_id, client_secret, scopes=scopes)

# API functions from the notebook
API_BASE_URL = "https://api.sparkcommodities.com"

def do_api_get_query(uri, access_token):
    """After receiving an Access Token, we can request information from the API."""
    url = urljoin(API_BASE_URL, uri)
    
    headers = {
        "Authorization": "Bearer {}".format(access_token),
        "accept": "application/json",
    }
    
    req = request.Request(url, headers=headers)
    try:
        response = request.urlopen(req)
    except HTTPError as e:
        st.error(f"HTTP Error: {e.code}")
        st.stop()
    
    resp_content = response.read()
    assert response.status == 200, resp_content
    content = json.loads(resp_content)
    return content

def fetch_price_releases(access_token, unit, limit=None, offset=None, terminal=None):
    """Fetch DES Hub Netbacks data with specified parameters."""
    query_params = "?unit={}".format(unit)
    if limit is not None:
        query_params += "&limit={}".format(limit)
    if offset is not None:
        query_params += "&offset={}".format(offset)
    if terminal is not None:
        query_params += "&terminal={}".format(terminal)

    content = do_api_get_query(
        uri="/beta/access/des-hub-netbacks/{}".format(query_params), access_token=access_token
    )
    return content

def organise_dataframe(data):
    """
    This function sorts the API content into a dataframe with DES Hub Netbacks data.
    """
    # create columns
    data_dict = {
        'Release Date':[],
        'Terminal':[],
        'Month Index':[],
        'Delivery Month':[],
        'DES Hub Netback - TTF Basis':[],
        'DES Hub Netback - Outright':[],
        'Total Regas':[],
        'Basic Slot (Berth)':[],
        'Basic Slot (Unload/Stor/Regas)':[],
        'Basic Slot (B/U/S/R)':[],
        'Additional Storage':[],
        'Additional Sendout':[],
        'Gas in Kind': [],
        'Entry Capacity':[],
        'Commodity Charge':[]
    }

    # loop for each data point
    for l in data['data']:
        # assigning values to each column
        data_dict['Release Date'].append(l["releaseDate"])
        data_dict['Terminal'].append(data['metaData']['terminals'][l['terminalUuid']])
        data_dict['Month Index'].append(l['monthIndex'])
        data_dict['Delivery Month'].append(l['deliveryMonth'])

        data_dict['DES Hub Netback - TTF Basis'].append(float(l['netbackTtfBasis']))
        data_dict['DES Hub Netback - Outright'].append(float(l['netbackOutright']))
        data_dict['Total Regas'].append(float(l['totalRegasificationCost']))
        data_dict['Basic Slot (Berth)'].append(float(l['slotBerth']))
        data_dict['Basic Slot (Unload/Stor/Regas)'].append(float(l['slotUnloadStorageRegas']))
        data_dict['Basic Slot (B/U/S/R)'].append(float(l['slotBerthUnloadStorageRegas']))
        data_dict['Additional Storage'].append(float(l['additionalStorage']))
        data_dict['Additional Sendout'].append(float(l['additionalSendout']))
        data_dict['Gas in Kind'].append(float(l['gasInKind']))
        data_dict['Entry Capacity'].append(float(l['entryCapacity']))
        data_dict['Commodity Charge'].append(float(l['commodityCharge']))
    
    # convert into dataframe
    df = pd.DataFrame(data_dict)
    
    if not df.empty:
        df['Delivery Month'] = pd.to_datetime(df['Delivery Month'])
        df['Release Date'] = pd.to_datetime(df['Release Date'])
    
    return df

def loop_historical_data(access_token, unit, n_offset, terminal=None):
    """Fetch multiple batches of historical data."""
    # initialise first set of historical data
    historical = fetch_price_releases(access_token, unit=unit, limit=30)
    hist_df = organise_dataframe(historical)

    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Looping through earlier historical data and adding to the historical dataframe
    for i in range(1, n_offset+1):
        status_text.text(f'Fetching batch {i}/{n_offset}...')
        progress_bar.progress(i / n_offset)
        
        historical = fetch_price_releases(access_token, unit=unit, limit=30, offset=i*30, terminal=terminal)
        new_data = organise_dataframe(historical)
        hist_df = pd.concat([hist_df, new_data], ignore_index=True)
    
    progress_bar.empty()
    status_text.empty()
    
    return hist_df

# Get available terminals by making an initial call
if 'terminal_data' not in st.session_state:
    with st.spinner("Loading terminal reference data..."):
        sample_data = fetch_price_releases(token, unit='usd-per-mmbtu', limit=1)
        st.session_state.terminal_data = sample_data['metaData']['terminals']

available_terminals = st.session_state.terminal_data
terminal_names = list(available_terminals.values())
terminal_names.sort()

# UI Controls
st.subheader("Configuration")

col1, col2 = st.columns(2)
with col1:
    unit = st.selectbox("Unit", ["usd-per-mmbtu", "eur-per-mwh"], index=0)
    
    terminal_filter = st.selectbox(
        "Terminal Filter", 
        ["All Terminals"] + terminal_names,
        index=0,
        help="Select a specific terminal or view all terminals"
    )
    selected_terminal = None if terminal_filter == "All Terminals" else terminal_filter

with col2:
    data_limit = st.slider("Number of releases per batch", min_value=1, max_value=30, value=10, step=1)
    
    use_extended_data = st.checkbox("Use extended historical data", value=False)
    if use_extended_data:
        n_offset = st.slider("Number of 30-release batches", min_value=1, max_value=50, value=2, step=1)

# Display configuration info
if selected_terminal:
    st.info(f"**Configuration:** Fetching {unit} data for **{selected_terminal}** terminal")
else:
    st.info(f"**Configuration:** Fetching {unit} data for **all terminals**")

if st.button("Fetch DES Hub Netbacks Data", type="primary"):
    with st.spinner("Fetching DES Hub Netbacks data..."):
        try:
            if use_extended_data:
                hist_df = loop_historical_data(token, unit, n_offset, terminal=selected_terminal)
            else:
                historical = fetch_price_releases(token, unit=unit, limit=data_limit, terminal=selected_terminal)
                hist_df = organise_dataframe(historical)
            
            if not hist_df.empty:
                st.success(f"Successfully fetched {len(hist_df)} rows of DES Hub Netbacks data!")
                
                # Store in session state
                st.session_state.deshub_df = hist_df
                
                # Display the DataFrame
                st.subheader("DES Hub Netbacks Data")
                st.dataframe(hist_df, use_container_width=True)
                
                # Download functionality
                @st.cache_data
                def convert_df(df):
                    return df.to_csv(index=False).encode('utf-8')
                
                csv = convert_df(hist_df)
                
                filename = f'des_hub_netbacks_{unit}'
                if selected_terminal:
                    filename += f'_{selected_terminal}'
                filename += '.csv'
                
                st.download_button(
                    label="📥 Download data as CSV",
                    data=csv,
                    file_name=filename,
                    mime='text/csv',
                    use_container_width=True
                )
                
                # Basic statistics
                st.subheader("Data Summary")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Records", len(hist_df))
                with col2:
                    st.metric("Terminals", hist_df['Terminal'].nunique())
                with col3:
                    st.metric("Release Dates", hist_df['Release Date'].nunique())
                with col4:
                    avg_netback = hist_df['DES Hub Netback - Outright'].mean()
                    unit_display = "$/MMBtu" if unit == "usd-per-mmbtu" else "€/MWh"
                    st.metric("Avg Netback", f"{avg_netback:.3f} {unit_display}")
                
                # Detailed statistics
                st.subheader("Price Statistics")
                numeric_cols = [
                    'DES Hub Netback - TTF Basis', 
                    'DES Hub Netback - Outright', 
                    'Total Regas',
                    'Basic Slot (B/U/S/R)',
                    'Gas in Kind',
                    'Entry Capacity'
                ]
                summary_stats = hist_df[numeric_cols].describe()
                st.dataframe(summary_stats, use_container_width=True)
                
                # Terminal breakdown if showing all terminals
                if not selected_terminal and hist_df['Terminal'].nunique() > 1:
                    st.subheader("By Terminal")
                    terminal_summary = hist_df.groupby('Terminal').agg({
                        'DES Hub Netback - Outright': ['mean', 'min', 'max'],
                        'Total Regas': 'mean',
                        'Release Date': 'count'
                    }).round(3)
                    terminal_summary.columns = ['Avg Netback', 'Min Netback', 'Max Netback', 'Avg Regas', 'Records']
                    st.dataframe(terminal_summary, use_container_width=True)
                
            else:
                st.warning("No data returned. Please check your parameters and try again.")
                
        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")

# Show existing data if available
if 'deshub_df' in st.session_state and not st.session_state.deshub_df.empty:
    st.subheader("Current Data Preview")
    df = st.session_state.deshub_df
    st.write(f"Showing {len(df)} records across {df['Terminal'].nunique()} terminal(s)")
    st.dataframe(df.head(10), use_container_width=True)