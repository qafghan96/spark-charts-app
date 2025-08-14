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

st.title("🏭 Access Terminal Costs")
st.caption("Access terminal cost data from the Spark API with downloadable DataFrame functionality.")

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

def fetch_price_releases(access_token, limit=4, offset=None):
    query_params = "?limit={}".format(limit)
    if offset is not None:
        query_params += "&offset={}".format(offset)
    
    content = do_api_get_query(
        uri="/beta/sparkr/releases/{}".format(query_params), access_token=access_token
    )
    return content["data"]

def organise_dataframe(latest):
    """
    This function sorts the API content into a dataframe. The columns available are Release Date, Terminal, Month, Vessel Size, $/MMBtu and €/MWh. 
    Essentially, this function parses the Access database using the Month, Terminal and Vessel size columns as reference.
    """
    # create columns
    data_dict = {
        'Release Date':[],
        'Terminal':[],
        'Month':[],
        'Vessel Size':[],
        'Total $/MMBtu':[],
        'Basic Slot (Berth)':[],
        'Basic Slot (Unload/Stor/Regas)':[],
        'Basic Slot (B/U/S/R)':[],
        'Additional Storage':[],
        'Additional Sendout':[],
        'Gas in Kind': [],
        'Entry Capacity':[],
        'Commodity Charge':[]
    }

    # loop for each Terminal
    for l in latest:
        sizes_available = list(latest[0]['perVesselSize'].keys())

        # loop for each available size
        for s in sizes_available:
            
            # loop for each month (in the form: YYYY-MM-DD)
            for month in range(len(l['perVesselSize'][f'{s}']['deliveryMonths'])):
                
                # assigning values to each column
                data_dict['Release Date'].append(l["releaseDate"])
                data_dict['Terminal'].append(l["terminalName"])
                data_dict['Month'].append(l['perVesselSize'][f'{s}']['deliveryMonths'][month]['month'])
                data_dict['Vessel Size'].append(s)
                data_dict['Total $/MMBtu'].append(float(l['perVesselSize'][f'{s}']['deliveryMonths'][month]["costsInUsdPerMmbtu"]["total"]))
                
                data_dict['Basic Slot (Berth)'].append(float(l['perVesselSize'][f'{s}']['deliveryMonths'][month]["costsInUsdPerMmbtu"]["breakdown"]['basic-slot-berth']['value']))
                data_dict['Basic Slot (Unload/Stor/Regas)'].append(float(l['perVesselSize'][f'{s}']['deliveryMonths'][month]["costsInUsdPerMmbtu"]["breakdown"]['basic-slot-unload-storage-regas']['value']))
                data_dict['Basic Slot (B/U/S/R)'].append(float(l['perVesselSize'][f'{s}']['deliveryMonths'][month]["costsInUsdPerMmbtu"]["breakdown"]['basic-slot-berth-unload-storage-regas']['value']))
                data_dict['Additional Storage'].append(float(l['perVesselSize'][f'{s}']['deliveryMonths'][month]["costsInUsdPerMmbtu"]["breakdown"]['additional-storage']['value']))
                data_dict['Additional Sendout'].append(float(l['perVesselSize'][f'{s}']['deliveryMonths'][month]["costsInUsdPerMmbtu"]["breakdown"]['additional-send-out']['value']))
                data_dict['Gas in Kind'].append(float(l['perVesselSize'][f'{s}']['deliveryMonths'][month]["costsInUsdPerMmbtu"]["breakdown"]['fuel-gas-losses-gas-in-kind']['value']))
                data_dict['Entry Capacity'].append(float(l['perVesselSize'][f'{s}']['deliveryMonths'][month]["costsInUsdPerMmbtu"]["breakdown"]['entry-capacity']['value']))
                data_dict['Commodity Charge'].append(float(l['perVesselSize'][f'{s}']['deliveryMonths'][month]["costsInUsdPerMmbtu"]["breakdown"]['commodity-charge']['value']))
                
    # convert into dataframe
    df = pd.DataFrame(data_dict)
    
    df['Month'] = pd.to_datetime(df['Month'])
    df['Release Date'] = pd.to_datetime(df['Release Date'])
    
    return df

def loop_historical_data(token, n30_offset):
    # initialise first set of historical data and initialising dataframe
    historical = fetch_price_releases(access_token=token, limit=30)
    hist_df = organise_dataframe(historical)

    # Looping through earlier historical data and adding to the historical dataframe
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(1, n30_offset+1):
        status_text.text(f'Fetching batch {i}/{n30_offset}...')
        progress_bar.progress(i / n30_offset)
        
        historical = fetch_price_releases(access_token=token, limit=30, offset=i*30)
        new_data = organise_dataframe(historical)
        hist_df = pd.concat([hist_df, new_data], ignore_index=True)
    
    progress_bar.empty()
    status_text.empty()
    
    return hist_df

# UI Controls
st.subheader("Data Configuration")

col1, col2 = st.columns(2)
with col1:
    data_limit = st.slider("Number of releases", min_value=1, max_value=30, value=3, step=1)
with col2:
    use_extended_data = st.checkbox("Use extended historical data", value=False)
    if use_extended_data:
        n30_offset = st.slider("Number of 30-release batches", min_value=1, max_value=50, value=1, step=1)

if st.button("Fetch Data"):
    with st.spinner("Fetching data from Spark API..."):
        try:
            if use_extended_data:
                hist_df = loop_historical_data(token, n30_offset)
            else:
                historical = fetch_price_releases(token, limit=data_limit)
                hist_df = organise_dataframe(historical)
            
            st.success(f"Successfully fetched {len(hist_df)} rows of data!")
            
            # Display the DataFrame
            st.subheader("Access Terminal Costs Data")
            st.dataframe(hist_df, use_container_width=True)
            
            # Download functionality
            @st.cache_data
            def convert_df(df):
                return df.to_csv(index=False).encode('utf-8')
            
            csv = convert_df(hist_df)
            
            st.download_button(
                label="📥 Download data as CSV",
                data=csv,
                file_name='access_terminal_costs.csv',
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
                st.metric("Vessel Sizes", hist_df['Vessel Size'].nunique())
            with col4:
                st.metric("Date Range", f"{hist_df['Release Date'].nunique()} releases")
                
        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")
            st.stop()