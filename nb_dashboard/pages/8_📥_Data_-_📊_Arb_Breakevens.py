import os
import sys
import streamlit as st
import pandas as pd
from io import StringIO

# Ensure we can import sibling module "utils.py" when run as a Streamlit page
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
APP_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if APP_ROOT not in sys.path:
    sys.path.insert(0, APP_ROOT)

from utils import (
    get_credentials,
    get_access_token,
)

st.title("Data Download - Arb Breakevens")

st.caption("Download arb breakevens data for JKM-TTF and freight breakevens using the netbacks API.")

client_id, client_secret = get_credentials()
if not client_id or not client_secret:
    st.error("Missing Spark API credentials. Set streamlit secrets 'spark.client_id' and 'spark.client_secret' (or env vars).")
    st.stop()

scopes = "read:netbacks,read:access,read:prices,read:routes"
token = get_access_token(client_id, client_secret, scopes=scopes)

# Import required libraries for API calls
import json
from base64 import b64encode
from urllib.parse import urljoin
from urllib import request
from urllib.error import HTTPError

API_BASE_URL = "https://api.sparkcommodities.com"

def do_api_get_query(uri, access_token, format='json'):
    """After receiving an Access Token, we can request information from the API."""
    url = urljoin(API_BASE_URL, uri)
    
    if format == 'json':
        headers = {
            "Authorization": "Bearer {}".format(access_token),
            "Accept": "application/json",
        }
    elif format == 'csv':
        headers = {
            "Authorization": "Bearer {}".format(access_token),
            "Accept": "text/csv"
        }
    
    req = request.Request(url, headers=headers)
    try:
        response = request.urlopen(req)
    except HTTPError as e:
        st.error(f"HTTP Error: {e.code}")
        st.error(e.read().decode())
        return None
    
    resp_content = response.read()
    
    if response.status != 200:
        st.error(f"API returned status {response.status}")
        return None
    
    if format == 'json':
        content = json.loads(resp_content)
    elif format == 'csv':
        content = resp_content
    
    return content

def list_netbacks(access_token):
    """Fetch available netbacks."""
    content = do_api_get_query(uri="/v1.0/netbacks/reference-data/", access_token=access_token)
    if content is None:
        return [], [], [], [], {}
    
    tickers = []
    fobPort_names = []
    availablevia = []
    
    for contract in content["data"]["staticData"]["fobPorts"]:
        tickers.append(contract["uuid"])
        fobPort_names.append(contract["name"])
        availablevia.append(contract["availableViaPoints"])
    
    reldates = content["data"]["staticData"]["sparkReleases"]
    dicto1 = content["data"]
    
    return tickers, fobPort_names, availablevia, reldates, dicto1

def format_store(available_via, fob_names, tickrs):
    """Format available netbacks data into DataFrame."""
    dict_store = {
        "Index": [],
        "Ports": [],
        "Ticker": [],
        "Available Via": []
    }
    
    c = 0
    for a in available_via:
        if len(a) != 0:
            dict_store['Index'].append(c)
            dict_store['Ports'].append(fob_names[c])
            dict_store['Ticker'].append(tickrs[c])
            dict_store['Available Via'].append(available_via[c])
        c += 1
    
    dict_df = pd.DataFrame(dict_store)
    return dict_df

def fetch_breakevens(access_token, ticker, via=None, breakeven='jkm-ttf', start=None, end=None, format='json'):
    """Fetch arb breakevens data."""
    query_params = breakeven + '/' + "?fob-port={}".format(ticker)
    
    if via is not None and via != "None":
        query_params += "&via-point={}".format(via)
    if start is not None:
        query_params += "&start={}".format(start)
    if end is not None:
        query_params += "&end={}".format(end)
    
    uri = "/v1.0/netbacks/arb-breakevens/{}".format(query_params)
    
    content = do_api_get_query(uri=uri, access_token=access_token, format=format)
    if content is None:
        return None
    
    if format == 'json':
        my_dict = content['data']
    else:
        my_dict = content.decode('utf-8')
        my_dict = pd.read_csv(StringIO(my_dict))
    
    return my_dict

# Fetch available netbacks
with st.spinner("Fetching available netbacks..."):
    tickers, fobPort_names, availablevia, reldates, dicto1 = list_netbacks(token)

if not tickers:
    st.error("Failed to fetch netbacks from API")
    st.stop()

# Create available ports dataframe
available_df = format_store(availablevia, fobPort_names, tickers)

# User inputs
st.subheader("Arb Breakevens Parameters")

col1, col2 = st.columns(2)

with col1:
    available_ports = sorted(available_df["Ports"].unique())
    port = st.selectbox("FOB Port", options=available_ports, 
                       index=available_ports.index("Sabine Pass") if "Sabine Pass" in available_ports else 0)

with col2:
    breakeven_type = st.selectbox("Breakeven Type", options=["jkm-ttf", "freight"], index=0)

# Get available via points for selected port
selected_port_data = available_df[available_df["Ports"] == port]
if not selected_port_data.empty:
    my_ticker = selected_port_data["Ticker"].iloc[0]
    possible_via = selected_port_data["Available Via"].iloc[0]
    
    # Convert None values to "None" string for selectbox
    via_display_options = []
    for v in possible_via:
        if v is None:
            via_display_options.append("None")
        else:
            via_display_options.append(v)
    
    my_via = st.selectbox("Via Point", options=via_display_options, index=0)
else:
    st.error(f"No data found for port {port}")
    st.stop()

# Date range selection
col3, col4 = st.columns(2)
with col3:
    start_date = st.date_input("Start Date", value=pd.to_datetime("2025-07-01"))
with col4:
    end_date = st.date_input("End Date", value=pd.to_datetime("2025-08-01"))

# Format selection
output_format = st.radio("Output Format", options=["csv", "json"], index=0)

st.write(f"**Selected Port:** {port}")
st.write(f"**Port UUID:** {my_ticker}")
st.write(f"**Via Point:** {my_via}")
st.write(f"**Breakeven Type:** {breakeven_type}")

# Fetch and display data
if st.button("Fetch Arb Breakevens Data", type="primary"):
    # Convert "None" string back to None for API call
    via_param = None if my_via == "None" else my_via
    
    with st.spinner("Fetching arb breakevens data..."):
        result = fetch_breakevens(
            access_token=token,
            ticker=my_ticker,
            via=via_param,
            breakeven=breakeven_type,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            format=output_format
        )
    
    if result is not None:
        if output_format == "csv":
            st.success(f"Successfully fetched {len(result)} data points!")
            
            # Display the dataframe
            st.subheader("Arb Breakevens Data")
            st.dataframe(result, use_container_width=True)
            
            # Download button
            csv_data = result.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name=f"arb_breakevens_{port}_{breakeven_type}_{my_via}_{start_date}_{end_date}.csv",
                mime="text/csv"
            )
            
            # Store in session state
            st.session_state.my_breakevens_df = result
            
        else:  # json format
            st.success("Successfully fetched JSON data!")
            st.subheader("Arb Breakevens Data (JSON)")
            st.json(result)
            
            # Convert to downloadable JSON
            json_data = json.dumps(result, indent=2)
            st.download_button(
                label="Download JSON",
                data=json_data,
                file_name=f"arb_breakevens_{port}_{breakeven_type}_{my_via}_{start_date}_{end_date}.json",
                mime="application/json"
            )
    else:
        st.error("Failed to fetch data from API")