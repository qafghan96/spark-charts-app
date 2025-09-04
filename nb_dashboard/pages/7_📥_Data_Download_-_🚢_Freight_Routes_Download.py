import os
import sys
import streamlit as st
import pandas as pd
import time

# Ensure we can import sibling module "utils.py" when run as a Streamlit page
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
APP_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if APP_ROOT not in sys.path:
    sys.path.insert(0, APP_ROOT)

from utils import (
    get_credentials,
    get_access_token,
)

st.title("Data Download - Freight Routes Download")

st.caption("Download freight route data with cost breakdowns using the routes API.")

client_id, client_secret = get_credentials()
if not client_id or not client_secret:
    st.error("Missing Spark API credentials. Set streamlit secrets 'spark.client_id' and 'spark.client_secret' (or env vars).")
    st.stop()

scopes = "read:prices,read:routes"
token = get_access_token(client_id, client_secret, scopes=scopes)

# Import required libraries for API calls
import json
from base64 import b64encode
from urllib.parse import urljoin
from urllib import request
from urllib.error import HTTPError

API_BASE_URL = "https://api.sparkcommodities.com"

def do_api_get_query(uri, access_token):
    """After receiving an Access Token, we can request information from the API."""
    url = urljoin(API_BASE_URL, uri)
    
    headers = {
        "Authorization": "Bearer {}".format(access_token),
        "Accept": "application/json",
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
    
    content = json.loads(resp_content)
    return content

def list_routes(access_token):
    """Fetch available routes."""
    content = do_api_get_query(uri="/v1.0/routes/", access_token=access_token)
    if content is None:
        return [], [], {}
    
    tickers = []
    for contract in content["data"]["routes"]:
        tickers.append(contract["uuid"])
    
    reldates = content["data"]["sparkReleaseDates"]
    dicto1 = content["data"]
    
    return tickers, reldates, dicto1

def check_and_store_characteristics(dict1):
    """Store route characteristics in a DataFrame."""
    routes_info = {
        "UUID": [],
        "Load Location": [],
        "Discharge Location": [],
        "Via": [],
        "Load Region": [],
        "Discharge Region": [],
        "Load UUID": [],
        "Discharge UUID": []
    }
    
    for route in dict1["routes"]:
        routes_info['UUID'].append(route["uuid"])
        routes_info['Via'].append(route["via"])
        routes_info['Load UUID'].append(route["loadPort"]["uuid"])
        routes_info['Load Location'].append(route["loadPort"]["name"])
        routes_info['Load Region'].append(route["loadPort"]["region"])
        routes_info['Discharge UUID'].append(route["dischargePort"]["uuid"])
        routes_info['Discharge Location'].append(route["dischargePort"]["name"])
        routes_info['Discharge Region'].append(route["dischargePort"]["region"])
    
    route_df = pd.DataFrame(routes_info)
    return route_df

def fetch_route_data(access_token, ticker, release, congestion_laden=None, congestion_ballast=None):
    """For a route, fetch then display the route details."""
    query_params = "?release-date={}".format(release)
    if congestion_laden is not None:
        query_params += "&congestion-laden-days={}".format(congestion_laden)
    if congestion_ballast is not None:
        query_params += "&congestion-ballast-days={}".format(congestion_ballast)
    
    uri = "/v1.0/routes/{}/{}".format(ticker, query_params)
    
    content = do_api_get_query(uri=uri, access_token=access_token)
    if content is None:
        return {}
    
    return content["data"]

def historical_routes(tick, unit, my_release, access_token):
    """Fetch historical route data and return as DataFrame."""
    my_route = {
        "Period": [],
        "Start Date": [],
        "End Date": [],
        "Total Cost": [],
        "Hire Cost": [],
        "Fuel Cost": [],
        "Canal Cost": [],
        "Port Cost": [],
        "Carbon Cost": [],
    }
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, r in enumerate(my_release):
        status_text.text(f'Fetching data for release {i+1}/{len(my_release)}: {r}')
        my_dict = fetch_route_data(access_token, tick, release=r)
        
        if my_dict and "dataPoints" in my_dict:
            for data in my_dict["dataPoints"]:
                my_route['Start Date'].append(data["deliveryPeriod"]["startAt"])
                my_route['End Date'].append(data["deliveryPeriod"]["endAt"])
                my_route['Period'].append(data["deliveryPeriod"]["name"])
                
                my_route['Total Cost'].append(data['costsIn' + unit]["total"])
                my_route['Hire Cost'].append(data['costsIn' + unit]["hire"])
                my_route['Fuel Cost'].append(data['costsIn' + unit]["fuel"])
                my_route['Canal Cost'].append(data['costsIn' + unit]["canal"])
                my_route['Port Cost'].append(data['costsIn' + unit]["port"])
                
                try:
                    my_route['Carbon Cost'].append(data['costsIn' + unit]["carbon"])
                except:
                    my_route['Carbon Cost'].append(0)
        
        progress_bar.progress((i + 1) / len(my_release))
        time.sleep(0.2)
    
    status_text.text('Processing data...')
    
    my_route_df = pd.DataFrame(my_route)
    
    # Convert string columns to numeric
    numeric_cols = ["Total Cost", "Hire Cost", "Fuel Cost", "Canal Cost", "Port Cost", "Carbon Cost"]
    for col in numeric_cols:
        my_route_df[col] = pd.to_numeric(my_route_df[col], errors='coerce')
    
    my_route_df = my_route_df.fillna(0)
    
    progress_bar.empty()
    status_text.empty()
    
    return my_route_df

# Fetch available routes
with st.spinner("Fetching available routes..."):
    routes, reldates, dicto1 = list_routes(token)

if not routes:
    st.error("Failed to fetch routes from API")
    st.stop()

# Create route dataframe
route_df = check_and_store_characteristics(dicto1)

# User inputs
st.subheader("Route Selection")

col1, col2, col3 = st.columns(3)

with col1:
    load_locations = sorted(route_df["Load Location"].unique())
    load = st.selectbox("Load Location", options=load_locations, 
                       index=load_locations.index("Sabine Pass") if "Sabine Pass" in load_locations else 0)

with col2:
    # Filter discharge locations based on selected load location
    filtered_routes = route_df[route_df["Load Location"] == load]
    discharge_locations = sorted(filtered_routes["Discharge Location"].unique())
    discharge = st.selectbox("Discharge Location", options=discharge_locations,
                           index=discharge_locations.index("Futtsu") if "Futtsu" in discharge_locations else 0)

with col3:
    via_options = ["cogh", "panama", "suez", None]
    via = st.selectbox("Via", options=via_options, index=0)

# Number of releases
num_releases = st.slider("Number of releases", min_value=5, max_value=50, value=10, step=5)
my_releases = reldates[:num_releases]

# Find the route UUID
matching_routes = route_df[(route_df["Load Location"] == load) & 
                          (route_df["Discharge Location"] == discharge) & 
                          (route_df['Via'] == via)]

if matching_routes.empty:
    st.error(f"No route found for {load} -> {discharge} via {via}")
    st.stop()

my_route = matching_routes['UUID'].values[0]

st.write(f"**Selected Route:** {load} → {discharge} (via {via})")
st.write(f"**Route UUID:** {my_route}")

# Fetch and display data
if st.button("Fetch Route Data", type="primary"):
    with st.spinner("Fetching historical route data..."):
        my_route_df = historical_routes(my_route, 'UsdPerMmbtu', my_releases, token)
    
    st.success(f"Successfully fetched {len(my_route_df)} data points!")
    
    # Display the dataframe
    st.subheader("Route Data")
    st.dataframe(my_route_df, use_container_width=True)
    
    # Download button
    csv = my_route_df.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name=f"route_data_{load}_{discharge}_{via}.csv",
        mime="text/csv"
    )
    
    # Store in session state for access
    st.session_state.my_route_df = my_route_df