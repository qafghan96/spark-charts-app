import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import time
import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure we can import sibling module "utils.py" when run as a Streamlit page
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
APP_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if APP_ROOT not in sys.path:
    sys.path.insert(0, APP_ROOT)

from utils import (
    get_credentials,
    get_access_token,
)

st.title("Analytics - DES Hub Marginal Terminal")

st.caption("Determine which terminals have been most marginal (WTP metric closest to 0) historically.")

st.info("**WTP (Willingness to Pay) metric defined as:**  \nFront Month hub price - variable regas costs - SparkNWE/SWE DES LNG price")

client_id, client_secret = get_credentials()
if not client_id or not client_secret:
    st.error("Missing Spark API credentials. Set streamlit secrets 'spark.client_id' and 'spark.client_secret' (or env vars).")
    st.stop()

scopes = "read:access,read:prices"
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
        "accept": "application/json",
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

def fetch_deshub_releases(access_token, unit, limit=None, offset=None, terminal=None):
    """Fetch DES hub releases data."""
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

def deshub_organise_dataframe(data):
    """Sort the API content into a dataframe."""
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

    for l in data['data']:
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
    
    df = pd.DataFrame(data_dict)
    
    df['Delivery Month'] = pd.to_datetime(df['Delivery Month'])
    df['Release Date'] = pd.to_datetime(df['Release Date'])
    
    # Variable Regas Costs only - treat slot costs as sunk
    df['DES Hub Netback - TTF Basis - Var Regas Costs Only'] = df['DES Hub Netback - TTF Basis'] \
                                                                + df['Basic Slot (B/U/S/R)'] \
                                                                + df['Basic Slot (Berth)'] \
                                                                + df['Basic Slot (B/U/S/R)']
    
    return df

def loop_historical_data(token, n_offset):
    """Loop through historical data with offset."""
    historical = fetch_deshub_releases(token, unit='usd-per-mmbtu', limit=30)
    if historical is None:
        return pd.DataFrame(), []
    
    hist_df = deshub_organise_dataframe(historical)
    terminal_list = list(historical['metaData']['terminals'].values())

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i in range(1, n_offset + 1):
        status_text.text(f'Fetching historical data batch {i+1}/{n_offset+1}')
        historical = fetch_deshub_releases(token, unit='usd-per-mmbtu', limit=30, offset=i*30)
        if historical is not None:
            hist_df = pd.concat([hist_df, deshub_organise_dataframe(historical)], ignore_index=True)
        
        progress_bar.progress(i / n_offset)
        time.sleep(0.2)
    
    progress_bar.empty()
    status_text.empty()
    
    return hist_df, terminal_list

def fetch_cargo_releases(access_token, ticker, limit=4, offset=None):
    """Fetch cargo price releases."""
    query_params = "?limit={}".format(limit)
    if offset is not None:
        query_params += "&offset={}".format(offset)

    content = do_api_get_query(
        uri="/v1.0/contracts/{}/price-releases/{}".format(ticker, query_params),
        access_token=access_token,
    )

    if content is None:
        return {}
    
    return content['data']

def cargo_to_dataframe(access_token, ticker, limit, month):
    """Convert cargo data to dataframe."""
    if month == 'M+1':
        full_tick = ticker + '-b-f'
        hist_data = fetch_cargo_releases(access_token, full_tick, limit)
    else:
        full_tick = ticker + '-b-fo'
        hist_data = fetch_cargo_releases(access_token, full_tick, limit)
    
    if not hist_data:
        return pd.DataFrame()

    release_dates = []
    period_start = []
    ticker_list = []
    spark = []

    for release in hist_data:
        release_date = release["releaseDate"]
        ticker_list.append(release['contractId'])
        release_dates.append(release_date)

        mi = int(month[-1]) - 2
        data_point = release['data'][0]['dataPoints'][mi]

        period_start_at = data_point["deliveryPeriod"]["startAt"]
        period_start.append(period_start_at)
        
        spark.append(data_point['derivedPrices']['usdPerMMBtu']['spark'])

    hist_df = pd.DataFrame({
        'Release Date': release_dates,
        'ticker': ticker_list,
        'Period Start': period_start,
        'Price': spark,
    })
    
    hist_df['Price'] = pd.to_numeric(hist_df['Price'])
    hist_df['Release Date'] = pd.to_datetime(hist_df['Release Date'])
    hist_df['Release Date'] = hist_df['Release Date'].dt.tz_localize(None)

    return hist_df

# Terminal region mapping
terminal_region_dict = {
    'gate': 'nwe',
    'grain-lng': 'nwe',
    'zeebrugge': 'nwe',
    'south-hook': 'nwe',
    'dunkerque': 'nwe',
    'le-havre': 'nwe',
    'montoir': 'nwe',
    'eems-energy-terminal': 'nwe',
    'brunsbuttel': 'nwe',
    'deutsche-ostsee': 'nwe',
    'wilhelmshaven': 'nwe',
    'wilhelmshaven-2': 'nwe',
    'stade': 'nwe',
    'fos-cavaou': 'swe',
    'adriatic': 'swe',
    'olt-toscana': 'swe',
    'piombino': 'swe',
    'ravenna': 'swe',
    'tvb': 'swe'
}

# User inputs
st.subheader("Marginal Terminal Analysis Parameters")

col1, col2 = st.columns(2)

with col1:
    month_options = ['M+1', 'M+2', 'M+3', 'M+4', 'M+5', 'M+6', 'M+7', 'M+8', 'M+9', 'M+10', 'M+11']
    month = st.selectbox("Contract Month", options=month_options, index=0)

with col2:
    loops = st.slider("Historical Data Batches", min_value=5, max_value=20, value=15, 
                     help="Each batch contains 30 data points. More batches = more historical data but longer processing time.")

# Terminal selection
st.subheader("Terminal Selection")
default_terminals = list(terminal_region_dict.keys())
selected_terminals = st.multiselect(
    "Select Terminals for Analysis", 
    options=default_terminals,
    default=default_terminals,
    help="Select which terminals to include in the marginal analysis"
)

if not selected_terminals:
    st.warning("Please select at least one terminal.")
    st.stop()

st.write(f"**Analysis Month:** {month}")
st.write(f"**Data Batches:** {loops} (≈{loops * 30} data points)")
st.write(f"**Selected Terminals:** {len(selected_terminals)} terminals")

# Generate analysis
if st.button("Generate Marginal Terminal Analysis", type="primary"):
    with st.spinner("Fetching DES Hub historical data..."):
        hdf, full_terms = loop_historical_data(token, loops)
    
    if hdf.empty:
        st.error("Failed to fetch DES Hub data")
        st.stop()
    
    # Filter terminals based on selection
    terms = [t for t in selected_terminals if t in full_terms]
    
    with st.spinner("Fetching SparkNWE and SparkSWE data..."):
        sparknwe = cargo_to_dataframe(token, 'sparknwe', loops*30, month=month)
        sparkswe = cargo_to_dataframe(token, 'sparkswe', loops*30, month=month)
    
    if sparknwe.empty or sparkswe.empty:
        st.error("Failed to fetch cargo price data")
        st.stop()
    
    # Process cargo data
    sparkswe = sparkswe[sparkswe['Release Date'] >= sparknwe['Release Date'].iloc[-1]].copy()
    cargo_df = pd.merge(sparknwe, sparkswe, how='left', on='Release Date')
    cargo_df['Price_y'] = cargo_df['Price_y'].bfill().copy()
    cargo_df = cargo_df[['Release Date', 'Price_x', 'Price_y']].copy()
    cargo_df = cargo_df.rename(columns={'Price_x': 'SparkNWE', 'Price_y': 'SparkSWE'})
    
    # Create month dataframe
    if 'gate' in terms:
        month_df = hdf[(hdf['Terminal'] == 'gate') & (hdf['Month Index'] == month)][['Release Date', 'Delivery Month', 'DES Hub Netback - TTF Basis - Var Regas Costs Only']]
        month_df = month_df.rename(columns={'DES Hub Netback - TTF Basis - Var Regas Costs Only':'gate'})
        
        terms2 = [x if x != 'gate' else None for x in terms]
        
        for t in terms2:
            if t is not None:
                tdf = hdf[(hdf['Terminal'] == t) & (hdf['Month Index'] == month)][['Release Date', 'DES Hub Netback - TTF Basis - Var Regas Costs Only']]
                month_df = month_df.merge(tdf, on='Release Date', how='left')
                month_df = month_df.rename(columns={'DES Hub Netback - TTF Basis - Var Regas Costs Only':t})
    else:
        # If gate not selected, use first available terminal as base
        base_terminal = terms[0]
        month_df = hdf[(hdf['Terminal'] == base_terminal) & (hdf['Month Index'] == month)][['Release Date', 'Delivery Month', 'DES Hub Netback - TTF Basis - Var Regas Costs Only']]
        month_df = month_df.rename(columns={'DES Hub Netback - TTF Basis - Var Regas Costs Only': base_terminal})
        
        for t in terms[1:]:
            tdf = hdf[(hdf['Terminal'] == t) & (hdf['Month Index'] == month)][['Release Date', 'DES Hub Netback - TTF Basis - Var Regas Costs Only']]
            month_df = month_df.merge(tdf, on='Release Date', how='left')
            month_df = month_df.rename(columns={'DES Hub Netback - TTF Basis - Var Regas Costs Only':t})
    
    # Calculate statistics
    month_df['Ave'] = month_df[terms].mean(axis=1)
    month_df['Min'] = month_df[terms].min(axis=1)
    month_df['Max'] = month_df[terms].max(axis=1)
    
    # Merge cargo prices
    month_df = month_df.merge(cargo_df, how='left', on='Release Date')
    month_df['SparkNWE'] = month_df['SparkNWE'].bfill().copy()
    month_df['SparkSWE'] = month_df['SparkSWE'].bfill().copy()
    
    # Create WTP dataframe
    wtp_df = month_df[['Release Date', 'Delivery Month', 'SparkNWE', 'SparkSWE']].copy()
    
    for t in terms:
        if t in terminal_region_dict:
            if terminal_region_dict[t] == 'nwe':
                wtp_df[t] = month_df[t].copy() - month_df['SparkNWE'].copy()
            elif terminal_region_dict[t] == 'swe':
                wtp_df[t] = month_df[t].copy() - month_df['SparkSWE'].copy()
            else:
                wtp_df[t] = month_df[t].copy() - month_df['SparkNWE'].copy()
        else:
            wtp_df[t] = month_df[t].copy() - month_df['SparkNWE'].copy()
    
    # Calculate WTP statistics
    wtp_df['Ave'] = wtp_df[terms].mean(axis=1)
    wtp_df['Min'] = wtp_df[terms].min(axis=1)
    wtp_df['Max'] = wtp_df[terms].max(axis=1)
    
    # Find marginal terminal
    wtp_df['Marginal Terminal'] = wtp_df[terms].abs().idxmin(axis="columns").copy()
    marg_list = list(wtp_df['Marginal Terminal'].unique())
    
    # Create plot
    sns.set_style('whitegrid')
    fig, ax = plt.subplots(figsize=(15, 7))
    
    ax.plot(wtp_df['Release Date'], wtp_df['Ave'], color='mediumseagreen', linewidth=2.0, label='European Average WTP')
    
    # Plot marginal terminals
    colors = plt.cm.Set3(np.linspace(0, 1, len(marg_list)))
    for i, m in enumerate(marg_list):
        tdf = wtp_df[wtp_df['Marginal Terminal'] == m][['Release Date', m]]
        ax.scatter(tdf['Release Date'], tdf[m], label=m, color=colors[i], alpha=0.7)
    
    # Shade negative area
    negrange = [wtp_df['Release Date'].iloc[-1] - pd.Timedelta(20, unit='day'), 
                wtp_df['Release Date'].iloc[0] + pd.Timedelta(20, unit='day')]
    ax.plot(negrange, [-1.0, -1.0], color='red', alpha=0.05)
    ax.plot(negrange, [0, 0], color='red', alpha=0.05)
    ax.fill_between(negrange, 0, -1.0, color='red', alpha=0.05)
    
    # Chart aesthetics
    ax.set_ylim(-0.5, 1.4)
    ax.axhline(0, color='black', linestyle='-', alpha=0.3)
    plt.title(f'European Average WTP & Range vs Marginal Cargo ({month})')
    plt.ylabel('$/MMBtu')
    plt.xlabel('Release Date')
    plt.legend()
    sns.despine(left=True, bottom=True)
    
    st.pyplot(fig)
    
    # Display summary statistics
    st.subheader("Analysis Summary")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        current_avg_wtp = wtp_df['Ave'].iloc[0] if len(wtp_df) > 0 else 0
        st.metric("Current Average WTP", f"${current_avg_wtp:.3f}/MMBtu")
    
    with col2:
        current_marginal = wtp_df['Marginal Terminal'].iloc[0] if len(wtp_df) > 0 else "N/A"
        st.metric("Current Marginal Terminal", current_marginal)
    
    with col3:
        if len(wtp_df) > 0 and current_marginal in wtp_df.columns:
            current_marginal_wtp = wtp_df[current_marginal].iloc[0]
            st.metric("Marginal Terminal WTP", f"${current_marginal_wtp:.3f}/MMBtu")
    
    # Marginal terminal frequency
    st.subheader("Marginal Terminal Frequency")
    marginal_counts = wtp_df['Marginal Terminal'].value_counts()
    marginal_pct = (marginal_counts / len(wtp_df) * 100).round(1)
    
    freq_df = pd.DataFrame({
        'Terminal': marginal_counts.index,
        'Count': marginal_counts.values,
        'Percentage': marginal_pct.values
    })
    
    st.dataframe(freq_df, use_container_width=True)
    
    # Download data
    st.subheader("Download Data")
    col1, col2 = st.columns(2)
    
    with col1:
        wtp_csv = wtp_df.to_csv(index=False)
        st.download_button(
            label="Download WTP Data CSV",
            data=wtp_csv,
            file_name=f"marginal_terminal_wtp_{month}.csv",
            mime="text/csv"
        )
    
    with col2:
        netbacks_csv = month_df.to_csv(index=False)
        st.download_button(
            label="Download Netbacks Data CSV",
            data=netbacks_csv,
            file_name=f"des_hub_netbacks_{month}.csv",
            mime="text/csv"
        )
    
    # Store in session state
    st.session_state.marginal_wtp_df = wtp_df
    st.session_state.marginal_netbacks_df = month_df
    
    st.success(f"Analysis complete! Processed {len(wtp_df)} data points across {len(terms)} terminals.")