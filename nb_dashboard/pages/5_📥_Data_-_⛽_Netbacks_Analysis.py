import os
import sys
import streamlit as st
import pandas as pd
import json
import numpy as np
from base64 import b64encode
from urllib.parse import urljoin
from urllib import request
from urllib.error import HTTPError
import time

# Ensure we can import sibling module "utils.py" when run as a Streamlit page
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
APP_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if APP_ROOT not in sys.path:
    sys.path.insert(0, APP_ROOT)

from utils import get_credentials, get_access_token

st.title("⛽ Netbacks Analysis")
st.caption("Analyze netback data from different FoB ports with various routing options.")

# Get credentials
client_id, client_secret = get_credentials()
if not client_id or not client_secret:
    st.error("Missing Spark API credentials. Set streamlit secrets 'spark.client_id' and 'spark.client_secret' (or env vars).")
    st.stop()

scopes = "read:netbacks,read:access,read:prices,read:routes"
token = get_access_token(client_id, client_secret, scopes=scopes)

# API functions from the notebook
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
        st.stop()
    
    resp_content = response.read()
    assert response.status == 200, resp_content
    content = json.loads(resp_content)
    return content

def list_netbacks(access_token):
    """Fetch available netbacks reference data."""
    content = do_api_get_query(uri="/v1.0/netbacks/reference-data/", access_token=access_token)
    
    tickers = []
    fobPort_names = []
    availablevia = []

    for contract in content["data"]['staticData']['fobPorts']:
        tickers.append(contract["uuid"])
        fobPort_names.append(contract['name'])
        availablevia.append(contract['availableViaPoints'])
    
    reldates = content["data"]['staticData']['sparkReleases']
    dicto1 = content["data"]
        
    return tickers, fobPort_names, availablevia, reldates, dicto1

def fetch_netback(access_token, ticker, release, via=None, laden=None, ballast=None):
    """For a FoB port, fetch netback details for a specific release."""
    query_params = "?fob-port={}".format(ticker)
    if release is not None:
        query_params += "&release-date={}".format(release)
    if via is not None:
        query_params += "&via-point={}".format(via)
    if laden is not None:
        query_params += "&laden-congestion-days={}".format(laden)
    if ballast is not None:
        query_params += "&ballast-congestion-days={}".format(ballast)
    
    url = "/v1.0/netbacks/{}".format(query_params)
    content = do_api_get_query(uri=url, access_token=access_token)
    
    return content['data']

def netbacks(access_token, tickers, fobPort_names, tick, my_releases, my_via=None, laden=None, ballast=None):
    """Process netbacks data and return as DataFrame."""
    months = []
    nea_outrights = []
    nea_ttfbasis = []
    nwe_outrights = []
    nwe_ttfbasis = []
    delta_outrights = []
    delta_ttfbasis = []
    release_date = []
    max_outrights = []
    max_ttfbasis = []
    port = []
    
    # NEA Meta fields
    nea_ttf_price = []
    nea_ttf_basis_meta = []
    nea_des_lng_price = []
    nea_route_cost = []
    nea_volume_adjustment = []
    
    # NWE Meta fields
    nwe_ttf_price = []
    nwe_ttf_basis_meta = []
    nwe_des_lng_price = []
    nwe_route_cost = []
    nwe_volume_adjustment = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for rel_idx, r in enumerate(my_releases):
        try:
            status_text.text(f'Processing release {rel_idx + 1}/{len(my_releases)}: {r}')
            progress_bar.progress((rel_idx + 1) / len(my_releases))
            
            my_dict = fetch_netback(access_token, tickers[tick], r, via=my_via, laden=laden, ballast=ballast)

            for m in my_dict['netbacks']:
                months.append(m['load']['month'])
                nea_outrights.append(float(m['nea']['outright']['usdPerMMBtu']))
                nea_ttfbasis.append(float(m['nea']['ttfBasis']['usdPerMMBtu']))
                nwe_outrights.append(float(m['nwe']['outright']['usdPerMMBtu']))
                nwe_ttfbasis.append(float(m['nwe']['ttfBasis']['usdPerMMBtu']))
                delta_outrights.append(float(m['neaMinusNwe']['outright']['usdPerMMBtu']))
                delta_ttfbasis.append(float(m['neaMinusNwe']['ttfBasis']['usdPerMMBtu']))
                max_outrights.append(float(m['max']['outright']['usdPerMMBtu']))
                max_ttfbasis.append(float(m['max']['ttfBasis']['usdPerMMBtu']))
                release_date.append(r)
                port.append(fobPort_names[tick])
                
                # Extract NEA Meta data
                nea_meta = m.get('neaMeta', {})
                nea_ttf_price.append(float(nea_meta.get('ttfPrice', {}).get('usdPerMMBtu', 0)))
                nea_ttf_basis_meta.append(float(nea_meta.get('ttfBasis', {}).get('usdPerMMBtu', 0)))
                nea_des_lng_price.append(float(nea_meta.get('desLngPrice', {}).get('usdPerMMBtu', 0)))
                nea_route_cost.append(float(nea_meta.get('routeCost', {}).get('usdPerMMBtu', 0)))
                nea_volume_adjustment.append(float(nea_meta.get('volumeAdjustment', {}).get('usdPerMMBtu', 0)))
                
                # Extract NWE Meta data
                nwe_meta = m.get('nweMeta', {})
                nwe_ttf_price.append(float(nwe_meta.get('ttfPrice', {}).get('usdPerMMBtu', 0)))
                nwe_ttf_basis_meta.append(float(nwe_meta.get('ttfBasis', {}).get('usdPerMMBtu', 0)))
                nwe_des_lng_price.append(float(nwe_meta.get('desLngPrice', {}).get('usdPerMMBtu', 0)))
                nwe_route_cost.append(float(nwe_meta.get('routeCost', {}).get('usdPerMMBtu', 0)))
                nwe_volume_adjustment.append(float(nwe_meta.get('volumeAdjustment', {}).get('usdPerMMBtu', 0)))
            
            time.sleep(0.5)  # Rate limiting
                
        except Exception as e:
            st.warning(f'Error processing date {r}: {str(e)}')
            continue
    
    progress_bar.empty()
    status_text.empty()
    
    historical_df = pd.DataFrame({
        'Release Date': release_date,
        'FoB Port': port,
        'Month': months,
        'NEA Netback (Outright)': nea_outrights,
        'NEA Netback (TTF Basis)': nea_ttfbasis,
        'NWE Netback (Outright)': nwe_outrights,
        'NWE Netback (TTF Basis)': nwe_ttfbasis,
        'NEA-NWE Arb': delta_outrights,
        'Max Outrights': max_outrights,
        'Max TTF Basis': max_ttfbasis,
        # NEA Meta fields
        'NEA TTF Price': nea_ttf_price,
        'NEA TTF Basis Meta': nea_ttf_basis_meta,
        'NEA DES LNG Price': nea_des_lng_price,
        'NEA Route Cost': nea_route_cost,
        'NEA Volume Adjustment': nea_volume_adjustment,
        # NWE Meta fields
        'NWE TTF Price': nwe_ttf_price,
        'NWE TTF Basis Meta': nwe_ttf_basis_meta,
        'NWE DES LNG Price': nwe_des_lng_price,
        'NWE Route Cost': nwe_route_cost,
        'NWE Volume Adjustment': nwe_volume_adjustment,
    })
    
    # Convert date columns
    if not historical_df.empty:
        historical_df['Release Date'] = pd.to_datetime(historical_df['Release Date'])
        historical_df['Month'] = pd.to_datetime(historical_df['Month'])
    
    return historical_df

# Initialize data
if 'netbacks_data' not in st.session_state:
    with st.spinner("Loading reference data..."):
        st.session_state.netbacks_data = list_netbacks(token)

tickers, fobPort_names, availablevia, reldates, dicto1 = st.session_state.netbacks_data

# UI Controls
st.subheader("Configuration")

col1, col2 = st.columns(2)
with col1:
    selected_port = st.selectbox("Select FoB Port", fobPort_names, index=fobPort_names.index('Sabine Pass') if 'Sabine Pass' in fobPort_names else 0)
    port_index = fobPort_names.index(selected_port)
    
    # Get available via points for selected port
    available_via_points = availablevia[port_index] if availablevia[port_index] else ['cogh']
    selected_via = st.selectbox("Via Point", available_via_points, index=0)

with col2:
    num_releases = st.slider("Number of releases", min_value=1, max_value=min(len(reldates), 500), value=10, step=1)
    
    # Advanced options
    with st.expander("Advanced Options"):
        laden_days = st.number_input("Laden Congestion Days", min_value=0, max_value=30, value=0, step=1)
        ballast_days = st.number_input("Ballast Congestion Days", min_value=0, max_value=30, value=0, step=1)

# Display selected releases
st.write(f"**Selected releases:** {reldates[:num_releases]}")

if st.button("Fetch Netbacks Data", type="primary"):
    with st.spinner("Fetching netbacks data..."):
        try:
            df = netbacks(
                token, 
                tickers, 
                fobPort_names, 
                port_index, 
                reldates[:num_releases], 
                my_via=selected_via,
                laden=laden_days if laden_days > 0 else None,
                ballast=ballast_days if ballast_days > 0 else None
            )
            
            if not df.empty:
                st.success(f"Successfully fetched {len(df)} rows of netbacks data!")
                
                # Store in session state
                st.session_state.netbacks_df = df
                
                # Display the DataFrame
                st.subheader(f"Netbacks Data - {selected_port}")
                st.dataframe(df, use_container_width=True)
                
                # Download functionality
                @st.cache_data
                def convert_df(df):
                    return df.to_csv(index=False).encode('utf-8')
                
                csv = convert_df(df)
                
                st.download_button(
                    label="📥 Download data as CSV",
                    data=csv,
                    file_name=f'netbacks_{selected_port.lower().replace(" ", "_")}.csv',
                    mime='text/csv',
                    use_container_width=True
                )
                
                # Basic statistics
                st.subheader("Data Summary")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Records", len(df))
                with col2:
                    st.metric("Unique Months", df['Month'].nunique())
                with col3:
                    st.metric("Date Range", f"{df['Release Date'].nunique()} releases")
                with col4:
                    avg_arb = df['NEA-NWE Arb'].mean()
                    st.metric("Avg NEA-NWE Arb", f"${avg_arb:.3f}/MMBtu")
                
                # Summary statistics
                st.subheader("Price Statistics ($/MMBtu)")
                
                # Display options
                show_meta = st.checkbox("Show detailed breakdown (Meta fields)", value=False)
                
                if show_meta:
                    # Detailed view with meta fields
                    st.subheader("Netback Components Analysis")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**NEA Components**")
                        nea_meta_stats = df[[
                            'NEA TTF Price', 'NEA TTF Basis Meta', 'NEA DES LNG Price', 
                            'NEA Route Cost', 'NEA Volume Adjustment', 'NEA Netback (Outright)'
                        ]].describe()
                        st.dataframe(nea_meta_stats, use_container_width=True)
                    
                    with col2:
                        st.write("**NWE Components**")
                        nwe_meta_stats = df[[
                            'NWE TTF Price', 'NWE TTF Basis Meta', 'NWE DES LNG Price', 
                            'NWE Route Cost', 'NWE Volume Adjustment', 'NWE Netback (Outright)'
                        ]].describe()
                        st.dataframe(nwe_meta_stats, use_container_width=True)
                    
                    # Component comparison
                    st.subheader("Route Cost & Volume Adjustment Comparison")
                    comp_df = pd.DataFrame({
                        'NEA Route Cost': df['NEA Route Cost'],
                        'NWE Route Cost': df['NWE Route Cost'],
                        'NEA Volume Adj': df['NEA Volume Adjustment'],
                        'NWE Volume Adj': df['NWE Volume Adjustment']
                    })
                    st.dataframe(comp_df.describe(), use_container_width=True)
                
                else:
                    # Standard view
                    summary_stats = df[['NEA Netback (Outright)', 'NWE Netback (Outright)', 'NEA-NWE Arb', 'Max Outrights']].describe()
                    st.dataframe(summary_stats, use_container_width=True)
                
            else:
                st.warning("No data returned. Please check your parameters and try again.")
                
        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")

# Show existing data if available
if 'netbacks_df' in st.session_state and not st.session_state.netbacks_df.empty:
    st.subheader("Current Data")
    st.write(f"Showing data for **{st.session_state.netbacks_df['FoB Port'].iloc[0]}** via **{selected_via}**")
    
    # Data view options
    view_option = st.radio(
        "Select data view:",
        ["Summary View", "Full Data with Meta", "Meta Components Only"],
        horizontal=True
    )
    
    if view_option == "Summary View":
        # Standard columns only
        summary_cols = ['Release Date', 'FoB Port', 'Month', 'NEA Netback (Outright)', 'NEA Netback (TTF Basis)', 
                       'NWE Netback (Outright)', 'NWE Netback (TTF Basis)', 'NEA-NWE Arb', 'Max Outrights', 'Max TTF Basis']
        st.dataframe(st.session_state.netbacks_df[summary_cols], use_container_width=True)
        
    elif view_option == "Full Data with Meta":
        # All columns
        st.dataframe(st.session_state.netbacks_df, use_container_width=True)
        
    else:  # Meta Components Only
        # Just the meta breakdown fields
        meta_cols = ['Release Date', 'Month', 'NEA TTF Price', 'NEA TTF Basis Meta', 'NEA DES LNG Price', 
                    'NEA Route Cost', 'NEA Volume Adjustment', 'NWE TTF Price', 'NWE TTF Basis Meta', 
                    'NWE DES LNG Price', 'NWE Route Cost', 'NWE Volume Adjustment']
        st.dataframe(st.session_state.netbacks_df[meta_cols], use_container_width=True)
        
        # Add explanation
        with st.expander("📖 Meta Field Explanations"):
            st.markdown("""
            **TTF Price**: The Title Transfer Facility (TTF) gas price component used in calculations
            
            **TTF Basis Meta**: The TTF basis component from the meta breakdown
            
            **DES LNG Price**: Delivered Ex Ship LNG price component
            
            **Route Cost**: The shipping/transportation cost for the specific route
            
            **Volume Adjustment**: Adjustments made for volume considerations in the netback calculation
            
            *Note: The final netback is calculated as: DES LNG Price - Route Cost + Volume Adjustment*
            """)