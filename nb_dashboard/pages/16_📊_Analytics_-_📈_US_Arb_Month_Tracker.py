import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import time
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
    add_axis_controls,
    apply_axis_limits,
)

st.title("Analytics - US Arb Month Tracker")

st.caption("Track the historical evolution of US arbitrage prices for a specific month across multiple years.")

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

def fetch_netback(access_token, ticker, release, via=None, laden=None, ballast=None):
    """Fetch netback data for a specific release."""
    query_params = "?fob-port={}".format(ticker)
    if release is not None:
        query_params += "&release-date={}".format(release)
    if via is not None:
        query_params += "&via-point={}".format(via)
    if laden is not None:
        query_params += "&laden-congestion-days={}".format(laden)
    if ballast is not None:
        query_params += "&ballast-congestion-days={}".format(ballast)
    
    content = do_api_get_query(
        uri="/v1.0/netbacks/{}".format(query_params),
        access_token=access_token,
    )
    
    if content is None:
        return {}
    
    return content['data']

def netbacks_history(access_token, ticker, reldates, fob_port_name, my_via=None, laden=None, ballast=None):
    """Get historical netbacks data."""
    months = []
    nea_outrights = []
    nea_ttfbasis = []
    nwe_outrights = []
    nwe_ttfbasis = []
    delta_outrights = []
    delta_ttfbasis = []
    release_date = []
    port = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, r in enumerate(reldates):
        status_text.text(f'Fetching data for release {i+1}/{len(reldates)}: {r}')
        
        try:
            my_dict = fetch_netback(access_token, ticker, release=r, via=my_via, laden=laden, ballast=ballast)
            
            if 'netbacks' in my_dict:
                for m in my_dict['netbacks']:
                    months.append(m['load']['month'])
                    nea_outrights.append(float(m['nea']['outright']['usdPerMMBtu']))
                    nea_ttfbasis.append(float(m['nea']['ttfBasis']['usdPerMMBtu']))
                    nwe_outrights.append(float(m['nwe']['outright']['usdPerMMBtu']))
                    nwe_ttfbasis.append(float(m['nwe']['ttfBasis']['usdPerMMBtu']))
                    delta_outrights.append(float(m['neaMinusNwe']['outright']['usdPerMMBtu']))
                    delta_ttfbasis.append(float(m['neaMinusNwe']['ttfBasis']['usdPerMMBtu']))
                    release_date.append(my_dict['releaseDate'])
                    port.append(fob_port_name)
        except Exception as e:
            st.warning(f'Bad Date: {r} - {str(e)}')
        
        progress_bar.progress((i + 1) / len(reldates))
        time.sleep(0.2)
    
    progress_bar.empty()
    status_text.empty()
    
    historical_df = pd.DataFrame({
        'Release Date': release_date,
        'FoB Port': port,
        'Month': months,
        'NEA Outrights': nea_outrights,
        'NEA TTF Basis': nea_ttfbasis,
        'NWE Outrights': nwe_outrights,
        'NWE TTF Basis': nwe_ttfbasis,
        'Delta Outrights': delta_outrights,
        'Delta TTF Basis': delta_ttfbasis,
    })
    
    historical_df['Release Date'] = pd.to_datetime(historical_df['Release Date'])
    historical_df['Month Start'] = pd.to_datetime(historical_df['Month'])
    
    return historical_df

def sort_years(df):
    """Calculate Day of Year for plotting multiple years on same timeline."""
    if 'Month Start' not in df.columns:
        df['Month Start'] = pd.to_datetime(df['Month'])
    
    df = df.copy()
    reldates = df['Release Date'].to_list()
    startdates = df['Month Start'].to_list()
    
    dayofyear = []
    
    for r in reldates:
        ir = reldates.index(r)
        if r.year - startdates[ir].year == -1:
            dayofyear.append(r.timetuple().tm_yday - 365)
        elif r.year - startdates[ir].year == -2:
            dayofyear.append(r.timetuple().tm_yday - 730)
        else:
            dayofyear.append(r.timetuple().tm_yday)
    
    df['Day of Year'] = dayofyear
    return df

# Fetch available netbacks
with st.spinner("Fetching available netbacks..."):
    tickers, fobPort_names, availablevia, reldates, dicto1 = list_netbacks(token)

if not tickers:
    st.error("Failed to fetch netbacks from API")
    st.stop()

# User inputs
st.subheader("US Arb Month Tracker Parameters")

col1, col2, col3 = st.columns(3)

with col1:
    # Filter for ports that have via points available
    available_ports = [name for i, name in enumerate(fobPort_names) if len(availablevia[i]) > 0]
    port = st.selectbox("FOB Port", options=available_ports, 
                       index=available_ports.index("Sabine Pass") if "Sabine Pass" in available_ports else 0)

with col2:
    via_options = ["cogh", "panama", "suez"]
    via = st.selectbox("Via Point", options=via_options, index=0)

with col3:
    # Month selection - create options for different months
    current_year = pd.Timestamp.now().year
    month_options = []
    for year in [current_year-2, current_year-1, current_year, current_year+1]:
        for month in range(1, 13):
            month_str = f"{year}-{month:02d}"
            month_options.append(month_str)
    
    # Default to July of current year
    default_month = f"{current_year}-07"
    selected_month = st.selectbox("Target Month", options=month_options,
                                 index=month_options.index(default_month) if default_month in month_options else 0)

# Number of releases
num_releases = st.slider("Number of releases", min_value=50, max_value=2000, value=200, step=50)
my_releases = reldates[:num_releases]

# Get port ticker
port_index = fobPort_names.index(port)
ticker = tickers[port_index]

st.write(f"**Selected Port:** {port}")
st.write(f"**Via Point:** {via}")
st.write(f"**Target Month:** {selected_month}")
st.write(f"**Port UUID:** {ticker}")

# Fetch and display data
if st.button("Generate US Arb Month Tracker", type="primary"):
    with st.spinner("Fetching historical netbacks data..."):
        df_data = netbacks_history(token, ticker, my_releases, port, my_via=via)
    
    if not df_data.empty:
        st.success(f"Successfully fetched {len(df_data)} data points!")
        
        # Filter data for the selected month across different years
        target_year = int(selected_month.split('-')[0])
        target_month = selected_month.split('-')[1]
        
        # Create month strings for different years
        months_to_plot = []
        years_to_plot = []
        for year_offset in [2, 1, 0]:  # 2 years ago, 1 year ago, current year
            year = target_year - year_offset
            month_str = f"{year}-{target_month}"
            if not df_data[df_data['Month'] == month_str].empty:
                months_to_plot.append(month_str)
                years_to_plot.append(year)
        
        if len(months_to_plot) == 0:
            st.warning(f"No data available for month {target_month} across the years.")
        else:
            # Filter and sort data for each year
            year_dataframes = []
            for month_str in months_to_plot:
                monthly_df = df_data[df_data['Month'] == month_str].copy()
                if not monthly_df.empty:
                    monthly_df = sort_years(monthly_df)
                    year_dataframes.append((month_str, monthly_df))
            
            # Add axis controls
            axis_controls = add_axis_controls(expanded=True)
            
            if st.button("Generate Chart", type="secondary"):
                # Create the plot
                sns.set_theme(style="whitegrid")
                fig, ax = plt.subplots(figsize=(15, 6))
                
                plt.axhline(0, color='grey')
                
                colors = ['darkorange', 'darkblue', 'firebrick']
                linewidths = [1, 1.5, 2]
                
                for i, (month_str, monthly_df) in enumerate(year_dataframes):
                    year = month_str.split('-')[0]
                    ax.plot(monthly_df['Day of Year'], monthly_df['Delta Outrights'], 
                           color=colors[i % len(colors)], 
                           label=year, 
                           linewidth=linewidths[i % len(linewidths)])
                
                # Apply axis limits using the utility function
                all_data = pd.concat([df for _, df in year_dataframes if not df.empty], ignore_index=True)
                apply_axis_limits(ax, axis_controls, data_df=all_data, y_cols=['Delta Outrights'])
                
                # Custom x-axis labels
                xlabels = ['Y-1', 'September', 'October', 'November', 'December', 'Y+0', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December', 'Year End']
                xpos = [-152, -121, -91, -60, -32, 0, 30, 60, 90, 121, 152, 182, 213, 244, 274, 305, 335, 365]
                
                plt.xticks(xpos, xlabels)
                plt.title(f'US Arb - {port} - Monthly Arb Evolution (Month: {target_month})')
                plt.ylabel('$/MMBtu')
                plt.xlabel('Release Date')
                
                # Set x-axis limits
                if not axis_controls['x_auto']:
                    ax.set_xlim(axis_controls['x_min'], axis_controls['x_max'])
                else:
                    if year_dataframes:
                        max_day = max([df['Day of Year'].max() for _, df in year_dataframes if not df.empty])
                        plt.xlim(-152, max_day + 10)
                
                ax.legend()
                sns.despine(left=True, bottom=True)
                
                st.pyplot(fig)
            
            # Display summary statistics
            st.subheader("Summary Statistics")
            for month_str, monthly_df in year_dataframes:
                year = month_str.split('-')[0]
                if not monthly_df.empty:
                    latest_arb = monthly_df['Delta Outrights'].iloc[0] if len(monthly_df) > 0 else None
                    avg_arb = monthly_df['Delta Outrights'].mean()
                    col1, col2 = st.columns(2)
                    with col1:
                        if latest_arb is not None:
                            st.metric(f"{year} Latest Arb", f"${latest_arb:.3f}/MMBtu")
                    with col2:
                        st.metric(f"{year} Average Arb", f"${avg_arb:.3f}/MMBtu")
        
        # Download option
        csv_data = df_data.to_csv(index=False)
        st.download_button(
            label="Download Full Dataset CSV",
            data=csv_data,
            file_name=f"us_arb_month_tracker_{port}_{via}_{selected_month}.csv",
            mime="text/csv"
        )
        
        # Store in session state
        st.session_state.us_arb_df = df_data
    else:
        st.error("No data returned from the API")