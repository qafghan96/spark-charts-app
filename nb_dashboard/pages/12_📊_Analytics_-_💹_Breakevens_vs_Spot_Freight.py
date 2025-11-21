import os
import sys
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import json
from io import StringIO

# Ensure we can import sibling module "utils.py" when run as a Streamlit page
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
APP_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if APP_ROOT not in sys.path:
    sys.path.insert(0, APP_ROOT)

from utils import (
    get_credentials,
    get_access_token,
    api_get,
    add_axis_controls,
    add_color_controls,
    apply_axis_limits,
)

st.title("💹 US Arb Freight Breakevens vs Spot Freight Rates")

st.caption("Compare US Arb Freight Breakevens with Spot Freight Rates.")

client_id, client_secret = get_credentials()
if not client_id or not client_secret:
    st.error("Missing Spark API credentials. Set streamlit secrets 'spark.client_id' and 'spark.client_secret' (or env vars).")
    st.stop()

scopes = "read:netbacks,read:access,read:prices,read:routes"
token = get_access_token(client_id, client_secret, scopes=scopes)

# Functions from the updated script
def list_netbacks(access_token):
    content = api_get("/v1.0/netbacks/reference-data/", access_token)
    
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

def fetch_breakevens(access_token, ticker, via=None, breakeven='freight', start=None, end=None, format='csv'):
    query_params = breakeven + '/' + "?fob-port={}".format(ticker)

    if via is not None:
        query_params += "&via-point={}".format(via)
    if start is not None:
        query_params += "&start={}".format(start)
    if end is not None:
        query_params += "&end={}".format(end)

    uri = "/v1.0/netbacks/arb-breakevens/{}".format(query_params)
    content = api_get(uri, access_token, format=format)
    
    if format == 'json':
        my_dict = content['data']
    else:
        # For CSV format, convert the data to DataFrame
        my_dict_temp = content.decode('utf-8')   
        my_dict = pd.read_csv(StringIO(my_dict_temp))

    return my_dict

def fetch_historical_freight_releases(access_token, ticker, limit=4, offset=None, vessel=None):
    query_params = "?limit={}".format(limit)
    if offset is not None:
        query_params += "&offset={}".format(offset)
    
    if vessel is not None:
        query_params += "&vessel-type={}".format(vessel)
    
    content = api_get("/v1.0/contracts/{}/price-releases/{}".format(ticker, query_params), access_token)
    my_dict = content['data']
    
    return my_dict

def fetch_freight_prices(access_token, ticker, my_lim, my_vessel=None):
    my_dict_hist = fetch_historical_freight_releases(access_token, ticker, limit=my_lim, vessel=my_vessel)
    
    release_dates = []
    period_start = []
    ticker_list = []
    usd_day = []
    day_min = []
    day_max = []

    for release in my_dict_hist:
        release_date = release["releaseDate"]
        ticker_list.append(release['contractId'])
        release_dates.append(release_date)

        data_points = release["data"][0]["dataPoints"]

        for data_point in data_points:
            period_start_at = data_point["deliveryPeriod"]["startAt"]
            period_start.append(period_start_at)

            usd_day.append(data_point['derivedPrices']['usdPerDay']['spark'])
            day_min.append(data_point['derivedPrices']['usdPerDay']['sparkMin'])
            day_max.append(data_point['derivedPrices']['usdPerDay']['sparkMax'])

    historical_df = pd.DataFrame({
        'Release Date': release_dates,
        'ticker': ticker_list,
        'Period Start': period_start,
        'USDperday': usd_day,
        'USDperdayMax': day_max,
        'USDperdayMin': day_min
    })

    historical_df['USDperday'] = pd.to_numeric(historical_df['USDperday'])
    historical_df['USDperdayMax'] = pd.to_numeric(historical_df['USDperdayMax'])
    historical_df['USDperdayMin'] = pd.to_numeric(historical_df['USDperdayMin'])
    historical_df['Release Date'] = pd.to_datetime(historical_df['Release Date'])
    
    return historical_df

# Get reference data
tickers, fobPort_names, availablevia, reldates, dicto1 = list_netbacks(token)
available_df = format_store(availablevia, fobPort_names, tickers)

# Configuration controls
st.subheader("Configuration")

col1, col2, col3 = st.columns(3)

with col1:
    # Port selection
    port = st.selectbox("Select Port", options=available_df["Ports"].tolist(), 
                       index=available_df["Ports"].tolist().index("Sabine Pass") if "Sabine Pass" in available_df["Ports"].tolist() else 0)

with col2:
    # Via point selection
    port_row = available_df[available_df["Ports"] == port].iloc[0]
    available_via_points = port_row["Available Via"]
    my_via = st.selectbox("Via Point", options=available_via_points, 
                         index=available_via_points.index('cogh') if 'cogh' in available_via_points else 0)

with col3:
    # Freight ticker selection
    freight_ticker = st.selectbox("Freight Ticker", options=['spark30s', 'spark25s'], index=0)

# Date range controls
st.subheader("Date Range")
col1, col2 = st.columns(2)

with col1:
    start_date = st.date_input(
        "Start Date", 
        value=datetime.datetime.today() - datetime.timedelta(days=380),
        help="Start date for the chart display range"
    )

with col2:
    end_date = st.date_input(
        "End Date", 
        value=datetime.datetime.today(),
        help="End date for the chart display range"
    )

# Get a filtered data sample for better axis defaults based on user's date range
@st.cache_data
def get_filtered_data_sample(token, port, my_via, freight_ticker, tickers, available_df, start_date, end_date):
    try:
        ti = int(available_df[available_df["Ports"] == port]["Index"].iloc[0])
        my_ticker = tickers[ti]
        
        # Fetch breakevens data
        break_df_sample = fetch_breakevens(token, my_ticker, via=my_via, breakeven='freight', format='csv')
        
        # Find date column
        date_column = None
        for col in break_df_sample.columns:
            if 'date' in col.lower() or 'release' in col.lower():
                date_column = col
                break
        
        if date_column:
            break_df_sample['ReleaseDate'] = pd.to_datetime(break_df_sample[date_column])
            
            # Get freight data with sufficient length
            sample_length = len(break_df_sample['ReleaseDate'].unique())
            freight_df_sample = fetch_freight_prices(token, freight_ticker, sample_length, my_vessel='174-2stroke')
            
            # Filter and merge data
            front_df_sample = break_df_sample[break_df_sample['LoadMonthIndex'] == "M+1"]
            freight_df_sample['Release Date'] = pd.to_datetime(freight_df_sample['Release Date'])
            merge_df_sample = pd.merge(freight_df_sample, front_df_sample, left_on='Release Date', right_on='ReleaseDate', how='inner')
            
            # Filter by user's date range
            start_datetime = pd.to_datetime(start_date)
            end_datetime = pd.to_datetime(end_date)
            date_filtered_df = merge_df_sample[
                (merge_df_sample['Release Date'] >= start_datetime) & 
                (merge_df_sample['Release Date'] <= end_datetime)
            ]
            
            return date_filtered_df[['USDperday', 'FreightBreakeven']] if not date_filtered_df.empty else pd.DataFrame()
        
        return pd.DataFrame()
    except:
        return pd.DataFrame()

# Get filtered data sample for axis defaults based on user's selected date range
data_sample = pd.DataFrame()
if port and my_via and freight_ticker:
    data_sample = get_filtered_data_sample(token, port, my_via, freight_ticker, tickers, available_df, start_date, end_date)

# Add color controls for the two data series
series_names = [f'{freight_ticker.upper()} (Atlantic)', 'US Arb [M+1] Freight Breakeven Level']
default_colors = ['#48C38D', '#4F41F4']  # Green and Blue (current chart colors)
color_controls = add_color_controls(series_names, default_colors, expanded=True)

# Add axis controls with data-driven defaults
axis_controls = add_axis_controls(expanded=True, data_df=data_sample, y_cols=['USDperday', 'FreightBreakeven'])

if st.button("Generate Chart", type="primary"):
    with st.spinner("Fetching data..."):
        try:
            # Get ticker for selected port
            ti = int(available_df[available_df["Ports"] == port]["Index"].iloc[0])
            my_ticker = tickers[ti]
            
            # Fetch breakevens data
            break_df = fetch_breakevens(token, my_ticker, via=my_via, breakeven='freight', format='csv')
            
            # Find the correct date column name
            date_column = None
            for col in break_df.columns:
                if 'date' in col.lower() or 'release' in col.lower():
                    date_column = col
                    break
            
            if date_column:
                break_df['ReleaseDate'] = pd.to_datetime(break_df[date_column])
            else:
                st.error("No date column found in breakevens data")
                st.stop()
            
            # Get length for freight data
            length = len(break_df['ReleaseDate'].unique())
            
            # Fetch freight prices
            freight_df = fetch_freight_prices(token, freight_ticker, length, my_vessel='174-2stroke')
            
            # Filter front month breakevens
            front_df = break_df[break_df['LoadMonthIndex'] == "M+1"]
            
            # Merge data
            freight_df['Release Date'] = pd.to_datetime(freight_df['Release Date'])
            merge_df = pd.merge(freight_df, front_df, left_on='Release Date', right_on='ReleaseDate', how='inner')
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.stop()

    # Create fig2 chart
    sns.set_style("whitegrid")
    fig2, ax2 = plt.subplots(figsize=(15, 7))

    # Get selected colors for the plot
    freight_color = color_controls[f'{freight_ticker.upper()} (Atlantic)']
    breakeven_color = color_controls['US Arb [M+1] Freight Breakeven Level']
    
    ax2.plot(merge_df['Release Date'], merge_df['USDperday'], 
             color=freight_color, linewidth=2.5, label=f'{freight_ticker.upper()} (Atlantic)')
    ax2.plot(merge_df['Release Date'], merge_df['FreightBreakeven'], 
             color=breakeven_color, linewidth=2, label='US Arb [M+1] Freight Breakeven Level')

    ax2.fill_between(merge_df['Release Date'], merge_df['USDperday'], merge_df['FreightBreakeven'],
                     where=merge_df['USDperday'] > merge_df['FreightBreakeven'], 
                     facecolor='red', interpolate=True, alpha=0.05)

    ax2.fill_between(merge_df['Release Date'], merge_df['USDperday'], merge_df['FreightBreakeven'],
                     where=merge_df['USDperday'] < merge_df['FreightBreakeven'], 
                     facecolor='green', interpolate=True, alpha=0.05)

    # Convert date inputs to datetime for matplotlib
    start_datetime = datetime.datetime.combine(start_date, datetime.time.min)
    end_datetime = datetime.datetime.combine(end_date, datetime.time.max)
    
    # Apply axis limits based on user controls
    if not axis_controls['x_auto']:
        # Use user-defined X limits
        ax2.set_xlim(axis_controls['x_min'], axis_controls['x_max'])
    else:
        # Use date range for X-axis
        ax2.set_xlim(start_datetime, end_datetime)
    
    # Filter the data by the selected date range for proper Y-axis auto-scaling
    date_filtered_merge_df = merge_df[
        (merge_df['Release Date'] >= start_datetime) & 
        (merge_df['Release Date'] <= end_datetime)
    ]
    
    # Apply Y-axis limits using the date-filtered data for auto-scaling
    apply_axis_limits(ax2, axis_controls, data_df=date_filtered_merge_df, y_cols=['USDperday', 'FreightBreakeven'])

    plt.title(f'{freight_ticker.upper()} (Atlantic) vs. US Arb [M+1] Freight Breakeven Level')
    sns.despine(left=True, bottom=True)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    st.pyplot(fig2)

    # Display merge_df table
    st.subheader("Merged Dataset")
    st.dataframe(merge_df, use_container_width=True)

st.markdown("---")
st.caption("This chart compares spot freight rates with US arbitrage freight breakevens.")