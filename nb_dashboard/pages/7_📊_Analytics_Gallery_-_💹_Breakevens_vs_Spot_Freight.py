import os
import sys
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from io import StringIO

# Ensure we can import sibling module "utils.py" when run as a Streamlit page
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
APP_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if APP_ROOT not in sys.path:
    sys.path.insert(0, APP_ROOT)

from utils import (
    get_credentials,
    get_access_token,
    list_netbacks_reference,
    api_get,
)

st.title("💹 US Arb Freight Breakevens vs Spot Freight Rates")

st.caption("Compare US Arb Freight Breakevens with Spot Freight Rates.")

client_id, client_secret = get_credentials()
if not client_id or not client_secret:
    st.error("Missing Spark API credentials. Set streamlit secrets 'spark.client_id' and 'spark.client_secret' (or env vars).")
    st.stop()

scopes = "read:netbacks,read:access,read:prices,read:routes"
token = get_access_token(client_id, client_secret, scopes=scopes)

# Get netbacks reference data
tickers, names, available_via, release_dates, _raw = list_netbacks_reference(token)

# Helper functions from the notebook
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

def fetch_breakevens(access_token, ticker, nea_via=None, nwe_via=None, format='csv'):
    query_params = "?fob-port={}".format(ticker)
    if nea_via is not None:
        query_params += "&nea-via-point={}".format(nea_via)
    if nwe_via is not None:
        query_params += "&nwe-via-point={}".format(nwe_via)
    
    try:
        content = api_get(f"/beta/netbacks/arb-breakevens/{query_params}", access_token)
        
        if format == 'json':
            return content
        else:
            # Convert JSON data to DataFrame
            data_list = content.get('data', [])
            if not data_list:
                return pd.DataFrame()
            
            df = pd.DataFrame(data_list)
            return df
            
    except Exception as e:
        st.error(f"Error fetching breakevens data: {e}")
        return pd.DataFrame()

def fetch_historical_price_releases(access_token, ticker, limit=4, offset=None, vessel=None):
    query_params = "?limit={}".format(limit)
    if offset is not None:
        query_params += "&offset={}".format(offset)
    
    if vessel is not None:
        query_params += "&vessel-type={}".format(vessel)
    
    content = api_get(f"/v1.0/contracts/{ticker}/price-releases/{query_params}", access_token)
    return content.get('data', [])

def fetch_prices(access_token, ticker, my_lim, my_vessel=None):
    my_dict_hist = fetch_historical_price_releases(access_token, ticker, limit=my_lim, vessel=my_vessel)
    
    release_dates = []
    period_start = []
    ticker_list = []
    usd_day = []
    usd_mmbtu = []
    day_min = []
    day_max = []
    cal_month = []

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
            usd_mmbtu.append(data_point['derivedPrices']['usdPerMMBtu']['spark'])
            cal_month.append(datetime.datetime.strptime(period_start_at, '%Y-%m-%d').strftime('%b-%Y'))

    historical_df = pd.DataFrame({
        'ticker': ticker_list,
        'Period Start': period_start,
        'USDperday': usd_day,
        'USDperdayMax': day_max,
        'USDperdayMin': day_min,
        'USDperMMBtu': usd_mmbtu,
        'Release Date': release_dates
    })

    historical_df['USDperday'] = pd.to_numeric(historical_df['USDperday'])
    historical_df['USDperdayMax'] = pd.to_numeric(historical_df['USDperdayMax'])
    historical_df['USDperdayMin'] = pd.to_numeric(historical_df['USDperdayMin'])
    historical_df['USDperMMBtu'] = pd.to_numeric(historical_df['USDperMMBtu'])
    historical_df['Release Datetime'] = pd.to_datetime(historical_df['Release Date'])
    
    return historical_df

# Create available ports dataframe
available_df = format_store(available_via, names, tickers)

# Configuration controls
st.subheader("Configuration")

col1, col2, col3 = st.columns(3)

with col1:
    # Port selection (matching script's "port" variable)
    available_ports = available_df["Ports"].tolist()
    default_port_idx = 0
    if "Sabine Pass" in available_ports:
        default_port_idx = available_ports.index("Sabine Pass")
    
    port = st.selectbox("Select Port", options=available_ports, index=default_port_idx)

with col2:
    # Get available via points for selected port
    port_row = available_df[available_df["Ports"] == port].iloc[0]
    available_via_points = eval(port_row["Available Via"]) if isinstance(port_row["Available Via"], str) else port_row["Available Via"]
    
    # Via point selection (matching script's "my_via" variable)
    if available_via_points:
        default_via_idx = 0
        if 'cogh' in available_via_points:
            default_via_idx = available_via_points.index('cogh')
        my_via = st.selectbox("Select Via Point", options=available_via_points, index=default_via_idx)
    else:
        my_via = st.text_input("Via Point", value="cogh")

with col3:
    # Freight ticker selection (matching script's freight ticker usage)
    freight_tickers = ['spark30s', 'spark25s']
    freight_ticker = st.selectbox("Select Freight Ticker", options=freight_tickers, index=0)
    
    # Vessel type for freight ticker
    vessel_type = st.selectbox("Vessel Type", options=['174-2stroke', '160-tfde'], index=0)

if st.button("Generate Breakevens vs Spot Freight Chart", type="primary"):
    with st.spinner("Fetching data..."):
        try:
            # Get ticker UUID for selected port
            ti = available_df[available_df["Ports"] == port]["Index"].iloc[0]
            my_ticker = tickers[ti]
            
            # Fetch breakevens data
            break_df = fetch_breakevens(token, my_ticker, nea_via=my_via, format='csv')
            
            if break_df.empty:
                st.error("No breakevens data available for selected parameters.")
                st.stop()
            
            # Debug: show columns to understand structure
            st.write("Breakevens DataFrame columns:", break_df.columns.tolist())
            st.write("Breakevens DataFrame sample:", break_df.head())
            
            # Check if ReleaseDate column exists, if not try alternatives
            if 'ReleaseDate' in break_df.columns:
                break_df['ReleaseDate'] = pd.to_datetime(break_df['ReleaseDate'])
            elif 'releaseDate' in break_df.columns:
                break_df['ReleaseDate'] = pd.to_datetime(break_df['releaseDate'])
            elif 'release_date' in break_df.columns:
                break_df['ReleaseDate'] = pd.to_datetime(break_df['release_date'])
            else:
                st.error(f"Could not find release date column. Available columns: {break_df.columns.tolist()}")
                st.stop()
            
            # Get length for freight data limit
            length = len(break_df['ReleaseDate'].unique())
            
            # Fetch spot freight prices
            freight_df = fetch_prices(token, freight_ticker, length, my_vessel=vessel_type)
            
            if freight_df.empty:
                st.error("No freight data available for selected ticker.")
                st.stop()
            
            # Filter to front month breakevens only
            front_df = break_df[break_df['LoadMonthIndex'] == "M+1"].copy()
            
            # Prepare data for merging
            freight_df['Release Date'] = pd.to_datetime(freight_df['Release Date'])
            merge_df = pd.merge(freight_df, front_df, left_on='Release Date', right_on='ReleaseDate', how='inner')
            
            if merge_df.empty:
                st.warning("No overlapping data found between freight and breakevens data.")
                st.stop()
            
        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")
            st.stop()

    # Create fig2 chart (the main result from the script)
    sns.set_style("whitegrid")
    fig2, ax2 = plt.subplots(figsize=(15, 7))

    # Plot the lines
    ax2.plot(merge_df['Release Date'], merge_df['USDperday'], 
             color='#48C38D', linewidth=2.5, label=f'{freight_ticker.upper()} ({vessel_type})')
    ax2.plot(merge_df['Release Date'], merge_df['FreightBreakevenUSDPerDay'], 
             color='#4F41F4', linewidth=2, label='US Arb [M+1] Freight Breakeven Level')

    # Add conditional shading
    ax2.fill_between(merge_df['Release Date'], merge_df['USDperday'], merge_df['FreightBreakevenUSDPerDay'],
                     where=merge_df['USDperday'] > merge_df['FreightBreakevenUSDPerDay'], 
                     facecolor='red', interpolate=True, alpha=0.05)

    ax2.fill_between(merge_df['Release Date'], merge_df['USDperday'], merge_df['FreightBreakevenUSDPerDay'],
                     where=merge_df['USDperday'] < merge_df['FreightBreakevenUSDPerDay'], 
                     facecolor='green', interpolate=True, alpha=0.05)

    # Set limits and formatting
    ax2.set_xlim(datetime.datetime.today() - datetime.timedelta(days=380), 
                 datetime.datetime.today())
    ax2.set_ylim(-100000, 120000)

    plt.title(f'{freight_ticker.upper()} vs. US Arb [M+1] Freight Breakeven Level - {port} via {my_via}')
    plt.ylabel('USD per Day')
    plt.xlabel('Release Date')

    sns.despine(left=True, bottom=True)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    st.pyplot(fig2)

    # Display the merge_df dataframe as requested
    st.subheader("Merged Dataset")
    st.caption("Combined freight prices and breakeven data used in the chart above")
    
    # Prepare display dataframe
    display_df = merge_df[['Release Date', 'USDperday', 'FreightBreakevenUSDPerDay', 
                          'ArbUSDPerMBBtu', 'LoadMonthIndex', 'FobPortSlug', 'NEAViaPoint']].copy()
    
    # Format columns for better display
    display_df['USDperday'] = display_df['USDperday'].apply(lambda x: f"${x:,.0f}")
    display_df['FreightBreakevenUSDPerDay'] = display_df['FreightBreakevenUSDPerDay'].apply(lambda x: f"${x:,.0f}")
    display_df['ArbUSDPerMBBtu'] = display_df['ArbUSDPerMBBtu'].apply(lambda x: f"${x:.2f}")
    
    # Rename columns for clarity
    display_df.columns = ['Release Date', 'Spot Freight (USD/day)', 'Breakeven Level (USD/day)', 
                         'Arb (USD/MMBtu)', 'Load Month', 'FoB Port', 'Via Point']
    
    st.dataframe(display_df, use_container_width=True)

    # Summary statistics
    st.subheader("Summary Statistics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_freight = merge_df['USDperday'].mean()
        st.metric("Average Spot Freight", f"${avg_freight:,.0f}")
    
    with col2:
        avg_breakeven = merge_df['FreightBreakevenUSDPerDay'].mean()
        st.metric("Average Breakeven", f"${avg_breakeven:,.0f}")
    
    with col3:
        spread = merge_df['USDperday'] - merge_df['FreightBreakevenUSDPerDay']
        avg_spread = spread.mean()
        st.metric("Average Spread", f"${avg_spread:,.0f}")

st.markdown("---")
st.caption("This chart compares spot freight rates with US arbitrage freight breakevens, with green shading indicating when freight is below breakeven (favorable for arbitrage) and red shading when above breakeven.")