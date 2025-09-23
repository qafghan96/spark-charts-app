import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import json
from base64 import b64encode
from urllib.parse import urljoin
from urllib import request
from urllib.error import HTTPError
from datetime import datetime, timedelta
import time

# Ensure we can import sibling module "utils.py" when run as a Streamlit page
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
APP_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if APP_ROOT not in sys.path:
    sys.path.insert(0, APP_ROOT)

from utils import get_credentials, get_access_token

st.title("📈 Friday Press Statement v2")
st.caption("Analyze freight and cargo price trends with historical comparisons for press statements.")

# Get credentials
client_id, client_secret = get_credentials()
if not client_id or not client_secret:
    st.error("Missing Spark API credentials. Set streamlit secrets 'spark.client_id' and 'spark.client_secret' (or env vars).")
    st.stop()

scopes = "read:prices"
token = get_access_token(client_id, client_secret, scopes=scopes)

# API functions
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

def fetch_historical_price_releases(access_token, ticker, limit=1000, vessel=None):
    """Fetch historical price releases for a contract."""
    query_params = "?limit={}".format(limit)
    if vessel is not None:
        query_params += "&vessel-type={}".format(vessel)
    
    content = do_api_get_query(
        uri="/v1.0/contracts/{}/price-releases/{}".format(ticker, query_params),
        access_token=access_token,
    )
    return content['data']

def hist_sort(data, product):
    """Process historical data into DataFrame."""
    release_dates = []
    period_start = []
    ticker = []
    usd_day = []
    cal_month = []

    for release in data:
        release_date = release["releaseDate"]
        ticker.append(release['contractId'])
        release_dates.append(release_date)

        data_points = release["data"][0]["dataPoints"]

        for data_point in data_points:
            period_start_at = data_point["deliveryPeriod"]["startAt"]
            period_start.append(period_start_at)

            if product == 'freight':
                usd_day.append(data_point['derivedPrices']['usdPerDay']['spark'])
            elif product == 'cargo':
                usd_day.append(data_point['derivedPrices']['usdPerMMBtu']['spark'])

            cal_month.append(datetime.strptime(period_start_at, '%Y-%m-%d').strftime('%b-%Y'))

    histdf = pd.DataFrame({
        'Release Date': release_dates,
        'ticker': ticker,
        'Period Start': period_start,
        'Price': usd_day
    })

    histdf['Price'] = pd.to_numeric(histdf['Price'])
    return histdf

def weekly(df, day, product):
    """Filter DataFrame to specific weekdays and calculate weekly differences."""
    df = df.copy()  # Fix pandas warning
    dates = df['Release Date'].to_list()
    dates_weekly = []
    
    for d in dates:
        d2 = datetime.strptime(d, '%Y-%m-%d')
        if product == 'freight':
            if d2.year < 2024:
                if day[1] == d2.strftime('%A'):
                    dates_weekly.append(d)
            else:
                if day[0] == d2.strftime('%A'):
                    dates_weekly.append(d)
        else:
            if day == d2.strftime('%A'):
                dates_weekly.append(d)
    
    new_df = df[df['Release Date'].isin(dates_weekly)].copy()
    
    prices = new_df['Price'].to_list()
    diffs = []
    for p in range(len(prices)):
        if p == len(prices) - 1:
            diffs.append(0)
        else:
            diffs.append(prices[p] - prices[p + 1])
    
    new_df['Diff'] = diffs
    new_df = new_df.drop(columns=['Period Start'])
    
    return new_df

def fuzzy_date_match(target_date, df, tolerance_days=7):
    """Find the closest date in DataFrame within tolerance."""
    target_dt = datetime.strptime(target_date, '%Y-%m-%d')
    
    # Try exact match first
    if target_date in df['Release Date'].values:
        return df[df['Release Date'] == target_date]['Price'].values[0]
    
    # Try fuzzy matching within tolerance
    for delta in range(1, tolerance_days + 1):
        # Try earlier dates
        earlier_date = (target_dt - timedelta(days=delta)).strftime('%Y-%m-%d')
        if earlier_date in df['Release Date'].values:
            return df[df['Release Date'] == earlier_date]['Price'].values[0]
        
        # Try later dates
        later_date = (target_dt + timedelta(days=delta)).strftime('%Y-%m-%d')
        if later_date in df['Release Date'].values:
            return df[df['Release Date'] == later_date]['Price'].values[0]
    
    return np.nan

def calculate_yearly_comparisons(df):
    """Calculate improved yearly comparisons with fuzzy date matching."""
    current_price = df['Price'].iloc[0]
    current_date = df['Release Date'].iloc[0]
    current_datetime = datetime.strptime(current_date, '%Y-%m-%d')
    
    yearly_data = {'Current': current_price}
    yearly_diffs = {'Current': current_price}
    
    for year_back in range(1, 6):
        weeks_back = 52 * year_back
        target_date = current_datetime - timedelta(weeks=weeks_back)
        target_date_str = target_date.strftime('%Y-%m-%d')
        
        # Use fuzzy matching with 7-day tolerance
        historical_price = fuzzy_date_match(target_date_str, df, tolerance_days=7)
        
        yearly_data[f'Year - {year_back}'] = historical_price
        if not np.isnan(historical_price):
            yearly_diffs[f'Year{year_back}-diff'] = current_price - historical_price
        else:
            yearly_diffs[f'Year{year_back}-diff'] = np.nan
    
    return pd.DataFrame([yearly_data]), pd.DataFrame([yearly_diffs])

def generate_statistics(df, dataset_name):
    """Generate key statistics for press statement."""
    current_price = df['Price'].iloc[0]
    
    # Record highs and lows
    min_record = df[df['Price'] <= current_price]
    max_record = df[df['Price'] >= current_price]
    
    # Weekly delta records
    max_delta_record = df[df['Diff'] >= df['Diff'].iloc[0]]
    min_delta_record = df[df['Diff'] <= df['Diff'].iloc[0]]
    
    stats = {
        'Latest Price': current_price,
        'Record Low Since': {
            'Date': min_record['Release Date'].iloc[1] if len(min_record) > 1 else 'N/A',
            'Price': min_record['Price'].iloc[1] if len(min_record) > 1 else 'N/A'
        },
        'Record High Since': {
            'Date': max_record['Release Date'].iloc[1] if len(max_record) > 1 else 'N/A',
            'Price': max_record['Price'].iloc[1] if len(max_record) > 1 else 'N/A'
        },
        'Highest Weekly Delta Since': {
            'Date': max_delta_record['Release Date'].iloc[1] if len(max_delta_record) > 1 else 'N/A',
            'Delta': max_delta_record['Diff'].iloc[1] if len(max_delta_record) > 1 else 'N/A'
        },
        'Lowest Weekly Delta Since': {
            'Date': min_delta_record['Release Date'].iloc[1] if len(min_delta_record) > 1 else 'N/A',
            'Delta': min_delta_record['Diff'].iloc[1] if len(min_delta_record) > 1 else 'N/A'
        }
    }
    
    return stats

# Initialize session state for data caching
if 'press_data_loaded' not in st.session_state:
    st.session_state.press_data_loaded = False
    st.session_state.datasets = {}

# Configuration
st.subheader("Configuration")

col1, col2 = st.columns(2)
with col1:
    selected_dataset = st.selectbox(
        "Select Dataset", 
        ["Spark30S (Freight)", "Spark25S (Freight)", "SparkNWE-B-F (Cargo Basis)", "SparkNWE-F (Cargo DES)"],
        index=0
    )

with col2:
    if st.button("Load All Data", type="primary"):
        with st.spinner("Loading historical data for all datasets..."):
            try:
                # Dataset configurations
                datasets_config = {
                    'Spark30S (Freight)': {'ticker': 'spark30s', 'vessel': '174-2stroke', 'weekdays': ['Friday', 'Tuesday'], 'product': 'freight'},
                    'Spark25S (Freight)': {'ticker': 'spark25s', 'vessel': '174-2stroke', 'weekdays': ['Friday', 'Tuesday'], 'product': 'freight'},
                    'SparkNWE-B-F (Cargo Basis)': {'ticker': 'sparknwe-b-f', 'vessel': None, 'weekdays': 'Thursday', 'product': 'cargo'},
                    'SparkNWE-F (Cargo DES)': {'ticker': 'sparknwe-f', 'vessel': None, 'weekdays': 'Thursday', 'product': 'cargo'}
                }
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, (name, config) in enumerate(datasets_config.items()):
                    status_text.text(f'Loading {name}...')
                    progress_bar.progress((i + 1) / len(datasets_config))
                    
                    # Fetch historical data
                    hist_data = fetch_historical_price_releases(
                        token, 
                        config['ticker'], 
                        vessel=config['vessel']
                    )
                    
                    # Process data
                    hist_df = hist_sort(hist_data, config['product'])
                    weekly_df = weekly(hist_df, config['weekdays'], config['product'])
                    
                    # Calculate yearly comparisons
                    years_df, yearsdiff_df = calculate_yearly_comparisons(weekly_df)
                    
                    # Generate statistics
                    stats = generate_statistics(weekly_df, name)
                    
                    # Store in session state
                    st.session_state.datasets[name] = {
                        'weekly_df': weekly_df,
                        'years_df': years_df,
                        'yearsdiff_df': yearsdiff_df,
                        'stats': stats,
                        'config': config
                    }
                    
                    time.sleep(0.1)  # Rate limiting
                
                progress_bar.empty()
                status_text.empty()
                st.session_state.press_data_loaded = True
                st.success("All datasets loaded successfully!")
                
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")

# Display data if loaded
if st.session_state.press_data_loaded and selected_dataset in st.session_state.datasets:
    data = st.session_state.datasets[selected_dataset]
    
    st.subheader(f"Analysis for {selected_dataset}")
    
    # Display key statistics
    st.subheader("📊 Key Statistics")
    stats = data['stats']
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Latest Price", f"${stats['Latest Price']:,.0f}")
    with col2:
        if stats['Record Low Since']['Price'] != 'N/A':
            st.metric("Record Low", f"${stats['Record Low Since']['Price']:,.0f}")
            st.caption(f"Since {stats['Record Low Since']['Date']}")
    with col3:
        if stats['Record High Since']['Price'] != 'N/A':
            st.metric("Record High", f"${stats['Record High Since']['Price']:,.0f}")
            st.caption(f"Since {stats['Record High Since']['Date']}")
    with col4:
        weekly_change = data['weekly_df']['Diff'].iloc[0]
        st.metric("Weekly Change", f"${weekly_change:,.0f}")
    
    # Yearly Comparisons
    st.subheader("📅 Yearly Price Comparisons")
    st.write("**Historical Price Comparison (Fuzzy Date Matched)**")
    st.dataframe(data['years_df'], use_container_width=True)
    
    st.write("**Price Differences vs Historical Years**")
    st.dataframe(data['yearsdiff_df'], use_container_width=True)
    
    # Weekly Delta Analysis
    st.subheader("📈 Weekly Delta Analysis")
    col1, col2 = st.columns(2)
    with col1:
        if stats['Highest Weekly Delta Since']['Delta'] != 'N/A':
            st.info(f"**Highest Weekly Gain**: ${stats['Highest Weekly Delta Since']['Delta']:,.0f} on {stats['Highest Weekly Delta Since']['Date']}")
    with col2:
        if stats['Lowest Weekly Delta Since']['Delta'] != 'N/A':
            st.info(f"**Largest Weekly Drop**: ${stats['Lowest Weekly Delta Since']['Delta']:,.0f} on {stats['Lowest Weekly Delta Since']['Date']}")
    
    # Recent Data Table
    st.subheader("📋 Recent Price Data")
    display_limit = st.slider("Number of recent records to display", 10, 100, 20)
    st.dataframe(data['weekly_df'].head(display_limit), use_container_width=True)
    
    # Download functionality
    @st.cache_data
    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')
    
    csv_weekly = convert_df(data['weekly_df'])
    csv_yearly = convert_df(data['years_df'])
    csv_yearly_diff = convert_df(data['yearsdiff_df'])
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.download_button(
            label="📥 Download Weekly Data",
            data=csv_weekly,
            file_name=f'{selected_dataset.lower().replace(" ", "_")}_weekly_data.csv',
            mime='text/csv'
        )
    with col2:
        st.download_button(
            label="📥 Download Yearly Comparison",
            data=csv_yearly,
            file_name=f'{selected_dataset.lower().replace(" ", "_")}_yearly_comparison.csv',
            mime='text/csv'
        )
    with col3:
        st.download_button(
            label="📥 Download Yearly Differences",
            data=csv_yearly_diff,
            file_name=f'{selected_dataset.lower().replace(" ", "_")}_yearly_differences.csv',
            mime='text/csv'
        )

elif not st.session_state.press_data_loaded:
    st.info("👆 Click 'Load All Data' to fetch and analyze the price data for all datasets.")

# Information section
with st.expander("ℹ️ About This Analysis"):
    st.markdown("""
    **Friday Press Statement v2** provides comprehensive analysis of freight and cargo price trends:
    
    **Data Sources:**
    - **Spark30S/25S**: LNG freight rates for different vessel sizes (174-2stroke)
    - **SparkNWE-B-F**: Northwest Europe cargo basis prices
    - **SparkNWE-F**: Northwest Europe DES cargo prices
    
    **Key Features:**
    - **Fuzzy Date Matching**: Automatically finds the closest available price data when exact historical dates fall on weekends
    - **Weekly Analysis**: Filters data to specific publication days (Friday/Tuesday for freight, Thursday for cargo)
    - **Multi-Year Comparison**: Compares current prices with equivalent periods from the last 5 years
    - **Statistical Analysis**: Identifies record highs, lows, and significant weekly changes
    
    **Weekly Day Logic:**
    - Freight contracts: Friday releases (2024+), Tuesday releases (pre-2024)
    - Cargo contracts: Thursday releases
    
    This analysis is designed to support press statement preparation with accurate historical context and market insights.
    """)