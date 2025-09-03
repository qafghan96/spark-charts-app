import os
import sys
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

# Ensure we can import sibling module "utils.py" when run as a Streamlit page
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
APP_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if APP_ROOT not in sys.path:
    sys.path.insert(0, APP_ROOT)

from utils import (
    get_credentials,
    get_access_token,
    api_get,
)

st.title("📈 Spot Seasonality Analysis")

st.caption("Plot seasonality charts for Spark30S and Spark25S freight rates with yearly comparisons.")

client_id, client_secret = get_credentials()
if not client_id or not client_secret:
    st.error("Missing Spark API credentials. Set streamlit secrets 'spark.client_id' and 'spark.client_secret' (or env vars).")
    st.stop()

scopes = "read:prices,read:routes"
token = get_access_token(client_id, client_secret, scopes=scopes)

# Functions from the script
def list_contracts(access_token):
    content = api_get("/v1.0/contracts/", access_token)
    tickers = []
    tick_names = []
    for contract in content["data"]:
        tickers.append(contract["id"])
        tick_names.append(contract["fullName"])
    return tickers, tick_names

def fetch_historical_price_releases(access_token, ticker, limit=4, offset=None, vessel=None):
    query_params = f"?limit={limit}"
    if offset is not None:
        query_params += f"&offset={offset}"
    if vessel is not None:
        query_params += f"&vessel-type={vessel}"
    
    content = api_get(f"/v1.0/contracts/{ticker}/price-releases/{query_params}", access_token)
    return content['data']

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
            cal_month.append(datetime.strptime(period_start_at, '%Y-%m-%d').strftime('%b-%Y'))

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

def sort_years(df):
    reldates = df['Release Date'].to_list()
    
    years = []
    months = []
    days = []
    dayofyear = []
    for r in reldates:
        rsplit = r.split('-')
        years.append(rsplit[0])
        months.append(rsplit[1])
        days.append(rsplit[2])
        dayofyear.append(datetime.strptime(r, '%Y-%m-%d').timetuple().tm_yday)

    df['Year'] = years
    df['Month'] = months
    df['Day'] = days
    df['Day of Year'] = dayofyear
    
    seas_check = [['04','05','06','07','08','09'], ['10','11','12','01','02','03']]
    quart_check = [['01','02','03'],['04','05','06'],['07','08','09'],['10','11','12']]

    seasons = []
    quarters = []

    for i in df['Month'].to_list():
        if i in quart_check[0]:
            quarters.append('Q1')
        elif i in quart_check[1]:
            quarters.append('Q2')
        elif i in quart_check[2]:
            quarters.append('Q3')
        elif i in quart_check[3]:
            quarters.append('Q4')

        if i in seas_check[0]:
            seasons.append('Summer')
        elif i in seas_check[1]:
            seasons.append('Winter')

    df['Quarters'] = quarters
    df['Seasons'] = seasons

    return df

# Configuration controls
st.subheader("Configuration")

col1, col2, col3 = st.columns(3)

with col1:
    contract_type = st.selectbox("Contract Type", options=["Spark30S", "Spark25S"], index=0)

with col2:
    vessel_type = st.selectbox("Vessel Type", options=["174-2stroke", "160-tfde"], index=0)

with col3:
    data_limit = st.number_input("Historical Data Limit", min_value=50, max_value=2000, value=1000, step=50,
                               help="Number of historical price releases to fetch")

if st.button("Generate Seasonality Analysis", type="primary"):
    with st.spinner("Fetching data..."):
        try:
            # Get contract ticker
            ticker = contract_type.lower() + 's'
            
            # Fetch price data
            price_df = fetch_prices(token, ticker, data_limit, my_vessel=vessel_type)
            
            # Sort by years and add seasonal columns
            price_df = sort_years(price_df)
            
            # Store data in session state
            st.session_state['price_df'] = price_df
            st.session_state['contract_type'] = contract_type
            st.session_state['vessel_type'] = vessel_type
            
        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")
            st.stop()

# Display data and chart if data exists
if 'price_df' in st.session_state:
    price_df = st.session_state['price_df']
    stored_contract_type = st.session_state['contract_type']
    stored_vessel_type = st.session_state['vessel_type']
    
    st.subheader(f"Historical Data - {stored_contract_type} ({stored_vessel_type})")
    st.dataframe(price_df, use_container_width=True)
    
    # Chart Configuration
    st.subheader("Seasonality Chart Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Year selection
        available_years = sorted(price_df['Year'].unique(), reverse=True)
        selected_years = st.multiselect(
            "Select years to display",
            options=available_years,
            default=available_years[:5] if len(available_years) >= 5 else available_years,
            help="Select which years to show on the seasonality chart"
        )
        
        # Current year highlighting
        highlight_year = st.selectbox(
            "Highlight current year",
            options=available_years,
            index=0,
            help="This year will be highlighted with a bold line and min/max range"
        )
    
    with col2:
        # Chart customization
        chart_style = st.selectbox("Chart Style", options=["Monthly Labels", "Bimonthly Labels"], index=0)
        
        show_range = st.checkbox("Show Min/Max Range", value=True,
                               help="Show min/max price range for the highlighted year")
        
        auto_scale_y = st.checkbox("Auto-scale Y-axis", value=True,
                                 help="Automatically scale Y-axis based on selected data")
    
    if st.button("Generate Chart", type="secondary") and selected_years:
        # Filter data to selected years
        filtered_df = price_df[price_df['Year'].isin(selected_years)]
        
        sns.set_style("whitegrid")
        fig, ax = plt.subplots(figsize=(15, 9))
        
        # Get year list
        year_list = sorted(selected_years, reverse=True)
        
        # Plot data for each year
        for y, year in enumerate(year_list):
            hdf = filtered_df[filtered_df['Year'] == year]
            
            if year == highlight_year:
                # Highlight the selected year
                ax.plot(hdf['Day of Year'], hdf['USDperday'], 
                       color='#48C38D', linewidth=3.0, label=year)
                
                if show_range:
                    ax.plot(hdf['Day of Year'], hdf['USDperdayMin'], 
                           color='#48C38D', alpha=0.1)
                    ax.plot(hdf['Day of Year'], hdf['USDperdayMax'], 
                           color='#48C38D', alpha=0.1)
                    ax.fill_between(hdf['Day of Year'], hdf['USDperdayMin'], 
                                  hdf['USDperdayMax'], color='#48C38D', alpha=0.2)
            else:
                # Other years with lower alpha
                ax.plot(hdf['Day of Year'], hdf['USDperday'], 
                       alpha=0.4, label=year, linewidth=1.5)
        
        # Set title and labels
        ax.set_title(f'Yearly Comparison of {stored_contract_type} ({stored_vessel_type})', fontsize=16)
        plt.xlabel('Date (Day of Year)', fontsize=12)
        plt.ylabel('USD per day', fontsize=12)
        plt.legend()
        
        # Set x-axis ticks and labels based on chart style
        if chart_style == "Monthly Labels":
            xlabels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Year End']
            xpos = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 365]
        else:  # Bimonthly Labels
            xlabels = ['January', 'March', 'May', 'July', 'September', 'November', 'Year End']
            xpos = [1, 60, 121, 182, 244, 305, 365]
        
        plt.xticks(xpos, xlabels, rotation=45 if chart_style == "Monthly Labels" else 0)
        
        # Format y-axis with currency
        current_values = plt.gca().get_yticks()
        plt.gca().set_yticklabels(['$ {:,.0f}'.format(x) for x in current_values])
        
        # Auto-scale y-axis if requested
        if auto_scale_y:
            y_values = filtered_df['USDperday'].dropna()
            if show_range and highlight_year in selected_years:
                highlight_df = filtered_df[filtered_df['Year'] == highlight_year]
                y_values_with_range = pd.concat([
                    y_values, 
                    highlight_df['USDperdayMin'].dropna(),
                    highlight_df['USDperdayMax'].dropna()
                ])
                y_min = y_values_with_range.min()
                y_max = y_values_with_range.max()
            else:
                y_min = y_values.min()
                y_max = y_values.max()
            
            y_range = y_max - y_min
            padding = y_range * 0.1 if y_range > 0 else 1000
            plt.ylim(max(0, y_min - padding), y_max + padding)
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        sns.despine(left=True, bottom=True)
        
        st.pyplot(fig)
        
        # Display summary statistics
        st.subheader("Summary Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Years", len(selected_years))
            st.metric("Total Data Points", len(filtered_df))
        
        with col2:
            if highlight_year in selected_years:
                highlight_data = filtered_df[filtered_df['Year'] == highlight_year]['USDperday']
                if not highlight_data.empty:
                    st.metric(f"{highlight_year} Average", f"${highlight_data.mean():,.0f}")
                    st.metric(f"{highlight_year} Range", 
                            f"${highlight_data.min():,.0f} - ${highlight_data.max():,.0f}")
        
        with col3:
            all_data = filtered_df['USDperday']
            if not all_data.empty:
                st.metric("Overall Average", f"${all_data.mean():,.0f}")
                st.metric("Overall Range", f"${all_data.min():,.0f} - ${all_data.max():,.0f}")

st.markdown("---")
st.caption("This seasonality analysis shows how freight rates vary throughout the year across different time periods, helping identify seasonal patterns and trends.")