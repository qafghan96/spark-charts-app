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
    add_axis_controls,
    add_color_controls,
    apply_axis_limits,
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
            ticker = contract_type.lower()
            
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
    
    # Add axis controls with data-driven defaults
    axis_controls = add_axis_controls(expanded=True, data_df=price_df, y_cols=['USDperday'])
    
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
        
    # Add color controls for selected years
    if selected_years:
        year_series_names = [f"Year {year}" for year in selected_years[:5]]  # Limit to first 5 for UI
        default_year_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        year_color_controls = add_color_controls(
            year_series_names, 
            default_year_colors[:len(year_series_names)], 
            expanded=True
        )
    
    with col2:
        # Chart customization
        chart_style = st.selectbox("Chart Style", options=["Monthly Labels", "Bimonthly Labels"], index=0)
        
        show_range = st.checkbox("Show Min/Max Range", value=True,
                               help="Show min/max price range for the highlighted year")
        
        # Chart range controls
        st.write("**Chart Range (Months)**")
        col_month_start, col_month_end = st.columns(2)
        
        month_names = ['January', 'February', 'March', 'April', 'May', 'June',
                      'July', 'August', 'September', 'October', 'November', 'December']
        
        with col_month_start:
            start_month = st.selectbox("Start Month", options=month_names, index=0,
                                     help="First month to display on chart")
        
        with col_month_end:
            end_month = st.selectbox("End Month", options=month_names, index=11,
                                   help="Last month to display on chart")
        
        # Y-axis controls
        st.write("**Y-Axis Range**")
        auto_scale_y = st.checkbox("Auto-scale Y-axis", value=True,
                                 help="Automatically scale Y-axis based on selected data")
        
        if not auto_scale_y:
            col_y_min, col_y_max = st.columns(2)
            with col_y_min:
                y_min_manual = st.number_input("Y-axis Min ($)", value=0, step=5000,
                                             help="Minimum value for Y-axis")
            with col_y_max:
                y_max_manual = st.number_input("Y-axis Max ($)", value=100000, step=5000,
                                             help="Maximum value for Y-axis")
    
    if st.button("Generate Chart", type="secondary") and selected_years:
        # Convert month names to numbers for filtering
        start_month_num = month_names.index(start_month) + 1
        end_month_num = month_names.index(end_month) + 1
        
        # Filter data to selected years
        filtered_df = price_df[price_df['Year'].isin(selected_years)].copy()
        
        # Filter by month range
        filtered_df['Month_num'] = filtered_df['Month'].astype(int)
        
        if start_month_num <= end_month_num:
            # Normal range (e.g., Jan to Jun)
            month_filtered_df = filtered_df[
                (filtered_df['Month_num'] >= start_month_num) & 
                (filtered_df['Month_num'] <= end_month_num)
            ]
        else:
            # Cross-year range (e.g., Oct to Mar)
            month_filtered_df = filtered_df[
                (filtered_df['Month_num'] >= start_month_num) | 
                (filtered_df['Month_num'] <= end_month_num)
            ]
        
        if month_filtered_df.empty:
            st.warning("No data available for the selected month range and years.")
            st.stop()
        
        sns.set_style("whitegrid")
        fig, ax = plt.subplots(figsize=(15, 9))
        
        # Get year list
        year_list = sorted(selected_years, reverse=True)
        
        # Plot data for each year
        for y, year in enumerate(year_list):
            hdf = month_filtered_df[month_filtered_df['Year'] == year]
            
            if hdf.empty:
                continue
                
            # Get color for this year
            year_color = '#1f77b4'  # Default color
            if f"Year {year}" in year_color_controls:
                year_color = year_color_controls[f"Year {year}"]
            
            if year == highlight_year:
                # Highlight the selected year
                ax.plot(hdf['Day of Year'], hdf['USDperday'], 
                       color=year_color, linewidth=3.0, label=year)
                
                if show_range:
                    ax.plot(hdf['Day of Year'], hdf['USDperdayMin'], 
                           color=year_color, alpha=0.1)
                    ax.plot(hdf['Day of Year'], hdf['USDperdayMax'], 
                           color=year_color, alpha=0.1)
                    ax.fill_between(hdf['Day of Year'], hdf['USDperdayMin'], 
                                  hdf['USDperdayMax'], color=year_color, alpha=0.2)
            else:
                # Other years with lower alpha
                ax.plot(hdf['Day of Year'], hdf['USDperday'], 
                       color=year_color, alpha=0.4, label=year, linewidth=1.5)
        
        # Set title and labels
        month_range_text = f"{start_month} to {end_month}" if start_month != end_month else start_month
        ax.set_title(f'Yearly Comparison of {stored_contract_type} ({stored_vessel_type}) - {month_range_text}', fontsize=16)
        plt.xlabel('Date (Day of Year)', fontsize=12)
        plt.ylabel('USD per day', fontsize=12)
        plt.legend()
        
        # Calculate day of year ranges for month filtering
        start_day_approx = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335][start_month_num - 1]
        end_day_approx = [31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365][end_month_num - 1]
        
        # Set x-axis limits based on month range
        if start_month_num <= end_month_num:
            x_min = start_day_approx
            x_max = end_day_approx
        else:
            # For cross-year ranges, show full year but highlight the range
            x_min = 1
            x_max = 365
        
        # Set x-axis ticks and labels based on chart style and month range
        if chart_style == "Monthly Labels":
            all_xlabels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            all_xpos = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
        else:  # Bimonthly Labels
            all_xlabels = ['Jan', 'Mar', 'May', 'Jul', 'Sep', 'Nov']
            all_xpos = [1, 60, 121, 182, 244, 305]
        
        # Filter ticks to show only relevant months
        if start_month_num <= end_month_num:
            if chart_style == "Monthly Labels":
                tick_indices = range(start_month_num - 1, end_month_num)
            else:
                tick_indices = [i for i in range(0, 6) if (i * 2 + 1) >= start_month_num and (i * 2 + 1) <= end_month_num]
        else:
            # Cross-year: show all ticks
            tick_indices = range(len(all_xlabels))
        
        filtered_xlabels = [all_xlabels[i] for i in tick_indices]
        filtered_xpos = [all_xpos[i] for i in tick_indices]
        
        plt.xticks(filtered_xpos, filtered_xlabels, rotation=45 if chart_style == "Monthly Labels" else 0)
        
        # Set x-axis limits
        if not axis_controls['x_auto']:
            ax.set_xlim(axis_controls['x_min'], axis_controls['x_max'])
        else:
            ax.set_xlim(x_min, x_max)
        
        # Apply axis limits using the utility function
        y_cols = ['USDperday']
        if show_range and highlight_year in selected_years:
            highlight_df = month_filtered_df[month_filtered_df['Year'] == highlight_year]
            if not highlight_df.empty:
                y_cols.extend(['USDperdayMin', 'USDperdayMax'])
        
        apply_axis_limits(ax, axis_controls, data_df=month_filtered_df, y_cols=y_cols)
        
        # Format y-axis with currency AFTER setting the limits
        current_values = plt.gca().get_yticks()
        plt.gca().set_yticklabels(['$ {:,.0f}'.format(x) for x in current_values])
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        sns.despine(left=True, bottom=True)
        
        st.pyplot(fig)
        
        # Display summary statistics
        st.subheader("Summary Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Years", len(selected_years))
            st.metric("Total Data Points", len(month_filtered_df))
            st.metric("Month Range", month_range_text)
        
        with col2:
            if highlight_year in selected_years:
                highlight_data = month_filtered_df[month_filtered_df['Year'] == highlight_year]['USDperday']
                if not highlight_data.empty:
                    st.metric(f"{highlight_year} Average", f"${highlight_data.mean():,.0f}")
                    st.metric(f"{highlight_year} Range", 
                            f"${highlight_data.min():,.0f} - ${highlight_data.max():,.0f}")
        
        with col3:
            all_data = month_filtered_df['USDperday']
            if not all_data.empty:
                st.metric("Overall Average", f"${all_data.mean():,.0f}")
                st.metric("Overall Range", f"${all_data.min():,.0f} - ${all_data.max():,.0f}")

st.markdown("---")
st.caption("This seasonality analysis shows how freight rates vary throughout the year across different time periods, helping identify seasonal patterns and trends.")