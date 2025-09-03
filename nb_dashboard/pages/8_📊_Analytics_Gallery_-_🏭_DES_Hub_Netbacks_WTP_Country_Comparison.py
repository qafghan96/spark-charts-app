import os
import sys
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import numpy as np

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

st.title("🏭 DES Hub Netbacks - WTP Country Comparison")

st.caption("Compare regas terminal competitiveness, averaged by country, across Europe.")

client_id, client_secret = get_credentials()
if not client_id or not client_secret:
    st.error("Missing Spark API credentials. Set streamlit secrets 'spark.client_id' and 'spark.client_secret' (or env vars).")
    st.stop()

scopes = "read:access,read:prices"
token = get_access_token(client_id, client_secret, scopes=scopes)

# Functions from the script
def fetch_deshub_releases(access_token, unit, limit=None, offset=None, terminal=None):
    query_params = f"?unit={unit}"
    if limit is not None:
        query_params += f"&limit={limit}"
    if offset is not None:
        query_params += f"&offset={offset}"
    if terminal is not None:
        query_params += f"&terminal={terminal}"

    content = api_get(f"/beta/access/des-hub-netbacks/{query_params}", access_token)
    return content

def deshub_organise_dataframe(data):
    data_dict = {
        'Release Date': [],
        'Terminal': [],
        'Month Index': [],
        'Delivery Month': [],
        'DES Hub Netback - TTF Basis': [],
        'DES Hub Netback - Outright': [],
        'Total Regas': [],
        'Basic Slot (Berth)': [],
        'Basic Slot (Unload/Stor/Regas)': [],
        'Basic Slot (B/U/S/R)': [],
        'Additional Storage': [],
        'Additional Sendout': [],
        'Gas in Kind': [],
        'Entry Capacity': [],
        'Commodity Charge': []
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

    # Create column that treats slot costs as sunk
    df['DES Hub Netback - TTF Basis - Var Regas Costs Only'] = df['DES Hub Netback - TTF Basis'] \
                                                                + df['Basic Slot (Unload/Stor/Regas)'] \
                                                                + df['Basic Slot (Berth)'] \
                                                                + df['Basic Slot (B/U/S/R)']
    return df

def loop_historical_data(token, n_offset):
    historical = fetch_deshub_releases(token, unit='usd-per-mmbtu', limit=30)
    hist_df = deshub_organise_dataframe(historical)
    terminal_list = list(historical['metaData']['terminals'].values())

    for i in range(1, n_offset + 1):
        historical = fetch_deshub_releases(token, unit='usd-per-mmbtu', limit=30, offset=i * 30)
        hist_df = pd.concat([hist_df, deshub_organise_dataframe(historical)])

    return hist_df, terminal_list

def fetch_cargo_releases(access_token, ticker, limit=4, offset=None):
    query_params = f"?limit={limit}"
    if offset is not None:
        query_params += f"&offset={offset}"

    content = api_get(f"/v1.0/contracts/{ticker}/price-releases/{query_params}", access_token)
    return content['data']

def cargo_to_dataframe(access_token, ticker, limit, month):
    if month == 'M+1':
        full_tick = ticker + '-b-f'
        hist_data = fetch_cargo_releases(access_token, full_tick, limit)
    else:
        full_tick = ticker + '-b-fo'
        hist_data = fetch_cargo_releases(access_token, full_tick, limit)

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

# Terminal mappings
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

terminal_country_dict = {
    'Netherlands': ['gate', 'eems-energy-terminal'],
    'UK': ['grain-lng', 'south-hook'],
    'Belgium': ['zeebrugge'],
    'France': ['dunkerque', 'le-havre', 'montoir', 'fos-cavaou'],
    'Germany': ['brunsbuttel', 'deutsche-ostsee', 'wilhelmshaven', 'wilhelmshaven-2', 'stade'],
    'Italy': ['adriatic', 'olt-toscana', 'piombino', 'ravenna'],
    'Spain': ['tvb']
}

# Configuration controls
st.subheader("Configuration")

col1, col2 = st.columns(2)

with col1:
    analysis_type = st.selectbox("Analysis Type", options=["By Terminal", "By Country"], index=0)

with col2:
    month = st.selectbox("Month Index", options=['M+1', 'M+2', 'M+3', 'M+4', 'M+5', 'M+6', 'M+7', 'M+8', 'M+9', 'M+10', 'M+11'], index=0)

if st.button("Generate Analysis", type="primary"):
    with st.spinner("Fetching data..."):
        try:
            # Fetch DES Hub data
            loops = 15
            hdf, deshub_terms = loop_historical_data(token, loops)
            
            # Fetch cargo prices
            sparknwe = cargo_to_dataframe(token, 'sparknwe', loops * 30, month=month)
            sparkswe = cargo_to_dataframe(token, 'sparkswe', loops * 30, month=month)
            
            sparkswe = sparkswe[sparkswe['Release Date'] >= sparknwe['Release Date'].iloc[-1]].copy()
            
            cargo_df = pd.merge(sparknwe, sparkswe, how='left', on='Release Date')
            cargo_df['Price_y'] = cargo_df['Price_y'].bfill().copy()
            
            cargo_df = cargo_df[['Release Date', 'Price_x', 'Price_y']].copy()
            cargo_df = cargo_df.rename(columns={'Price_x': 'SparkNWE', 'Price_y': 'SparkSWE'})
            
            # Create month_df
            month_df = hdf[(hdf['Terminal'] == 'gate') & (hdf['Month Index'] == month)][['Release Date', 'Delivery Month', 'DES Hub Netback - TTF Basis - Var Regas Costs Only']]
            month_df = month_df.rename(columns={'DES Hub Netback - TTF Basis - Var Regas Costs Only': 'gate'})
            
            terms2 = [x if x != 'gate' else None for x in deshub_terms]
            
            for t in terms2:
                if t is not None:
                    tdf = hdf[(hdf['Terminal'] == t) & (hdf['Month Index'] == month)][['Release Date', 'Delivery Month', 'DES Hub Netback - TTF Basis - Var Regas Costs Only']]
                    month_df = month_df.merge(tdf, on='Release Date', how='left')
                    month_df = month_df.rename(columns={'DES Hub Netback - TTF Basis - Var Regas Costs Only': t})
            
            month_df['Ave'] = month_df[deshub_terms].mean(axis=1)
            month_df['Min'] = month_df[deshub_terms].min(axis=1)
            month_df['Max'] = month_df[deshub_terms].max(axis=1)
            
            month_df = month_df.merge(cargo_df, how='left', on='Release Date')
            month_df['SparkNWE'] = month_df['SparkNWE'].bfill().copy()
            month_df['SparkSWE'] = month_df['SparkSWE'].bfill().copy()
            
            # Create WTP dataframe
            wtp_df = month_df[['Release Date', 'Delivery Month', 'SparkNWE', 'SparkSWE']].copy()
            
            for t in deshub_terms:
                if terminal_region_dict[t] == 'nwe':
                    wtp_df[t] = month_df[t].copy() - month_df['SparkNWE'].copy()
                elif terminal_region_dict[t] == 'swe':
                    wtp_df[t] = month_df[t].copy() - month_df['SparkSWE'].copy()
                else:
                    wtp_df[t] = month_df[t].copy() - month_df['SparkNWE'].copy()
            
            # Create countries dataframe
            countries = list(terminal_country_dict.keys())
            countries_df = wtp_df[['Release Date', 'Delivery Month']].copy()
            
            for c in countries:
                countries_df[c + ' Ave'] = wtp_df[terminal_country_dict[c]].mean(axis=1)
                countries_df[c + ' Min'] = wtp_df[terminal_country_dict[c]].min(axis=1)
                countries_df[c + ' Max'] = wtp_df[terminal_country_dict[c]].max(axis=1)
            
            # Store data in session state
            st.session_state['wtp_df'] = wtp_df
            st.session_state['countries_df'] = countries_df
            st.session_state['deshub_terms'] = deshub_terms
            st.session_state['countries'] = countries
            st.session_state['month'] = month
            st.session_state['analysis_type'] = analysis_type
            
        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")
            st.stop()

# Display data and chart controls if data exists
if 'wtp_df' in st.session_state and 'countries_df' in st.session_state:
    wtp_df = st.session_state['wtp_df']
    countries_df = st.session_state['countries_df']
    deshub_terms = st.session_state['deshub_terms']
    countries = st.session_state['countries']
    stored_month = st.session_state['month']
    stored_analysis_type = st.session_state['analysis_type']
    
    if stored_analysis_type == "By Terminal":
        st.subheader("WTP DataFrame (By Terminal)")
        st.dataframe(wtp_df, use_container_width=True)
        
        # Chart Configuration
        st.subheader("Chart Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            selected_terminals = st.multiselect(
                "Select terminals to chart",
                options=deshub_terms,
                default=deshub_terms[:3],
                help="Select which terminals to display in the chart"
            )
        
        with col2:
            # Date range controls
            st.write("**Chart Date Range**")
            col_start, col_end = st.columns(2)
            
            with col_start:
                start_date = st.date_input(
                    "Start Date", 
                    value=wtp_df['Release Date'].min().date(),
                    min_value=wtp_df['Release Date'].min().date(),
                    max_value=wtp_df['Release Date'].max().date(),
                    help="Start date for the chart display range"
                )
            
            with col_end:
                end_date = st.date_input(
                    "End Date", 
                    value=wtp_df['Release Date'].max().date(),
                    min_value=wtp_df['Release Date'].min().date(),
                    max_value=wtp_df['Release Date'].max().date(),
                    help="End date for the chart display range"
                )
        
        if st.button("Generate Terminal Chart", type="secondary") and selected_terminals:
            # Convert date inputs to datetime for filtering
            start_datetime = pd.Timestamp(start_date)
            end_datetime = pd.Timestamp(end_date)
            
            # Filter data to the selected date range
            date_filtered_df = wtp_df[
                (wtp_df['Release Date'] >= start_datetime) & 
                (wtp_df['Release Date'] <= end_datetime)
            ]
            
            sns.set_style("whitegrid")
            fig, ax = plt.subplots(figsize=(15, 7))
            
            ax.hlines(0, date_filtered_df['Release Date'].min(), date_filtered_df['Release Date'].max(), 
                     color='grey', linewidth=1)
            
            colors = plt.cm.tab10(np.linspace(0, 1, len(selected_terminals)))
            
            # Plot data and collect y-values for auto-scaling
            all_y_values = []
            for i, terminal in enumerate(selected_terminals):
                terminal_data = date_filtered_df[terminal].dropna()
                if not terminal_data.empty:
                    all_y_values.extend(terminal_data.tolist())
                    ax.plot(date_filtered_df['Release Date'], date_filtered_df[terminal], 
                           linewidth=2.0, label=terminal, color=colors[i])
                    # Highlight latest point
                    latest_idx = date_filtered_df['Release Date'].idxmax()
                    if not pd.isna(date_filtered_df.loc[latest_idx, terminal]):
                        ax.scatter(date_filtered_df.loc[latest_idx, 'Release Date'], 
                                 date_filtered_df.loc[latest_idx, terminal], 
                                 color=colors[i], marker='o', s=80)
            
            # Auto-scale y-axis based on filtered data
            if all_y_values:
                y_min = min(all_y_values)
                y_max = max(all_y_values)
                y_range = y_max - y_min
                padding = y_range * 0.1 if y_range > 0 else 0.1
                y_min_padded = y_min - padding
                y_max_padded = y_max + padding
                ax.set_ylim(y_min_padded, y_max_padded)
            
            # Set x-axis limits
            ax.set_xlim(start_datetime, end_datetime)
            
            # Negative shading
            if all_y_values and min(all_y_values) < 0:
                ax.fill_between([start_datetime, end_datetime], 
                               0, min(y_min_padded, -0.1), 
                               color='red', alpha=0.05)
            
            plt.title(f'Terminal WTP - {stored_month}')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            sns.despine(left=True, bottom=True)
            
            st.pyplot(fig)
    
    else:  # By Country
        st.subheader("WTP DataFrame (By Country)")
        st.dataframe(countries_df, use_container_width=True)
        
        # Chart Configuration
        st.subheader("Chart Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            selected_countries = st.multiselect(
                "Select countries to chart",
                options=countries,
                default=countries[:3],
                help="Select which countries to display in the chart"
            )
        
        with col2:
            # Date range controls
            st.write("**Chart Date Range**")
            col_start, col_end = st.columns(2)
            
            with col_start:
                start_date = st.date_input(
                    "Start Date", 
                    value=countries_df['Release Date'].min().date(),
                    min_value=countries_df['Release Date'].min().date(),
                    max_value=countries_df['Release Date'].max().date(),
                    help="Start date for the chart display range",
                    key="country_start_date"
                )
            
            with col_end:
                end_date = st.date_input(
                    "End Date", 
                    value=countries_df['Release Date'].max().date(),
                    min_value=countries_df['Release Date'].min().date(),
                    max_value=countries_df['Release Date'].max().date(),
                    help="End date for the chart display range",
                    key="country_end_date"
                )
        
        if st.button("Generate Country Chart", type="secondary") and selected_countries:
            # Convert date inputs to datetime for filtering
            start_datetime = pd.Timestamp(start_date)
            end_datetime = pd.Timestamp(end_date)
            
            # Filter data to the selected date range
            date_filtered_df = countries_df[
                (countries_df['Release Date'] >= start_datetime) & 
                (countries_df['Release Date'] <= end_datetime)
            ]
            
            sns.set_style("whitegrid")
            fig, ax = plt.subplots(figsize=(15, 7))
            
            ax.hlines(0, date_filtered_df['Release Date'].min(), date_filtered_df['Release Date'].max(), 
                     color='grey', linewidth=1)
            
            colors = plt.cm.tab10(np.linspace(0, 1, len(selected_countries)))
            
            st.write(f"**Latest Assessment:** {date_filtered_df['Release Date'].max().strftime('%Y-%m-%d')}")
            
            # Plot data and collect y-values for auto-scaling
            all_y_values = []
            for i, country in enumerate(selected_countries):
                country_ave = country + ' Ave'
                country_min = country + ' Min'
                country_max = country + ' Max'
                
                # Collect y-values for scaling
                ave_data = date_filtered_df[country_ave].dropna()
                min_data = date_filtered_df[country_min].dropna()
                max_data = date_filtered_df[country_max].dropna()
                
                if not ave_data.empty:
                    all_y_values.extend(ave_data.tolist())
                if not min_data.empty:
                    all_y_values.extend(min_data.tolist())
                if not max_data.empty:
                    all_y_values.extend(max_data.tolist())
                
                # Plot average line
                ax.plot(date_filtered_df['Release Date'], date_filtered_df[country_ave], 
                       linewidth=2.0, label=country, color=colors[i])
                
                # Plot min/max range
                ax.plot(date_filtered_df['Release Date'], date_filtered_df[country_min], 
                       linewidth=1.0, alpha=0.06, color=colors[i])
                ax.plot(date_filtered_df['Release Date'], date_filtered_df[country_max], 
                       linewidth=1.0, alpha=0.06, color=colors[i])
                ax.fill_between(date_filtered_df['Release Date'], 
                               date_filtered_df[country_min], 
                               date_filtered_df[country_max], 
                               alpha=0.2, color=colors[i])
                
                # Highlight latest point and show value
                latest_idx = date_filtered_df['Release Date'].idxmax()
                if not pd.isna(date_filtered_df.loc[latest_idx, country_ave]):
                    ax.scatter(date_filtered_df.loc[latest_idx, 'Release Date'], 
                             date_filtered_df.loc[latest_idx, country_ave], 
                             color=colors[i], marker='o', s=80)
                    st.write(f"**{country}:** {date_filtered_df.loc[latest_idx, country_ave]:.3f}")
            
            # Auto-scale y-axis based on filtered data
            if all_y_values:
                y_min = min(all_y_values)
                y_max = max(all_y_values)
                y_range = y_max - y_min
                padding = y_range * 0.1 if y_range > 0 else 0.1
                y_min_padded = y_min - padding
                y_max_padded = y_max + padding
                ax.set_ylim(y_min_padded, y_max_padded)
            
            # Set x-axis limits
            ax.set_xlim(start_datetime, end_datetime)
            
            # Negative shading
            if all_y_values and min(all_y_values) < 0:
                ax.fill_between([start_datetime, end_datetime], 
                               0, min(y_min_padded, -0.1), 
                               color='red', alpha=0.05)
            
            plt.title(f'Country Average WTP & Range - {stored_month}')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            sns.despine(left=True, bottom=True)
            
            st.pyplot(fig)

st.markdown("---")
st.caption("This analysis compares regas terminal competitiveness (WTP - Willingness to Pay) across European terminals and countries.")