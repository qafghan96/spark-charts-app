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
                    tdf = hdf[(hdf['Terminal'] == t) & (hdf['Month Index'] == month)][['Release Date', 'DES Hub Netback - TTF Basis - Var Regas Costs Only']]
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
            
        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")
            st.stop()

    if analysis_type == "By Terminal":
        st.subheader("WTP DataFrame (By Terminal)")
        st.dataframe(wtp_df, use_container_width=True)
        
        # Terminal selection for chart
        st.subheader("Terminal Chart")
        selected_terminals = st.multiselect(
            "Select terminals to chart",
            options=deshub_terms,
            default=deshub_terms[:3],
            help="Select which terminals to display in the chart"
        )
        
        if selected_terminals:
            sns.set_style("whitegrid")
            fig, ax = plt.subplots(figsize=(15, 7))
            
            ax.hlines(0, wtp_df['Release Date'].iloc[-1], wtp_df['Release Date'].iloc[0], color='grey')
            
            colors = plt.cm.tab10(np.linspace(0, 1, len(selected_terminals)))
            
            for i, terminal in enumerate(selected_terminals):
                ax.plot(wtp_df['Release Date'], wtp_df[terminal], 
                       linewidth=2.0, label=terminal, color=colors[i])
                ax.scatter(wtp_df['Release Date'].iloc[0], wtp_df[terminal].iloc[0], 
                          color=colors[i], marker='o', s=80)
            
            negrange = [wtp_df['Release Date'].iloc[-1] - pd.Timedelta(20, unit='day'), 
                       wtp_df['Release Date'].iloc[0] + pd.Timedelta(20, unit='day')]
            
            ax.plot(negrange, [-2.0, -2.0], color='red', alpha=0.05)
            ax.plot(negrange, [0, 0], color='red', alpha=0.05)
            ax.fill_between(negrange, 0, -2.0, color='red', alpha=0.05)
            
            plt.xlim([wtp_df['Release Date'].iloc[-1] - pd.Timedelta(7, unit='day'), 
                     wtp_df['Release Date'].iloc[0] + pd.Timedelta(20, unit='day')])
            plt.ylim(-1.7, 1)
            
            plt.title(f'Terminal WTP - {month}')
            plt.legend()
            plt.grid()
            sns.despine(left=True, bottom=True)
            
            st.pyplot(fig)
    
    else:  # By Country
        st.subheader("WTP DataFrame (By Country)")
        st.dataframe(countries_df, use_container_width=True)
        
        # Country selection for chart
        st.subheader("Country Chart")
        selected_countries = st.multiselect(
            "Select countries to chart",
            options=countries,
            default=countries[:3],
            help="Select which countries to display in the chart"
        )
        
        if selected_countries:
            sns.set_style("whitegrid")
            fig, ax = plt.subplots(figsize=(15, 7))
            
            ax.hlines(0, countries_df['Release Date'].iloc[-1], countries_df['Release Date'].iloc[0], color='grey')
            
            colors = plt.cm.tab10(np.linspace(0, 1, len(selected_countries)))
            
            st.write(f"Latest Assessment: {countries_df['Release Date'].iloc[0]}")
            
            for i, country in enumerate(selected_countries):
                ax.scatter(countries_df['Release Date'].iloc[0], countries_df[country + ' Ave'].iloc[0], 
                          color=colors[i], marker='o', s=80)
                st.write(f"{country} = {countries_df[country + ' Ave'].iloc[0]:.3f}")
                
                ax.plot(countries_df['Release Date'], countries_df[country + ' Ave'], 
                       linewidth=2.0, label=country, color=colors[i])
                ax.plot(countries_df['Release Date'], countries_df[country + ' Min'], 
                       linewidth=1.0, alpha=0.06, color=colors[i])
                ax.plot(countries_df['Release Date'], countries_df[country + ' Max'], 
                       linewidth=1.0, alpha=0.06, color=colors[i])
                ax.fill_between(countries_df['Release Date'], countries_df[country + ' Min'], 
                               countries_df[country + ' Max'], alpha=0.2, color=colors[i])
            
            negrange = [countries_df['Release Date'].iloc[-1] - pd.Timedelta(20, unit='day'), 
                       countries_df['Release Date'].iloc[0] + pd.Timedelta(20, unit='day')]
            
            ax.plot(negrange, [-2.0, -2.0], color='red', alpha=0.05)
            ax.plot(negrange, [0, 0], color='red', alpha=0.05)
            ax.fill_between(negrange, 0, -2.0, color='red', alpha=0.05)
            
            plt.xlim([countries_df['Release Date'].iloc[-1] - pd.Timedelta(7, unit='day'), 
                     countries_df['Release Date'].iloc[0] + pd.Timedelta(20, unit='day')])
            plt.ylim(-1.7, 1)
            
            plt.title(f'Country Average WTP & Range - {month}')
            plt.legend()
            plt.grid()
            sns.despine(left=True, bottom=True)
            
            st.pyplot(fig)

st.markdown("---")
st.caption("This analysis compares regas terminal competitiveness (WTP - Willingness to Pay) across European terminals and countries.")