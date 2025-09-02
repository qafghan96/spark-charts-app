import os
import sys
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import time

# Ensure we can import sibling module "utils.py" when run as a Streamlit page
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
APP_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if APP_ROOT not in sys.path:
    sys.path.insert(0, APP_ROOT)

from utils import (
    get_credentials,
    get_access_token,
    list_netbacks_reference,
    fetch_netback,
    api_get,
)

st.title("🌏 US Front Month Historical Arb")

st.caption("Plot US Front Month Historical Arb across different via-points with forward curves.")

client_id, client_secret = get_credentials()
if not client_id or not client_secret:
    st.error("Missing Spark API credentials. Set streamlit secrets 'spark.client_id' and 'spark.client_secret' (or env vars).")
    st.stop()

scopes = "read:netbacks,read:access,read:prices,read:routes"
token = get_access_token(client_id, client_secret, scopes=scopes)

# Get netbacks reference data
tickers, names, available_via, release_dates, _raw = list_netbacks_reference(token)

# Configuration controls
st.subheader("Configuration")

col1, col2 = st.columns(2)
with col1:
    port_options = {name: (uuid, vias) for uuid, name, vias in zip(tickers, names, available_via)}
    selected_port = st.selectbox("Select FoB Port", options=list(port_options.keys()), 
                                index=names.index("Sabine Pass") if "Sabine Pass" in names else 0)
    
    uuid, available_vias = port_options[selected_port]

with col2:
    num_releases = st.slider("Number of Historical Releases", min_value=50, max_value=300, value=200, step=25)

# Via point selections
st.subheader("Via Points to Compare")
selected_vias = []
via_colors = ['#FFC217', '#4F41F4', 'forestgreen', 'firebrick']
via_labels = []

for i, via in enumerate(['cogh', 'suez', 'panama']):
    if via in (available_vias or []):
        include_via = st.checkbox(f"Include {via.upper()}", value=True, key=f"via_{via}")
        if include_via:
            selected_vias.append(via)
            via_labels.append(f"NEA via {via.upper()}")

# Panama with delays option
if 'panama' in (available_vias or []) and 'panama' in selected_vias:
    include_panama_delays = st.checkbox("Include Panama with 7/7 congestion delays", value=True)
    if include_panama_delays:
        selected_vias.append('panama_delays')
        via_labels.append("NEA via Panama - 7/7")

show_forward_curves = st.checkbox("Show Forward Curves", value=True)

if st.button("Generate Historical Arb Chart", type="primary"):
    if not selected_vias:
        st.warning("Please select at least one via point to compare.")
        st.stop()

    def netbacks_history(ticker_idx, rel_dates, my_via=None, laden=None, ballast=None):
        """Fetch historical netbacks data"""
        months = []
        delta_outrights = []
        release_date = []
        port = []

        progress_bar = st.progress(0)
        total_requests = len(rel_dates)

        for i, r in enumerate(rel_dates):
            try:
                query_params = f"?fob-port={tickers[ticker_idx]}"
                if r:
                    query_params += f"&release-date={r}"
                if my_via and my_via != 'panama_delays':
                    query_params += f"&via-point={my_via}"
                elif my_via == 'panama_delays':
                    query_params += f"&via-point=panama"
                if laden:
                    query_params += f"&laden-congestion-days={laden}"
                if ballast:
                    query_params += f"&ballast-congestion-days={ballast}"

                content = api_get(f"/v1.0/netbacks/{query_params}", token)
                
                if content and 'netbacks' in content and content['netbacks']:
                    m = content['netbacks'][0]
                    months.append(m['load']['month'])
                    delta_outrights.append(float(m['neaMinusNwe']['outright']['usdPerMMBtu']))
                    release_date.append(content['releaseDate'])
                    port.append(names[ticker_idx])
                    
            except Exception as e:
                st.warning(f'Bad Date: {r} - {str(e)}')
            
            progress_bar.progress((i + 1) / total_requests)
            time.sleep(0.1)  # Small delay to avoid rate limits

        progress_bar.empty()
        
        df = pd.DataFrame({
            'Release Date': release_date,
            'FoB Port': port,
            'Month': months,
            'Delta Outrights': delta_outrights,
        })
        
        if not df.empty:
            df['Release Date'] = pd.to_datetime(df['Release Date'])
        
        return df

    def fetch_forward_curve(ticker_idx, my_via=None, my_release=None, laden=None, ballast=None):
        """Fetch forward curve data"""
        try:
            query_params = f"?fob-port={tickers[ticker_idx]}"
            if my_release:
                query_params += f"&release-date={my_release}"
            if my_via and my_via != 'panama_delays':
                query_params += f"&via-point={my_via}"
            elif my_via == 'panama_delays':
                query_params += f"&via-point=panama"
            if laden:
                query_params += f"&laden-congestion-days={laden}"
            if ballast:
                query_params += f"&ballast-congestion-days={ballast}"

            content = api_get(f"/v1.0/netbacks/{query_params}", token)
            
            months = []
            delta_outrights = []
            
            if content and 'netbacks' in content:
                for m in content['netbacks']:
                    months.append(m['load']['month'])
                    delta_outrights.append(float(m['neaMinusNwe']['outright']['usdPerMMBtu']))
            
            df = pd.DataFrame({
                'Month': months,
                'Delta Outrights': delta_outrights,
            })
            
            if not df.empty:
                df['Month'] = pd.to_datetime(df['Month'])
            
            return df
        except Exception as e:
            st.error(f"Error fetching forward curve: {e}")
            return pd.DataFrame()

    # Get port index
    port_idx = names.index(selected_port)
    my_releases = release_dates[:num_releases]

    with st.spinner("Fetching historical data..."):
        historical_data = {}
        forward_data = {}
        
        for i, via in enumerate(selected_vias):
            if via == 'panama_delays':
                df_hist = netbacks_history(port_idx, my_releases, my_via='panama_delays', laden=7, ballast=7)
                if show_forward_curves:
                    df_fo = fetch_forward_curve(port_idx, my_via='panama_delays', my_release=release_dates[0], laden=7, ballast=7)
            else:
                df_hist = netbacks_history(port_idx, my_releases, my_via=via)
                if show_forward_curves:
                    df_fo = fetch_forward_curve(port_idx, my_via=via, my_release=release_dates[0])
            
            historical_data[via] = df_hist
            if show_forward_curves:
                forward_data[via] = df_fo

    # Create the visualization
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(15, 6))
    
    plt.axhline(0, color='grey')
    
    # Plot historical data
    for i, via in enumerate(selected_vias):
        if via in historical_data and not historical_data[via].empty:
            df = historical_data[via]
            color = via_colors[i % len(via_colors)]
            label = via_labels[i] if i < len(via_labels) else f"NEA via {via.upper()}"
            
            ax.plot(df['Release Date'], df['Delta Outrights'], 
                   color=color, linewidth=2, label=label)
    
    # Plot forward curves if enabled
    if show_forward_curves:
        for i, via in enumerate(selected_vias):
            if via in forward_data and not forward_data[via].empty:
                df_fo = forward_data[via]
                color = via_colors[i % len(via_colors)]
                
                ax.plot(df_fo['Month'], df_fo['Delta Outrights'], 
                       color=color, linewidth=2, linestyle='--', alpha=0.8)
    
    # Add vertical line for today
    ax.plot([dt.datetime.today(), dt.datetime.today()], [-3, 3], '--', color='darkgray')
    
    ax.legend(loc=4)
    plt.title(f'Netbacks - {selected_port} (Front Loading Month)')
    plt.ylabel('$/MMBtu')
    plt.xlabel('Release Date')
    plt.ylim(-2.5, 2.5)
    
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    
    st.pyplot(fig)
    
    # Display summary statistics
    st.subheader("Summary Statistics")
    summary_cols = st.columns(len(selected_vias))
    
    for i, via in enumerate(selected_vias):
        if via in historical_data and not historical_data[via].empty:
            df = historical_data[via]
            with summary_cols[i]:
                st.metric(
                    label=via_labels[i] if i < len(via_labels) else f"NEA via {via.upper()}",
                    value=f"${df['Delta Outrights'].iloc[-1]:.2f}" if len(df) > 0 else "N/A",
                    delta=f"{df['Delta Outrights'].mean():.2f} avg"
                )
    
    # Show data table
    with st.expander("View Historical Data"):
        for via in selected_vias:
            if via in historical_data and not historical_data[via].empty:
                st.subheader(f"{via.upper()} Data")
                st.dataframe(historical_data[via].head(10))

st.markdown("---")
st.caption("This chart shows historical US arbitrage opportunities across different shipping routes with optional forward curve overlay.")