import os
import sys
import streamlit as st
import pandas as pd
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
    api_get,
)

st.title("🏭 DES Hub Netbacks - WTP Country Comparison")

st.caption("Compare Willingness to Pay (WTP) netbacks across different countries and terminals.")

client_id, client_secret = get_credentials()
if not client_id or not client_secret:
    st.error("Missing Spark API credentials. Set streamlit secrets 'spark.client_id' and 'spark.client_secret' (or env vars).")
    st.stop()

scopes = "read:access"
token = get_access_token(client_id, client_secret, scopes=scopes)

# Get reference data for DES hubs
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_des_hub_reference():
    try:
        content = api_get("/v1.0/des-hub-netbacks/reference-data/", token)
        return content.get('data', {})
    except Exception as e:
        st.error(f"Error fetching reference data: {e}")
        return {}

reference_data = get_des_hub_reference()

if not reference_data:
    st.error("Unable to fetch reference data. Please check your API credentials and try again.")
    st.stop()

# Extract available options
static_data = reference_data.get('staticData', {})
terminals = static_data.get('terminals', [])
fob_ports = static_data.get('fobPorts', [])
release_dates = static_data.get('sparkReleases', [])

# Configuration controls
st.subheader("Configuration")

col1, col2 = st.columns(2)
with col1:
    # Terminal/Country selection
    terminal_options = {}
    for terminal in terminals:
        country = terminal.get('country', 'Unknown')
        name = terminal.get('name', 'Unknown')
        terminal_id = terminal.get('uuid', '')
        
        if country not in terminal_options:
            terminal_options[country] = []
        terminal_options[country].append((name, terminal_id))
    
    selected_countries = st.multiselect(
        "Select Countries/Regions", 
        options=list(terminal_options.keys()),
        default=list(terminal_options.keys())[:3] if len(terminal_options) > 3 else list(terminal_options.keys())
    )

with col2:
    # FoB Port selection
    fob_port_options = {port.get('name', 'Unknown'): port.get('uuid', '') for port in fob_ports}
    selected_fob_port_name = st.selectbox(
        "Select FoB Port",
        options=list(fob_port_options.keys()),
        index=0 if fob_port_options else None
    )
    selected_fob_port_id = fob_port_options.get(selected_fob_port_name)

# Additional parameters
num_releases = st.slider("Number of Recent Releases", min_value=5, max_value=50, value=20)
selected_releases = release_dates[:num_releases] if release_dates else []

if st.button("Generate Country Comparison", type="primary") and selected_countries and selected_fob_port_id:
    
    @st.cache_data(ttl=1800)  # Cache for 30 minutes
    def fetch_des_hub_netbacks(fob_port_id, terminal_id, release_date):
        try:
            query_params = f"?fob-port={fob_port_id}&terminal={terminal_id}&release-date={release_date}"
            content = api_get(f"/v1.0/des-hub-netbacks/{query_params}", token)
            return content.get('data', {})
        except Exception as e:
            return None

    with st.spinner("Fetching DES Hub netbacks data..."):
        # Collect data for all selected countries
        country_data = {}
        
        for country in selected_countries:
            country_terminals = terminal_options[country]
            country_wtp_data = []
            
            # Get data for each terminal in the country
            for terminal_name, terminal_id in country_terminals:
                terminal_data = []
                
                for release_date in selected_releases:
                    netback_data = fetch_des_hub_netbacks(selected_fob_port_id, terminal_id, release_date)
                    
                    if netback_data and 'netbacks' in netback_data:
                        for netback in netback_data['netbacks']:
                            if 'willingnessToPayPremiumToNwe' in netback:
                                wtp_value = netback['willingnessToPayPremiumToNwe'].get('usdPerMMBtu')
                                if wtp_value is not None:
                                    terminal_data.append({
                                        'Release Date': release_date,
                                        'Terminal': terminal_name,
                                        'Country': country,
                                        'WTP Premium': float(wtp_value),
                                        'Month': netback.get('load', {}).get('month', '')
                                    })
                
                country_wtp_data.extend(terminal_data)
            
            if country_wtp_data:
                country_data[country] = pd.DataFrame(country_wtp_data)

    if country_data:
        # Create visualization
        sns.set_style("whitegrid")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Average WTP by Country
        avg_wtp_by_country = []
        countries_with_data = []
        
        for country, df in country_data.items():
            if not df.empty:
                avg_wtp = df['WTP Premium'].mean()
                avg_wtp_by_country.append(avg_wtp)
                countries_with_data.append(country)
        
        if countries_with_data:
            colors = plt.cm.Set3(range(len(countries_with_data)))
            bars = ax1.bar(countries_with_data, avg_wtp_by_country, color=colors, alpha=0.7)
            ax1.set_title('Average Willingness to Pay Premium by Country')
            ax1.set_ylabel('USD per MMBtu')
            ax1.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, avg_wtp_by_country):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'${value:.2f}', ha='center', va='bottom', fontweight='bold')

        # Plot 2: WTP Distribution by Country (Box Plot)
        all_data = []
        all_countries = []
        
        for country, df in country_data.items():
            if not df.empty:
                all_data.extend(df['WTP Premium'].tolist())
                all_countries.extend([country] * len(df))
        
        if all_data:
            box_df = pd.DataFrame({'WTP Premium': all_data, 'Country': all_countries})
            box_plot = ax2.boxplot([country_data[country]['WTP Premium'].values 
                                  for country in countries_with_data],
                                 labels=countries_with_data, patch_artist=True)
            
            # Color the box plots
            for patch, color in zip(box_plot['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax2.set_title('WTP Premium Distribution by Country')
            ax2.set_ylabel('USD per MMBtu')
            ax2.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        st.pyplot(fig)

        # Summary Statistics
        st.subheader("Summary Statistics")
        summary_data = []
        
        for country, df in country_data.items():
            if not df.empty:
                summary_data.append({
                    'Country': country,
                    'Terminals': df['Terminal'].nunique(),
                    'Avg WTP Premium': f"${df['WTP Premium'].mean():.2f}",
                    'Min WTP Premium': f"${df['WTP Premium'].min():.2f}",
                    'Max WTP Premium': f"${df['WTP Premium'].max():.2f}",
                    'Std Dev': f"${df['WTP Premium'].std():.2f}",
                    'Data Points': len(df)
                })
        
        if summary_data:
            st.dataframe(pd.DataFrame(summary_data), use_container_width=True)

        # Time series comparison
        st.subheader("WTP Premium Trends Over Time")
        
        fig2, ax = plt.subplots(figsize=(15, 6))
        
        for i, (country, df) in enumerate(country_data.items()):
            if not df.empty:
                # Group by release date and take average
                time_series = df.groupby('Release Date')['WTP Premium'].mean().reset_index()
                time_series['Release Date'] = pd.to_datetime(time_series['Release Date'])
                time_series = time_series.sort_values('Release Date')
                
                ax.plot(time_series['Release Date'], time_series['WTP Premium'], 
                       marker='o', linewidth=2, label=country, 
                       color=colors[i % len(colors)])
        
        ax.set_title(f'WTP Premium Trends - {selected_fob_port_name}')
        ax.set_ylabel('USD per MMBtu')
        ax.set_xlabel('Release Date')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig2)

        # Detailed data view
        with st.expander("View Detailed Data"):
            for country, df in country_data.items():
                if not df.empty:
                    st.subheader(f"{country} Data")
                    st.dataframe(df.sort_values('Release Date', ascending=False))

    else:
        st.warning("No data available for the selected parameters. Try adjusting your selections.")

st.markdown("---")
st.caption("This analysis compares willingness to pay premiums across different countries and terminals for DES hub netbacks.")