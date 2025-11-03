import os
import sys
import streamlit as st

# Ensure we can import sibling module "utils.py" when run as a Streamlit page
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
APP_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if APP_ROOT not in sys.path:
    sys.path.insert(0, APP_ROOT)

from utils import (
    get_credentials,
    get_access_token,
    fetch_price_releases,
    add_axis_controls,
    apply_axis_limits,
)

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime

st.title("Weekly SparkNWE")

st.caption("Displays SparkNWE basis (sparknwe-b-f) and outright (sparknwe-f) price datasets with chart visualization.")

# Get credentials
client_id, client_secret = get_credentials()
if not client_id or not client_secret:
    st.error("Missing Spark API credentials. Set streamlit secrets 'spark.client_id' and 'spark.client_secret' (or env vars).")
    st.stop()

# Get access token
scopes = "read:lng-freight-prices,read:routes"
token = get_access_token(client_id, client_secret, scopes=scopes)

# Configuration
limit = st.slider("Number of releases", min_value=50, max_value=500, value=250, step=50)

# Add axis controls
axis_controls = add_axis_controls(expanded=True)

sns.set_theme(style="whitegrid")

def build_sparknwe_price_df(access_token: str, ticker: str, limit: int = 60) -> pd.DataFrame:
    """Build price dataframe specifically for SparkNWE contracts"""
    releases = fetch_price_releases(access_token, ticker, limit=limit)

    release_dates = []
    period_start = []
    period_end = []
    period_name = []
    cal_month = []
    tickers = []
    usd = []
    usd_min = []
    usd_max = []

    for release in releases:
        release_date = release["releaseDate"]
        for d in release.get("data", []):
            for data_point in d.get("dataPoints", []):
                start_at = data_point["deliveryPeriod"]["startAt"]
                end_at = data_point["deliveryPeriod"]["endAt"]
                period_start.append(start_at)
                period_end.append(end_at)
                period_name.append(data_point["deliveryPeriod"]["name"])
                release_dates.append(release_date)
                tickers.append(release.get("contractId"))
                cal_month.append(datetime.strptime(start_at, "%Y-%m-%d").strftime("%b-%Y"))
                
                # Use usdPerMMBtu instead of usdPerDay for SparkNWE
                derived = data_point.get("derivedPrices", {}).get("usdPerMMBtu", {})
                usd.append(float(derived.get("spark", "nan")))
                usd_min.append(float(derived.get("sparkMin", "nan")))
                usd_max.append(float(derived.get("sparkMax", "nan")))

    df = pd.DataFrame(
        {
            "Release Date": release_dates,
            "ticker": tickers,
            "Period Name": period_name,
            "Period Start": period_start,
            "Period End": period_end,
            "Calendar Month": cal_month,
            "Spark": usd,
            "SparkMin": usd_min,
            "SparkMax": usd_max,
        }
    )
    if not df.empty:
        df["Release Date"] = pd.to_datetime(df["Release Date"], format="%Y-%m-%d")
    return df

if st.button("Generate Chart", type="primary"):
    with st.spinner("Fetching data..."):
        # Fetch data for both datasets
        sparkbasis = build_sparknwe_price_df(token, 'sparknwe-b-f', limit=limit)
        sparkoutright = build_sparknwe_price_df(token, 'sparknwe-f', limit=limit)
        
        if sparkoutright.empty:
            st.error("No data available for SparkNWE outright prices")
            st.stop()
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(16, 7))
        ax.set_xlabel('Release Date')
        
        # Plot SparkNWE outright with confidence bands
        ax.plot(sparkoutright['Release Date'], sparkoutright['Spark'], 
                color='#4F41F4', linewidth=2.5, label='SparkNWE-F')
        ax.plot(sparkoutright['Release Date'], sparkoutright['SparkMin'], 
                color='#4F41F4', alpha=0.1)
        ax.plot(sparkoutright['Release Date'], sparkoutright['SparkMax'], 
                color='#4F41F4', alpha=0.1)
        ax.fill_between(sparkoutright['Release Date'], sparkoutright['SparkMin'], 
                       sparkoutright['SparkMax'], alpha=0.2, color='#4F41F4')
        
        # Add latest point marker
        ax.scatter(sparkoutright['Release Date'].iloc[0], sparkoutright['Spark'].iloc[0], 
                  color='#4F41F4', marker='o', s=120)
        
        # Apply axis limits using the utility function
        all_data = [sparkoutright]
        if not sparkbasis.empty:
            all_data.append(sparkbasis)
        
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            apply_axis_limits(ax, axis_controls, data_df=combined_data, y_cols=['Spark'])
        
        sns.despine(left=True, bottom=True)
        ax.legend()
        
        st.pyplot(fig)
        
        # Store data for display
        st.session_state.sparknwe_data = {
            'outright': sparkoutright,
            'basis': sparkbasis
        }

# Display latest prices if available
if 'sparknwe_data' in st.session_state:
    data = st.session_state.sparknwe_data
    
    st.subheader("Latest Release Date Prices")
    col1, col2 = st.columns(2)
    
    if not data['outright'].empty:
        with col1:
            latest_price = data['outright']['Spark'].iloc[0]
            latest_date = data['outright']['Release Date'].iloc[0]
            st.metric(
                label="SparkNWE-F (Outright)",
                value=f"${latest_price:.3f}/MMBtu",
                help=f"Latest price as of {latest_date.strftime('%Y-%m-%d')}"
            )
    
    if not data['basis'].empty:
        with col2:
            latest_price = data['basis']['Spark'].iloc[0]
            latest_date = data['basis']['Release Date'].iloc[0]
            st.metric(
                label="SparkNWE-B-F (Basis)",
                value=f"${latest_price:.3f}/MMBtu",
                help=f"Latest price as of {latest_date.strftime('%Y-%m-%d')}"
            )

# Data preview section
if 'sparknwe_data' in st.session_state:
    data = st.session_state.sparknwe_data
    
    st.subheader("Data Preview")
    
    tab1, tab2 = st.tabs(["SparkNWE Outright", "SparkNWE Basis"])
    
    with tab1:
        if not data['outright'].empty:
            st.dataframe(data['outright'].head(10), use_container_width=True)
        else:
            st.info("No outright data available")
    
    with tab2:
        if not data['basis'].empty:
            st.dataframe(data['basis'].head(10), use_container_width=True)
        else:
            st.info("No basis data available")