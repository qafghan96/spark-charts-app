import os
import sys
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

# Ensure we can import sibling module "utils.py" when run as a Streamlit page
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
APP_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if APP_ROOT not in sys.path:
    sys.path.insert(0, APP_ROOT)

from utils import (
    get_credentials,
    get_access_token,
    list_netbacks_reference,
    netbacks_history,
    api_get,
    build_price_df,
)

st.title("💹 US Arb Freight Breakevens vs Spot Freight Rates")

st.caption("Compare US Arb Freight Breakevens with Spark30S Spot Freight Rates.")

client_id, client_secret = get_credentials()
if not client_id or not client_secret:
    st.error("Missing Spark API credentials. Set streamlit secrets 'spark.client_id' and 'spark.client_secret' (or env vars).")
    st.stop()

scopes = "read:lng-freight-prices,read:routes,read:netbacks"
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
    uuid, vias = port_options[selected_port]
    selected_via = st.selectbox("Via Point", options=vias or ["cogh"], index=0)

with col2:
    num_releases = st.slider("Number of releases", min_value=10, max_value=200, value=50, step=10)
    vessel_type = st.selectbox("Vessel Type", options=["174-2stroke", "160-tfde"], index=0)

if st.button("Generate Chart", type="primary"):
    with st.spinner("Fetching breakevens data..."):
        # Fetch breakevens data
        try:
            query_params = f"?fob-port={uuid}"
            if selected_via:
                query_params += f"&nea-via-point={selected_via}"
            
            breakevens_data = api_get(f"/beta/netbacks/arb-breakevens/{query_params}", token)
            break_df = pd.DataFrame(breakevens_data['data'])
            break_df['ReleaseDate'] = pd.to_datetime(break_df['ReleaseDate'])
            
            # Filter to front month only
            front_df = break_df[break_df['LoadMonthIndex'] == "M+1"].copy()
            
        except Exception as e:
            st.error(f"Error fetching breakevens data: {e}")
            st.stop()

    with st.spinner("Fetching spot freight prices..."):
        # Fetch Spark30S data
        try:
            spark30_df = build_price_df(token, 'spark30s', limit=num_releases)
            spark30_df = spark30_df[spark30_df['ticker'] == 'spark30s'].copy()
            
        except Exception as e:
            st.error(f"Error fetching spot freight data: {e}")
            st.stop()

    # Create visualization
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(15, 7))

    # Plot the data
    ax.plot(spark30_df['Release Date'], spark30_df['Spark'], 
           color='#48C38D', linewidth=2.5, label='Spark30S (Atlantic)')
    ax.plot(front_df['ReleaseDate'], front_df['FreightBreakevenUSDPerDay'], 
           color='#4F41F4', linewidth=2, label='US Arb [M+1] Freight Breakeven Level')

    # Set limits and formatting
    ax.set_xlim(datetime.datetime.today() - datetime.timedelta(days=380), 
                datetime.datetime.today())
    ax.set_ylim(-100000, 120000)

    plt.title(f'Spark30S (Atlantic) vs. US Arb [M+1] Freight Breakeven Level - {selected_port} via {selected_via}')
    plt.ylabel('USD per Day')
    plt.xlabel('Release Date')

    sns.despine(left=True, bottom=True)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    st.pyplot(fig)

    # Show conditional shading chart
    st.subheader("Chart with Conditional Shading")
    
    # Merge dataframes for conditional shading
    spark30_df['Release Date'] = pd.to_datetime(spark30_df['Release Date'])
    merge_df = pd.merge(spark30_df, front_df, left_on='Release Date', right_on='ReleaseDate', how='inner')

    if not merge_df.empty:
        fig2, ax2 = plt.subplots(figsize=(15, 7))

        ax2.plot(merge_df['Release Date'], merge_df['Spark'], 
                color='#48C38D', linewidth=2.5, label='Spark30S (Atlantic)')
        ax2.plot(merge_df['Release Date'], merge_df['FreightBreakevenUSDPerDay'], 
                color='#4F41F4', linewidth=2, label='US Arb [M+1] Freight Breakeven Level')

        # Add conditional shading
        ax2.fill_between(merge_df['Release Date'], merge_df['Spark'], merge_df['FreightBreakevenUSDPerDay'],
                        where=merge_df['Spark'] > merge_df['FreightBreakevenUSDPerDay'], 
                        facecolor='red', interpolate=True, alpha=0.05)
        
        ax2.fill_between(merge_df['Release Date'], merge_df['Spark'], merge_df['FreightBreakevenUSDPerDay'],
                        where=merge_df['Spark'] < merge_df['FreightBreakevenUSDPerDay'], 
                        facecolor='green', interpolate=True, alpha=0.05)

        ax2.set_xlim(datetime.datetime.today() - datetime.timedelta(days=380), 
                    datetime.datetime.today())
        ax2.set_ylim(-100000, 120000)

        plt.title(f'Spark30S vs. US Arb Freight Breakeven with Conditional Shading - {selected_port} via {selected_via}')
        plt.ylabel('USD per Day')
        plt.xlabel('Release Date')

        sns.despine(left=True, bottom=True)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        st.pyplot(fig2)

        # Display summary statistics
        st.subheader("Summary Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Average Spark30S", f"${merge_df['Spark'].mean():,.0f}")
        with col2:
            st.metric("Average Breakeven", f"${merge_df['FreightBreakevenUSDPerDay'].mean():,.0f}")
        with col3:
            spread = merge_df['Spark'] - merge_df['FreightBreakevenUSDPerDay']
            st.metric("Average Spread", f"${spread.mean():,.0f}")

st.markdown("---")
st.caption("This chart compares US arbitrage freight breakevens with spot freight rates to identify market opportunities.")