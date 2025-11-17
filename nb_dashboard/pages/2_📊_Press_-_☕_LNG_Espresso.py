import os
import sys
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Ensure we can import sibling module "utils.py" when run as a Streamlit page
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
APP_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if APP_ROOT not in sys.path:
    sys.path.insert(0, APP_ROOT)

from utils import (
    get_credentials,
    get_access_token,
    build_price_df,
    add_axis_controls,
    add_color_controls,
    apply_axis_limits,
)

st.title("LNG Espresso")

st.caption("Replicates the LNG Espresso chart: Spark25S Pacific and Spark30S Atlantic.")

client_id, client_secret = get_credentials()
if not client_id or not client_secret:
    st.error("Missing Spark API credentials. Set streamlit secrets 'spark.client_id' and 'spark.client_secret' (or env vars).")
    st.stop()

scopes = "read:lng-freight-prices,read:routes"
token = get_access_token(client_id, client_secret, scopes=scopes)

contracts = {
    "Spark25S Pacific": "spark25s",
    "Spark30S Atlantic": "spark30s",
}

limit = st.slider("Number of releases", min_value=10, max_value=310, value=20, step=15)

# Add color controls for the two freight series
series_names = ["Spark25S Pacific", "Spark30S Atlantic"]
default_colors = ["#4F41F4", "#48C38D"]  # Blue and Green (current chart colors)
color_controls = add_color_controls(series_names, default_colors, expanded=True)

# Add axis controls
axis_controls = add_axis_controls(expanded=True)

sns.set_theme(style="whitegrid")

# Initialize price_data
price_data = {}

if st.button("Generate Chart", type="primary"):
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.set_xlabel("Release Date")

    # Store dataframes and latest prices
    price_data = {}
    all_data = []
    
    for name, ticker in contracts.items():
        df = build_price_df(token, ticker, limit=limit)
        if df.empty:
            continue
        # Get selected color for this series
        series_color = color_controls.get(name, "#333")
        ax.plot(df["Release Date"], df["Spark"], color=series_color, linewidth=3.0, label=name)
        ax.scatter(df["Release Date"].iloc[0], df["Spark"].iloc[0], color=series_color, s=120)
        
        # Store data for axis limits
        all_data.append(df)
        
        # Store latest price data
        price_data[name] = {
            "latest_price": df["Spark"].iloc[0],
            "latest_date": df["Release Date"].iloc[0]
        }
    
    # Apply axis limits using the utility function
    if all_data:
        combined_data = pd.concat(all_data, ignore_index=True)
        apply_axis_limits(ax, axis_controls, data_df=combined_data, y_cols=['Spark'])
    
    sns.despine(left=True, bottom=True)
    ax.legend()
    st.pyplot(fig)
    
    # Store price data for display outside the conditional
    st.session_state.price_data = price_data

# Display latest prices if they exist
if 'price_data' in st.session_state and st.session_state.price_data:
    price_data = st.session_state.price_data

# Display latest prices
st.subheader("Latest Release Date Prices")
col1, col2 = st.columns(2)

if "Spark25S Pacific" in price_data:
    with col1:
        latest_price = price_data["Spark25S Pacific"]["latest_price"]
        latest_date = price_data["Spark25S Pacific"]["latest_date"]
        st.metric(
            label="Spark25S Pacific",
            value=f"${latest_price:,.0f}",
            help=f"Latest price as of {latest_date.strftime('%Y-%m-%d')}"
        )

if "Spark30S Atlantic" in price_data:
    with col2:
        latest_price = price_data["Spark30S Atlantic"]["latest_price"]
        latest_date = price_data["Spark30S Atlantic"]["latest_date"]
        st.metric(
            label="Spark30S Atlantic", 
            value=f"${latest_price:,.0f}",
            help=f"Latest price as of {latest_date.strftime('%Y-%m-%d')}"
        )



