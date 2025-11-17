import os
import sys
import streamlit as st
import pandas as pd
import json
from io import StringIO

# Ensure we can import sibling module "utils.py" when run as a Streamlit page
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
APP_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if APP_ROOT not in sys.path:
    sys.path.insert(0, APP_ROOT)

from utils import get_credentials, get_access_token, api_get

st.title("🏭 LNG Hubs (USGC) Data")
st.caption("Access currently active and historical USGC LNG Hub posts including both swap and outright transactions.")

# Get credentials
client_id, client_secret = get_credentials()
if not client_id or not client_secret:
    st.error("Missing Spark API credentials. Set streamlit secrets 'spark.client_id' and 'spark.client_secret' (or env vars).")
    st.stop()

# Get access token
scopes = "read:lng-freight-prices,read:routes"
token = get_access_token(client_id, client_secret, scopes=scopes)

# Configuration
st.subheader("Data Selection")
data_type = st.radio(
    "Choose data type to download:",
    options=["Live", "Historical"],
    help="Live: Currently active hub posts. Historical: All expired posts."
)

# Functions for fetching LNG Hubs data
def fetch_live_hubs(access_token, format='csv'):
    """Fetch currently live/active LNG Hub posts"""
    uri = "/beta/lng/hubs/usgc/live/"
    
    if format == 'csv':
        content = api_get(uri, access_token, format='csv')
        return pd.read_csv(StringIO(content.decode('utf-8')))
    else:
        return api_get(uri, access_token, format='json')

def fetch_historical_hubs(access_token, format='csv'):
    """Fetch historical/expired LNG Hub posts"""
    uri = "/beta/lng/hubs/usgc/historical/"
    
    if format == 'csv':
        content = api_get(uri, access_token, format='csv')
        return pd.read_csv(StringIO(content.decode('utf-8')))
    else:
        return api_get(uri, access_token, format='json')

@st.cache_data
def get_hub_metadata(access_token):
    """Get metadata including terminal codes"""
    data = api_get("/beta/lng/hubs/usgc/live/", access_token, format='json')
    return data.get('metaData', {})

if st.button("Fetch Data", type="primary"):
    with st.spinner(f"Fetching {data_type.lower()} LNG Hubs data..."):
        try:
            if data_type == "Live":
                df = fetch_live_hubs(token, format='csv')
                data_title = "Live/Active LNG Hub Posts"
            else:
                df = fetch_historical_hubs(token, format='csv')
                data_title = "Historical/Expired LNG Hub Posts"
            
            if df.empty:
                st.warning(f"No {data_type.lower()} data available.")
            else:
                # Display summary statistics
                st.success(f"Successfully fetched {len(df)} {data_title.lower()}!")
                
                # Display metadata
                with st.expander("📋 Terminal Codes Reference", expanded=False):
                    try:
                        metadata = get_hub_metadata(token)
                        terminal_codes = metadata.get('terminalCodes', {})
                        if terminal_codes:
                            st.write("**Terminal Code Mappings:**")
                            for uuid, code in terminal_codes.items():
                                st.write(f"• **{code}**: `{uuid}`")
                        else:
                            st.info("No terminal code metadata available.")
                    except Exception as e:
                        st.warning(f"Could not fetch metadata: {e}")
                
                # Display data summary
                st.subheader(f"📊 {data_title} Summary")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Records", len(df))
                
                with col2:
                    if 'InterestType' in df.columns:
                        interest_counts = df['InterestType'].value_counts()
                        st.metric("Interest Types", len(interest_counts))
                        for interest, count in interest_counts.items():
                            st.write(f"• {interest.title()}: {count}")
                
                with col3:
                    if 'OrderType' in df.columns:
                        order_counts = df['OrderType'].value_counts()
                        st.metric("Order Types", len(order_counts))
                        for order, count in order_counts.items():
                            st.write(f"• {order.title()}: {count}")
                
                with col4:
                    if 'Status' in df.columns:
                        status_counts = df['Status'].value_counts()
                        st.metric("Status Types", len(status_counts))
                        for status, count in status_counts.items():
                            st.write(f"• {status.title()}: {count}")
                
                # Display the full dataframe
                st.subheader(f"📋 {data_title}")
                st.dataframe(df, use_container_width=True, height=400)
                
                # Download functionality
                @st.cache_data
                def convert_df_to_csv(dataframe):
                    return dataframe.to_csv(index=False).encode('utf-8')
                
                csv_data = convert_df_to_csv(df)
                filename = f"lng_hubs_{data_type.lower()}_data.csv"
                
                st.download_button(
                    label=f"📥 Download {data_title} as CSV",
                    data=csv_data,
                    file_name=filename,
                    mime='text/csv',
                    help=f"Download the complete {data_title.lower()} dataset as a CSV file"
                )
                
                # Column information
                with st.expander("📋 Column Information", expanded=False):
                    st.write("**Key Columns:**")
                    col_descriptions = {
                        'Id': 'Unique identifier for the hub post',
                        'HubRegion': 'Hub region (USGC for US Gulf Coast)',
                        'InterestType': 'Type of interest (swap, outright)',
                        'OrderType': 'Order type (offered, requested, bid)',
                        'PosterName': 'Name of the poster (often Anonymous)',
                        'Status': 'Status of the post (active, expired)',
                        'TerminalCodes': 'Comma-separated list of terminal codes',
                        'PostedAtUtc': 'When the post was created (UTC)',
                        'ValidUntilUtc': 'When the post expires (UTC)',
                        'LastUpdatedAtUtc': 'When the post was last updated (UTC)'
                    }
                    
                    for col, desc in col_descriptions.items():
                        if col in df.columns:
                            st.write(f"• **{col}**: {desc}")
                    
                    st.write(f"\n**Total Columns**: {len(df.columns)}")
                    st.write("**All Columns**: " + ", ".join(df.columns.tolist()))

        except Exception as e:
            st.error(f"Error fetching {data_type.lower()} data: {e}")

# Information section
st.markdown("---")
st.subheader("ℹ️ About LNG Hubs Data")

with st.expander("Understanding the Data", expanded=False):
    st.markdown("""
    **LNG Hubs (USGC) Data** provides information about liquefied natural gas trading posts in the US Gulf Coast region.
    
    **Data Types:**
    - **Live Data**: Currently active hub posts that are still valid for trading
    - **Historical Data**: Expired posts that provide historical trading activity
    
    **Key Features:**
    - **Swap Transactions**: Exchange of LNG cargoes between different terminals
    - **Outright Transactions**: Direct purchase/sale of LNG cargoes
    - **Terminal Information**: Specific terminals involved in each transaction
    - **Timing Data**: Post creation, expiry, and update timestamps
    
    **Use Cases:**
    - Monitor current LNG trading activity
    - Analyze historical trading patterns
    - Track terminal utilization and activity
    - Study pricing trends and market dynamics
    """)

st.caption("Data source: Spark Commodities LNG Hubs API")