import os
import sys
import streamlit as st
import pandas as pd
from io import StringIO

# Ensure we can import sibling module "utils.py" when run as a Streamlit page
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
APP_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if APP_ROOT not in sys.path:
    sys.path.insert(0, APP_ROOT)

from utils import get_credentials, get_access_token, api_get

st.title("🏭 LNG Hubs Data")
st.caption("Access currently active and historical USGC LNG Hub posts including both swap and outright transactions.")

# Get credentials
client_id, client_secret = get_credentials()
if not client_id or not client_secret:
    st.error("Missing Spark API credentials. Set streamlit secrets 'spark.client_id' and 'spark.client_secret' (or env vars).")
    st.stop()

# Get access token (no specific scopes required for LNG Hubs endpoint)
token = get_access_token(client_id, client_secret)

# Configuration
st.subheader("Data Selection")
data_type = st.radio(
    "Choose data type to download:",
    options=["Live", "Historical"],
    help="Live: Currently active hub posts. Historical: All expired posts."
)

# Functions for fetching LNG Hubs data
def fetch_live_hubs(access_token):
    """Fetch currently live/active LNG Hub posts"""
    content = api_get("/beta/lng/hubs/fob/usg/live/", access_token, format='csv')
    return pd.read_csv(StringIO(content.decode('utf-8')))

def fetch_historical_hubs(access_token):
    """Fetch historical/expired LNG Hub posts"""
    content = api_get("/beta/lng/hubs/fob/usg/historical/", access_token, format='csv')
    return pd.read_csv(StringIO(content.decode('utf-8')))

if st.button("Fetch Data", type="primary"):
    with st.spinner(f"Fetching {data_type.lower()} LNG Hubs data..."):
        try:
            if data_type == "Live":
                df = fetch_live_hubs(token)
                data_title = "Live/Active LNG Hub Posts"
            else:
                df = fetch_historical_hubs(token)
                data_title = "Historical/Expired LNG Hub Posts"

            if df.empty:
                st.warning(f"No {data_type.lower()} data available.")
            else:
                # Display summary statistics
                st.success(f"Successfully fetched {len(df)} {data_title.lower()}!")

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
                
                # Detailed Statistics Section
                st.subheader("📊 Detailed Field Statistics")
                
                # Helper function to create statistics for categorical fields
                def create_field_stats(df, field_name, max_display=10):
                    """Create statistics for a categorical field"""
                    if field_name not in df.columns:
                        return None
                    
                    # Handle special cases for fields that might contain multiple values
                    if field_name == 'TerminalCodes':
                        # Split comma-separated values and count individual terminals
                        all_terminals = []
                        for terminals_str in df[field_name].dropna():
                            if pd.notna(terminals_str) and str(terminals_str).strip():
                                terminals = [t.strip() for t in str(terminals_str).split(',')]
                                all_terminals.extend(terminals)
                        
                        if all_terminals:
                            terminal_counts = pd.Series(all_terminals).value_counts()
                            stats_df = pd.DataFrame({
                                'Value': terminal_counts.index[:max_display],
                                'Count': terminal_counts.values[:max_display],
                                'Percentage': (terminal_counts.values[:max_display] / len(all_terminals) * 100).round(2)
                            })
                            return stats_df, len(all_terminals), len(terminal_counts)
                        return None
                    
                    # Standard categorical field handling
                    value_counts = df[field_name].value_counts()
                    if len(value_counts) == 0:
                        return None
                    
                    stats_df = pd.DataFrame({
                        'Value': value_counts.index[:max_display],
                        'Count': value_counts.values[:max_display],
                        'Percentage': (value_counts.values[:max_display] / len(df) * 100).round(2)
                    })
                    
                    return stats_df, len(df), len(value_counts)
                
                # Helper function to create statistics for numeric fields
                def create_numeric_stats(df, field_name):
                    """Create statistics for numeric fields"""
                    if field_name not in df.columns:
                        return None
                    
                    numeric_data = pd.to_numeric(df[field_name], errors='coerce').dropna()
                    if len(numeric_data) == 0:
                        return None
                    
                    stats = {
                        'Count': len(numeric_data),
                        'Mean': numeric_data.mean().round(2),
                        'Median': numeric_data.median().round(2),
                        'Std Dev': numeric_data.std().round(2),
                        'Min': numeric_data.min(),
                        'Max': numeric_data.max(),
                        'Q25': numeric_data.quantile(0.25).round(2),
                        'Q75': numeric_data.quantile(0.75).round(2)
                    }
                    
                    return pd.DataFrame([stats])
                
                # Identify categorical and numeric fields
                categorical_fields = []
                numeric_fields = []
                datetime_fields = []
                
                for col in df.columns:
                    # Skip ID fields and very long text fields
                    if col.lower() in ['id', 'uuid'] or df[col].astype(str).str.len().max() > 100:
                        continue
                    
                    # Check for datetime fields
                    if 'utc' in col.lower() or 'date' in col.lower() or 'time' in col.lower():
                        datetime_fields.append(col)
                    # Check for numeric fields
                    elif pd.api.types.is_numeric_dtype(df[col]) or df[col].astype(str).str.replace('.', '').str.isdigit().any():
                        numeric_data = pd.to_numeric(df[col], errors='coerce')
                        if not numeric_data.isna().all():
                            numeric_fields.append(col)
                    else:
                        categorical_fields.append(col)
                
                # Display categorical field statistics
                if categorical_fields:
                    st.write("**📋 Categorical Fields Statistics:**")
                    
                    # Create tabs for different categorical fields
                    tabs = st.tabs([f"{field}" for field in categorical_fields[:5]])  # Limit to 5 tabs
                    
                    for i, field in enumerate(categorical_fields[:5]):
                        with tabs[i]:
                            result = create_field_stats(df, field)
                            if result:
                                stats_df, total_records, unique_values = result
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Total Records", total_records)
                                with col2:
                                    st.metric("Unique Values", unique_values)
                                with col3:
                                    if len(stats_df) < unique_values:
                                        st.metric("Showing Top", len(stats_df))
                                
                                st.dataframe(stats_df, use_container_width=True, hide_index=True)
                                
                                # Show additional fields if there are more than 5
                                if i == 4 and len(categorical_fields) > 5:
                                    with st.expander(f"📊 Additional Categorical Fields ({len(categorical_fields) - 5} more)", expanded=False):
                                        for extra_field in categorical_fields[5:]:
                                            st.write(f"**{extra_field}:**")
                                            result = create_field_stats(df, extra_field, max_display=5)
                                            if result:
                                                stats_df, total_records, unique_values = result
                                                st.write(f"Unique values: {unique_values} | Total records: {total_records}")
                                                st.dataframe(stats_df, use_container_width=True, hide_index=True)
                            else:
                                st.info(f"No valid data found for {field}")
                
                # Display numeric field statistics
                if numeric_fields:
                    st.write("**🔢 Numeric Fields Statistics:**")
                    for field in numeric_fields:
                        with st.expander(f"📊 {field}", expanded=False):
                            result = create_numeric_stats(df, field)
                            if result:
                                st.dataframe(result, use_container_width=True, hide_index=True)
                            else:
                                st.info(f"No valid numeric data found for {field}")
                
                # Display datetime field information
                if datetime_fields:
                    st.write("**📅 DateTime Fields Information:**")
                    for field in datetime_fields:
                        with st.expander(f"📊 {field}", expanded=False):
                            try:
                                datetime_data = pd.to_datetime(df[field], errors='coerce').dropna()
                                if len(datetime_data) > 0:
                                    stats = {
                                        'Count': len(datetime_data),
                                        'Earliest': datetime_data.min(),
                                        'Latest': datetime_data.max(),
                                        'Range (Days)': (datetime_data.max() - datetime_data.min()).days
                                    }
                                    stats_df = pd.DataFrame([stats])
                                    st.dataframe(stats_df, use_container_width=True, hide_index=True)
                                else:
                                    st.info(f"No valid datetime data found for {field}")
                            except Exception as e:
                                st.warning(f"Could not analyze datetime field {field}: {e}")
                
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