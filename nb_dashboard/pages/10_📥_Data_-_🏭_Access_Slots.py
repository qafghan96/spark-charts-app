import os
import sys
import streamlit as st
import pandas as pd
from io import StringIO
import time
import requests
from urllib.parse import urljoin

# Ensure we can import sibling module "utils.py" when run as a Streamlit page
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
APP_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if APP_ROOT not in sys.path:
    sys.path.insert(0, APP_ROOT)

from utils import (
    get_credentials,
    get_access_token,
)

st.title("Data Download - Access Slots")

st.caption("Download terminal slot availability data using the access slots API.")

client_id, client_secret = get_credentials()
if not client_id or not client_secret:
    st.error("Missing Spark API credentials. Set streamlit secrets 'spark.client_id' and 'spark.client_secret' (or env vars).")
    st.stop()

scopes = "read:access"
token = get_access_token(client_id, client_secret, scopes=scopes)

API_BASE_URL = "https://api.sparkcommodities.com"

def get_latest_slots(access_token):
    """Get the latest slot release."""
    uri = urljoin(API_BASE_URL, '/beta/terminal-slots/releases/latest/')
    headers = {
        "Authorization": "Bearer {}".format(access_token),
        "accept": "text/csv"
    }
    response = requests.get(uri, headers=headers)
    if response.status_code == 200:
        df = response.content.decode('utf-8')
        df = pd.read_csv(StringIO(df))
        return df
    else:
        st.error('Bad Request for latest slots')
        return None

def get_slot_releases(access_token, date):
    """Get slot releases for a specific date."""
    uri = urljoin(API_BASE_URL, f'/beta/terminal-slots/releases/{date}/')
    headers = {
        "Authorization": "Bearer {}".format(access_token),
        "accept": "text/csv"
    }
    response = requests.get(uri, headers=headers)
    
    if response.status_code == 200:
        df = response.content.decode('utf-8')
        df = pd.read_csv(StringIO(df))
        return df
    elif response.content == b'{"errors":[{"code":"object_not_found","detail":"Object not found"}]}':
        st.error('Invalid date - no data available for this date')
        return None
    else:
        st.error('Bad Request')
        return None

def get_terminal_list(access_token):
    """Get list of terminals and their UUIDs."""
    uri = urljoin(API_BASE_URL, 'beta/terminal-slots/terminals/')
    headers = {
        "Authorization": "Bearer {}".format(access_token),
        "accept": "text/csv"
    }
    response = requests.get(uri, headers=headers)
    if response.status_code == 200:
        df = response.content.decode('utf-8')
        df = pd.read_csv(StringIO(df))
        return df
    else:
        st.error('Bad Request for terminal list')
        return None

def get_individual_terminal(access_token, terminal_uuid):
    """Get historical slots data for a specific terminal."""
    uri = urljoin(API_BASE_URL, f'/beta/terminal-slots/terminals/{terminal_uuid}/')
    headers = {
        "Authorization": "Bearer {}".format(access_token),
        "accept": "text/csv"
    }
    response = requests.get(uri, headers=headers)
    if response.status_code == 200:
        df = response.content.decode('utf-8')
        df = pd.read_csv(StringIO(df))
        return df
    elif response.content == b'{"errors":[{"code":"object_not_found","detail":"Object not found"}]}':
        st.error('Invalid terminal UUID')
        return None
    else:
        st.error('Bad Request')
        return None

def get_all_terminal_data(access_token, terminal_list):
    """Get historical data for all terminals."""
    terminals_all = pd.DataFrame()
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(len(terminal_list)):
        terminal_name = terminal_list['TerminalName'].iloc[i]
        status_text.text(f'Fetching data for {terminal_name} ({i+1}/{len(terminal_list)})')
        
        terminal_df = get_individual_terminal(access_token, terminal_list['TerminalUUID'].iloc[i])
        if terminal_df is not None:
            terminals_all = pd.concat([terminals_all, terminal_df], ignore_index=True)
        
        progress_bar.progress((i + 1) / len(terminal_list))
        time.sleep(0.1)
    
    progress_bar.empty()
    status_text.empty()
    return terminals_all

# Fetch terminal list
with st.spinner("Fetching terminal list..."):
    terminal_list = get_terminal_list(token)

if terminal_list is None:
    st.error("Failed to fetch terminal list")
    st.stop()

# User interface
st.subheader("Data Selection")

data_type = st.selectbox("Data Type", options=[
    "Latest Slot Release",
    "Specific Release Date", 
    "Individual Terminal Historical",
    "All Terminals Historical"
], index=0)

if data_type == "Latest Slot Release":
    if st.button("Fetch Latest Slots Data", type="primary"):
        with st.spinner("Fetching latest slots data..."):
            latest_df = get_latest_slots(token)
        
        if latest_df is not None:
            st.success(f"Successfully fetched latest slots data for {len(latest_df)} terminals!")
            
            st.subheader("Latest Slots Data")
            st.dataframe(latest_df, use_container_width=True)
            
            # Download button
            csv_data = latest_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name="latest_slots_data.csv",
                mime="text/csv"
            )
            
            st.session_state.slots_df = latest_df

elif data_type == "Specific Release Date":
    release_date = st.date_input("Release Date", value=pd.to_datetime("2024-10-22"))
    
    if st.button("Fetch Release Data", type="primary"):
        with st.spinner(f"Fetching slots data for {release_date}..."):
            release_df = get_slot_releases(token, release_date.strftime("%Y-%m-%d"))
        
        if release_df is not None:
            st.success(f"Successfully fetched slots data for {release_date}!")
            
            st.subheader(f"Slots Data for {release_date}")
            st.dataframe(release_df, use_container_width=True)
            
            # Download button
            csv_data = release_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name=f"slots_data_{release_date}.csv",
                mime="text/csv"
            )
            
            st.session_state.slots_df = release_df

elif data_type == "Individual Terminal Historical":
    terminal_name = st.selectbox("Terminal", options=sorted(terminal_list["TerminalName"].unique()))
    
    if st.button("Fetch Terminal Data", type="primary"):
        selected_terminal = terminal_list[terminal_list['TerminalName'] == terminal_name]
        if not selected_terminal.empty:
            terminal_uuid = selected_terminal['TerminalUUID'].iloc[0]
            
            with st.spinner(f"Fetching historical data for {terminal_name}..."):
                terminal_df = get_individual_terminal(token, terminal_uuid)
            
            if terminal_df is not None:
                st.success(f"Successfully fetched {len(terminal_df)} records for {terminal_name}!")
                
                st.subheader(f"Historical Data for {terminal_name}")
                st.dataframe(terminal_df, use_container_width=True)
                
                # Download button
                csv_data = terminal_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv_data,
                    file_name=f"terminal_data_{terminal_name.replace(' ', '_')}.csv",
                    mime="text/csv"
                )
                
                st.session_state.slots_df = terminal_df

elif data_type == "All Terminals Historical":
    st.warning("This will fetch historical data for all terminals and may take several minutes.")
    
    if st.button("Fetch All Terminal Data", type="primary"):
        with st.spinner("Fetching historical data for all terminals..."):
            all_terminals_df = get_all_terminal_data(token, terminal_list)
        
        if not all_terminals_df.empty:
            st.success(f"Successfully fetched {len(all_terminals_df)} total records from all terminals!")
            
            st.subheader("Historical Data for All Terminals")
            st.dataframe(all_terminals_df, use_container_width=True)
            
            # Download button
            csv_data = all_terminals_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name="all_terminals_historical_data.csv",
                mime="text/csv"
            )
            
            st.session_state.slots_df = all_terminals_df

# Display terminal list for reference
with st.expander("View Available Terminals"):
    st.dataframe(terminal_list, use_container_width=True)