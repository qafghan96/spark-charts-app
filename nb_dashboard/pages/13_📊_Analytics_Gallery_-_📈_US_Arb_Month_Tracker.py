import os
import sys
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt

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
)

st.title("📈 US Arb Month Tracker")

st.caption("Track US arbitrage opportunities month-by-month with detailed breakdowns.")

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
    us_ports = [name for name in port_options.keys() if any(x in name.lower() for x in ['sabine', 'cameron', 'cheniere', 'freeport', 'gulf', 'corpus'])]
    
    if us_ports:
        default_port = us_ports[0]
    else:
        default_port = list(port_options.keys())[0] if port_options else ""
    
    selected_port = st.selectbox("Select US FoB Port", options=list(port_options.keys()), 
                                index=names.index(default_port) if default_port in names else 0)
    
    uuid, available_vias = port_options[selected_port]

with col2:
    selected_via = st.selectbox("Via Point", options=available_vias or ["cogh"], index=0)
    tracking_months = st.slider("Months to Track", min_value=6, max_value=24, value=12)

# Display options
display_options = st.columns(3)
with display_options[0]:
    show_components = st.checkbox("Show Price Components", value=True)
with display_options[1]:
    show_trends = st.checkbox("Show Trend Analysis", value=True)
with display_options[2]:
    show_volatility = st.checkbox("Show Volatility Metrics", value=False)

if st.button("Generate Month Tracker", type="primary"):
    port_idx = names.index(selected_port)
    my_releases = release_dates[:tracking_months * 4]  # Approximate 4 releases per month

    with st.spinner(f"Fetching {tracking_months} months of data for {selected_port}..."):
        df = netbacks_history(token, uuid, selected_port, my_releases, via=selected_via, delay_seconds=0.1)
        
        if df.empty:
            st.warning("No data available for the selected parameters.")
            st.stop()
        
        # Add month-year grouping
        df['Month_Year'] = df['Release Date'].dt.to_period('M')
        df['Month_Name'] = df['Release Date'].dt.strftime('%b %Y')
        
        # Group by month and calculate statistics
        monthly_stats = df.groupby(['Month_Year', 'Month_Name']).agg({
            'Delta Outrights': ['mean', 'std', 'min', 'max', 'count'],
            'NEA Outrights': ['mean'],
            'NWE Outrights': ['mean'],
            'NEA TTF Basis': ['mean'],
            'NWE TTF Basis': ['mean'],
            'Delta TTF Basis': ['mean']
        }).reset_index()
        
        # Flatten column names
        monthly_stats.columns = ['Month_Year', 'Month_Name', 'Delta_Mean', 'Delta_Std', 'Delta_Min', 'Delta_Max', 'Count',
                                'NEA_Mean', 'NWE_Mean', 'NEA_TTF_Mean', 'NWE_TTF_Mean', 'Delta_TTF_Mean']
        
        monthly_stats = monthly_stats.sort_values('Month_Year')

    # Main visualization
    sns.set_style("whitegrid")
    
    if show_components:
        # Plot with components
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
        
        # Plot 1: Delta Outrights with error bars
        ax1.errorbar(range(len(monthly_stats)), monthly_stats['Delta_Mean'], 
                    yerr=monthly_stats['Delta_Std'], fmt='o-', linewidth=2, 
                    color='#4F41F4', capsize=5, capthick=2)
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
        ax1.set_title(f'{selected_port} - US Arb Delta (NEA - NWE) via {selected_via.upper()}')
        ax1.set_ylabel('$/MMBtu')
        ax1.set_xticks(range(len(monthly_stats)))
        ax1.set_xticklabels(monthly_stats['Month_Name'], rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Add shaded regions for positive/negative arb
        for i in range(len(monthly_stats)):
            color = 'lightgreen' if monthly_stats.iloc[i]['Delta_Mean'] > 0 else 'lightcoral'
            ax1.axvspan(i-0.4, i+0.4, alpha=0.1, color=color)
        
        # Plot 2: NEA vs NWE Components
        ax2.plot(range(len(monthly_stats)), monthly_stats['NEA_Mean'], 
                marker='o', linewidth=2, label='NEA Outrights', color='orange')
        ax2.plot(range(len(monthly_stats)), monthly_stats['NWE_Mean'], 
                marker='s', linewidth=2, label='NWE Outrights', color='blue')
        ax2.set_title('NEA vs NWE Outright Prices')
        ax2.set_ylabel('$/MMBtu')
        ax2.set_xticks(range(len(monthly_stats)))
        ax2.set_xticklabels(monthly_stats['Month_Name'], rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: TTF Basis Components
        ax3.plot(range(len(monthly_stats)), monthly_stats['NEA_TTF_Mean'], 
                marker='o', linewidth=2, label='NEA TTF Basis', color='lightblue')
        ax3.plot(range(len(monthly_stats)), monthly_stats['NWE_TTF_Mean'], 
                marker='s', linewidth=2, label='NWE TTF Basis', color='lightcoral')
        ax3.plot(range(len(monthly_stats)), monthly_stats['Delta_TTF_Mean'], 
                marker='^', linewidth=2, label='Delta TTF Basis', color='green')
        ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
        ax3.set_title('TTF Basis Components')
        ax3.set_ylabel('$/MMBtu')
        ax3.set_xlabel('Month')
        ax3.set_xticks(range(len(monthly_stats)))
        ax3.set_xticklabels(monthly_stats['Month_Name'], rotation=45)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
    else:
        # Simplified plot - just delta
        fig, ax1 = plt.subplots(1, 1, figsize=(15, 6))
        
        ax1.plot(range(len(monthly_stats)), monthly_stats['Delta_Mean'], 
                marker='o', linewidth=3, color='#4F41F4')
        ax1.fill_between(range(len(monthly_stats)), 
                        monthly_stats['Delta_Mean'] - monthly_stats['Delta_Std'],
                        monthly_stats['Delta_Mean'] + monthly_stats['Delta_Std'],
                        alpha=0.2, color='#4F41F4')
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
        ax1.set_title(f'{selected_port} - US Arb Month Tracker via {selected_via.upper()}')
        ax1.set_ylabel('$/MMBtu')
        ax1.set_xlabel('Month')
        ax1.set_xticks(range(len(monthly_stats)))
        ax1.set_xticklabels(monthly_stats['Month_Name'], rotation=45)
        ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)

    # Summary metrics
    st.subheader("Monthly Performance Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        avg_delta = monthly_stats['Delta_Mean'].mean()
        st.metric("Average Delta", f"${avg_delta:.2f}", 
                 delta=f"{'+' if avg_delta > 0 else ''}{avg_delta:.2f}")
    
    with col2:
        positive_months = len(monthly_stats[monthly_stats['Delta_Mean'] > 0])
        positive_pct = (positive_months / len(monthly_stats)) * 100
        st.metric("Positive Arb Months", f"{positive_months}/{len(monthly_stats)}", 
                 delta=f"{positive_pct:.1f}%")
    
    with col3:
        max_arb = monthly_stats['Delta_Mean'].max()
        max_month = monthly_stats.loc[monthly_stats['Delta_Mean'].idxmax(), 'Month_Name']
        st.metric("Best Month", max_month, delta=f"${max_arb:.2f}")
    
    with col4:
        volatility = monthly_stats['Delta_Mean'].std()
        st.metric("Monthly Volatility", f"${volatility:.2f}")

    if show_trends:
        st.subheader("Trend Analysis")
        
        # Calculate 3-month moving average
        monthly_stats['MA_3'] = monthly_stats['Delta_Mean'].rolling(window=3).mean()
        monthly_stats['Trend'] = monthly_stats['Delta_Mean'].diff()
        
        fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8))
        
        # Moving average plot
        ax1.plot(range(len(monthly_stats)), monthly_stats['Delta_Mean'], 
                marker='o', linewidth=2, label='Monthly Average', alpha=0.7)
        ax1.plot(range(len(monthly_stats)), monthly_stats['MA_3'], 
                linewidth=3, label='3-Month MA', color='red')
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
        ax1.set_title('Delta with 3-Month Moving Average')
        ax1.set_ylabel('$/MMBtu')
        ax1.set_xticks(range(len(monthly_stats)))
        ax1.set_xticklabels(monthly_stats['Month_Name'], rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Trend changes
        trend_colors = ['green' if x > 0 else 'red' for x in monthly_stats['Trend'].fillna(0)]
        ax2.bar(range(1, len(monthly_stats)), monthly_stats['Trend'].iloc[1:], 
               color=trend_colors[1:], alpha=0.7)
        ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.7)
        ax2.set_title('Month-to-Month Changes')
        ax2.set_ylabel('Change ($/MMBtu)')
        ax2.set_xlabel('Month')
        ax2.set_xticks(range(1, len(monthly_stats)))
        ax2.set_xticklabels(monthly_stats['Month_Name'].iloc[1:], rotation=45)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig2)

    if show_volatility:
        st.subheader("Volatility Analysis")
        
        fig3, ax = plt.subplots(figsize=(15, 6))
        
        # Volatility by month
        ax.bar(range(len(monthly_stats)), monthly_stats['Delta_Std'], 
              alpha=0.7, color='purple')
        ax.set_title('Monthly Volatility (Standard Deviation)')
        ax.set_ylabel('Volatility ($/MMBtu)')
        ax.set_xlabel('Month')
        ax.set_xticks(range(len(monthly_stats)))
        ax.set_xticklabels(monthly_stats['Month_Name'], rotation=45)
        ax.grid(True, alpha=0.3)
        
        # Add average volatility line
        avg_vol = monthly_stats['Delta_Std'].mean()
        ax.axhline(y=avg_vol, color='red', linestyle='--', 
                  label=f'Average: ${avg_vol:.2f}')
        ax.legend()
        
        plt.tight_layout()
        st.pyplot(fig3)

    # Data table
    st.subheader("Monthly Data Table")
    
    # Prepare display table
    display_table = monthly_stats[['Month_Name', 'Delta_Mean', 'Delta_Std', 'Delta_Min', 'Delta_Max', 'Count']].copy()
    display_table.columns = ['Month', 'Avg Delta', 'Std Dev', 'Min', 'Max', 'Data Points']
    
    # Format currency columns
    for col in ['Avg Delta', 'Std Dev', 'Min', 'Max']:
        display_table[col] = display_table[col].apply(lambda x: f"${x:.2f}")
    
    st.dataframe(display_table, use_container_width=True)

    # Raw data download
    with st.expander("Download Raw Data"):
        csv_data = df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name=f"us_arb_tracker_{selected_port}_{selected_via}_{dt.datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

st.markdown("---")
st.caption("Track US arbitrage performance month-by-month to identify seasonal patterns and market opportunities.")