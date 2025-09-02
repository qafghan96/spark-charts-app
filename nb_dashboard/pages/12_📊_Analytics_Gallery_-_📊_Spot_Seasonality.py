import os
import sys
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

# Ensure we can import sibling module "utils.py" when run as a Streamlit page
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
APP_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if APP_ROOT not in sys.path:
    sys.path.insert(0, APP_ROOT)

from utils import (
    get_credentials,
    get_access_token,
    list_contracts,
    build_price_df,
)

st.title("📊 Spot Price Seasonality Analysis")

st.caption("Analyze seasonal patterns in spot freight prices across different contract types.")

client_id, client_secret = get_credentials()
if not client_id or not client_secret:
    st.error("Missing Spark API credentials. Set streamlit secrets 'spark.client_id' and 'spark.client_secret' (or env vars).")
    st.stop()

scopes = "read:lng-freight-prices,read:routes"
token = get_access_token(client_id, client_secret, scopes=scopes)

# Get available contracts
contracts = list_contracts(token)
spot_contracts = [c for c in contracts if 'spot' in c[1].lower() or any(x in c[0].lower() for x in ['25s', '30s'])]

# Configuration controls
st.subheader("Configuration")

col1, col2 = st.columns(2)
with col1:
    # Contract selection
    if spot_contracts:
        contract_options = {name: ticker for ticker, name in spot_contracts}
        selected_contract_name = st.selectbox("Select Spot Contract", 
                                             options=list(contract_options.keys()),
                                             index=0)
        selected_contract = contract_options.get(selected_contract_name)
    else:
        st.error("No spot contracts available.")
        st.stop()

with col2:
    analysis_type = st.selectbox("Analysis Type", 
                                options=["Monthly Seasonality", "Quarterly Patterns", "Year-over-Year"], 
                                index=0)

# Advanced options
with st.expander("Advanced Options"):
    data_limit = st.slider("Historical Data Limit", min_value=200, max_value=2000, value=1000, step=100)
    min_years = st.slider("Minimum Years for Analysis", min_value=2, max_value=5, value=3)
    smoothing = st.checkbox("Apply Moving Average Smoothing", value=True)
    if smoothing:
        window_size = st.slider("Moving Average Window", min_value=5, max_value=30, value=10)

if st.button("Generate Seasonality Analysis", type="primary") and selected_contract:
    with st.spinner(f"Fetching {selected_contract} historical data..."):
        try:
            # Fetch historical price data
            df = build_price_df(token, selected_contract, limit=data_limit)
            
            if df.empty:
                st.warning("No data available for the selected contract.")
                st.stop()
            
            # Add time-based features
            df['Year'] = df['Release Date'].dt.year
            df['Month'] = df['Release Date'].dt.month
            df['Quarter'] = df['Release Date'].dt.quarter
            df['DayOfYear'] = df['Release Date'].dt.dayofyear
            df['WeekOfYear'] = df['Release Date'].dt.isocalendar().week
            
            # Filter for sufficient data
            year_counts = df['Year'].value_counts()
            valid_years = year_counts[year_counts >= 50].index.tolist()  # At least 50 data points per year
            
            if len(valid_years) < min_years:
                st.warning(f"Insufficient data. Only {len(valid_years)} years with adequate data, minimum {min_years} required.")
                st.stop()
            
            df_filtered = df[df['Year'].isin(valid_years)].copy()
            
            # Apply smoothing if requested
            if smoothing:
                df_filtered = df_filtered.sort_values('Release Date')
                df_filtered['Spark_Smoothed'] = df_filtered['Spark'].rolling(window=window_size, center=True).mean()
                price_col = 'Spark_Smoothed'
            else:
                price_col = 'Spark'
            
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            st.stop()

    # Generate analysis based on selected type
    if analysis_type == "Monthly Seasonality":
        # Monthly seasonality analysis
        sns.set_style("whitegrid")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        # Plot 1: Monthly patterns by year
        for year in sorted(valid_years):
            year_data = df_filtered[df_filtered['Year'] == year]
            monthly_avg = year_data.groupby('Month')[price_col].mean()
            
            ax1.plot(monthly_avg.index, monthly_avg.values, 
                    marker='o', linewidth=2, label=str(year), alpha=0.7)
        
        ax1.set_title(f'{selected_contract_name} - Monthly Seasonality by Year')
        ax1.set_xlabel('Month')
        ax1.set_ylabel('Average Price (USD/day)')
        ax1.set_xticks(range(1, 13))
        ax1.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                            'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Average monthly pattern with confidence intervals
        monthly_stats = df_filtered.groupby('Month')[price_col].agg(['mean', 'std', 'count']).reset_index()
        monthly_stats['se'] = monthly_stats['std'] / np.sqrt(monthly_stats['count'])
        monthly_stats['ci_lower'] = monthly_stats['mean'] - 1.96 * monthly_stats['se']
        monthly_stats['ci_upper'] = monthly_stats['mean'] + 1.96 * monthly_stats['se']
        
        ax2.plot(monthly_stats['Month'], monthly_stats['mean'], 
                marker='o', linewidth=3, color='darkblue', label='Average')
        ax2.fill_between(monthly_stats['Month'], monthly_stats['ci_lower'], 
                        monthly_stats['ci_upper'], alpha=0.2, color='blue', label='95% CI')
        
        ax2.set_title(f'{selected_contract_name} - Average Monthly Seasonality Pattern')
        ax2.set_xlabel('Month')
        ax2.set_ylabel('Average Price (USD/day)')
        ax2.set_xticks(range(1, 13))
        ax2.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                            'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Monthly statistics table
        st.subheader("Monthly Statistics")
        monthly_display = monthly_stats.copy()
        monthly_display['Month_Name'] = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        monthly_display = monthly_display[['Month_Name', 'mean', 'std', 'count']]
        monthly_display.columns = ['Month', 'Average Price', 'Std Dev', 'Data Points']
        monthly_display['Average Price'] = monthly_display['Average Price'].apply(lambda x: f"${x:,.0f}")
        monthly_display['Std Dev'] = monthly_display['Std Dev'].apply(lambda x: f"${x:,.0f}")
        st.dataframe(monthly_display, use_container_width=True)

    elif analysis_type == "Quarterly Patterns":
        # Quarterly analysis
        sns.set_style("whitegrid")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Quarterly trends by year
        for year in sorted(valid_years):
            year_data = df_filtered[df_filtered['Year'] == year]
            quarterly_avg = year_data.groupby('Quarter')[price_col].mean()
            
            ax1.plot(quarterly_avg.index, quarterly_avg.values, 
                    marker='o', linewidth=2, label=str(year), alpha=0.7)
        
        ax1.set_title(f'{selected_contract_name} - Quarterly Patterns by Year')
        ax1.set_xlabel('Quarter')
        ax1.set_ylabel('Average Price (USD/day)')
        ax1.set_xticks([1, 2, 3, 4])
        ax1.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4'])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Average quarterly pattern
        quarterly_stats = df_filtered.groupby('Quarter')[price_col].agg(['mean', 'std']).reset_index()
        
        bars = ax2.bar(quarterly_stats['Quarter'], quarterly_stats['mean'], 
                      yerr=quarterly_stats['std'], capsize=5, alpha=0.7,
                      color=['lightblue', 'lightgreen', 'lightcoral', 'lightsalmon'])
        
        ax2.set_title(f'{selected_contract_name} - Average Quarterly Pattern')
        ax2.set_xlabel('Quarter')
        ax2.set_ylabel('Average Price (USD/day)')
        ax2.set_xticks([1, 2, 3, 4])
        ax2.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4'])
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, quarterly_stats['mean']):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + quarterly_stats['std'].mean() * 0.1,
                    f'${value:,.0f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig)

    elif analysis_type == "Year-over-Year":
        # Year-over-Year comparison
        sns.set_style("whitegrid")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Plot 1: Annual average prices
        annual_stats = df_filtered.groupby('Year')[price_col].agg(['mean', 'std', 'min', 'max']).reset_index()
        
        bars = ax1.bar(annual_stats['Year'], annual_stats['mean'], 
                      yerr=annual_stats['std'], capsize=5, alpha=0.7)
        
        ax1.set_title(f'{selected_contract_name} - Year-over-Year Comparison')
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Average Price (USD/day)')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, annual_stats['mean']):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + annual_stats['std'].mean() * 0.1,
                    f'${value:,.0f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: Price volatility by year
        ax2.bar(annual_stats['Year'], annual_stats['std'], alpha=0.7, color='orange')
        ax2.set_title(f'{selected_contract_name} - Price Volatility by Year')
        ax2.set_xlabel('Year')
        ax2.set_ylabel('Price Volatility (Std Dev)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Annual statistics table
        st.subheader("Annual Statistics")
        annual_display = annual_stats.copy()
        annual_display.columns = ['Year', 'Average', 'Std Dev', 'Min', 'Max']
        for col in ['Average', 'Std Dev', 'Min', 'Max']:
            annual_display[col] = annual_display[col].apply(lambda x: f"${x:,.0f}")
        st.dataframe(annual_display, use_container_width=True)

    # Overall statistics
    st.subheader("Overall Analysis Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Data Points", f"{len(df_filtered):,}")
    with col2:
        st.metric("Years Analyzed", len(valid_years))
    with col3:
        st.metric("Average Price", f"${df_filtered[price_col].mean():,.0f}")
    with col4:
        st.metric("Price Range", f"${df_filtered[price_col].max() - df_filtered[price_col].min():,.0f}")

    # Raw data view
    with st.expander("View Raw Data Sample"):
        st.dataframe(df_filtered[['Release Date', 'Spark', 'Year', 'Month', 'Quarter']].head(20))

st.markdown("---")
st.caption("This analysis identifies seasonal patterns in spot freight prices to help with market timing and forecasting.")