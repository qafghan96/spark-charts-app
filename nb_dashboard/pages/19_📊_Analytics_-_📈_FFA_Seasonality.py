import os
import sys
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
    add_axis_controls,
    add_color_controls,
    apply_axis_limits,
)

st.title("📈 FFA Seasonality Charts")

st.caption("Plot seasonality charts for specific contract months from Spark FFA freight rates.")

client_id, client_secret = get_credentials()
if not client_id or not client_secret:
    st.error("Missing Spark API credentials. Set streamlit secrets 'spark.client_id' and 'spark.client_secret' (or env vars).")
    st.stop()

scopes = "read:lng-freight-prices,read:routes"
token = get_access_token(client_id, client_secret, scopes=scopes)

# Get available contracts
contracts = list_contracts(token)
ffa_contracts = [c for c in contracts if 'ffa' in c[0].lower()]

# Configuration controls
st.subheader("Configuration")

col1, col2 = st.columns(2)
with col1:
    # Contract selection
    contract_options = {name: ticker for ticker, name in ffa_contracts}
    selected_contract_name = st.selectbox("Select FFA Contract", 
                                         options=list(contract_options.keys()),
                                         index=0 if contract_options else None)
    selected_contract = contract_options.get(selected_contract_name) if selected_contract_name else None

with col2:
    # Month selection
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    selected_month = st.selectbox("Contract Month to Analyze", options=months, index=11)  # Default to Dec
    
limit = st.slider("Historical Data Limit", min_value=100, max_value=1000, value=900, step=50)

# Get a data sample for better axis defaults
@st.cache_data
def get_ffa_data_sample(token, contract, limit_sample=50):
    try:
        return build_price_df(token, contract, limit=limit_sample)
    except:
        return pd.DataFrame()

# Get data sample for axis defaults if contract is selected
data_sample = pd.DataFrame()
if selected_contract:
    data_sample = get_ffa_data_sample(token, selected_contract, limit_sample=50)

# Add axis controls with data-driven defaults
axis_controls = add_axis_controls(expanded=True, data_df=data_sample, y_cols=['Spark'])

if st.button("Generate Seasonality Chart", type="primary") and selected_contract:
    with st.spinner(f"Fetching {selected_contract} data..."):
        try:
            # Fetch historical price data
            df = build_price_df(token, selected_contract, limit=limit)
            
            if df.empty:
                st.warning("No data available for the selected contract.")
                st.stop()
            
            # Add day of year calculation for seasonality analysis
            def sort_years(df):
                reldates = df['Release Date'].to_list()
                startdates = df['Period Start'].to_list()
                
                dayofyear = []
                for i, r in enumerate(reldates):
                    start_date = pd.to_datetime(startdates[i])
                    if r.year - start_date.year == -1:
                        dayofyear.append(r.timetuple().tm_yday - 365)
                    elif r.year - start_date.year == -2:
                        dayofyear.append(r.timetuple().tm_yday - 730)
                    else:
                        dayofyear.append(r.timetuple().tm_yday)
                        
                df['Day of Year'] = dayofyear
                return df

            df = sort_years(df)
            
            # Group by contract month
            groups = df.groupby('Calendar Month')
            years = sorted(df['Release Date'].dt.year.unique())
            
            # Add color controls for recent years (last 5 years)
            recent_years = years[-5:] if len(years) >= 5 else years
            year_series_names = [f"{selected_month}{year}" for year in recent_years]
            
            # Create default color palette for years
            default_year_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
            year_color_controls = add_color_controls(
                year_series_names, 
                default_year_colors[:len(year_series_names)], 
                expanded=True
            )
            
            # Create the seasonality plot
            sns.set_style("whitegrid")
            fig, ax = plt.subplots(figsize=(14, 7))
            
            max_dates = []
            colors = plt.cm.tab10(range(len(years)))
            
            # Plot each year's data for the selected month
            for i, year in enumerate(years):
                month_year = f"{selected_month}-{year}"
                if month_year in groups.groups:
                    year_df = groups.get_group(month_year).copy()
                    year_df = sort_years(year_df)
                    
                    # Use custom color for recent years, default colormap for older years
                    series_name = f"{selected_month}{year}"
                    if series_name in year_color_controls:
                        line_color = year_color_controls[series_name]
                        linewidth = 2.5  # Slightly thicker for selected years
                    else:
                        line_color = colors[i % len(colors)]
                        linewidth = 1.5  # Thinner for older years
                    
                    ax.plot(year_df["Day of Year"], year_df["Spark"], 
                           label=series_name, 
                           color=line_color, linewidth=linewidth)
                    max_dates.append(year_df["Day of Year"].max())

            # Customize the plot
            plt.xlabel("Release Date")
            plt.ylabel("Cost in USD/day")
            plt.title(f"{selected_contract_name} - {selected_month} Contract Seasonality")

            # Set custom x-axis labels
            xlabels = ['Y-1', 'Mar', 'May', 'Jul', 'Sep', 'Nov', 'Y+0', 'Mar', 'May', 'Jul', 'Sep', 'Nov', 'Year End']
            xpos = [-365, -305, -244, -183, -121, -60, 0, 60, 121, 182, 244, 305, 365]

            # Format y-axis with currency
            current_values = plt.gca().get_yticks()
            plt.gca().set_yticklabels(['$ {:,.0f}'.format(x) for x in current_values])

            plt.xticks(xpos, xlabels)
            
            # Apply axis limits based on user controls
            if not axis_controls['x_auto']:
                ax.set_xlim(axis_controls['x_min'], axis_controls['x_max'])
            elif max_dates:
                ax.set_xlim(-365, max(max_dates))
            
            # Apply Y-axis limits using the utility function
            apply_axis_limits(ax, axis_controls, data_df=df, y_cols=['Spark'])

            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()

            st.pyplot(fig)

            # Display statistics
            st.subheader("Contract Statistics")
            
            # Calculate average price evolution
            all_month_data = []
            for year in years:
                month_year = f"{selected_month}-{year}"
                if month_year in groups.groups:
                    year_df = groups.get_group(month_year)
                    all_month_data.extend(year_df["Spark"].tolist())
            
            if all_month_data:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Average Price", f"${pd.Series(all_month_data).mean():,.0f}")
                with col2:
                    st.metric("Price Volatility (Std)", f"${pd.Series(all_month_data).std():,.0f}")
                with col3:
                    st.metric("Years Analyzed", len([y for y in years if f"{selected_month}-{y}" in groups.groups]))

            # Show data summary table
            with st.expander("View Data Summary"):
                summary_data = []
                for year in years[-5:]:  # Show last 5 years
                    month_year = f"{selected_month}-{year}"
                    if month_year in groups.groups:
                        year_df = groups.get_group(month_year)
                        summary_data.append({
                            'Year': year,
                            'Avg Price': f"${year_df['Spark'].mean():,.0f}",
                            'Min Price': f"${year_df['Spark'].min():,.0f}",
                            'Max Price': f"${year_df['Spark'].max():,.0f}",
                            'Data Points': len(year_df)
                        })
                
                if summary_data:
                    st.dataframe(pd.DataFrame(summary_data))

        except Exception as e:
            st.error(f"Error generating seasonality chart: {e}")

st.markdown("---")
st.caption("This chart shows how FFA contract prices evolve seasonally, helping identify patterns in freight rate movements.")