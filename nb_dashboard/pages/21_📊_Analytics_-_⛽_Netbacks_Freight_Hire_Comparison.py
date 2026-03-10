import os
import sys
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from typing import Literal

# Ensure we can import sibling module "utils.py" when run as a Streamlit page
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
APP_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if APP_ROOT not in sys.path:
    sys.path.insert(0, APP_ROOT)

from utils import (
    get_credentials,
    get_access_token,
    api_get,
    list_netbacks_reference,
    add_axis_controls,
    add_color_controls,
    apply_axis_limits,
)

st.title("⛽ Netbacks Freight Hire Comparison")

st.caption("Plot multiple percentages of freight hire included in a given Netback to compare arbitrage opportunities.")

st.info("📋 **Note**: This analysis requires a Cargo subscription for full functionality.")

client_id, client_secret = get_credentials()
if not client_id or not client_secret:
    st.error("Missing Spark API credentials. Set streamlit secrets 'spark.client_id' and 'spark.client_secret' (or env vars).")
    st.stop()

scopes = "read:netbacks"
token = get_access_token(client_id, client_secret, scopes=scopes)

# Functions from the script
def fetch_netback(access_token, ticker, release, via=None, laden=None, ballast=None, percent_hire: Literal[0, 100] = 100):
    query_params = f"?fob-port={ticker}"
    if release is not None:
        query_params += f"&release-date={release}"
    if via is not None:
        query_params += f"&via-point={via}"
    if laden is not None:
        query_params += f"&laden-congestion-days={laden}"
    if ballast is not None:
        query_params += f"&ballast-congestion-days={ballast}"
    if percent_hire in [0, 100]:
        query_params += f"&percent-hire={percent_hire}"
    
    content = api_get(f"/v1.0/netbacks/{query_params}", access_token)
    return content['data']

def format_store(available_via, fob_names, tickers):
    dict_store = {
        "Index": [],
        "Callable Ports": [],
        "Corresponding Ticker": [],
        "Available Via": []
    }
    
    c = 0
    for a in available_via:
        if len(a) != 0:
            dict_store['Index'].append(c)
            dict_store['Callable Ports'].append(fob_names[c])
            dict_store['Corresponding Ticker'].append(tickers[c])
            dict_store['Available Via'].append(available_via[c])
        c += 1
    
    dict_df = pd.DataFrame(dict_store)
    return dict_df

def netbacks_history(access_token, tickers, fob_names, tick, reldates, my_via=None, laden=None, ballast=None, percent_hire: Literal[0, 100] = 100, delay_seconds=0.2):
    months = []
    nea_outrights = []
    nwe_outrights = []
    release_date = []
    port = []
    failed_dates = []

    for r in reldates:
        try:
            my_dict = fetch_netback(access_token, tickers[tick], release=r, via=my_via, laden=laden, ballast=ballast, percent_hire=percent_hire)
            
            if my_dict.get('netbacks') and len(my_dict['netbacks']) > 0:
                m = my_dict['netbacks'][0]
                
                months.append(m['load']['month'])
                nea_outrights.append(float(m['nea']['outright']['usdPerMMBtu']))
                nwe_outrights.append(float(m['nwe']['outright']['usdPerMMBtu']))
                release_date.append(my_dict['releaseDate'])
                port.append(fob_names[tick])
            else:
                failed_dates.append(r)
                
        except Exception as e:
            failed_dates.append(r)
        
        # Rate limiting
        if delay_seconds:
            time.sleep(delay_seconds)
        
    if failed_dates:
        st.warning(f"Failed to fetch data for {len(failed_dates)} dates")
        
    historical_df = pd.DataFrame({
        'Release Date': release_date,
        'FoB Port': port,
        'Month': months,
        'NEA Outrights': nea_outrights,
        'NWE Outrights': nwe_outrights,
    })
    
    if not historical_df.empty:
        historical_df['Release Date'] = pd.to_datetime(historical_df['Release Date'])
    
    return historical_df

def calculate_netbacks(my_dict_0, my_dict_100, percent_hire_list):
    if my_dict_0.empty or my_dict_100.empty:
        return pd.DataFrame()
        
    m = pd.merge(my_dict_0, my_dict_100, how='inner', on=['Release Date', 'FoB Port', 'Month'], suffixes=(" 0%", " 100%"))
    
    if m.empty:
        return m
        
    m['NEA Base Costs'] = m['NEA Outrights 100%'] - m['NEA Outrights 0%']
    m['NWE Base Costs'] = m['NWE Outrights 100%'] - m['NWE Outrights 0%']

    for percent_hire in percent_hire_list:
        m[f'NEA Outright {percent_hire}%'] = m['NEA Base Costs'] * (percent_hire/100) + m['NEA Outrights 0%']
        m[f'NWE Outright {percent_hire}%'] = m['NWE Base Costs'] * (percent_hire/100) + m['NWE Outrights 0%']
        m[f'Arb {percent_hire}%'] = m[f'NEA Outright {percent_hire}%'] - m[f'NWE Outright {percent_hire}%']

    return m

# Get netbacks reference data
try:
    tickers, fob_names, available_via, release_dates, raw_dict = list_netbacks_reference(token)
    available_df = format_store(available_via, fob_names, tickers)
except Exception as e:
    st.error(f"Error fetching netbacks reference data: {str(e)}")
    st.stop()

# Configuration controls
st.subheader("Configuration")

col1, col2, col3 = st.columns(3)

with col1:
    # Port selection
    if not available_df.empty:
        selected_port = st.selectbox(
            "Select FoB Port", 
            options=available_df["Callable Ports"].tolist(),
            index=available_df["Callable Ports"].tolist().index("Sabine Pass") if "Sabine Pass" in available_df["Callable Ports"].tolist() else 0
        )
        
        # Get port index and available via points
        port_row = available_df[available_df["Callable Ports"] == selected_port].iloc[0]
        port_index = port_row["Index"]
        available_via_points = port_row["Available Via"]
    else:
        st.error("No netbacks data available")
        st.stop()

with col2:
    # Via point selection
    selected_via = st.selectbox(
        "Via Point", 
        options=available_via_points,
        index=available_via_points.index('cogh') if 'cogh' in available_via_points else 0
    )

with col3:
    # Data limit
    data_limit = st.number_input(
        "Number of Releases", 
        min_value=10, 
        max_value=500, 
        value=200, 
        step=10,
        help="Number of historical price releases to fetch"
    )

# Freight hire percentages
st.subheader("Freight Hire Percentages")

col1, col2 = st.columns(2)

with col1:
    # Predefined percentages
    use_predefined = st.checkbox("Use Predefined Percentages", value=True)
    
    if use_predefined:
        predefined_percentages = st.multiselect(
            "Select Percentages",
            options=[0, 25, 50, 75, 100],
            default=[0, 50, 100],
            help="Standard freight hire percentages to compare"
        )
        percent_hires = predefined_percentages
    else:
        # Custom percentages
        custom_percentages = st.text_input(
            "Custom Percentages (comma-separated)",
            value="0,30,70,100",
            help="Enter custom percentages separated by commas (e.g., 0,30,70,100)"
        )
        try:
            percent_hires = [int(x.strip()) for x in custom_percentages.split(',') if x.strip()]
            percent_hires = [x for x in percent_hires if 0 <= x <= 100]  # Validate range
        except:
            st.error("Invalid percentage format. Please use comma-separated integers.")
            percent_hires = [0, 50, 100]

with col2:
    # Optional parameters
    st.write("**Optional Parameters**")
    
    laden_days = st.number_input(
        "Laden Congestion Days", 
        min_value=0, 
        max_value=30, 
        value=0,
        help="Additional laden congestion days"
    )
    
    ballast_days = st.number_input(
        "Ballast Congestion Days", 
        min_value=0, 
        max_value=30, 
        value=0,
        help="Additional ballast congestion days"
    )

if st.button("Generate Freight Hire Comparison", type="primary"):
    if not percent_hires or 0 not in percent_hires or 100 not in percent_hires:
        st.error("Please ensure both 0% and 100% are included in the percentage list (required for calculations).")
        st.stop()
        
    with st.spinner("Fetching netbacks data... This may take a few minutes due to API rate limiting."):
        try:
            # Get subset of release dates
            my_rels = release_dates[:data_limit]
            
            # Fetch data for 0% and 100% freight hire
            progress_bar = st.progress(0)
            progress_text = st.empty()
            
            progress_text.text("Fetching 0% freight hire data...")
            my_dict_0 = netbacks_history(
                token, tickers, fob_names, port_index, my_rels, 
                my_via=selected_via, 
                laden=laden_days if laden_days > 0 else None,
                ballast=ballast_days if ballast_days > 0 else None,
                percent_hire=0,
                delay_seconds=0.2  # Rate limiting
            )
            progress_bar.progress(50)
            
            progress_text.text("Fetching 100% freight hire data...")
            my_dict_100 = netbacks_history(
                token, tickers, fob_names, port_index, my_rels, 
                my_via=selected_via,
                laden=laden_days if laden_days > 0 else None,
                ballast=ballast_days if ballast_days > 0 else None,
                percent_hire=100,
                delay_seconds=0.2  # Rate limiting
            )
            progress_bar.progress(100)
            progress_text.text("Processing data...")
            
            # Calculate netbacks for all percentages
            final_df = calculate_netbacks(my_dict_0, my_dict_100, percent_hires)
            
            if final_df.empty:
                st.error("No data could be processed. Please check your port selection and try again.")
                st.stop()
            
            # Store in session state
            st.session_state['netbacks_df'] = final_df
            st.session_state['percent_hires'] = percent_hires
            st.session_state['selected_port'] = selected_port
            st.session_state['selected_via'] = selected_via
            
            progress_bar.empty()
            progress_text.empty()
            st.success(f"Successfully processed {len(final_df)} data points!")
            
        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")
            st.stop()

# Display results if data exists
if 'netbacks_df' in st.session_state:
    final_df = st.session_state['netbacks_df']
    stored_percent_hires = st.session_state['percent_hires']
    stored_port = st.session_state['selected_port']
    stored_via = st.session_state['selected_via']
    
    st.subheader("Netbacks Data")
    st.dataframe(final_df, use_container_width=True)
    
    # Chart Configuration
    st.subheader("Chart Configuration")
    
    # Add axis controls
    axis_controls = add_axis_controls(expanded=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Select which percentages to plot
        percentages_to_plot = st.multiselect(
            "Select Percentages to Plot",
            options=stored_percent_hires,
            default=stored_percent_hires[:3] if len(stored_percent_hires) >= 3 else stored_percent_hires,
            help="Choose which freight hire percentages to display on the chart"
        )
        
        # Date range filtering
        if not final_df.empty:
            date_range = st.date_input(
                "Date Range",
                value=(final_df['Release Date'].min().date(), final_df['Release Date'].max().date()),
                min_value=final_df['Release Date'].min().date(),
                max_value=final_df['Release Date'].max().date(),
                help="Filter data to specific date range"
            )
    
    with col2:
        # Chart styling options
        chart_title = st.text_input(
            "Chart Title",
            value=f"US Arb via {stored_via.upper()} - {stored_port}",
            help="Custom title for the chart"
        )
        
        show_grid = st.checkbox("Show Grid", value=True)
        
        line_width = st.slider("Line Width", min_value=1, max_value=5, value=2)
        
        # Y-axis controls
        auto_scale_y = st.checkbox("Auto-scale Y-axis", value=True)
        
        if not auto_scale_y:
            col_y_min, col_y_max = st.columns(2)
            with col_y_min:
                y_min_manual = st.number_input("Y-axis Min", value=-1.0, step=0.1)
            with col_y_max:
                y_max_manual = st.number_input("Y-axis Max", value=1.0, step=0.1)
    
    # Add color controls for freight hire percentages
    if percentages_to_plot:
        series_names = [f'Arb - {percent}% Freight Hire' for percent in percentages_to_plot]
        # Use husl color palette as default (same as original)
        default_colors = sns.color_palette("husl", len(percentages_to_plot)).as_hex()
        color_controls = add_color_controls(series_names, default_colors, expanded=True)
    
    if st.button("Generate Chart", type="secondary") and percentages_to_plot:
        # Filter data by date range
        if len(date_range) == 2:
            start_date, end_date = date_range
            filtered_df = final_df[
                (final_df['Release Date'].dt.date >= start_date) &
                (final_df['Release Date'].dt.date <= end_date)
            ]
        else:
            filtered_df = final_df
        
        if filtered_df.empty:
            st.warning("No data available for the selected date range.")
            st.stop()
        
        # Sort by date
        filtered_df = filtered_df.sort_values('Release Date', ascending=True)
        
        # Create the chart
        sns.set_theme(style="whitegrid" if show_grid else "white")
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Zero line
        ax.axhline(0, color='grey', linewidth=1, alpha=0.7)
        
        # Line styles
        line_styles = ['-', '--', ':', '-.', '-']  # Different line styles
        
        # Get all y-values for selected percentages (needed for legend positioning and y-axis scaling)
        all_y_values = []
        for percent in percentages_to_plot:
            arb_column = f'Arb {percent}%'
            if arb_column in filtered_df.columns:
                y_data = filtered_df[arb_column].dropna()
                if not y_data.empty:
                    all_y_values.extend(y_data.tolist())
        
        # Plot each percentage
        for i, percent in enumerate(percentages_to_plot):
            arb_column = f'Arb {percent}%'
            if arb_column in filtered_df.columns:
                line_style = line_styles[i % len(line_styles)]
                series_name = f'Arb - {percent}% Freight Hire'
                
                # Get selected color for this series
                if series_name in color_controls:
                    line_color = color_controls[series_name]
                else:
                    # Fallback to default color palette
                    default_colors = sns.color_palette("husl", len(percentages_to_plot))
                    line_color = default_colors[i]
                
                ax.plot(
                    filtered_df['Release Date'], 
                    filtered_df[arb_column],
                    color=line_color,
                    label=f'Arb - {percent}% Freight Hire Included',
                    linewidth=line_width,
                    linestyle=line_style
                )
        
        # Formatting
        ax.set_title(chart_title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Release Date', fontsize=12)
        ax.set_ylabel('$/MMBtu', fontsize=12)
        
        # Position legend inside the chart in the best location
        if all_y_values:
            # Get the data range to determine best legend position
            y_min_data = min(all_y_values)
            y_max_data = max(all_y_values)
            
            # If most data is negative, place legend at top
            if y_max_data < 0:
                ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, framealpha=0.9)
            # If most data is positive, place legend at bottom right
            elif y_min_data > 0:
                ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True, framealpha=0.9)
            # If data spans both positive and negative, use upper left
            else:
                ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True, framealpha=0.9)
        else:
            # Default position if no data
            ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, framealpha=0.9)
        
        # Apply axis limits using the utility function
        arb_cols = [f'Arb {percent}%' for percent in percentages_to_plot]
        apply_axis_limits(ax, axis_controls, data_df=filtered_df, y_cols=arb_cols)
        
        plt.grid(show_grid, alpha=0.3)
        plt.tight_layout()
        sns.despine(left=True, bottom=True)
        
        st.pyplot(fig)
        
        # Summary Statistics
        st.subheader("Summary Statistics")
        
        stats_cols = st.columns(len(percentages_to_plot))
        
        for i, percent in enumerate(percentages_to_plot):
            arb_column = f'Arb {percent}%'
            if arb_column in filtered_df.columns:
                with stats_cols[i]:
                    arb_data = filtered_df[arb_column].dropna()
                    if not arb_data.empty:
                        st.metric(f"{percent}% Freight Hire", f"${arb_data.iloc[-1]:.3f}")
                        st.caption(f"Avg: ${arb_data.mean():.3f}")
                        st.caption(f"Range: ${arb_data.min():.3f} to ${arb_data.max():.3f}")

st.markdown("---")
st.caption("This analysis compares arbitrage opportunities at different freight hire inclusion levels, helping identify optimal freight cost allocation strategies.")