import os
import sys
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Ensure we can import sibling module "utils.py" when run as a Streamlit page
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
APP_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if APP_ROOT not in sys.path:
    sys.path.insert(0, APP_ROOT)

from utils import (
    get_credentials,
    get_access_token,
    api_get,
)

st.title("🛣️ Global Route Costs Analysis")

st.caption("Analyze and compare shipping route costs across different global LNG routes.")

client_id, client_secret = get_credentials()
if not client_id or not client_secret:
    st.error("Missing Spark API credentials. Set streamlit secrets 'spark.client_id' and 'spark.client_secret' (or env vars).")
    st.stop()

scopes = "read:routes"
token = get_access_token(client_id, client_secret, scopes=scopes)

# Get routes reference data
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_routes_reference():
    try:
        content = api_get("/v1.0/routes/reference-data/", token)
        return content.get('data', {})
    except Exception as e:
        st.error(f"Error fetching routes reference data: {e}")
        return {}

reference_data = get_routes_reference()

if not reference_data:
    st.error("Unable to fetch routes reference data. Please check your API credentials and try again.")
    st.stop()

# Extract available routes
static_data = reference_data.get('staticData', {})
routes = static_data.get('routes', [])
release_dates = static_data.get('sparkReleases', [])

# Configuration controls
st.subheader("Configuration")

if not routes:
    st.error("No routes available in the reference data.")
    st.stop()

# Create route options for selection
route_options = {}
for route in routes:
    route_name = f"{route.get('loadPort', {}).get('name', 'Unknown')} → {route.get('dischargePort', {}).get('name', 'Unknown')}"
    if route.get('viaPoint'):
        route_name += f" (via {route.get('viaPoint')})"
    route_options[route_name] = route.get('uuid', '')

col1, col2 = st.columns(2)
with col1:
    selected_routes = st.multiselect(
        "Select Routes to Compare",
        options=list(route_options.keys()),
        default=list(route_options.keys())[:5] if len(route_options) > 5 else list(route_options.keys())
    )

with col2:
    num_releases = st.slider("Number of Recent Releases", min_value=5, max_value=50, value=20)
    vessel_type = st.selectbox("Vessel Type", options=["174-2stroke", "160-tfde"], index=0)

selected_releases = release_dates[:num_releases] if release_dates else []

if st.button("Generate Route Costs Analysis", type="primary") and selected_routes:
    
    @st.cache_data(ttl=1800)  # Cache for 30 minutes
    def fetch_route_costs(route_uuid, release_date, vessel_type):
        try:
            query_params = f"?release-date={release_date}"
            if vessel_type:
                query_params += f"&vessel-type={vessel_type}"
            
            content = api_get(f"/v1.0/routes/{route_uuid}/{query_params}", token)
            return content.get('data', {})
        except Exception as e:
            return None

    with st.spinner("Fetching route costs data..."):
        routes_data = {}
        
        for route_name in selected_routes:
            route_uuid = route_options[route_name]
            route_costs = []
            
            for release_date in selected_releases:
                cost_data = fetch_route_costs(route_uuid, release_date, vessel_type)
                
                if cost_data and 'routes' in cost_data:
                    for route_info in cost_data['routes']:
                        # Extract cost components
                        total_cost = route_info.get('freightCost', {}).get('usdPerDay', {}).get('spark')
                        ballast_cost = route_info.get('ballastCost', {}).get('usdPerDay', {}).get('spark')
                        laden_cost = route_info.get('ladenCost', {}).get('usdPerDay', {}).get('spark')
                        
                        if total_cost is not None:
                            route_costs.append({
                                'Release Date': release_date,
                                'Route': route_name,
                                'Total Cost': float(total_cost),
                                'Ballast Cost': float(ballast_cost) if ballast_cost is not None else 0,
                                'Laden Cost': float(laden_cost) if laden_cost is not None else 0,
                                'Duration Days': route_info.get('durationDays'),
                                'Distance Miles': route_info.get('distanceMiles')
                            })
            
            if route_costs:
                routes_data[route_name] = pd.DataFrame(route_costs)

    if routes_data:
        # Create visualizations
        sns.set_style("whitegrid")
        
        # Plot 1: Total Route Costs Comparison
        fig1, ax1 = plt.subplots(figsize=(15, 6))
        
        for i, (route_name, df) in enumerate(routes_data.items()):
            if not df.empty:
                df['Release Date'] = pd.to_datetime(df['Release Date'])
                df_sorted = df.sort_values('Release Date')
                
                color = plt.cm.tab10(i % 10)
                ax1.plot(df_sorted['Release Date'], df_sorted['Total Cost'], 
                        marker='o', linewidth=2, label=route_name.split(' →')[0][:15] + '...', 
                        color=color)
        
        ax1.set_title(f'Route Costs Over Time ({vessel_type})')
        ax1.set_ylabel('USD per Day')
        ax1.set_xlabel('Release Date')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig1)

        # Plot 2: Average Route Costs Bar Chart
        fig2, ax2 = plt.subplots(figsize=(12, 8))
        
        avg_costs = []
        route_names_short = []
        
        for route_name, df in routes_data.items():
            if not df.empty:
                avg_cost = df['Total Cost'].mean()
                avg_costs.append(avg_cost)
                # Shorten route names for better display
                short_name = route_name.split(' →')[0][:20]
                route_names_short.append(short_name)
        
        if avg_costs:
            colors = plt.cm.Set3(range(len(avg_costs)))
            bars = ax2.barh(route_names_short, avg_costs, color=colors, alpha=0.8)
            
            # Add value labels on bars
            for bar, value in zip(bars, avg_costs):
                ax2.text(bar.get_width() + max(avg_costs) * 0.01, bar.get_y() + bar.get_height()/2,
                        f'${value:,.0f}', ha='left', va='center', fontweight='bold')
            
            ax2.set_title('Average Route Costs Comparison')
            ax2.set_xlabel('USD per Day')
            ax2.set_ylabel('Routes')
            
        plt.tight_layout()
        st.pyplot(fig2)

        # Plot 3: Cost Components Analysis
        if len(routes_data) <= 3:  # Only show detailed breakdown for few routes
            fig3, axes = plt.subplots(1, len(routes_data), figsize=(5*len(routes_data), 6))
            if len(routes_data) == 1:
                axes = [axes]
            
            for i, (route_name, df) in enumerate(routes_data.items()):
                if not df.empty:
                    avg_ballast = df['Ballast Cost'].mean()
                    avg_laden = df['Laden Cost'].mean()
                    
                    components = ['Ballast Cost', 'Laden Cost']
                    values = [avg_ballast, avg_laden]
                    colors = ['lightblue', 'lightcoral']
                    
                    axes[i].pie(values, labels=components, colors=colors, autopct='%1.1f%%')
                    axes[i].set_title(f'{route_name.split(" →")[0][:15]}...\nCost Breakdown')
            
            plt.tight_layout()
            st.pyplot(fig3)

        # Summary Statistics
        st.subheader("Route Performance Metrics")
        summary_data = []
        
        for route_name, df in routes_data.items():
            if not df.empty:
                summary_data.append({
                    'Route': route_name,
                    'Avg Total Cost': f"${df['Total Cost'].mean():,.0f}",
                    'Min Cost': f"${df['Total Cost'].min():,.0f}",
                    'Max Cost': f"${df['Total Cost'].max():,.0f}",
                    'Avg Duration': f"{df['Duration Days'].mean():.1f} days" if 'Duration Days' in df.columns else 'N/A',
                    'Avg Distance': f"{df['Distance Miles'].mean():,.0f} miles" if 'Distance Miles' in df.columns else 'N/A',
                    'Cost Volatility': f"${df['Total Cost'].std():,.0f}",
                    'Data Points': len(df)
                })
        
        if summary_data:
            st.dataframe(pd.DataFrame(summary_data), use_container_width=True)

        # Cost efficiency analysis
        st.subheader("Cost Efficiency Analysis")
        
        efficiency_data = []
        for route_name, df in routes_data.items():
            if not df.empty and 'Distance Miles' in df.columns and 'Duration Days' in df.columns:
                avg_cost_per_mile = df['Total Cost'].mean() / df['Distance Miles'].mean()
                avg_cost_per_day = df['Total Cost'].mean() / df['Duration Days'].mean()
                
                efficiency_data.append({
                    'Route': route_name,
                    'Cost per Mile': f"${avg_cost_per_mile:.2f}",
                    'Cost per Day': f"${avg_cost_per_day:,.0f}",
                    'Utilization Score': f"{(1 / (avg_cost_per_mile * avg_cost_per_day)) * 1000000:.2f}"
                })
        
        if efficiency_data:
            st.dataframe(pd.DataFrame(efficiency_data), use_container_width=True)

        # Detailed data view
        with st.expander("View Detailed Route Data"):
            for route_name, df in routes_data.items():
                if not df.empty:
                    st.subheader(f"{route_name}")
                    st.dataframe(df.sort_values('Release Date', ascending=False))

    else:
        st.warning("No data available for the selected routes and parameters. Try adjusting your selections.")

st.markdown("---")
st.caption("This analysis provides comprehensive insights into global LNG shipping route costs and efficiency metrics.")