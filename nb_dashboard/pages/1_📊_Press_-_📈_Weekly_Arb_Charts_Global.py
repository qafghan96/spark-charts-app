import os
import sys
import streamlit as st

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
    add_axis_controls,
    add_color_controls,
    apply_axis_limits,
)

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

st.title("Press - Weekly Arb Charts - Global")

st.caption("Replicates Weekly Arb Charts Global using netbacks reference and history APIs.")

client_id, client_secret = get_credentials()
if not client_id or not client_secret:
    st.error("Missing Spark API credentials. Set streamlit secrets 'spark.client_id' and 'spark.client_secret' (or env vars).")
    st.stop()

scopes = "read:lng-freight-prices,read:routes"
token = get_access_token(client_id, client_secret, scopes=scopes)

tickers, names, available_via, release_dates, _raw = list_netbacks_reference(token)

# Choose ports and via routes similar to the notebook example
port_options = {name: (uuid, vias) for uuid, name, vias in zip(tickers, names, available_via)}

left, right = st.columns(2)
with left:
    port_a = st.selectbox("Port A 🟡", options=list(port_options.keys()), index=names.index("Bonny LNG") if "Bonny LNG" in names else 0)
    via_a = st.selectbox("Port A via-point", options=port_options[port_a][1] or ["cogh"], index=0)
with right:
    port_b = st.selectbox("Port B 🟣", options=list(port_options.keys()), index=names.index("Sabine Pass") if "Sabine Pass" in names else 0)
    via_b_options = port_options[port_b][1] or ["cogh", "panama"]
    panama_index = via_b_options.index("panama") if "panama" in via_b_options else 0
    via_b = st.selectbox("Port B via-point", options=via_b_options, index=panama_index)

include_c = st.checkbox("Include Port C", value=True)
if include_c:
    c1, c2 = st.columns(2)
    with c1:
        default_c_idx = names.index("Sabine Pass") if "Sabine Pass" in names else 0
        port_c = st.selectbox("Port C 🟢", options=list(port_options.keys()), index=default_c_idx, key="port_c")
    with c2:
        via_c = st.selectbox(
            "Port C via-point",
            options=port_options[port_c][1] or ["panama", "cogh"],
            index=0,
            key="via_c",
        )

num_releases = st.slider("Number of releases", min_value=10, max_value=200, value=30, step=5)
my_releases = release_dates[:num_releases]

# Get a data sample for better axis defaults
@st.cache_data
def get_data_sample(port_a, port_b, port_c, via_a, via_b, via_c, include_c, my_releases, token):
    uuid_a, _ = port_options[port_a]
    uuid_b, _ = port_options[port_b]
    
    sample_releases = my_releases[:5]  # Use first 5 releases for sample
    sample_data_a = netbacks_history(token, uuid_a, port_a, sample_releases, via=via_a)
    sample_data_b = netbacks_history(token, uuid_b, port_b, sample_releases, via=via_b)
    sample_data_c = pd.DataFrame()
    
    if include_c:
        uuid_c, _ = port_options[port_c]
        sample_data_c = netbacks_history(token, uuid_c, port_c, sample_releases, via=via_c)
    
    return pd.concat([df for df in [sample_data_a, sample_data_b, sample_data_c] if not df.empty], ignore_index=True)

# Get data sample for axis defaults
try:
    data_sample = get_data_sample(port_a, port_b, port_c if include_c else "", via_a, via_b, via_c if include_c else "", include_c, my_releases, token)
except:
    data_sample = pd.DataFrame()

# Prepare series names for color controls
series_names = [f"{port_a} ({via_a})", f"{port_b} ({via_b})"]
if include_c:
    series_names.append(f"{port_c} ({via_c})")

# Set up default colors (current chart colors)
default_colors = ['#FFC217', '#4F41F4', '#48C38D']  # Gold, Blue, Green

# Add color controls
color_controls = add_color_controls(series_names, default_colors[:len(series_names)], expanded=True)

# Add axis controls with data-driven defaults
axis_controls = add_axis_controls(expanded=True, data_df=data_sample, x_col='Release Date', y_cols=['Delta Outrights'])

sns.set_theme(style="whitegrid")

if st.button("Generate Chart", type="primary"):
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.axhline(0, color='grey')

    uuid_a, _ = port_options[port_a]
    uuid_b, _ = port_options[port_b]

    df_a = netbacks_history(token, uuid_a, port_a, my_releases, via=via_a)
    df_b = netbacks_history(token, uuid_b, port_b, my_releases, via=via_b)
    df_c = None
    if include_c:
        uuid_c, _ = port_options[port_c]
        df_c = netbacks_history(token, uuid_c, port_c, my_releases, via=via_c)

    if df_a.empty and df_b.empty and (df_c is None or df_c.empty):
        st.warning("No data returned for the selected ports.")
    else:
        # Store price data for display
        port_prices = []
        
        if not df_a.empty:
            color_a = color_controls[f"{port_a} ({via_a})"]
            ax.plot(df_a['Release Date'], df_a['Delta Outrights'], color=color_a, label=f"{port_a} ({via_a})", linewidth=3.0)
            ax.scatter(df_a['Release Date'].iloc[0], df_a['Delta Outrights'].iloc[0], color=color_a, s=120)
            port_prices.append({
                "port": f"{port_a} ({via_a})",
                "price": df_a['Delta Outrights'].iloc[0],
                "date": df_a['Release Date'].iloc[0],
                "color": "🟡"
            })
        
        if not df_b.empty:
            color_b = color_controls[f"{port_b} ({via_b})"]
            ax.plot(df_b['Release Date'], df_b['Delta Outrights'], color=color_b, label=f"{port_b} ({via_b})", linewidth=3.0)
            ax.scatter(df_b['Release Date'].iloc[0], df_b['Delta Outrights'].iloc[0], color=color_b, s=120)
            port_prices.append({
                "port": f"{port_b} ({via_b})",
                "price": df_b['Delta Outrights'].iloc[0],
                "date": df_b['Release Date'].iloc[0],
                "color": "🟣"
            })
        
        if include_c and df_c is not None and not df_c.empty:
            color_c = color_controls[f"{port_c} ({via_c})"]
            ax.plot(df_c['Release Date'], df_c['Delta Outrights'], color=color_c, label=f"{port_c} ({via_c})", linewidth=3.0)
            ax.scatter(df_c['Release Date'].iloc[0], df_c['Delta Outrights'].iloc[0], color=color_c, s=120)
            port_prices.append({
                "port": f"{port_c} ({via_c})",
                "price": df_c['Delta Outrights'].iloc[0],
                "date": df_c['Release Date'].iloc[0],
                "color": "🟢"
            })

        # Shaded band similar to notebook
        # Choose reference df for band/x-limits: prefer B, then A, then C
        ref_df = None
        for candidate in [df_b, df_a, (df_c if include_c else None)]:
            if candidate is not None and not candidate.empty:
                ref_df = candidate
                break
        if not ref_df.empty:
            negrange = [ref_df['Release Date'].iloc[-1] - pd.Timedelta(20, unit='day'), ref_df['Release Date'].iloc[0] + pd.Timedelta(20, unit='day')]
            ax.plot(negrange, [-3.0, -3.0], color='red', alpha=0.05)
            ax.plot(negrange, [0, 0], color='red', alpha=0.05)
            ax.fill_between(negrange, 0, -3.0, color='red', alpha=0.05)

        ax.set_ylabel('$/MMBtu')
        ax.set_xlabel('Release Date')
        sns.despine(left=True, bottom=True)
        
        # Apply axis limits using the utility function
        all_data = pd.concat([df for df in [df_a, df_b, df_c] if df is not None and not df.empty], ignore_index=True)
        apply_axis_limits(ax, axis_controls, data_df=all_data, y_cols=['Delta Outrights'])
        
        # Set x-axis limits if not auto
        if not axis_controls['x_auto']:
            ax.set_xlim(axis_controls['x_min'], axis_controls['x_max'])
        else:
            # View window similar to notebook
            if not ref_df.empty:
                plt.xlim([ref_df['Release Date'].iloc[-1]-pd.Timedelta(1, unit='day'), ref_df['Release Date'].iloc[0]+pd.Timedelta(6, unit='day')])
                
        plt.tight_layout()
        st.pyplot(fig)
        
        # Display latest prices and store for access outside the conditional
        st.session_state.port_prices = port_prices

# Display latest prices if they exist
if 'port_prices' in st.session_state and st.session_state.port_prices:
    port_prices = st.session_state.port_prices
    
    # Display latest prices
    st.subheader("Latest Release Date Prices")
    if len(port_prices) == 1:
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            price_data = port_prices[0]
            st.metric(
                label=f"{price_data['color']} {price_data['port']}",
                value=f"${price_data['price']:.3f}/MMBtu",
                help=f"Latest price as of {price_data['date'].strftime('%Y-%m-%d')}"
            )
    elif len(port_prices) == 2:
        col1, col2 = st.columns(2)
        for i, price_data in enumerate(port_prices):
            with [col1, col2][i]:
                st.metric(
                    label=f"{price_data['color']} {price_data['port']}",
                    value=f"${price_data['price']:.3f}/MMBtu",
                    help=f"Latest price as of {price_data['date'].strftime('%Y-%m-%d')}"
                )
    elif len(port_prices) == 3:
        col1, col2, col3 = st.columns(3)
        for i, price_data in enumerate(port_prices):
            with [col1, col2, col3][i]:
                st.metric(
                    label=f"{price_data['color']} {price_data['port']}",
                    value=f"${price_data['price']:.3f}/MMBtu",
                    help=f"Latest price as of {price_data['date'].strftime('%Y-%m-%d')}"
                )


