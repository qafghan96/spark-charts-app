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
)

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
    port_a = st.selectbox("Port A", options=list(port_options.keys()), index=names.index("Bonny LNG") if "Bonny LNG" in names else 0)
    via_a = st.selectbox("Port A via-point", options=port_options[port_a][1] or ["cogh"], index=0)
with right:
    port_b = st.selectbox("Port B", options=list(port_options.keys()), index=names.index("Sabine Pass") if "Sabine Pass" in names else 0)
    via_b = st.selectbox("Port B via-point", options=port_options[port_b][1] or ["cogh", "panama"], index=0)

include_c = st.checkbox("Include Port C", value=True)
if include_c:
    c1, c2 = st.columns(2)
    with c1:
        default_c_idx = names.index("Sabine Pass") if "Sabine Pass" in names else 0
        port_c = st.selectbox("Port C", options=list(port_options.keys()), index=default_c_idx, key="port_c")
    with c2:
        via_c = st.selectbox(
            "Port C via-point",
            options=port_options[port_c][1] or ["panama", "cogh"],
            index=0,
            key="via_c",
        )

num_releases = st.slider("Number of releases", min_value=10, max_value=90, value=30, step=5)
my_releases = release_dates[:num_releases]

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")
import pandas as pd

fig, ax = plt.subplots(figsize=(12, 6))
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
    if not df_a.empty:
        ax.plot(df_a['Release Date'], df_a['Delta Outrights'], color='#FFC217', label=f"{port_a} ({via_a})", linewidth=3.0)
        ax.scatter(df_a['Release Date'].iloc[0], df_a['Delta Outrights'].iloc[0], color='#FFC217', s=120)
    if not df_b.empty:
        ax.plot(df_b['Release Date'], df_b['Delta Outrights'], color='#4F41F4', label=f"{port_b} ({via_b})", linewidth=3.0)
        ax.scatter(df_b['Release Date'].iloc[0], df_b['Delta Outrights'].iloc[0], color='#4F41F4', s=120)
    if include_c and df_c is not None and not df_c.empty:
        ax.plot(df_c['Release Date'], df_c['Delta Outrights'], color='#48C38D', label=f"{port_c} ({via_c})", linewidth=3.0)
        ax.scatter(df_c['Release Date'].iloc[0], df_c['Delta Outrights'].iloc[0], color='#48C38D', s=120)

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
    # View window similar to notebook
    if not ref_df.empty:
        plt.xlim([ref_df['Release Date'].iloc[-1]-pd.Timedelta(1, unit='day'), ref_df['Release Date'].iloc[0]+pd.Timedelta(6, unit='day')])
    plt.ylim(-0.5, 0.5)
    plt.tight_layout()
    st.pyplot(fig)


