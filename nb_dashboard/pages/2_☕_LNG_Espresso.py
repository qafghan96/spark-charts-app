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
    build_price_df,
)

st.title("LNG Espresso")

st.caption("Replicates the LNG Espresso chart: Spark25S Pacific and Spark30S Atlantic.")

client_id, client_secret = get_credentials()
if not client_id or not client_secret:
    st.error("Missing Spark API credentials. Set streamlit secrets 'spark.client_id' and 'spark.client_secret' (or env vars).")
    st.stop()

scopes = "read:lng-freight-prices,read:routes"
token = get_access_token(client_id, client_secret, scopes=scopes)

contracts = {
    "Spark25S Pacific": "spark25s",
    "Spark30S Atlantic": "spark30s",
}

limit = st.slider("Number of releases", min_value=12, max_value=120, value=60, step=12)

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")

fig, ax = plt.subplots(figsize=(15, 7))
ax.set_xlabel("Release Date")

colors = {
    "Spark25S Pacific": "#4F41F4",
    "Spark30S Atlantic": "#48C38D",
}

for name, ticker in contracts.items():
    df = build_price_df(token, ticker, limit=limit)
    if df.empty:
        continue
    ax.plot(df["Release Date"], df["Spark"], color=colors.get(name, "#333"), linewidth=3.0, label=name)
    ax.scatter(df["Release Date"].iloc[0], df["Spark"].iloc[0], color=colors.get(name, "#333"), s=120)

ax.set_ylim(10000, 60000)
sns.despine(left=True, bottom=True)
ax.legend()
st.pyplot(fig)



