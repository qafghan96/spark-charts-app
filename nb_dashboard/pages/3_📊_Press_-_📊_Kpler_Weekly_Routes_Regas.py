import os
import sys
import pandas as pd
import streamlit as st

# Ensure we can import sibling module "utils.py"
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
APP_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if APP_ROOT not in sys.path:
    sys.path.insert(0, APP_ROOT)

from utils import (
    get_credentials,
    get_access_token,
    api_get,
)


st.title("Kpler Weekly: Routes + Regas Data")
st.caption("Replicates df2 from Kpler_weekly_routesregas_data.ipynb: first datapoint per release for Regas and Routes.")


# --- Functions adapted from the notebook ---

def fetch_price_releases_access(access_token: str, limit: int = 8, offset: int | None = None) -> list[dict]:
    query = f"?limit={limit}"
    if offset is not None:
        query += f"&offset={offset}"
    content = api_get(f"/beta/sparkr/releases/{query}", access_token)
    return content.get("data", [])


def organise_access_dataframe(latest: list[dict]) -> pd.DataFrame:
    data_dict: dict[str, list] = {
        "Release Date": [],
        "Terminal": [],
        "Month": [],
        "Vessel Size": [],
        "Total $/MMBtu": [],
        "Basic Slot (Berth)": [],
        "Basic Slot (Unload/Stor/Regas)": [],
        "Basic Slot (B/U/S/R)": [],
        "Additional Storage": [],
        "Additional Sendout": [],
    }
    if not latest:
        return pd.DataFrame(data_dict)
    sizes_available = list(latest[0].get("perVesselSize", {}).keys())
    for item in latest:
        for s in sizes_available:
            delivery_months = item.get("perVesselSize", {}).get(s, {}).get("deliveryMonths", [])
            for month in delivery_months:
                data_dict["Release Date"].append(item.get("releaseDate"))
                data_dict["Terminal"].append(item.get("terminalName"))
                data_dict["Month"].append(month.get("month"))
                data_dict["Vessel Size"].append(s)
                costs_mmbtu = month.get("costsInUsdPerMmbtu", {})
                data_dict["Total $/MMBtu"].append(float(costs_mmbtu.get("total", 0)))
                breakdown = costs_mmbtu.get("breakdown", {})
                data_dict["Basic Slot (Berth)"].append(float(breakdown.get("basic-slot-berth", {}).get("value", 0)))
                data_dict["Basic Slot (Unload/Stor/Regas)"].append(float(breakdown.get("basic-slot-unload-storage-regas", {}).get("value", 0)))
                data_dict["Basic Slot (B/U/S/R)"].append(float(breakdown.get("basic-slot-berth-unload-storage-regas", {}).get("value", 0)))
                data_dict["Additional Storage"].append(float(breakdown.get("additional-storage", {}).get("value", 0)))
                data_dict["Additional Sendout"].append(float(breakdown.get("additional-send-out", {}).get("value", 0)))
    df = pd.DataFrame(data_dict)
    if not df.empty:
        df["Month"] = pd.to_datetime(df["Month"])  # type: ignore
        df["Release Date"] = pd.to_datetime(df["Release Date"])  # type: ignore
    return df


def list_routes(access_token: str) -> tuple[list[str], list[str]]:
    content = api_get("/v1.0/routes/", access_token)
    data = content.get("data", {})
    routes = data.get("routes", [])
    reldates = data.get("sparkReleaseDates", [])
    # Build a DataFrame-like structure for route selection
    labels: list[str] = []
    ids: list[str] = []
    for r in routes:
        load_name = r.get("loadPort", {}).get("name")
        discharge_name = r.get("dischargePort", {}).get("name")
        via = r.get("via")
        route_id = r.get("uuid")
        label = f"{load_name} → {discharge_name}" + (f" (via {via})" if via else "")
        labels.append(label)
        ids.append(route_id)
    return list(zip(ids, labels)), reldates


def fetch_route_data(access_token: str, route_uuid: str, release: str, congestion_laden: float | None = None,
                     congestion_ballast: float | None = None) -> dict:
    query = f"?release-date={release}"
    if congestion_laden is not None:
        query += f"&congestion-laden-days={congestion_laden}"
    if congestion_ballast is not None:
        query += f"&congestion-ballast-days={congestion_ballast}"
    content = api_get(f"/v1.0/routes/{route_uuid}/{query}", access_token)
    return content.get("data", {})


def routes_history(access_token: str, route_uuid: str, release_dates: list[str], laden: float | None = None,
                   ballast: float | None = None) -> pd.DataFrame:
    import time as _time
    import datetime as dt

    my_route = {
        "Period": [],
        "Start Date": [],
        "End Date": [],
        "Total Cost USD": [],
        "Cost USDperMMBtu": [],
        "Hire USDperMMBtu": [],
        "Fuel USDperMMBtu": [],
        "Port USDperMMBtu": [],
        "Canal USDperMMBtu": [],
        "Congestion USDperMMBtu": [],
        "Release Date": [],
        "Cal Month": [],
    }
    for r in release_dates:
        try:
            doc = fetch_route_data(access_token, route_uuid, release=r, congestion_laden=laden, congestion_ballast=ballast)
        except Exception:
            continue
        for data in doc.get("dataPoints", []):
            my_route["Start Date"].append(data.get("deliveryPeriod", {}).get("startAt"))
            my_route["End Date"].append(data.get("deliveryPeriod", {}).get("endAt"))
            my_route["Period"].append(data.get("deliveryPeriod", {}).get("name"))
            my_route["Total Cost USD"].append(data.get("costsInUsd", {}).get("total"))
            mmbtu = data.get("costsInUsdPerMmbtu", {})
            my_route["Cost USDperMMBtu"].append(mmbtu.get("total"))
            my_route["Hire USDperMMBtu"].append(mmbtu.get("hire"))
            my_route["Fuel USDperMMBtu"].append(mmbtu.get("fuel"))
            my_route["Port USDperMMBtu"].append(mmbtu.get("port"))
            my_route["Canal USDperMMBtu"].append(mmbtu.get("canal"))
            my_route["Congestion USDperMMBtu"].append(mmbtu.get("congestion", 0))
            my_route["Release Date"].append(r)
            start_at = data.get("deliveryPeriod", {}).get("startAt")
            if start_at:
                my_route["Cal Month"].append(dt.datetime.strptime(start_at, "%Y-%m-%d").strftime("%b%y"))
            else:
                my_route["Cal Month"].append("")
        _time.sleep(0.2)

    df = pd.DataFrame(my_route)
    for col in [
        "Total Cost USD",
        "Cost USDperMMBtu",
        "Hire USDperMMBtu",
        "Fuel USDperMMBtu",
        "Port USDperMMBtu",
        "Canal USDperMMBtu",
        "Congestion USDperMMBtu",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["Release Date", "Start Date", "End Date"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    df["Time Diff"] = (df["Release Date"] - df["Start Date"]).dt.days
    return df


# --- Page logic ---

client_id, client_secret = get_credentials()
if not client_id or not client_secret:
    st.error("Missing Spark API credentials. Use the homepage to set them in session, or configure secrets.")
    st.stop()

scopes = "read:access,read:prices,read:routes"
token = get_access_token(client_id, client_secret, scopes=scopes)

# Load route list
route_pairs, reldates = list_routes(token)
labels = [label for _id, label in route_pairs]
ids = [_id for _id, label in route_pairs]

col_a, col_b = st.columns([2, 1])
with col_a:
    route_sel = st.selectbox("Select route", options=labels)
with col_b:
    num_releases = st.number_input("Releases", min_value=5, max_value=50, value=10, step=1)

route_id = ids[labels.index(route_sel)] if labels else None

terminal = st.text_input("Regas terminal name (for Access data)", value="Gate")

if route_id is None:
    st.warning("No routes available.")
    st.stop()

# Fetch Access historical and organise to dataframe
historical = fetch_price_releases_access(token, limit=int(num_releases))
access_df = organise_access_dataframe(historical)

# Fetch route history for the selected route
histdf = routes_history(token, route_id, reldates[: int(num_releases)])

# Build df2: first datapoint per release
# Access (Regas) side
a = access_df[access_df["Terminal"].str.lower() == terminal.lower()].copy()
if a.empty:
    st.warning("No Access data for the selected terminal.")
    st.stop()

# unique release dates (descending)
rels = list(pd.to_datetime(a["Release Date"]).dropna().sort_values(ascending=False).unique())
ag = a.groupby("Release Date")
prices: list[float] = []
for r in rels:
    try:
        g = ag.get_group(r)
    except Exception:
        prices.append(float("nan"))
        continue
    # first datapoint for this release
    prices.append(float(g["Total $/MMBtu"].iloc[0]))

# Routes side (Spot only), first datapoint per release
spot = histdf[histdf["Period"] == "Spot (Physical)"][["Release Date", "Cost USDperMMBtu"]].copy()
spot["Release Date"] = pd.to_datetime(spot["Release Date"], errors="coerce")
rts_map = spot.dropna().drop_duplicates(subset=["Release Date"]).set_index("Release Date")["Cost USDperMMBtu"].to_dict()
rts = [float(rts_map.get(r, float("nan"))) for r in rels]

# Compose df2
df2 = pd.DataFrame({
    "Release Date": rels,
    "Regas Cost": prices,
    "Routes Cost": rts,
})

st.subheader("df2 (first datapoint per release)")
st.dataframe(df2, use_container_width=True)


