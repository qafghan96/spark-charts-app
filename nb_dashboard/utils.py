import os
import io
import base64
from typing import List, Tuple, Set

import streamlit as st

try:
    import plotly.io as pio
except Exception:
    pio = None

# ------- Chart Axis Controls -------
def add_axis_controls(chart_type="matplotlib", expanded=False, data_df=None, x_col=None, y_cols=None):
    """
    Add axis limit controls to Streamlit sidebar or main area.
    
    Args:
        chart_type (str): Type of chart ('matplotlib', 'plotly', etc.)
        expanded (bool): Whether to show controls in an expanded section
        data_df (pd.DataFrame, optional): DataFrame for calculating data-driven defaults
        x_col (str, optional): Column name for x-axis data
        y_cols (list, optional): List of column names for y-axis data
    
    Returns:
        dict: Dictionary containing axis limits and auto settings
            {
                'x_auto': bool,
                'y_auto': bool, 
                'x_min': float or None,
                'x_max': float or None,
                'y_min': float or None,
                'y_max': float or None
            }
    """
    if expanded:
        with st.expander("📊 Chart Axis Controls", expanded=False):
            return _create_axis_controls(data_df, x_col, y_cols)
    else:
        st.subheader("📊 Chart Axis Controls")
        return _create_axis_controls(data_df, x_col, y_cols)

def _create_axis_controls(data_df=None, x_col=None, y_cols=None):
    """Internal function to create axis control widgets."""
    
    # Calculate data-driven defaults - use sensible fallbacks instead of 0-100
    default_x_min, default_x_max = -10.0, 10.0  # More sensible for price/date data
    default_y_min, default_y_max = -5.0, 5.0    # Better for financial data
    
    if data_df is not None and not data_df.empty:
        try:
            # Calculate X-axis defaults
            if x_col and x_col in data_df.columns:
                x_data = data_df[x_col].dropna()
                if len(x_data) > 0:
                    if x_data.dtype.kind in 'iufc':  # numeric data
                        x_range = x_data.max() - x_data.min()
                        padding = x_range * 0.05
                        default_x_min = float(x_data.min() - padding)
                        default_x_max = float(x_data.max() + padding)
                    else:
                        # For non-numeric data (dates, etc.), use indices
                        default_x_min = 0.0
                        default_x_max = float(len(x_data) - 1)
            
            # Calculate Y-axis defaults
            if y_cols:
                all_y_values = []
                for col in y_cols:
                    if col in data_df.columns:
                        y_data = data_df[col].dropna()
                        if y_data.dtype.kind in 'iufc':  # numeric data
                            all_y_values.extend(y_data.tolist())
                
                if all_y_values:
                    y_min = min(all_y_values)
                    y_max = max(all_y_values)
                    y_range = y_max - y_min
                    padding = y_range * 0.05 if y_range > 0 else 1.0
                    default_y_min = float(y_min - padding)
                    default_y_max = float(y_max + padding)
        except Exception:
            # Fall back to default values if calculation fails
            pass
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**X-Axis**")
        x_auto = st.checkbox("Auto X-Axis", value=True, help="Automatically set X-axis limits based on data")
        
        x_min = None
        x_max = None
        if not x_auto:
            x_min = st.number_input("X-Axis Min", value=default_x_min, help="Minimum value for X-axis")
            x_max = st.number_input("X-Axis Max", value=default_x_max, help="Maximum value for X-axis")
    
    with col2:
        st.write("**Y-Axis**")
        y_auto = st.checkbox("Auto Y-Axis", value=True, help="Automatically set Y-axis limits based on data")
        
        y_min = None
        y_max = None
        if not y_auto:
            y_min = st.number_input("Y-Axis Min", value=default_y_min, help="Minimum value for Y-axis")
            y_max = st.number_input("Y-Axis Max", value=default_y_max, help="Maximum value for Y-axis")
    
    return {
        'x_auto': x_auto,
        'y_auto': y_auto,
        'x_min': x_min,
        'x_max': x_max,
        'y_min': y_min,
        'y_max': y_max
    }

def add_color_controls(series_names, default_colors=None, expanded=False):
    """
    Add color selection controls for chart series.
    
    Args:
        series_names (list): List of series names to add color controls for
        default_colors (list, optional): List of default hex colors for each series
        expanded (bool): Whether to show controls in an expanded section
    
    Returns:
        dict: Dictionary mapping series names to selected colors
    """
    # Standard color palette with common chart colors
    color_palette = {
        'Gold': '#FFC217',
        'Blue': '#4F41F4', 
        'Green': '#48C38D',
        'Red': '#FF4B4B',
        'Orange': '#FF8C00',
        'Purple': '#8B5CF6',
        'Pink': '#F472B6',
        'Teal': '#14B8A6',
        'Indigo': '#6366F1',
        'Lime': '#84CC16',
        'Rose': '#F43F5E',
        'Cyan': '#06B6D4',
        'Amber': '#F59E0B',
        'Emerald': '#10B981',
        'Violet': '#8B5CF6',
        'Sky': '#0EA5E9'
    }
    
    # If no default colors provided, use the standard sequence
    if default_colors is None:
        default_sequence = ['#FFC217', '#4F41F4', '#48C38D', '#FF4B4B', '#FF8C00']
        default_colors = [default_sequence[i % len(default_sequence)] for i in range(len(series_names))]
    
    # Find default color names (or use hex if not in palette)
    default_names = []
    for color in default_colors:
        color_name = None
        for name, hex_val in color_palette.items():
            if hex_val.lower() == color.lower():
                color_name = name
                break
        default_names.append(color_name if color_name else color)
    
    if expanded:
        with st.expander("🎨 Chart Color Controls", expanded=False):
            return _create_color_controls(series_names, color_palette, default_names, default_colors)
    else:
        st.subheader("🎨 Chart Color Controls")
        return _create_color_controls(series_names, color_palette, default_names, default_colors)

def _create_color_controls(series_names, color_palette, default_names, default_colors):
    """Internal function to create color control widgets."""
    selected_colors = {}
    
    st.write("**Select colors for each data series:**")
    
    # Create columns for color controls
    num_series = len(series_names)
    if num_series <= 3:
        cols = st.columns(num_series)
    else:
        cols = st.columns(3)
    
    for i, series_name in enumerate(series_names):
        col_idx = i % len(cols)
        with cols[col_idx]:
            # Create selectbox for color choice
            color_options = list(color_palette.keys()) + ['Custom']
            default_idx = 0
            
            # Find index of default color
            if i < len(default_names) and default_names[i] in color_options:
                default_idx = color_options.index(default_names[i])
            
            selected_color_name = st.selectbox(
                f"{series_name}",
                options=color_options,
                index=default_idx,
                key=f"color_{series_name}_{i}"
            )
            
            if selected_color_name == 'Custom':
                # Show color picker for custom color
                default_hex = default_colors[i] if i < len(default_colors) else '#FFC217'
                selected_colors[series_name] = st.color_picker(
                    f"Custom color for {series_name}",
                    value=default_hex,
                    key=f"custom_color_{series_name}_{i}"
                )
            else:
                # Use predefined color
                selected_colors[series_name] = color_palette[selected_color_name]
            
            # Show color preview
            st.markdown(
                f'<div style="width: 100%; height: 20px; background-color: {selected_colors[series_name]}; '
                f'border: 1px solid #ccc; border-radius: 4px; margin-top: 5px;"></div>',
                unsafe_allow_html=True
            )
    
    return selected_colors

def apply_axis_limits(ax, axis_controls, data_df=None, x_col=None, y_cols=None):
    """
    Apply axis limits to matplotlib axes based on user controls.
    
    Args:
        ax: matplotlib axes object
        axis_controls (dict): Result from add_axis_controls()
        data_df (pd.DataFrame, optional): DataFrame for auto-scaling
        x_col (str, optional): Column name for x-axis data
        y_cols (list, optional): List of column names for y-axis data
    """
    # Apply X-axis limits
    if not axis_controls['x_auto']:
        if axis_controls['x_min'] is not None and axis_controls['x_max'] is not None:
            ax.set_xlim(axis_controls['x_min'], axis_controls['x_max'])
    
    # Apply Y-axis limits
    if not axis_controls['y_auto']:
        if axis_controls['y_min'] is not None and axis_controls['y_max'] is not None:
            ax.set_ylim(axis_controls['y_min'], axis_controls['y_max'])
    else:
        # Auto-scaling with padding if data is provided
        if data_df is not None and y_cols is not None:
            try:
                all_y_values = []
                for col in y_cols:
                    if col in data_df.columns:
                        all_y_values.extend(data_df[col].dropna().tolist())
                
                if all_y_values:
                    y_min = min(all_y_values)
                    y_max = max(all_y_values)
                    y_range = y_max - y_min
                    padding = y_range * 0.05  # 5% padding
                    ax.set_ylim(y_min - padding, y_max + padding)
            except Exception:
                pass  # Fall back to matplotlib's default scaling

# ------- Spark API helpers (self-contained) -------
import json
import requests
from base64 import b64encode
from datetime import datetime
import pandas as pd
import time


API_BASE_URL = "https://api.sparkcommodities.com"


def get_credentials() -> Tuple[str, str]:
    """Resolve credentials in the following order:
    1) Values set in st.session_state (from the homepage inputs)
    2) Streamlit secrets (spark.client_id / spark.client_secret)
    3) Environment variables (SPARK_CLIENT_ID / SPARK_CLIENT_SECRET)
    """
    # 1) Session state (set via homepage input form)
    client_id = st.session_state.get("spark_client_id")
    client_secret = st.session_state.get("spark_client_secret")

    # 2) Streamlit secrets
    if not client_id or not client_secret:
        try:
            secrets = st.secrets.get("spark", {})
            client_id = client_id or secrets.get("client_id")
            client_secret = client_secret or secrets.get("client_secret")
        except Exception:
            pass

    # 3) Environment variables
    if not client_id:
        client_id = os.getenv("SPARK_CLIENT_ID")
    if not client_secret:
        client_secret = os.getenv("SPARK_CLIENT_SECRET")

    return client_id, client_secret


def get_access_token(client_id: str, client_secret: str, scopes: str | None = None) -> str:
    payload = f"{client_id}:{client_secret}".encode()
    headers = {
        "Authorization": b64encode(payload).decode(),
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    body = {"grantType": "clientCredentials"}
    if scopes:
        body["scopes"] = scopes
    url = f"{API_BASE_URL}/oauth/token/"
    r = requests.post(url, headers=headers, data=json.dumps(body))
    r.raise_for_status()
    content = r.json()
    return content["accessToken"]


def api_get(uri: str, access_token: str, format: str = 'json'):
    url = f"{API_BASE_URL}{uri}"
    
    if format == 'json':
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/json",
        }
    elif format == 'csv':
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Accept": "text/csv"
        }
    else:
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/json",
        }
    
    r = requests.get(url, headers=headers)
    r.raise_for_status()
    
    # Return response based on format
    if format == 'json':
        return r.json()
    elif format == 'csv':
        return r.content
    else:
        return r.json()


def list_contracts(access_token: str) -> List[Tuple[str, str]]:
    data = api_get("/v1.0/contracts/", access_token)
    # Return list of (id, fullName)
    return [(c["id"], c["fullName"]) for c in data.get("data", [])]


def list_netbacks_reference(access_token: str):
    """Return reference data for netbacks, including FoB ports and release dates.
    Returns (tickers, fob_port_names, available_via, release_dates, raw_dict)
    """
    data = api_get("/v1.0/netbacks/reference-data/", access_token)
    static = data.get("data", {}).get("staticData", {})
    fob_ports = static.get("fobPorts", [])
    tickers: list[str] = []
    names: list[str] = []
    available_via: list[list[str]] = []
    for fp in fob_ports:
        tickers.append(fp.get("uuid"))
        names.append(fp.get("name"))
        available_via.append(fp.get("availableViaPoints", []))
    release_dates = static.get("sparkReleases", [])
    return tickers, names, available_via, release_dates, data.get("data", {})


def fetch_netback(access_token: str, fob_port_uuid: str, release: str, via: str | None = None,
                  laden: float | None = None, ballast: float | None = None) -> dict:
    """Fetch a single netback snapshot for a FoB port and release date."""
    query_params = f"?fob-port={fob_port_uuid}"
    if release is not None:
        query_params += f"&release-date={release}"
    if via is not None:
        query_params += f"&via-point={via}"
    if laden is not None:
        query_params += f"&laden-congestion-days={laden}"
    if ballast is not None:
        query_params += f"&ballast-congestion-days={ballast}"
    content = api_get(f"/v1.0/netbacks/{query_params}", access_token)
    return content.get("data", {})


def netbacks_history(access_token: str, fob_port_uuid: str, fob_port_name: str, release_dates: list[str],
                     via: str | None = None, laden: float | None = None, ballast: float | None = None,
                     delay_seconds: float = 0.2) -> pd.DataFrame:
    months: list[str] = []
    nea_outrights: list[float] = []
    nea_ttfbasis: list[float] = []
    nwe_outrights: list[float] = []
    nwe_ttfbasis: list[float] = []
    delta_outrights: list[float] = []
    delta_ttfbasis: list[float] = []
    release_date: list[str] = []
    port: list[str] = []

    for r in release_dates:
        try:
            doc = fetch_netback(access_token, fob_port_uuid, release=r, via=via, laden=laden, ballast=ballast)
            m = doc.get("netbacks", [{}])[0]
            months.append(m.get("load", {}).get("month"))
            nea = m.get("nea", {})
            nwe = m.get("nwe", {})
            delta = m.get("neaMinusNwe", {})
            def _val(bucket: dict, key: str) -> float:
                return float(bucket.get(key, {}).get("usdPerMMBtu", "nan"))
            nea_outrights.append(_val(nea, "outright"))
            nea_ttfbasis.append(_val(nea, "ttfBasis"))
            nwe_outrights.append(_val(nwe, "outright"))
            nwe_ttfbasis.append(_val(nwe, "ttfBasis"))
            delta_outrights.append(_val(delta, "outright"))
            delta_ttfbasis.append(_val(delta, "ttfBasis"))
            release_date.append(doc.get("releaseDate"))
            port.append(fob_port_name)
        except Exception:
            # Skip bad dates
            continue
        if delay_seconds:
            time.sleep(delay_seconds)

    df = pd.DataFrame(
        {
            "Release Date": release_date,
            "FoB Port": port,
            "Month": months,
            "NEA Outrights": nea_outrights,
            "NEA TTF Basis": nea_ttfbasis,
            "NWE Outrights": nwe_outrights,
            "NWE TTF Basis": nwe_ttfbasis,
            "Delta Outrights": delta_outrights,
            "Delta TTF Basis": delta_ttfbasis,
        }
    )
    if not df.empty:
        df["Release Date"] = pd.to_datetime(df["Release Date"])
    return df

def fetch_price_releases(access_token: str, contract_id: str, limit: int = 60, offset: int | None = None) -> List[dict]:
    query = f"?limit={limit}"
    if offset is not None:
        query += f"&offset={offset}"
    data = api_get(f"/v1.0/contracts/{contract_id}/price-releases/{query}", access_token)
    return data.get("data", [])


def fetch_netback(access_token: str, fob_port_uuid: str, release: str, via: str | None = None,
                  laden: float | None = None, ballast: float | None = None) -> dict:
    """Fetch a single netback snapshot for a FoB port and release date."""
    query_params = f"?fob-port={fob_port_uuid}"
    if release is not None:
        query_params += f"&release-date={release}"
    if via is not None:
        query_params += f"&via-point={via}"
    if laden is not None:
        query_params += f"&laden-congestion-days={laden}"
    if ballast is not None:
        query_params += f"&ballast-congestion-days={ballast}"
    content = api_get(f"/v1.0/netbacks/{query_params}", access_token)
    return content.get("data", {})


def build_price_df(access_token: str, ticker: str, limit: int = 60) -> pd.DataFrame:
    releases = fetch_price_releases(access_token, ticker, limit=limit)

    release_dates: list[str] = []
    period_start: list[str] = []
    period_end: list[str] = []
    period_name: list[str] = []
    cal_month: list[str] = []
    tickers: list[str] = []
    usd: list[float] = []
    usd_min: list[float] = []
    usd_max: list[float] = []

    for release in releases:
        release_date = release["releaseDate"]
        for d in release.get("data", []):
            for data_point in d.get("dataPoints", []):
                start_at = data_point["deliveryPeriod"]["startAt"]
                end_at = data_point["deliveryPeriod"]["endAt"]
                period_start.append(start_at)
                period_end.append(end_at)
                period_name.append(data_point["deliveryPeriod"]["name"])
                release_dates.append(release_date)
                tickers.append(release.get("contractId"))
                cal_month.append(datetime.strptime(start_at, "%Y-%m-%d").strftime("%b-%Y"))
                derived = data_point.get("derivedPrices", {}).get("usdPerDay", {})
                usd.append(float(derived.get("spark", "nan")))
                usd_min.append(float(derived.get("sparkMin", "nan")))
                usd_max.append(float(derived.get("sparkMax", "nan")))

    df = pd.DataFrame(
        {
            "Release Date": release_dates,
            "ticker": tickers,
            "Period Name": period_name,
            "Period Start": period_start,
            "Period End": period_end,
            "Calendar Month": cal_month,
            "Spark": usd,
            "SparkMin": usd_min,
            "SparkMax": usd_max,
        }
    )
    if not df.empty:
        df["Release Date"] = pd.to_datetime(df["Release Date"], format="%Y-%m-%d")
    return df


# ------- (Optional) legacy notebook helpers used in earlier version -------
try:
    import nbformat  # type: ignore
    from nbclient import ExecutePreprocessor  # type: ignore
except Exception:
    nbformat = None
    ExecutePreprocessor = None


def execute_notebook(nb_path: str, timeout_sec: int = 1200, allow_errors: bool = False):
    if nbformat is None or ExecutePreprocessor is None:
        raise RuntimeError("Please install nbclient and nbformat to execute notebooks.")
    nb = nbformat.read(nb_path, as_version=4)
    ep = ExecutePreprocessor(timeout=timeout_sec, kernel_name="python3", allow_errors=allow_errors)
    resources = {"metadata": {"path": os.path.dirname(nb_path) or "."}}
    ep.preprocess(nb, resources=resources)
    return nb


def extract_last_charts(nb) -> List[Tuple[str, object]]:
    images: List[Tuple[str, object]] = []
    last_idx = -1
    for idx, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        for out in cell.get("outputs", []):
            if out.get("output_type") in ("display_data", "execute_result"):
                data = out.get("data", {})
                if any(k in data for k in ("image/png", "image/svg+xml", "application/vnd.plotly.v1+json", "text/html")):
                    last_idx = idx
    for idx, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        if idx != last_idx:
            continue
        for out in cell.get("outputs", []):
            if out.get("output_type") not in ("display_data", "execute_result"):
                continue
            data = out.get("data", {})
            if "image/png" in data:
                images.append(("image/png", data["image/png"]))
            elif "image/svg+xml" in data:
                images.append(("image/svg+xml", data["image/svg+xml"]))
            elif "application/vnd.plotly.v1+json" in data:
                images.append(("plotly", data["application/vnd.plotly.v1+json"]))
            elif "text/html" in data:
                images.append(("html", data["text/html"]))
    return images


def render_charts(charts: List[Tuple[str, object]]):
    for kind, payload in charts:
        if kind == "image/png":
            try:
                img_bytes = base64.b64decode(payload)
            except Exception:
                img_bytes = payload if isinstance(payload, (bytes, bytearray)) else None
            if img_bytes:
                st.image(io.BytesIO(img_bytes))
        elif kind == "image/svg+xml":
            st.image(payload)
        elif kind == "plotly" and pio is not None:
            try:
                fig = pio.from_json(payload)
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                pass
        elif kind == "html":
            st.components.v1.html(payload, height=600, scrolling=True)


def scrub_notebook_outputs(nb, keep_kinds: Set[str]) -> str:
    import nbformat  # local import to ensure availability
    nb_copy = nbformat.from_dict(nb)
    for cell in nb_copy.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        cell["execution_count"] = None
        new_outputs = []
        for out in cell.get("outputs", []):
            if out.get("output_type") not in ("display_data", "execute_result"):
                continue
            data = out.get("data", {})
            kept = {k: v for k, v in data.items() if k in keep_kinds}
            if kept:
                out2 = dict(out)
                out2["data"] = kept
                out2.pop("text", None)
                new_outputs.append(out2)
        cell["outputs"] = new_outputs
    return nbformat.writes(nb_copy)


