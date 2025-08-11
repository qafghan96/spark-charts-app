import os
import io
import base64
import glob
from typing import List, Tuple, Set

import streamlit as st

# Optional imports for notebook execution
try:
    import nbformat
    from nbclient import ExecutePreprocessor
except Exception as import_err:  # pragma: no cover
    nbformat = None
    ExecutePreprocessor = None

# Optional plotly support (only used if present)
try:  # pragma: no cover
    import plotly.io as pio
except Exception:
    pio = None


st.set_page_config(page_title="Notebook Chart Runner", layout="wide")

st.title("Notebook Chart Runner")

st.markdown(
    "Select one or more notebooks, execute them, and display the charts produced at the end of each script."
)

# -------------------------------
# Helpers
# -------------------------------

def list_notebooks(base_dirs: List[str]) -> List[str]:
    notebooks: List[str] = []
    for d in base_dirs:
        if not os.path.isdir(d):
            continue
        # Recursively find .ipynb files
        for path in glob.glob(os.path.join(d, "**", "*.ipynb"), recursive=True):
            # Skip common caches/backups
            name = os.path.basename(path).lower()
            if name.startswith("._") or name.endswith("-checkpoint.ipynb"):
                continue
            notebooks.append(path)
    notebooks.sort(key=lambda p: p.lower())
    return notebooks


def execute_notebook(nb_path: str, timeout_sec: int = 1200, allow_errors: bool = False):
    if nbformat is None or ExecutePreprocessor is None:
        raise RuntimeError(
            "Missing dependencies to execute notebooks. Please install: nbformat nbclient"
        )

    nb = nbformat.read(nb_path, as_version=4)
    ep = ExecutePreprocessor(timeout=timeout_sec, kernel_name="python3", allow_errors=allow_errors)
    resources = {"metadata": {"path": os.path.dirname(nb_path) or "."}}
    ep.preprocess(nb, resources=resources)
    return nb


def extract_chart_outputs(nb, only_last: bool = True) -> List[Tuple[str, object]]:
    """
    Return a list of (kind, payload) where kind in {"image/png", "image/svg+xml", "plotly", "html"}
    """
    images: List[Tuple[str, object]] = []

    # Find last cell index that produced any visual output
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
        if only_last and idx != last_idx:
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


def scrub_notebook_outputs(nb, keep_kinds: Set[str]) -> str:
    """
    Remove stdout/stderr/text outputs and keep only chart-like outputs by MIME kind.
    Return the scrubbed notebook JSON string.
    """
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
            # Filter data keys by keep_kinds
            kept = {k: v for k, v in data.items() if k in keep_kinds}
            if kept:
                out2 = dict(out)
                out2["data"] = kept
                # Remove text/plain fallbacks if present
                out2.pop("text", None)
                new_outputs.append(out2)
        cell["outputs"] = new_outputs
    return nbformat.writes(nb_copy)


@st.cache_data(show_spinner=False)
def exec_and_collect(nb_path: str, only_last: bool, timeout_sec: int, allow_errors: bool):
    nb = execute_notebook(nb_path, timeout_sec=timeout_sec, allow_errors=allow_errors)
    charts = extract_chart_outputs(nb, only_last=only_last)
    nb_json = nbformat.writes(nb)
    return charts, nb_json, nb


# -------------------------------
# Sidebar configuration
# -------------------------------

with st.sidebar:
    st.header("Configuration")

    default_dirs = [
        "Press",
        "Testing",
        "data-team",
        "Gallery",
        "Public Files Testing",
        "Internal Analytics",
        "Clients",
    ]

    base_dirs = st.multiselect(
        "Search in directories",
        options=sorted([d for d in default_dirs if os.path.isdir(d)]),
        default=[d for d in default_dirs if os.path.isdir(d)],
    )

    search_term = st.text_input("Filter notebooks by name (optional)", value="")

    only_last = st.checkbox("Show only the final chart per notebook", value=True)
    allow_errors = st.checkbox("Allow notebook errors (continue execution)", value=False)
    timeout_sec = st.number_input("Execution timeout (seconds)", min_value=60, max_value=7200, value=1200, step=60)

    scrub_outputs = st.checkbox(
        "Scrub text outputs in downloaded notebook (keep only charts)", value=True,
        help="Removes stdout/stderr and text outputs to avoid exposing printed credentials or tokens."
    )

    st.markdown("---")
    st.caption(
        "These notebooks now include hard-coded API credentials. The app will execute them as-is.\n"
        "If notebooks print sensitive values to stdout, enable output scrubbing to keep only charts in the exported file."
    )

# List notebooks
all_notebooks = list_notebooks(base_dirs)
if search_term:
    all_notebooks = [p for p in all_notebooks if search_term.lower() in os.path.basename(p).lower()]

selected = st.multiselect(
    "Select notebooks to run",
    options=all_notebooks,
    default=[p for p in all_notebooks if p.endswith("Press/Weekly Arb Charts - Global.ipynb")][:1],
)

run = st.button("Run selected notebooks", type="primary")

# -------------------------------
# Execution
# -------------------------------

if run:
    if not selected:
        st.info("Please select at least one notebook.")
    else:
        progress = st.progress(0, text="Starting...")
        for idx, nb_path in enumerate(selected, start=1):
            st.markdown(f"### {os.path.relpath(nb_path)}")
            try:
                with st.spinner(f"Executing {os.path.basename(nb_path)}..."):
                    charts, nb_json, nb_obj = exec_and_collect(
                        nb_path=nb_path,
                        only_last=only_last,
                        timeout_sec=timeout_sec,
                        allow_errors=allow_errors,
                    )
            except Exception as e:  # noqa: BLE001
                st.error(f"Execution failed: {e}")
                progress.progress(idx / len(selected), text=f"Failed: {os.path.basename(nb_path)}")
                continue

            if not charts:
                st.warning("No chart outputs found.")
            else:
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

            # Prepare executed notebook for download (optionally scrub)
            with st.expander("Download executed notebook and cell outputs"):
                if scrub_outputs and nbformat is not None:
                    keep = {"image/png", "image/svg+xml", "application/vnd.plotly.v1+json", "text/html"}
                    nb_scrubbed = scrub_notebook_outputs(nb_obj, keep)
                    download_json = nb_scrubbed
                    fname = os.path.basename(nb_path).replace(".ipynb", "_executed_scrubbed.ipynb")
                else:
                    download_json = nb_json
                    fname = os.path.basename(nb_path).replace(".ipynb", "_executed.ipynb")

                st.download_button(
                    label="Download .ipynb",
                    data=download_json,
                    file_name=fname,
                    mime="application/json",
                )

            progress.progress(idx / len(selected), text=f"Completed: {os.path.basename(nb_path)}")

        progress.empty()
        st.success("Finished running selected notebooks.")

# Guidance when imports are missing
if nbformat is None or ExecutePreprocessor is None:
    st.warning(
        "Notebook execution dependencies not found. Install with:\n\n"
        "pip install nbclient nbformat\n\n"
        "Then rerun the app."
    )