import streamlit as st

st.set_page_config(page_title="Notebook Dashboard", layout="wide")

st.title("Notebook Dashboard")

with st.expander("API Credentials (Spark)"):
    st.caption("These are stored in your session only. Prefer Streamlit secrets or env vars in production.")
    # Use separate widget keys to avoid conflicts with saved session keys
    cid = st.text_input(
        "Client ID",
        key="spark_client_id_input",
        value=st.session_state.get("spark_client_id", ""),
    )
    csec = st.text_input(
        "Client Secret",
        key="spark_client_secret_input",
        value=st.session_state.get("spark_client_secret", ""),
        type="password",
    )
    col1, col2 = st.columns([1,1])
    with col1:
        if st.button("Save to session"):
            st.session_state["spark_client_id"] = st.session_state.get("spark_client_id_input", "")
            st.session_state["spark_client_secret"] = st.session_state.get("spark_client_secret_input", "")
            st.success("Saved credentials in session.")
    with col2:
        if st.button("Clear from session"):
            st.session_state.pop("spark_client_id", None)
            st.session_state.pop("spark_client_secret", None)
            st.success("Cleared credentials from session.")

st.markdown(
    "This multi-page app replicates selected notebooks as dedicated pages, calling the Spark API directly to build charts."
)

st.markdown("---")
st.subheader("Included pages")
st.markdown(
    "- Run: Press - Weekly Arb Charts - Global\n"
    "- Run: LNG Espresso\n"
)

st.caption(
    "Notes: Notebooks execute in their source directories. If they require credentials, ensure your environment has them available."
)


