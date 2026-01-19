import streamlit as st
import pandas as pd
import io
from reco_logic import process_reco


# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="GST 2B vs Books Reconciliation",
    layout="wide",
)


# --------------------------------------------------
# Sidebar
# --------------------------------------------------
with st.sidebar:
    st.title("⚙️ Settings")
    threshold = st.slider("Fuzzy Match Threshold (%)", 50, 100, 90)
    st.divider()
    st.info("Ensure correct column names before upload.")


# --------------------------------------------------
# Header
# --------------------------------------------------
st.title("⚖️ GST Reconciliation Engine")
st.caption("2B vs Purchase Register — Exact & Fuzzy Matching")


# --------------------------------------------------
# Upload Section
# --------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    gst_file = st.file_uploader("Upload GSTR-2B Excel", type=["xlsx"])

with col2:
    pur_file = st.file_uploader("Upload Purchase Register Excel", type=["xlsx"])


# --------------------------------------------------
# Execution
# --------------------------------------------------
if gst_file and pur_file:

    gst_df = pd.read_excel(gst_file)
    pur_df = pd.read_excel(pur_file)

    if st.button("🚀 Run Reconciliation"):

        with st.spinner("Processing reconciliation..."):
            try:
                result = process_reco(gst_df, pur_df, threshold)

                st.success("Reconciliation completed")

                # Summary
                c1, c2, c3 = st.columns(3)
                c1.metric("Total Records", len(result))
                c2.metric(
                    "Matched",
                    (result["Match_Status"].isin(["Exact Match", "Fuzzy Match"])).sum(),
                )
                c3.metric(
                    "Action Required",
                    (result["Match_Status"].isin(["Open in 2B", "Open in Books"])).sum(),
                )

                st.divider()
                st.dataframe(result, use_container_width=True)

                # Download
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                    result.to_excel(writer, index=False, sheet_name="GST_Reco")

                st.download_button(
                    "Download Excel Report",
                    output.getvalue(),
                    file_name="GST_Reconciliation.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

            except Exception as e:
                st.error(str(e))

else:
    st.warning("Upload both files to begin reconciliation.")
