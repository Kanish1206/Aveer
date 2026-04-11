import streamlit as st
import pandas as pd
from io import BytesIO

# import your function
from reco import process_reco   # <-- rename your file to reco.py

st.title("GST Reconciliation Tool")

st.write("Upload 2B and Purchase files to reconcile")

gst_file = st.file_uploader("Upload 2B File", type=["xlsx", "csv"])
pur_file = st.file_uploader("Upload Purchase File", type=["xlsx", "csv"])

if gst_file and pur_file:

    gst_df = pd.read_excel(gst_file)
    pur_df = pd.read_excel(pur_file)

    st.success("Files uploaded successfully")

    if st.button("Run Reconciliation"):

        with st.spinner("Processing..."):
            result = process_reco(gst_df, pur_df)

        st.success("Reconciliation Completed")

        st.dataframe(result)

        # Download button
        def convert_df(df):
            output = BytesIO()
            df.to_excel(output, index=False)
            return output.getvalue()

        st.download_button(
            label="Download Result",
            data=convert_df(result),
            file_name="reconciliation_output.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
