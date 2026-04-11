import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

# -----------------------------
# Simple Reco Function
# -----------------------------
def process_reco(gst_df, pur_df):

    gst_df["key"] = gst_df["Supplier GSTIN"].astype(str) + gst_df["Document Number"].astype(str)
    pur_df["key"] = pur_df["GSTIN Of Vendor/Customer"].astype(str) + pur_df["Reference Document No."].astype(str)

    merged = gst_df.merge(
        pur_df,
        left_on="key",
        right_on="key",
        how="outer",
        indicator=True
    )

    merged["Match_Status"] = merged["_merge"].map({
        "both": "Exact Match",
        "left_only": "Open in 2B",
        "right_only": "Open in Books"
    })

    return merged


# -----------------------------
# UI
# -----------------------------
st.title("GST Reconciliation Tool")

gst_file = st.file_uploader("Upload 2B File", type=["xlsx"])
pur_file = st.file_uploader("Upload Purchase File", type=["xlsx"])

if gst_file and pur_file:

    gst_df = pd.read_excel(gst_file)
    pur_df = pd.read_excel(pur_file)

    if st.button("Run Reconciliation"):

        result = process_reco(gst_df, pur_df)

        st.dataframe(result)

        # download
        output = BytesIO()
        result.to_excel(output, index=False)

        st.download_button(
            "Download Result",
            data=output.getvalue(),
            file_name="output.xlsx"
        )
