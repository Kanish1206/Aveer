import streamlit as st
import pandas as pd
import numpy as np
from rapidfuzz import process, fuzz
from io import BytesIO

# -------------------------------------------------
# CONSTANTS
# -------------------------------------------------
MATCH_EXACT = "Exact Match"
MATCH_VALUE_MISMATCH = "Value Mismatch"
MATCH_OPEN_2B = "Open in 2B"
MATCH_OPEN_BOOKS = "Open in Books"
MATCH_FUZZY = "Fuzzy Match"
MATCH_FUZZY_CONSUMED = "Fuzzy Consumed"
MATCH_GSTIN_MISMATCH = "GSTIN Mismatch"
MATCH_PAN = "PAN Match (GSTIN Variation)"
MATCH_PAN_CONSUMED = "PAN Consumed"

# -------------------------------------------------
# HELPERS
# -------------------------------------------------
def normalize_doc(series):
    return (
        series.fillna("")
        .astype(str)
        .str.upper()
        .str.replace(r"[^A-Z0-9]", "", regex=True)
    )

def validate_columns(df, required_cols, df_name):
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        st.error(f"{df_name} missing columns: {missing}")
        st.stop()

def compute_diffs(df):
    df["IGST Diff"] = df["IGST Amount_PUR"] - df["IGST Amount_2B"]
    df["CGST Diff"] = df["CGST Amount_PUR"] - df["CGST Amount_2B"]
    df["SGST Diff"] = df["SGST Amount_PUR"] - df["SGST Amount_2B"]
    return df

# -------------------------------------------------
# MAIN FUNCTION (FIXED)
# -------------------------------------------------
def process_reco(gst_df, pur_df):

    gst = gst_df.copy()
    pur = pur_df.copy()

    doc_type_map = {
        "INVOICE": "R",
        "CREDIT NOTE": "C",
        "DEBIT NOTE": "D",
    }

    pur["Document Type"] = pur["Invoice Type"].map(doc_type_map).fillna("UNKNOWN")

    validate_columns(gst, [
        "Supplier GSTIN","Document Number","Document Date","Return Period",
        "Taxable Value","Supplier Name","IGST Amount","CGST Amount",
        "SGST Amount","Invoice Value","Document Type"
    ], "2B File")

    validate_columns(pur, [
        "GSTIN Of Vendor/Customer","Reference Document No.","Taxable Amount",
        "Document Date","Vendor/Customer Name","IGST Amount",
        "CGST Amount","SGST Amount","Invoice Value","Invoice Type"
    ], "Purchase File")

    pur.rename(columns={"GSTIN Of Vendor/Customer": "Supplier GSTIN"}, inplace=True)

    gst["doc_norm"] = normalize_doc(gst["Document Number"])
    pur["doc_norm"] = normalize_doc(pur["Reference Document No."])

    gst_agg = gst.groupby(
        ["Supplier GSTIN","doc_norm","Document Type"], as_index=False
    ).sum(numeric_only=True)

    pur_agg = pur.groupby(
        ["Supplier GSTIN","doc_norm","Document Type"], as_index=False
    ).sum(numeric_only=True)

    merged = gst_agg.merge(
        pur_agg,
        on=["Supplier GSTIN","doc_norm","Document Type"],
        how="outer",
        suffixes=("_2B","_PUR"),
        indicator=True
    )

    for col in merged.columns:
        merged[col] = pd.to_numeric(merged[col], errors="ignore")

    merged = compute_diffs(merged)

    merged["Match_Status"] = np.where(
        merged["_merge"] == "both",
        MATCH_EXACT,
        np.where(merged["_merge"] == "left_only", MATCH_OPEN_2B, MATCH_OPEN_BOOKS)
    )

    merged.drop(columns=["_merge"], inplace=True)

    return merged

# -------------------------------------------------
# UI
# -------------------------------------------------
st.set_page_config(page_title="GST Reconciliation", layout="wide")

st.title("📊 GST 2B vs Purchase Reconciliation")

gst_file = st.file_uploader("Upload 2B File", type=["xlsx"])
pur_file = st.file_uploader("Upload Purchase File", type=["xlsx"])

if gst_file and pur_file:

    gst_df = pd.read_excel(gst_file)
    pur_df = pd.read_excel(pur_file)

    st.success("Files uploaded successfully ✅")

    if st.button("🚀 Run Reconciliation"):

        with st.spinner("Processing..."):
            result = process_reco(gst_df, pur_df)

        st.success("Done ✅")

        st.subheader("Result Preview")
        st.dataframe(result, use_container_width=True)

        # Summary
        st.subheader("Summary")
        summary = result["Match_Status"].value_counts()
        st.bar_chart(summary)

        # Download
        output = BytesIO()
        result.to_excel(output, index=False)

        st.download_button(
            "⬇ Download Excel",
            data=output.getvalue(),
            file_name="reco_output.xlsx"
        )import streamlit as st
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
