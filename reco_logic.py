import pandas as pd
import numpy as np
from rapidfuzz import process, fuzz


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
def normalize_doc(series):
    return (
        series.fillna("")
        .astype(str)
        .str.upper()
        .str.replace(r"[^A-Z0-9]", "", regex=True)
    )


# -------------------------------------------------
def validate_columns(df, required_cols, df_name):
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"{df_name} missing columns: {missing}")


# -------------------------------------------------
def compute_diffs(df):
    df["IGST Diff"] = df["IGST Amount_PUR"] - df["IGST Amount_2B"]
    df["CGST Diff"] = df["CGST Amount_PUR"] - df["CGST Amount_2B"]
    df["SGST Diff"] = df["SGST Amount_PUR"] - df["SGST Amount_2B"]
    return df


# -------------------------------------------------
# SAFE COPY FUNCTION (no logic change)
def copy_data(merged, left_idx, right_idx):

    cols = [
        "Reference Document No.",
        "FI Document Number",
        "Vendor/Customer GSTIN",
        "Vendor/Customer Name",
        "IGST Amount_PUR",
        "CGST Amount_PUR",
        "SGST Amount_PUR",
        "Taxable Amount",
        "Invoice Value_PUR"
    ]

    for col in cols:
        if col in merged.columns:
            val = merged.at[right_idx, col]
            if pd.notna(val):
                merged.at[left_idx, col] = val


# -------------------------------------------------
def process_reco(gst_df, pur_df, doc_threshold=60, tax_tolerance=10):

    gst = gst_df.copy()
    pur = pur_df.copy()

    pur["Vendor/Customer GSTIN"] = pur["GSTIN Of Vendor/Customer"]

    gst["doc_norm"] = normalize_doc(gst["Document Number"])
    pur["doc_norm"] = normalize_doc(pur["Reference Document No."])

    pur.rename(columns={"GSTIN Of Vendor/Customer": "Supplier GSTIN"}, inplace=True)

    # ---------------- AGG ----------------
    gst_agg = gst.groupby(["Supplier GSTIN", "doc_norm"], as_index=False).agg({
        "Document Number": "first",
        "Return Period": "first",
        "Supplier Name": "first",
        "Document Date": "first",
        "IGST Amount": "sum",
        "CGST Amount": "sum",
        "SGST Amount": "sum",
        "Taxable Value": "sum",
        "Invoice Value": "sum",
    })

    pur_agg = pur.groupby(["Supplier GSTIN", "doc_norm"], as_index=False).agg({
        "Reference Document No.": "first",
        "FI Document Number": "first",
        "Vendor/Customer GSTIN": "first",
        "Vendor/Customer Name": "first",
        "Document Date": "first",
        "Taxable Amount": "sum",
        "IGST Amount": "sum",
        "CGST Amount": "sum",
        "SGST Amount": "sum",
        "Invoice Value": "sum",
    })

    # ---------------- MERGE ----------------
    merged = gst_agg.merge(
        pur_agg,
        on=["Supplier GSTIN", "doc_norm"],
        how="outer",
        suffixes=["_2B", "_PUR"],
        indicator=True,
    )

    # ✅ FIXED FILLNA (SAFE)
    numeric_cols = merged.select_dtypes(include=[np.number]).columns
    merged[numeric_cols] = merged[numeric_cols].fillna(0)

    object_cols = merged.select_dtypes(include=["object"]).columns
    merged[object_cols] = merged[object_cols].fillna("")

    merged = compute_diffs(merged)

    merged["Match_Status"] = None
    merged["Fuzzy Score"] = 0.0

    merged.loc[merged["_merge"] == "both", "Match_Status"] = MATCH_EXACT
    merged.loc[merged["_merge"] == "left_only", "Match_Status"] = MATCH_OPEN_2B
    merged.loc[merged["_merge"] == "right_only", "Match_Status"] = MATCH_OPEN_BOOKS

    # ---------------- FUZZY ----------------
    open_2b = merged[merged["Match_Status"] == MATCH_OPEN_2B]
    open_books = merged[merged["Match_Status"] == MATCH_OPEN_BOOKS]

    for left_idx in open_2b.index:

        left_doc = str(merged.at[left_idx, "Document Number"])

        match = process.extractOne(
            left_doc,
            dict(zip(open_books.index, open_books["Reference Document No."])),
            scorer=fuzz.partial_token_set_ratio,
            score_cutoff=doc_threshold
        )

        if match:
            _, score, right_idx = match

            copy_data(merged, left_idx, right_idx)

            merged.at[left_idx, "Match_Status"] = MATCH_FUZZY
            merged.at[left_idx, "Fuzzy Score"] = score
            merged.at[right_idx, "Match_Status"] = MATCH_FUZZY_CONSUMED

    # ---------------- GSTIN MISMATCH ----------------
    open_2b = merged[merged["Match_Status"] == MATCH_OPEN_2B]
    open_books = merged[merged["Match_Status"] == MATCH_OPEN_BOOKS]

    for left_idx in open_2b.index:
        doc = merged.at[left_idx, "doc_norm"]

        candidates = open_books[open_books["doc_norm"] == doc]

        for right_idx in candidates.index:
            copy_data(merged, left_idx, right_idx)

            merged.at[left_idx, "Match_Status"] = MATCH_GSTIN_MISMATCH
            merged.at[right_idx, "Match_Status"] = MATCH_GSTIN_MISMATCH
            break

    # ---------------- PAN MATCH ----------------
    merged["PAN_2B"] = merged["Supplier GSTIN"].str[2:12]
    merged["PAN_PUR"] = merged["Vendor/Customer GSTIN"].str[2:12]

    for left_idx in merged.index:
        if merged.at[left_idx, "Match_Status"] != MATCH_OPEN_2B:
            continue

        pan = merged.at[left_idx, "PAN_2B"]
        doc = merged.at[left_idx, "doc_norm"]

        candidates = merged[
            (merged["PAN_PUR"] == pan) &
            (merged["doc_norm"] == doc) &
            (merged["Match_Status"] == MATCH_OPEN_BOOKS)
        ]

        for right_idx in candidates.index:
            copy_data(merged, left_idx, right_idx)

            merged.at[left_idx, "Match_Status"] = MATCH_PAN
            merged.at[right_idx, "Match_Status"] = MATCH_PAN_CONSUMED
            break

    # ---------------- CLEAN ----------------
    merged = merged[~merged["Match_Status"].isin([
        MATCH_FUZZY_CONSUMED,
        MATCH_PAN_CONSUMED
    ])]

    merged = compute_diffs(merged)

    return merged
