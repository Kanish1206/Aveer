import pandas as pd
import numpy as np
from rapidfuzz import process, fuzz

# -------------------------------------------------
# 1️⃣ NORMALIZE DOCUMENT NUMBER
# -------------------------------------------------
def normalize_doc(series):
    return (
        series.fillna("")
        .astype(str)
        .str.upper()
        .str.replace(r"[^A-Z0-9]", "", regex=True)
    )


# -------------------------------------------------
# 2️⃣ CLEAN NUMERIC COLUMNS
# -------------------------------------------------
def clean_numeric(df, cols):

    for col in cols:
        if col not in df.columns:
            df[col] = 0

        df[col] = (
            df[col]
            .astype(str)
            .str.replace(",", "")
            .str.replace(" ", "")
        )

        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df


# -------------------------------------------------
# 3️⃣ COLUMN VALIDATION
# -------------------------------------------------
def validate_columns(df, required_cols, name):

    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        raise ValueError(f"{name} missing columns: {missing}")


# -------------------------------------------------
# 4️⃣ DUPLICATE DETECTION
# -------------------------------------------------
def detect_duplicates(df, keys):

    dup_mask = df.duplicated(subset=keys, keep=False)

    df["Duplicate Flag"] = np.where(
        dup_mask,
        "Duplicate",
        "Unique"
    )

    return df


# -------------------------------------------------
# 5️⃣ INTELLIGENT INVOICE CLUSTERING
# -------------------------------------------------
def create_invoice_clusters(df):

    df["Invoice Cluster"] = (
        df["Supplier GSTIN"].astype(str)
        + "_"
        + df["Invoice Value"].round(0).astype(str)
    )

    return df


# -------------------------------------------------
# 6️⃣ MAIN RECON ENGINE
# -------------------------------------------------
def process_reco(gst_df, pur_df,
                 doc_threshold=80,
                 tax_tolerance=10,
                 cluster_tolerance=50):

    gst = gst_df.copy()
    pur = pur_df.copy()

    # ---------------- REQUIRED COLUMNS ----------------
    gst_required = [
        "Supplier GSTIN",
        "Document Number",
        "Document Date",
        "Taxable Value",
        "Invoice Value",
        "IGST Amount",
        "CGST Amount",
        "SGST Amount",
        "Supplier Name"
    ]

    pur_required = [
        "GSTIN Of Vendor/Customer",
        "Reference Document No.",
        "Document Date",
        "Taxable Amount",
        "Invoice Value",
        "IGST Amount",
        "CGST Amount",
        "SGST Amount",
        "Vendor/Customer Name"
    ]

    validate_columns(gst, gst_required, "2B")
    validate_columns(pur, pur_required, "Books")

    # ---------------- RENAME ----------------
    pur.rename(columns={
        "GSTIN Of Vendor/Customer": "Supplier GSTIN"
    }, inplace=True)

    # ---------------- NORMALIZE DOC ----------------
    gst["doc_norm"] = normalize_doc(gst["Document Number"])
    pur["doc_norm"] = normalize_doc(pur["Reference Document No."])

    # ---------------- NUMERIC CLEANING ----------------
    numeric_cols_gst = [
        "Invoice Value", "Taxable Value",
        "IGST Amount", "CGST Amount", "SGST Amount"
    ]

    numeric_cols_pur = [
        "Invoice Value", "Taxable Amount",
        "IGST Amount", "CGST Amount", "SGST Amount"
    ]

    gst = clean_numeric(gst, numeric_cols_gst)
    pur = clean_numeric(pur, numeric_cols_pur)

    # ---------------- DUPLICATE DETECTION ----------------
    gst = detect_duplicates(
        gst,
        ["Supplier GSTIN", "doc_norm", "Invoice Value"]
    )

    pur = detect_duplicates(
        pur,
        ["Supplier GSTIN", "doc_norm", "Invoice Value"]
    )

    # ---------------- INVOICE CLUSTERING ----------------
    gst = create_invoice_clusters(gst)
    pur = create_invoice_clusters(pur)

    # ---------------- AGGREGATION ----------------
    gst_agg = gst.groupby(
        ["Supplier GSTIN", "doc_norm"],
        as_index=False
    ).agg({
        "Document Number": "first",
        "Document Date": "first",
        "Supplier Name": "first",
        "Taxable Value": "sum",
        "Invoice Value": "sum",
        "IGST Amount": "sum",
        "CGST Amount": "sum",
        "SGST Amount": "sum",
        "Duplicate Flag": "first"
    })

    pur_agg = pur.groupby(
        ["Supplier GSTIN", "doc_norm"],
        as_index=False
    ).agg({
        "Reference Document No.": "first",
        "Document Date": "first",
        "Vendor/Customer Name": "first",
        "Taxable Amount": "sum",
        "Invoice Value": "sum",
        "IGST Amount": "sum",
        "CGST Amount": "sum",
        "SGST Amount": "sum",
        "Duplicate Flag": "first"
    })

    # ---------------- MERGE ----------------
    merged = gst_agg.merge(
        pur_agg,
        on=["Supplier GSTIN", "doc_norm"],
        how="outer",
        suffixes=("_2B", "_Books"),
        indicator=True
    )

    # ---------------- DIFF CALCULATION ----------------
    merged["Invoice Diff"] = (
        merged["Invoice Value_Books"]
        - merged["Invoice Value_2B"]
    )

    merged["Taxable Diff"] = (
        merged["Taxable Amount"]
        - merged["Taxable Value"]
    )

    merged["IGST Diff"] = (
        merged["IGST Amount_Books"]
        - merged["IGST Amount_2B"]
    )

    merged["CGST Diff"] = (
        merged["CGST Amount_Books"]
        - merged["CGST Amount_2B"]
    )

    merged["SGST Diff"] = (
        merged["SGST Amount_Books"]
        - merged["SGST Amount_2B"]
    )

    # ---------------- MATCH STATUS ----------------
    merged["Match_Status"] = None
    merged["Fuzzy Score"] = 0

    both = merged["_merge"] == "both"

    tolerance_mask = (
        merged["Invoice Diff"].abs() <= tax_tolerance
    ) & (
        merged["Taxable Diff"].abs() <= tax_tolerance
    )

    merged.loc[both & tolerance_mask,
               "Match_Status"] = "Exact Match"

    merged.loc[both & ~tolerance_mask,
               "Match_Status"] = "Value Mismatch"

    merged.loc[merged["_merge"] == "left_only",
               "Match_Status"] = "Open in 2B"

    merged.loc[merged["_merge"] == "right_only",
               "Match_Status"] = "Open in Books"

    # -------------------------------------------------
    # 7️⃣ LIGHTNING FAST FUZZY MATCHING
    # -------------------------------------------------
    open_2b = merged[merged["Match_Status"] == "Open in 2B"]
    open_books = merged[merged["Match_Status"] == "Open in Books"]

    book_docs = open_books["doc_norm"].tolist()
    book_index = open_books.index.tolist()

    for idx, row in open_2b.iterrows():

        match = process.extractOne(
            row["doc_norm"],
            book_docs,
            scorer=fuzz.ratio,
            score_cutoff=doc_threshold
        )

        if match:

            matched_doc, score, pos = match
            right_idx = book_index[pos]

            merged.at[idx, "Match_Status"] = "Fuzzy Match"
            merged.at[idx, "Fuzzy Score"] = score

            merged.at[right_idx, "Match_Status"] = "Fuzzy Consumed"

    merged = merged[merged["Match_Status"] != "Fuzzy Consumed"]

    # -------------------------------------------------
    # 8️⃣ GSTIN / VENDOR MISMATCH DETECTION
    # -------------------------------------------------
    open_2b = merged[merged["Match_Status"] == "Open in 2B"]
    open_books = merged[merged["Match_Status"] == "Open in Books"]

    for left in open_2b.index:

        doc = merged.at[left, "doc_norm"]
        val = merged.at[left, "Invoice Value_2B"]

        candidates = open_books[
            open_books["doc_norm"] == doc
        ]

        for right in candidates.index:

            val_books = merged.at[right,
                                  "Invoice Value_Books"]

            if abs(val - val_books) <= cluster_tolerance:

                merged.at[left,
                          "Match_Status"] = "GSTIN Mismatch"

                merged.at[right,
                          "Match_Status"] = "GSTIN Mismatch"

    merged.drop(columns=["_merge"], inplace=True)

    return merged
