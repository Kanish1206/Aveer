import pandas as pd
import numpy as np
from rapidfuzz import process, fuzz


# -------------------------------------------------
# 1️⃣ CLEAN COLUMN NAMES
# -------------------------------------------------
def clean_columns(df):

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
        .str.replace("\n","",regex=False)
        .str.replace("\t","",regex=False)
    )

    return df


# -------------------------------------------------
# 2️⃣ NORMALIZE DOCUMENT
# -------------------------------------------------
def normalize_doc(series):

    return (
        series.fillna("")
        .astype(str)
        .str.upper()
        .str.replace(r"[^A-Z0-9]", "", regex=True)
    )


# -------------------------------------------------
# 3️⃣ CLEAN NUMERIC
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
# 4️⃣ DUPLICATE DETECTION
# -------------------------------------------------
def detect_duplicates(df, keys):

    dup = df.duplicated(subset=keys, keep=False)

    df["Duplicate Flag"] = np.where(dup,"Duplicate","Unique")

    return df


# -------------------------------------------------
# 5️⃣ MAIN RECON FUNCTION
# -------------------------------------------------
def process_reco(gst_df, pur_df,
                 doc_threshold=75,
                 tax_tolerance=10,
                 gstin_tolerance=20):

    # ---------------- CLEAN COLUMN NAMES ----------------
    gst = clean_columns(gst_df.copy())
    pur = clean_columns(pur_df.copy())

    # ---------------- REQUIRED COLUMNS ----------------
    gst_required = [
        "Supplier GSTIN",
        "Document Number",
        "Taxable Value",
        "IGST Amount",
        "CGST Amount",
        "SGST Amount",
        "Invoice Value"
    ]

    pur_required = [
        "GSTIN Of Vendor/Customer",
        "Reference Document No.",
        "Taxable Amount",
        "IGST Amount",
        "CGST Amount",
        "SGST Amount",
        "Invoice Value"
    ]

    for col in gst_required:
        if col not in gst.columns:
            raise ValueError(f"2B missing column: {col}")

    for col in pur_required:
        if col not in pur.columns:
            raise ValueError(f"Books missing column: {col}")

    # ---------------- NORMALIZE DOC ----------------
    gst["doc_norm"] = normalize_doc(gst["Document Number"])
    pur["doc_norm"] = normalize_doc(pur["Reference Document No."])

    # ---------------- NUMERIC CLEAN ----------------
    gst = clean_numeric(gst, [
        "Taxable Value","IGST Amount","CGST Amount",
        "SGST Amount","Invoice Value"
    ])

    pur = clean_numeric(pur, [
        "Taxable Amount","IGST Amount","CGST Amount",
        "SGST Amount","Invoice Value"
    ])

    # ---------------- DUPLICATE DETECTION ----------------
    gst = detect_duplicates(
        gst,
        ["Supplier GSTIN","doc_norm","Invoice Value"]
    )

    pur = detect_duplicates(
        pur,
        ["GSTIN Of Vendor/Customer","doc_norm","Invoice Value"]
    )

    # ---------------- GSTIN ALIGN ----------------
    pur["Vendor/Customer GSTIN"] = pur["GSTIN Of Vendor/Customer"]

    pur.rename(
        columns={"GSTIN Of Vendor/Customer":"Supplier GSTIN"},
        inplace=True
    )

    # ---------------- AGGREGATION ----------------
    gst_agg = gst.groupby(
        ["Supplier GSTIN","doc_norm"],
        as_index=False
    ).agg({

        "Document Number":"first",
        "Taxable Value":"sum",
        "IGST Amount":"sum",
        "CGST Amount":"sum",
        "SGST Amount":"sum",
        "Invoice Value":"sum",
        "Duplicate Flag":"first"
    })

    pur_agg = pur.groupby(
        ["Supplier GSTIN","doc_norm"],
        as_index=False
    ).agg({

        "Reference Document No.":"first",
        "Taxable Amount":"sum",
        "IGST Amount":"sum",
        "CGST Amount":"sum",
        "SGST Amount":"sum",
        "Invoice Value":"sum",
        "Vendor/Customer GSTIN":"first",
        "Duplicate Flag":"first"
    })

    # ---------------- MERGE ----------------
    merged = gst_agg.merge(

        pur_agg,
        on=["Supplier GSTIN","doc_norm"],
        how="outer",
        suffixes=("_2B","_PUR"),
        indicator=True
    )

    # ---------------- DIFF CALC ----------------
    merged["Invoice Diff"] = (
        merged["Invoice Value_PUR"]
        - merged["Invoice Value_2B"]
    )

    merged["Taxable Diff"] = (
        merged["Taxable Amount"]
        - merged["Taxable Value"]
    )

    merged["IGST Diff"] = (
        merged["IGST Amount_PUR"]
        - merged["IGST Amount_2B"]
    )

    merged["CGST Diff"] = (
        merged["CGST Amount_PUR"]
        - merged["CGST Amount_2B"]
    )

    merged["SGST Diff"] = (
        merged["SGST Amount_PUR"]
        - merged["SGST Amount_2B"]
    )

    # ---------------- STATUS ----------------
    merged["Match_Status"] = None
    merged["Fuzzy Score"] = 0

    both = merged["_merge"]=="both"

    tax_condition = (
        merged["Invoice Diff"].abs() <= tax_tolerance
    ) & (
        merged["Taxable Diff"].abs() <= tax_tolerance
    )

    merged.loc[both & tax_condition,"Match_Status"]="Exact Match"

    merged.loc[both & ~tax_condition,"Match_Status"]="Value Mismatch"

    merged.loc[
        merged["_merge"]=="left_only",
        "Match_Status"
    ]="Open in 2B"

    merged.loc[
        merged["_merge"]=="right_only",
        "Match_Status"
    ]="Open in Books"

    # -------------------------------------------------
    # 6️⃣ FUZZY MATCH
    # -------------------------------------------------
    # -------------------------------------------------
# 6️⃣ FIXED FUZZY MATCH
# -------------------------------------------------

open_2b = merged[merged["Match_Status"] == "Open in 2B"]
open_books = merged[merged["Match_Status"] == "Open in Books"]

for left_idx in open_2b.index:

    left_doc = merged.at[left_idx, "doc_norm"]
    left_val = merged.at[left_idx, "Invoice Value_2B"]

    # 🔹 Reduce search space using invoice value
    candidates = open_books[
        (open_books["Invoice Value_PUR"] - left_val).abs() <= tax_tolerance
    ]

    if candidates.empty:
        continue

    candidate_docs = candidates["doc_norm"].tolist()
    candidate_index = candidates.index.tolist()

    match = process.extractOne(
        left_doc,
        candidate_docs,
        scorer=fuzz.ratio
    )

    if match and match[1] >= doc_threshold:

        matched_doc, score, position = match
        right_idx = candidate_index[position]

        merged.at[left_idx, "Match_Status"] = "Fuzzy Match"
        merged.at[left_idx, "Fuzzy Score"] = score

        merged.at[right_idx, "Match_Status"] = "Fuzzy Consumed"

        # remove matched row from candidate pool
        open_books = open_books.drop(right_idx)

    # -------------------------------------------------
    # 7️⃣ GSTIN MISMATCH
    # -------------------------------------------------
    open_2b = merged[merged["Match_Status"]=="Open in 2B"]
    open_books = merged[merged["Match_Status"]=="Open in Books"]

    for left in open_2b.index:

        doc = merged.at[left,"doc_norm"]
        val = merged.at[left,"Invoice Value_2B"]

        candidates = open_books[
            open_books["doc_norm"]==doc
        ]

        for right in candidates.index:

            val2 = merged.at[right,"Invoice Value_PUR"]

            if abs(val-val2) <= gstin_tolerance:

                merged.at[left,"Match_Status"]="GSTIN Mismatch"
                merged.at[right,"Match_Status"]="GSTIN Mismatch"

    merged.drop(columns="_merge",inplace=True)

    return merged

