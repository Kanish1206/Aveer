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
# 1️⃣ NORMALIZE DOCUMENT
# -------------------------------------------------
def normalize_doc(series):
    return (
        series.fillna("")
        .astype(str)
        .str.upper()
        .str.replace(r"[^A-Z0-9]", "", regex=True)
    )


# -------------------------------------------------
# 2️⃣ COLUMN VALIDATION
# -------------------------------------------------
def validate_columns(df, required_cols, df_name):
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"{df_name} is missing required columns: {missing}")


# -------------------------------------------------
# 3️⃣ DIFF CALCULATION
# -------------------------------------------------
def compute_diffs(df):
    df["IGST Diff"] = df["IGST(Cr)"] - df["Integrated Tax(₹)"]
    df["CGST Diff"] = df["CGST(Cr)"] - df["Central Tax(₹)"]
    df["SGST Diff"] = df["SGST(Cr)"] - df["State/UT Tax(₹)"]
    return df


# -------------------------------------------------
# 4️⃣ MAIN FUNCTION
# -------------------------------------------------
def process_reco(
    gst_df,
    pur_df,
    doc_threshold=60,
    tax_tolerance=10,
    gstin_mismatch_tolerance=5,
):
    gst = gst_df.copy()
    pur = pur_df.copy()

    # ---------------- VALIDATION ----------------
    gst_required = [
        "GSTIN of supplier", "Invoice number", "Invoice Date",
        "Taxable Value (₹)", "Trade/Legal name", "Remark 2B",
        "Integrated Tax(₹)", "Central Tax(₹)", "State/UT Tax(₹)", "Invoice Value(₹)"
    ]
    pur_required = [
        "Supplier GSTIN", "Reference Document No.",
        "Taxable Amount", "Document Date",
        "Vendor/Customer Name", "IGST(Cr)", "CGST(Cr)",
        "SGST(Cr)", "Total"
    ]

    validate_columns(gst, gst_required, "2B File")
    validate_columns(pur, pur_required, "Purchase File")

    # ---------------- PREP ----------------
    pur["Vendor/Customer GSTIN"] = pur["Supplier GSTIN"]

    gst["doc_norm"] = normalize_doc(gst["Invoice number"])
    pur["doc_norm"] = normalize_doc(pur["Reference Document No."])

    # Rename GSTIN of supplier to common key for merging
    gst.rename(columns={"GSTIN of supplier": "Supplier GSTIN"}, inplace=True)

    # ---------------- AGGREGATION ----------------
    gst_agg = gst.groupby(
        ["Supplier GSTIN", "doc_norm"], as_index=False 
    ).agg({
        "Invoice number": "first",
        "Trade/Legal name": "first",
        "Invoice Date": "first",
        "Remark 2B": "first",
        "Integrated Tax(₹)": "sum",
        "Central Tax(₹)": "sum",
        "State/UT Tax(₹)": "sum",
        "Taxable Value (₹)": "sum",
        "Invoice Value(₹)": "sum",
    }) 
    
    pur_agg = pur.groupby(
        ["Supplier GSTIN", "doc_norm"], as_index=False
    ).agg({
        "Reference Document No.": "first",
        "Vendor/Customer GSTIN": "first",
        "FI Document Number": "first",
        "Vendor/Customer Name": "first",
        "Document Date": "first",
        "Taxable Amount": "sum",
        "IGST(Cr)": "sum",
        "CGST(Cr)": "sum",
        "SGST(Cr)": "sum",
        "Total": "sum",
    })

    # ---------------- MERGE ----------------
    merged = gst_agg.merge(
        pur_agg,
        on=["Supplier GSTIN", "doc_norm"],
        how="outer",
        indicator=True,
    )

    # ---------------- NUMERIC CLEAN ----------------
    numeric_cols = [
        "Integrated Tax(₹)", "Central Tax(₹)", "State/UT Tax(₹)",
        "Invoice Value(₹)", "Taxable Value (₹)",
        "IGST(Cr)", "CGST(Cr)", "SGST(Cr)",
        "Total", "Taxable Amount",
    ]

    for col in numeric_cols:
        if col not in merged.columns:
            merged[col] = 0
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0)

    # ---------------- INITIAL MATCH ----------------
    merged = compute_diffs(merged)

    merged["Match_Status"] = None
    merged["Fuzzy Score"] = 0.0

    both_mask = merged["_merge"] == "both"

    tax_condition = (
        (merged["IGST Diff"].abs() <= tax_tolerance) &
        (merged["CGST Diff"].abs() <= tax_tolerance) &
        (merged["SGST Diff"].abs() <= tax_tolerance)
    )

    merged.loc[both_mask & tax_condition, "Match_Status"] = MATCH_EXACT
    merged.loc[both_mask & ~tax_condition, "Match_Status"] = MATCH_VALUE_MISMATCH
    merged.loc[merged["_merge"] == "left_only", "Match_Status"] = MATCH_OPEN_2B
    merged.loc[merged["_merge"] == "right_only", "Match_Status"] = MATCH_OPEN_BOOKS

    # ---------------- FUZZY MATCH ----------------
    for gstin in merged["Supplier GSTIN"].dropna().unique():

        open_2b = merged[
            (merged["Supplier GSTIN"] == gstin) &
            (merged["Match_Status"] == MATCH_OPEN_2B)
        ]

        open_books = merged[
            (merged["Supplier GSTIN"] == gstin) &
            (merged["Match_Status"] == MATCH_OPEN_BOOKS)
        ]

        for left_idx in open_2b.index:

            left_doc = str(merged.at[left_idx, "Invoice number"])
            if not left_doc or open_books.empty:
                continue

            candidates = open_books.copy()

            candidates["tax_score"] = (
                (candidates["IGST(Cr)"] - merged.at[left_idx, "Integrated Tax(₹)"]).abs() +
                (candidates["CGST(Cr)"] - merged.at[left_idx, "Central Tax(₹)"]).abs() +
                (candidates["SGST(Cr)"] - merged.at[left_idx, "State/UT Tax(₹)"]).abs()
            )

            candidates = candidates[candidates["tax_score"] <= tax_tolerance * 2]

            if candidates.empty:
                continue

            candidate_dict = dict(zip(
                candidates.index,
                candidates["Reference Document No."].astype(str)
            ))

            match = process.extractOne(
                left_doc,
                candidate_dict,
                scorer=fuzz.partial_token_set_ratio,
                score_cutoff=doc_threshold
            )

            if match:
                _, score, right_idx = match

                for col in [
                    "Reference Document No.",
                    "FI Document Number",
                    "Vendor/Customer GSTIN",
                    "Vendor/Customer Name",
                    "Document Date",
                    "IGST(Cr)",
                    "CGST(Cr)",
                    "SGST(Cr)",
                    "Taxable Amount",
                    "Total"
                ]:
                    merged.at[left_idx, col] = merged.at[right_idx, col]

                merged.at[left_idx, "Match_Status"] = MATCH_FUZZY
                merged.at[left_idx, "Fuzzy Score"] = score
                merged.at[right_idx, "Match_Status"] = MATCH_FUZZY_CONSUMED

                open_books = open_books.drop(index=right_idx)

    # ---------------- GSTIN MISMATCH ----------------
    open_2b = merged[merged["Match_Status"] == MATCH_OPEN_2B]
    open_books = merged[merged["Match_Status"] == MATCH_OPEN_BOOKS]

    for left_idx in open_2b.index:

        doc = merged.at[left_idx, "doc_norm"]
        if not doc:
            continue

        left_val = merged.at[left_idx, "Invoice Value(₹)"]
        left_igst = merged.at[left_idx, "Integrated Tax(₹)"]
        left_cgst = merged.at[left_idx, "Central Tax(₹)"]
        left_sgst = merged.at[left_idx, "State/UT Tax(₹)"]

        possible = open_books[open_books["doc_norm"] == doc]

        if len(possible) > 3:
            continue

        for right_idx in possible.index:

            right_val = merged.at[right_idx, "Total"]
            right_igst = merged.at[right_idx, "IGST(Cr)"]
            right_cgst = merged.at[right_idx, "CGST(Cr)"]
            right_sgst = merged.at[right_idx, "SGST(Cr)"]

            if (
                abs(left_val - right_val) <= gstin_mismatch_tolerance and
                abs(left_igst - right_igst) <= tax_tolerance and
                abs(left_cgst - right_cgst) <= tax_tolerance and
                abs(left_sgst - right_sgst) <= tax_tolerance
            ):
                merged.at[left_idx, "Match_Status"] = MATCH_GSTIN_MISMATCH
                merged.at[right_idx, "Match_Status"] = MATCH_GSTIN_MISMATCH
                break

    # ---------------- PAN MATCH ----------------

    open_2b_for_pan = merged[merged["Match_Status"] == "Open in 2B"]
    open_books_for_pan = merged[merged["Match_Status"] == "Open in Books"]

    merged["PAN_2B"] = (
        merged["Supplier GSTIN"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
        .str[2:12]
    )

    merged["PAN_PUR"] = (
        merged["Vendor/Customer GSTIN"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
        .str[2:12]
    )

    for left_idx in open_2b_for_pan.index:

        pan_2b = merged.at[left_idx, "PAN_2B"]
        doc_2b = merged.at[left_idx, "doc_norm"]

        if not pan_2b or not doc_2b:
            continue

        igst_2b = merged.at[left_idx, "Integrated Tax(₹)"]
        cgst_2b = merged.at[left_idx, "Central Tax(₹)"]
        sgst_2b = merged.at[left_idx, "State/UT Tax(₹)"]

        candidates = merged[
            (merged.index.isin(open_books_for_pan.index)) &
            (merged["PAN_PUR"] == pan_2b) &
            (merged["doc_norm"] == doc_2b)
        ].copy()

        if candidates.empty:
            continue

        candidates["tax_diff"] = (
            (candidates["IGST(Cr)"] - igst_2b).abs() +
            (candidates["CGST(Cr)"] - cgst_2b).abs() +
            (candidates["SGST(Cr)"] - sgst_2b).abs()
        )

        candidates = candidates.sort_values("tax_diff")

        for right_idx in candidates.index:

            if candidates.at[right_idx, "tax_diff"] > tax_tolerance * 3:
                continue

            merged.at[left_idx, "Match_Status"] = "PAN Match (GSTIN Variation)"
            merged.at[right_idx, "Match_Status"] = "PAN Consumed"

            # Books columns to copy to 2B side
            pur_cols = [
                "Reference Document No.", "FI Document Number",
                "Vendor/Customer Name", "Vendor/Customer GSTIN",
                "Document Date", "IGST(Cr)", "CGST(Cr)", "SGST(Cr)",
                "Taxable Amount", "Total"
            ]

            for col in pur_cols:
                if col in merged.columns:
                    merged.at[left_idx, col] = merged.at[right_idx, col]

            open_books_for_pan = open_books_for_pan.drop(index=right_idx)
            break

    # Clean consumed rows
    merged = merged[~merged["Match_Status"].isin(["PAN Consumed"])]
    merged.drop(columns=["PAN_2B", "PAN_PUR"], inplace=True, errors="ignore")
    
    # ---------------- FINAL MATCH (IGNORE GSTIN) ----------------
    open_2b_final = merged[merged["Match_Status"] == MATCH_OPEN_2B]
    open_books_final = merged[merged["Match_Status"] == MATCH_OPEN_BOOKS]

    for left_idx in open_2b_final.index:

        doc_2b = merged.at[left_idx, "doc_norm"]
        if not doc_2b:
            continue

        igst_2b = merged.at[left_idx, "Integrated Tax(₹)"]
        cgst_2b = merged.at[left_idx, "Central Tax(₹)"]
        sgst_2b = merged.at[left_idx, "State/UT Tax(₹)"]

        candidates = open_books_final[
            open_books_final["doc_norm"] == doc_2b
        ].copy()

        if candidates.empty:
            continue

        candidates["tax_diff"] = (
            (candidates["IGST(Cr)"] - igst_2b).abs() +
            (candidates["CGST(Cr)"] - cgst_2b).abs() +
            (candidates["SGST(Cr)"] - sgst_2b).abs()
        )

        candidates = candidates.sort_values("tax_diff")

        for right_idx in candidates.index:

            if candidates.at[right_idx, "tax_diff"] > tax_tolerance * 3:
                continue

            merged.at[left_idx, "Match_Status"] = "Doc Match (Ignore GSTIN)"
            merged.at[right_idx, "Match_Status"] = "Doc Consumed (Ignore GSTIN)"

            pur_cols = [
                "Reference Document No.", "FI Document Number",
                "Vendor/Customer Name", "Vendor/Customer GSTIN",
                "Document Date", "IGST(Cr)", "CGST(Cr)", "SGST(Cr)",
                "Taxable Amount", "Total"
            ]

            for col in pur_cols:
                if col in merged.columns:
                    merged.at[left_idx, col] = merged.at[right_idx, col]

            open_books_final = open_books_final.drop(index=right_idx)
            break

    # ---------------- CLEANUP ----------------
    merged = merged[~merged["Match_Status"].isin([
        MATCH_FUZZY_CONSUMED,
        MATCH_PAN_CONSUMED,
        "Doc Consumed (Ignore GSTIN)"
    ])]
    
    merged = compute_diffs(merged)
    merged.drop(columns=["_merge"], inplace=True, errors="ignore")
    
    priority_cols = [
        "Supplier GSTIN",
        "doc_norm",
        "Document Type",
        "Invoice number",
        "Return Period",
        "Trade/Legal name",
        "Invoice Date",
        "Remark 2B",
        "Integrated Tax(₹)",
        "Central Tax(₹)",
        "State/UT Tax(₹)",
        "Taxable Value (₹)",
        "Invoice Value(₹)",
        "FI Document Number",
        "Reference Document No.",
        "Vendor/Customer GSTIN",
        "Vendor/Customer Name",
        "Document Date",
        "IGST(Cr)",
        "CGST(Cr)",
        "SGST(Cr)",
        "Taxable Amount",
        "Total",
        "IGST Diff",
        "CGST Diff",
        "SGST Diff",
        "Match_Status",
        "Fuzzy Score",
    ]

    # Keep only columns that exist (avoids errors)
    priority_cols = [col for col in priority_cols if col in merged.columns]
    merged = merged[priority_cols + [col for col in merged.columns if col not in priority_cols]]
    
    return merged
