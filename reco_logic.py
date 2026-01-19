import pandas as pd
import numpy as np
import re
from rapidfuzz import process, fuzz


# --------------------------------------------------
# Helpers
# --------------------------------------------------
def normalize_doc_no(x: str) -> str:
    """
    Normalizes invoice / document numbers for fuzzy matching
    Example:
    002/2025-02 -> "2 2025 2"
    """
    if pd.isna(x):
        return ""

    x = str(x)

    # Replace separators with space
    x = re.sub(r"[\/\-_]", " ", x)

    # Extract numeric tokens
    nums = re.findall(r"\d+", x)

    # Remove leading zeros safely
    clean_nums = []
    for n in nums:
        try:
            clean_nums.append(str(int(n)))
        except ValueError:
            pass

    return " ".join(clean_nums)


# --------------------------------------------------
# Core Reconciliation Function
# --------------------------------------------------
def process_reco(gst: pd.DataFrame, pur: pd.DataFrame, threshold: int = 90) -> pd.DataFrame:
    """
    GST 2B vs Purchase Register reconciliation engine
    """

    # --------------------------------------------------
    # 1. REQUIRED COLUMNS CHECK
    # --------------------------------------------------
    gst_required = [
        "GSTIN of supplier",
        "Invoice number",
        "Invoice Date",
        "Trade/Legal name",
        "Integrated Tax(₹)",
        "Central Tax(₹)",
        "State/UT Tax(₹)",
    ]

    pur_required = [
        "Supplier GSTIN",
        "Reference Document No.",
        "FI Document Number",
        "Vendor/Customer Name",
        "IGST(Cr)",
        "CGST(Cr)",
        "SGST(Cr)",
    ]

    missing_gst = [c for c in gst_required if c not in gst.columns]
    missing_pur = [c for c in pur_required if c not in pur.columns]

    if missing_gst:
        raise ValueError(f"Missing columns in 2B file: {missing_gst}")

    if missing_pur:
        raise ValueError(f"Missing columns in Books file: {missing_pur}")

    # --------------------------------------------------
    # 2. CLEANING
    # --------------------------------------------------
    gst["Invoice number"] = gst["Invoice number"].astype(str)
    pur["Reference Document No."] = pur["Reference Document No."].astype(str)

    gst["Doc_norm"] = gst["Invoice number"].apply(normalize_doc_no)
    pur["Doc_norm"] = pur["Reference Document No."].apply(normalize_doc_no)

    # --------------------------------------------------
    # 3. AGGREGATION
    # --------------------------------------------------
    gst_agg = (
        gst.groupby(["GSTIN of supplier", "Doc_norm"], as_index=False)
        .agg(
            Supplier_Name_2B=("Trade/Legal name", "first"),
            Doc No.=("Invoice number","first"),
            Invoice_Date_2B=("Invoice Date", "first"),
            IGST_2B=("Integrated Tax(₹)", "sum"),
            CGST_2B=("Central Tax(₹)", "sum"),
            SGST_2B=("State/UT Tax(₹)", "sum"),
            #Doc_norm=("Doc_norm", "first"),
        )
    )

    pur_agg = (
        pur.groupby(
            ["Supplier GSTIN", "Doc_norm", "FI Document Number"],
            as_index=False,
        )
        .agg(
            Supplier_Name_PUR=("Vendor/Customer Name", "first"),
            Doc No.=("Reference Document No.","first"),
            IGST_PUR=("IGST(Cr)", "sum"),
            CGST_PUR=("CGST(Cr)", "sum"),
            SGST_PUR=("SGST(Cr)", "sum"),
            #Doc_norm=("Doc_norm", "first"),
        )
        .rename(
            columns={
                "Supplier GSTIN": "GSTIN of supplier",
                "Reference Document No.": "Invoice number",
            }
        )
    )

    # --------------------------------------------------
    # 4. EXACT MATCH
    # --------------------------------------------------
    merged = gst_agg.merge(
        pur_agg,
        on=["GSTIN of supplier", "Invoice number"],
        how="outer",
        suffixes=("_2B", "_PUR"),
        indicator=True,
    )

    merged["Match_Status"] = merged["_merge"].map({
        "both": "Exact Match",
        "left_only": "Open in 2B",
        "right_only": "Open in Books",
    })

    merged["Matched_Doc_no_other_side"] = None
    merged["Fuzzy_Score"] = 0

    # --------------------------------------------------
    # 5. FUZZY MATCH (GSTIN SCOPED)
    # --------------------------------------------------
    left_df = merged[merged["_merge"] == "left_only"].copy()
    right_df = merged[merged["_merge"] == "right_only"].copy()

    used_right_indices = set()

    for gstin in set(left_df["GSTIN of supplier"]) & set(right_df["GSTIN of supplier"]):

        left_subset = left_df[left_df["GSTIN of supplier"] == gstin]
        right_subset = right_df[right_df["GSTIN of supplier"] == gstin]

        choices = right_subset["Doc_norm"].tolist()
        choice_index_map = dict(enumerate(right_subset.index))

        for left_idx, row in left_subset.iterrows():
            query = row["Doc_norm"]
            if not query:
                continue

            result = process.extractOne(
                query,
                choices,
                scorer=fuzz.token_set_ratio,
                score_cutoff=threshold,
            )

            if result:
                _, score, pos = result
                right_idx = choice_index_map[pos]

                if right_idx in used_right_indices:
                    continue

                # Mark fuzzy match
                merged.loc[left_idx, "Match_Status"] = "Fuzzy Match"
                merged.loc[left_idx, "Matched_Doc_no_other_side"] = merged.loc[
                    right_idx, "Invoice number"
                ]
                merged.loc[left_idx, "Fuzzy_Score"] = score

                # Copy PUR columns
                pur_cols = [c for c in merged.columns if c.endswith("_PUR")]
                for c in pur_cols:
                    merged.loc[left_idx, c] = merged.loc[right_idx, c]

                used_right_indices.add(right_idx)

    # --------------------------------------------------
    # 6. DIFFERENCE CALCULATION
    # --------------------------------------------------
    merged["Diff_IGST"] = merged["IGST_PUR"].fillna(0) - merged["IGST_2B"].fillna(0)
    merged["Diff_CGST"] = merged["CGST_PUR"].fillna(0) - merged["CGST_2B"].fillna(0)
    merged["Diff_SGST"] = merged["SGST_PUR"].fillna(0) - merged["SGST_2B"].fillna(0)

    # --------------------------------------------------
    # 7. CLEANUP
    # --------------------------------------------------
    merged.drop(columns=["_merge", "Doc_norm"], errors="ignore", inplace=True)

    return merged


