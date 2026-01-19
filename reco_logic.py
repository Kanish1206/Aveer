import pandas as pd
import re
from rapidfuzz import process, fuzz


# --------------------------------------------------
# Helper: Normalize Document Number
# --------------------------------------------------
def normalize_doc_no(x) -> str:
    """
    Normalize invoice/document numbers for robust matching.
    Example:
    '002/2025-02' -> '2 2025 2'
    """
    if pd.isna(x):
        return ""

    x = str(x)

    # Replace separators with space
    x = re.sub(r"[\/\-_]", " ", x)

    # Extract numeric tokens
    nums = re.findall(r"\d+", x)

    clean_nums = []
    for n in nums:
        try:
            clean_nums.append(str(int(n)))  # remove leading zeros
        except ValueError:
            pass

    return " ".join(clean_nums)


# --------------------------------------------------
# Core Reconciliation Engine
# --------------------------------------------------
def process_reco(gst: pd.DataFrame, pur: pd.DataFrame, threshold: int = 90) -> pd.DataFrame:
    """
    GST 2B vs Purchase Register reconciliation
    """

    # --------------------------------------------------
    # 1. Column cleanup (VERY IMPORTANT)
    # --------------------------------------------------
    gst.columns = gst.columns.str.strip()
    pur.columns = pur.columns.str.strip()

    # --------------------------------------------------
    # 2. Required columns validation
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
    # 3. Cleaning & normalization
    # --------------------------------------------------
    gst["Invoice number"] = gst["Invoice number"].astype(str)
    pur["Reference Document No."] = pur["Reference Document No."].astype(str)

    gst["Doc_norm"] = gst["Invoice number"].apply(normalize_doc_no)
    pur["Doc_norm"] = pur["Reference Document No."].apply(normalize_doc_no)

    # --------------------------------------------------
    # 4. Aggregation
    # --------------------------------------------------
    gst_agg = (
        gst.groupby(["GSTIN of supplier", "Doc_norm"], as_index=False)
        .agg(
            Doc_No_2B=("Invoice number", "first"),
            Supplier_Name_2B=("Trade/Legal name", "first"),
            Invoice_Date_2B=("Invoice Date", "first"),
            IGST_2B=("Integrated Tax(₹)", "sum"),
            CGST_2B=("Central Tax(₹)", "sum"),
            SGST_2B=("State/UT Tax(₹)", "sum"),
        )
    )

    pur_agg = (
        pur.groupby(
            ["Supplier GSTIN", "Doc_norm", "FI Document Number"],
            as_index=False,
        )
        .agg(
            Doc_No_PUR=("Reference Document No.", "first"),
            Supplier_Name_PUR=("Vendor/Customer Name", "first"),
            IGST_PUR=("IGST(Cr)", "sum"),
            CGST_PUR=("CGST(Cr)", "sum"),
            SGST_PUR=("SGST(Cr)", "sum"),
        )
        .rename(columns={"Supplier GSTIN": "GSTIN of supplier"})
    )

    # --------------------------------------------------
    # 5. Exact Match (GSTIN + Normalized Doc No)
    # --------------------------------------------------
    merged = gst_agg.merge(
        pur_agg,
        on=["GSTIN of supplier", "Doc_norm"],
        how="outer",
        suffixes=("_2B", "_PUR"),
        indicator=True,
    )

    merged["Match_Status"] = pd.Categorical(
        merged["_merge"].map({
            "both": "Exact Match",
            "left_only": "Open in 2B",
            "right_only": "Open in Books",
        }),
        categories=[
            "Exact Match",
            "Fuzzy Match",
            "Open in 2B",
            "Open in Books",
        ],
    )

    merged["Matched_Doc_no_other_side"] = None
    merged["Fuzzy_Score"] = 0

    # --------------------------------------------------
    # 6. Fuzzy Matching (GSTIN scoped)
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

                # Assign fuzzy match
                merged.loc[left_idx, "Match_Status"] = "Fuzzy Match"
                merged.loc[left_idx, "Matched_Doc_no_other_side"] = merged.loc[
                    right_idx, "Doc_No_PUR"
                ]
                merged.loc[left_idx, "Fuzzy_Score"] = score

                # Copy Purchase values
                pur_cols = [c for c in merged.columns if c.endswith("_PUR")]
                for c in pur_cols:
                    merged.loc[left_idx, c] = merged.loc[right_idx, c]

                used_right_indices.add(right_idx)

    # --------------------------------------------------
    # 7. Difference Calculation
    # --------------------------------------------------
    merged["Diff_IGST"] = merged["IGST_PUR"].fillna(0) - merged["IGST_2B"].fillna(0)
    merged["Diff_CGST"] = merged["CGST_PUR"].fillna(0) - merged["CGST_2B"].fillna(0)
    merged["Diff_SGST"] = merged["SGST_PUR"].fillna(0) - merged["SGST_2B"].fillna(0)

    # --------------------------------------------------
    # 8. Cleanup
    # --------------------------------------------------
    merged.drop(columns=["_merge", "Doc_norm"], errors="ignore", inplace=True)

    return merged
