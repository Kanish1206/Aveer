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
        gst.groupby(["GSTIN of supplier", "Invoice number"], as_index=False)
        .agg(
            Supplier_Name_2B=("Trade/Legal name", "first"),
            Invoice_Date_2B=("Invoice Date", "first"),
            IGST_2B=("Integrated Tax(₹)", "sum"),
            CGST_2B=("Central Tax(₹)", "sum"),
            SGST_2B=("State/UT Tax(₹)", "sum"),
            Doc_norm=("Doc_norm", "first"),
        )
    )

    pur_agg = (
        pur.groupby(
            ["Supplier GSTIN", "Reference Document No.", "FI Document Number"],
            as_index=False,
        )
        .agg(
            Supplier_Name_PUR=("Vendor/Customer Name", "first"),
            IGST_PUR=("IGST(Cr)", "sum"),
            CGST_PUR=("CGST(Cr)", "sum"),
            SGST_PUR=("SGST(Cr)", "sum"),
            Doc_norm=("Doc_norm", "first"),
        )
        .rename(
            columns={
                "Supplier GSTIN": "GSTIN of supplier",
                "Reference Document No.": "Invoice number",
            }
        )
    )

if merged is None or merged.empty:
        raise ValueError("Reconciliation produced no output. Check input data.")

    return merged

    # ---------------------------------

