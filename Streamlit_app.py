import streamlit as st
import pandas as pd

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="GST Reconciliation Dashboard",
    page_icon="📘",
    layout="wide"
)

# ---------------- HEADER ----------------
st.markdown("""
    <h1 style='text-align: center; color: #1f4e79;'>
        📘 GST Reconciliation Dashboard
    </h1>
    <p style='text-align: center;'>
        Smart comparison of GSTR-2B & Purchase Register
    </p>
""", unsafe_allow_html=True)

# ---------------- FILE UPLOAD ----------------
col1, col2 = st.columns(2)

with col1:
    gst_file = st.file_uploader("📄 Upload GSTR-2B", type=["xlsx"])

with col2:
    purchase_file = st.file_uploader("📄 Upload Purchase Register", type=["xlsx"])

# ---------------- PROCESS FUNCTION ----------------
def process_reco(gst, books):
    result = gst.copy()

    result["Match_Status"] = result["Invoice"].isin(books["Invoice"]).map({
        True: "Matched",
        False: "Unmatched"
    })

    return result

# ---------------- RUN BUTTON ----------------
if st.button("🚀 Run Reconciliation"):

    if gst_file is None or purchase_file is None:
        st.error("❗ Please upload both files")
    else:
        with st.spinner("Processing..."):

            gst = pd.read_excel(gst_file)
            books = pd.read_excel(purchase_file)

            result = process_reco(gst, books)

        # ---------------- SUMMARY ----------------
        total = len(result)
        matched = (result["Match_Status"] == "Matched").sum()
        unmatched = total - matched
        pct = (matched / total * 100) if total > 0 else 0

        st.markdown("## 📊 Summary")

        c1, c2, c3 = st.columns(3)

        c1.metric("📄 Total Records", total)
        c2.metric("✅ Matched", matched, f"{pct:.1f}%")
        c3.metric("❌ Unmatched", unmatched)

        # ---------------- TABLE ----------------
        st.markdown("## 📋 Detailed Results")

        def highlight(row):
            if row["Match_Status"] == "Matched":
                return ["background-color: #e6ffed"] * len(row)
            else:
                return ["background-color: #ffe6e6"] * len(row)

        styled_df = result.style.apply(highlight, axis=1)

        st.dataframe(styled_df, use_container_width=True)

        # ---------------- DOWNLOAD ----------------
        csv = result.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="⬇️ Download Results",
            data=csv,
            file_name="GST_Reconciliation.csv",
            mime="text/csv"
        )
