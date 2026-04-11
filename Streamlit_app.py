import streamlit as st
import pandas as pd
import io
import reco_logic as reco_logic

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="GST Reco Pro",
    page_icon="📘",
    layout="wide"
)

# ---------------- HEADER ----------------
st.markdown("""
    <h1 style='margin-bottom:5px;'>📘 GST Reconciliation Dashboard</h1>
    <p style='color:gray;'>Compare GSTR-2B with Purchase Register</p>
""", unsafe_allow_html=True)

st.divider()

# ---------------- FILE UPLOAD ----------------
st.subheader("📂 Upload Files")

col1, col2 = st.columns(2)

with col1:
    gst_file = st.file_uploader("Upload GSTR-2B Excel", type=["xlsx"])

with col2:
    pur_file = st.file_uploader("Upload Purchase Register Excel", type=["xlsx"])

st.divider()

# ---------------- MAIN LOGIC ----------------
if gst_file and pur_file:

    try:
        df_2b = pd.read_excel(gst_file)
        df_books = pd.read_excel(pur_file)

        df_2b.columns = df_2b.columns.str.strip()
        df_books.columns = df_books.columns.str.strip()

        st.success("✅ Files uploaded successfully")

        if st.button("🚀 Run Reconciliation", use_container_width=True):

            with st.spinner("Processing reconciliation... ⏳"):
                result_df = reco_logic.process_reco(df_2b, df_books)

            # ---------------- SUMMARY ----------------
            st.subheader("📊 Summary")

            total = len(result_df)
            matched = result_df["Match_Status"].str.contains("Match", case=False, na=False).sum()
            unmatched = total - matched

            c1, c2, c3 = st.columns(3)

            c1.metric("📄 Total Records", total)
            c2.metric("✅ Matched", matched)
            c3.metric("❌ Unmatched", unmatched)

            st.divider()

            # ---------------- TABLE ----------------
            st.subheader("📋 Detailed Results")

            st.dataframe(result_df, use_container_width=True)

            st.divider()

            # ---------------- DOWNLOAD ----------------
            st.subheader("📥 Download Report")

            output = io.BytesIO()
            with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                result_df.to_excel(writer, index=False)

            st.download_button(
                "⬇️ Download Excel",
                data=output.getvalue(),
                file_name="GST_Reconciliation.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

else:
    st.info("👆 Please upload both files to start reconciliation.")
