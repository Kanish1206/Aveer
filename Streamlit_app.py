import streamlit as st
import pandas as pd
import io
import reconciliation_logic as reco_logic

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="GST Reco Pro",
    page_icon="📘",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.main {
    background-color: #f7f9fc;
}
h1 {
    color: #1f4e79;
}
.block-container {
    padding-top: 2rem;
}

.metric-card {
    background: white;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.05);
    text-align: center;
}

.upload-box {
    background: white;
    padding: 20px;
    border-radius: 12px;
    border: 2px dashed #d0d7e2;
    text-align: center;
}

button[kind="primary"] {
    background-color: #1f77b4;
    border-radius: 10px;
    height: 3em;
    font-weight: bold;
}

.stDownloadButton button {
    background-color: #28a745 !important;
    color: white !important;
    border-radius: 10px;
    height: 3em;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("""
<h1>📘 GST Reconciliation Dashboard</h1>
<p style='color:gray;font-size:16px;'>Compare GSTR-2B with Purchase Register easily</p>
""", unsafe_allow_html=True)

st.divider()

# ---------------- FILE UPLOAD ----------------
st.subheader("📂 Upload Files")

col1, col2 = st.columns(2)

with col1:
    st.markdown("<div class='upload-box'>", unsafe_allow_html=True)
    gst_file = st.file_uploader("📄 Upload GSTR-2B Excel", type=["xlsx"])
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='upload-box'>", unsafe_allow_html=True)
    pur_file = st.file_uploader("📄 Upload Purchase Register Excel", type=["xlsx"])
    st.markdown("</div>", unsafe_allow_html=True)

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

            with c1:
                st.markdown(f"<div class='metric-card'><h3>📄 Total</h3><h2>{total}</h2></div>", unsafe_allow_html=True)

            with c2:
                st.markdown(f"<div class='metric-card'><h3>✅ Matched</h3><h2>{matched}</h2></div>", unsafe_allow_html=True)

            with c3:
                st.markdown(f"<div class='metric-card'><h3>❌ Unmatched</h3><h2>{unmatched}</h2></div>", unsafe_allow_html=True)

            st.divider()

            # ---------------- TABLE ----------------
            st.subheader("📋 Detailed Results")
            st.dataframe(result_df, use_container_width=True, height=500)

            st.divider()

            # ---------------- DOWNLOAD ----------------
            st.subheader("📥 Download Report")

            output = io.BytesIO()
            with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                result_df.to_excel(writer, index=False)

            st.download_button(
                "⬇️ Download Excel Report",
                data=output.getvalue(),
                file_name="GST_Reconciliation.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

else:
    st.info("👆 Please upload both files to start reconciliation.")
