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

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

body {
    background-color: #f4f7fb;
}

/* Header */
.header-box {
    background: linear-gradient(90deg, #1f4e79, #4fa3d1);
    padding: 25px;
    border-radius: 12px;
    color: white;
}

/* Card UI */
.metric-card {
    background: white;
    padding: 22px;
    border-radius: 14px;
    box-shadow: 0px 6px 18px rgba(0,0,0,0.08);
    text-align: center;
    transition: 0.3s;
}
.metric-card:hover {
    transform: translateY(-5px);
}

/* Upload box */
.upload-box {
    background: #ffffff;
    padding: 25px;
    border-radius: 14px;
    border: 2px dashed #cbd5e1;
    text-align: center;
    transition: 0.3s;
}
.upload-box:hover {
    border-color: #1f77b4;
    background: #f0f8ff;
}

/* Buttons */
button[kind="primary"] {
    background: linear-gradient(90deg, #1f77b4, #4fa3d1);
    border-radius: 12px;
    height: 3em;
    font-weight: bold;
    border: none;
}

/* Download button */
.stDownloadButton button {
    background: linear-gradient(90deg, #28a745, #5cd65c) !important;
    color: white !important;
    border-radius: 12px;
    height: 3em;
    font-weight: bold;
}

/* Divider spacing */
hr {
    margin-top: 30px;
    margin-bottom: 30px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("""
<div class="header-box">
    <h1 style="margin-bottom:5px;">📘 GST Reconciliation Dashboard</h1>
    <p style="margin:0;font-size:16px;">
        Smart comparison of GSTR-2B & Purchase Register
    </p>
</div>
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
        st.toast("Files ready! Click 'Run Reconciliation' 🚀")

        if st.button("🚀 Run Reconciliation", use_container_width=True):

            progress = st.progress(0)

            with st.spinner("Processing reconciliation... ⏳"):
                progress.progress(30)
                result_df = reco_logic.process_reco(df_2b, df_books)
                progress.progress(100)

            # ---------------- SUMMARY ----------------
            st.subheader("📊 Summary")

            total = len(result_df)
            matched = result_df["Match_Status"].str.contains("Match", case=False, na=False).sum()
            unmatched = total - matched
            match_pct = (matched / total * 100) if total else 0

            c1, c2, c3 = st.columns(3)

            with c1:
                st.markdown(f"""
                <div class='metric-card'>
                    <h4>📄 Total Records</h4>
                    <h2>{total}</h2>
                </div>
                """, unsafe_allow_html=True)

            with c2:
                st.markdown(f"""
                <div class='metric-card'>
                    <h4>✅ Matched</h4>
                    <h2>{matched}</h2>
                    <p style='color:green'>{match_pct:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)

            with c3:
                st.markdown(f"""
                <div class='metric-card'>
                    <h4>❌ Unmatched</h4>
                    <h2>{unmatched}</h2>
                </div>
                """, unsafe_allow_html=True)

            st.divider()

            # ---------------- CHART ----------------
            st.subheader("📈 Match Overview")

            chart_data = pd.DataFrame({
                "Status": ["Matched", "Unmatched"],
                "Count": [matched, unmatched]
            })

            st.bar_chart(chart_data.set_index("Status"))

            st.divider()

            # ---------------- TABLE ----------------
            st.subheader("📋 Detailed Results")

            def highlight_rows(row):
                if "match" in str(row["Match_Status"]).lower():
                    return ["background-color: #e6ffed"] * len(row)
                else:
                    return ["background-color: #ffe6e6"] * len(row)

            st.dataframe(
                result_df.style.apply(highlight_rows, axis=1),
                use_container_width=True,
                height=500
            )

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

# ---------------- FOOTER ----------------
st.markdown("""
<hr>
<p style='text-align:center;color:gray;font-size:14px;'>
GST Reco Pro • Built with Streamlit 🚀
</p>
""", unsafe_allow_html=True)
