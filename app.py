"""Corporate Leads Pipeline. Run with:  streamlit run app.py

app.py is the entry point and router. It runs on every interaction, so the
sidebar date input below appears on all screens. Screens live in pages/.
"""

import streamlit as st

import engine
import ui

st.set_page_config(page_title="Corporate Leads Pipeline", page_icon="📈", layout="wide")
ui.inject_css()

with st.sidebar:
    st.markdown("### Corporate Leads")
    st.date_input("Today (reference date)", value=engine.TODAY, key="today_ref",
                  format="YYYY-MM-DD")
    st.caption(
        "Defaults to **2026-11-01**, which matches the brief. The app never reads "
        "the system clock, so every overdue / going-cold number is measured from this date."
    )
    st.divider()

pages = [
    st.Page("pages/1_Pipeline.py", title="Pipeline", icon="🗂️", default=True),
    st.Page("pages/2_Lead_detail.py", title="Lead detail", icon="🔎"),
    st.Page("pages/3_Data_quality.py", title="Data quality", icon="🧹"),
    st.Page("pages/4_Ask_your_pipeline.py", title="Ask your pipeline", icon="💬"),
    st.Page("pages/5_Export_and_summary.py", title="Export & weekly summary", icon="📤"),
]
st.navigation(pages).run()
