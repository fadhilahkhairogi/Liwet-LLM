import streamlit as st

st.set_page_config(
    page_title="Liwet-LLM",
    layout="wide",
    initial_sidebar_state="expanded",
)

navigation = st.navigation(
    [
        st.Page("chat.py", title="Chat", default=True),
        st.Page("arena.py", title="Arena 3 Skala"),
        st.Page("parameter_testing.py", title="Testing Parameter"),
        st.Page("analysis.py", title="Analisis & Evaluasi"),
    ]
)
navigation.run()
