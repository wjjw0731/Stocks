import streamlit as st


def sidebar_navigation():
    page = st.sidebar.radio("导航", ["🏠 主页",  "📊 回测"])

    return page
