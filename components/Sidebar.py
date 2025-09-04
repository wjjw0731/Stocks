import streamlit as st


def sidebar_navigation():
    page = st.sidebar.radio(
        label="导航",  # 明确label参数（可选，但更规范）
        options=["🏠 主页", "📊 回测"],
        key="sidebar_nav_radio"  # 核心：添加唯一key，避免ID重复
    )
    return page
