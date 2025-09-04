import streamlit as st
from components.Sidebar import sidebar_navigation

# 初始化页面
st.set_page_config(
    page_title="A股量化--回测",
    page_icon="📈",
    layout="wide",
    # initial_sidebar_state="expanded"  # 侧边栏默认展开
)
# 侧边栏导航
page = sidebar_navigation()

# 根据选择的页面加载不同内容
if page == "🏠 主页":
    from pages import Home
    Home.show()
elif page == "📊 回测":
    from pages import Backtrade
    Backtrade.show()
# elif page == "📖 重新量化":
#     if sub_page == "📂 输入数据分析":
#         from pages import Introduction
#
#         Introduction.show()
