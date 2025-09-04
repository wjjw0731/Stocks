import streamlit as st
import akshare as ak
from datetime import datetime
import time
from requests.exceptions import ConnectionError, Timeout

def show():
    # -------------------------- 1. 初始化SessionState（避免KeyError）--------------------------
    init_keys = {
        "show_overview": False,
        "show_stock_detail": False,
        "stock_zh_a_spot_em_df": None,
        "num": 25,
        "symbol": "600000",
        "start_date": datetime(2020, 1, 1),
        "end_date": datetime.now(),
        "stock_df": None,
        "date_col": "date"  # 预设日期列名为英文（AKshare主流返回格式）
    }
    for key, value in init_keys.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # -------------------------- 2. 页面标题与功能按钮（兼容Streamlit警告）--------------------------
    with st.container():
        st.title("📊 A股量化--回测在线网站")
        st.markdown("### 数据来源：AKshare（东方财富接口）")
        st.markdown("---")

    # 按钮用width='stretch'替代use_container_width=True（解决弃用警告）
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔍 查看今日大盘", width='stretch'):
            st.session_state.show_overview = True
            st.session_state.show_stock_detail = False
    with col2:
        if st.button("📈 查看股票具体数据", width='stretch'):
            st.session_state.show_stock_detail = True
            st.session_state.show_overview = False

    # -------------------------- 3. 今日大盘数据（增强网络错误处理）--------------------------
    if st.session_state.show_overview:
        st.subheader("今日A股大盘数据")
        if st.session_state.stock_zh_a_spot_em_df is None or st.button("🔄 重新加载大盘数据", width='stretch'):
            with st.spinner("正在加载数据...（若缓慢请重试）"):
                retry_count = 0
                max_retries = 3
                while retry_count < max_retries:
                    try:
                        df = ak.stock_zh_a_spot_em()
                        st.session_state.stock_zh_a_spot_em_df = df
                        st.success("✅ 大盘数据加载完成！")
                        break
                    except (ConnectionError, Timeout):
                        retry_count += 1
                        wait_time = 2 * retry_count  # 重试间隔递增（2s→4s→6s）
                        st.warning(f"⚠️ 网络连接超时，{wait_time}秒后进行第{retry_count+1}次重试...")
                        time.sleep(wait_time)
                    except Exception as e:
                        st.error(f"❌ 大盘数据加载失败：{str(e)}")
                        break
                if retry_count >= max_retries:
                    st.error("❌ 多次网络请求失败，请检查网络或稍后再试！")

        # 显示大盘数据（只展示关键列，避免表格过宽）
        if st.session_state.stock_zh_a_spot_em_df is not None:
            st.session_state.num = st.slider(
                "选择显示条数",
                min_value=10,
                max_value=100,
                step=10,
                value=st.session_state.num
            )
            show_cols = ["序号", "代码", "名称", "最新价", "涨跌幅", "成交量", "成交额"]
            # 确保列名存在（兼容AKshare接口列名变化）
            valid_cols = [col for col in show_cols if col in st.session_state.stock_zh_a_spot_em_df.columns]
            st.dataframe(
                st.session_state.stock_zh_a_spot_em_df[valid_cols].head(st.session_state.num),
                height=600,
                use_container_width=True
            )

    # -------------------------- 4. 个股数据（核心修复：列名适配+精准错误判断）--------------------------
    if st.session_state.show_stock_detail:
        st.subheader("个股历史数据查询")
        # 输入股票代码
        st.session_state.symbol = st.text_input(
            "请输入股票代码（例：600000=浦发银行）",
            value=st.session_state.symbol,
            help="沪市6开头，深市0/3开头，创业板30开头"
        )
        # 日期选择
        st.session_state.start_date = st.date_input(
            "开始日期",
            value=st.session_state.start_date,
            min_value=datetime(2000, 1, 1),
            max_value=datetime.now()
        )
        st.session_state.end_date = st.date_input(
            "结束日期",
            value=st.session_state.end_date,
            min_value=st.session_state.start_date,
            max_value=datetime.now()
        )
        start_str = st.session_state.start_date.strftime("%Y%m%d")
        end_str = st.session_state.end_date.strftime("%Y%m%d")

        # 获取个股数据（精准处理不同错误类型）
        if st.button("📥 获取个股数据", width='stretch'):
            with st.spinner(f"正在获取 {st.session_state.symbol} 的数据..."):
                st.session_state.stock_df = None  # 清空旧数据
                retry_count = 0
                max_retries = 3
                while retry_count < max_retries:
                    try:
                        # 调用AKshare接口（获取后复权数据）
                        df = ak.stock_zh_a_hist(
                            symbol=st.session_state.symbol,
                            period="daily",
                            start_date=start_str,
                            end_date=end_str,
                            adjust="hfq"
                        )
                        # 自动检测日期列名（兼容中文"日期"和英文"date"）
                        if "日期" in df.columns:
                            st.session_state.date_col = "日期"
                        elif "date" in df.columns:
                            st.session_state.date_col = "date"
                        else:
                            st.error("❌ 未找到该股票信息请重新输入")
                            break

                        st.session_state.stock_df = df
                        st.success(f"✅ 成功获取 {st.session_state.symbol} 的数据（共{len(df)}条）！")
                        break

                    # 单独处理网络错误（不与股票代码错误混淆）
                    except (ConnectionError, Timeout):
                        retry_count += 1
                        wait_time = 2 * retry_count
                        st.warning(f"⚠️ 网络连接超时，{wait_time}秒后第{retry_count+1}次重试...")
                        time.sleep(wait_time)

                    # 处理股票代码错误（AKshare接口特定错误信息）
                    except Exception as e:
                        error_msg = str(e).lower()  # 转为小写，避免大小写匹配问题
                        # 匹配AKshare返回的"股票代码不存在"相关错误
                        if any(keyword in error_msg for keyword in ["不存在", "无效", "invalid", "not exist"]):
                            st.error(f"❌ 股票代码 {st.session_state.symbol} 不存在，请重新输入！")
                        else:
                            st.error(f"❌ 数据获取失败：{str(e)}")
                        break
                if retry_count >= max_retries:
                    st.error("❌ 多次网络请求失败，请检查网络或稍后再试！")

        # 显示个股数据（用动态日期列名，避免KeyError）
        if st.session_state.stock_df is not None:
            df = st.session_state.stock_df
            # 按日期倒序显示（用检测到的日期列名）
            df_sorted = df.sort_values(by=st.session_state.date_col, ascending=False)
            st.dataframe(df_sorted, height=500, use_container_width=True)

            # 显示数据统计（用动态日期列名）
            with st.expander("📊 数据统计摘要"):
                date_min = df[st.session_state.date_col].min()
                date_max = df[st.session_state.date_col].max()
                highest = df["最高"].max() if "最高" in df.columns else "未知"
                lowest = df["最低"].min() if "最低" in df.columns else "未知"
                total_volume = df["成交量"].sum() / 10000 if "成交量" in df.columns else 0

                st.write(f"**时间范围**：{date_min} ~ {date_max}")
                st.write(f"**最高股价**：{highest:.2f} 元" if highest != "未知" else "**最高股价**：数据缺失")
                st.write(f"**最低股价**：{lowest:.2f} 元" if lowest != "未知" else "**最低股价**：数据缺失")
                st.write(f"**总成交量**：{total_volume:.2f} 万手")

