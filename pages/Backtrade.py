import streamlit as st
import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime
import time
from requests.exceptions import ConnectionError, Timeout
# 导入自定义工具函数（需确保utils文件夹存在对应文件）
from utils import feature_engineering, data_clean


# 计算最大回撤（风险指标）
def calculate_max_drawdown(return_series):
    if len(return_series) < 2:
        return 0.0
    peak_series = return_series.cummax()  # 历史峰值
    drawdown_series = (return_series - peak_series) / peak_series  # 每日回撤
    return round(drawdown_series.min() * 100, 2)  # 最大回撤百分比


# 计算策略胜率（盈利交易占比）
def calculate_win_rate(signal_df):
    signal_points = signal_df[signal_df["买卖信号"] != 0].copy()
    if len(signal_points) < 2:
        return 0.0

    win_count = 0
    total_trades = 0
    buy_price = None

    for _, row in signal_points.iterrows():
        if row["买卖信号"] == 1:
            buy_price = row["收盘"]
        elif row["买卖信号"] == -1 and buy_price is not None:
            total_trades += 1
            if row["收盘"] > buy_price:
                win_count += 1
            buy_price = None

    return round((win_count / total_trades) * 100, 2) if total_trades > 0 else 0.0


def show():
    # ---------------------- 1. 初始化会话状态 ----------------------
    session_vars = {
        "symbol": "600000",  # 默认股票：浦发银行
        "start_date": datetime(2020, 1, 1),
        "end_date": datetime.now(),
        "stock_df": None,  # 原始数据
        "date_col": "日期",
        "feature_df": None,  # 特征工程数据
        "df_clean": None,  # 清洗后数据
        "df": None,  # 最终回测数据（含信号+收益）
        "backtest_result": None  # 回测指标
    }
    for key, value in session_vars.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # ---------------------- 2. 页面标题与说明 ----------------------
    with st.container():
        st.title("📊 A股MACD策略回测平台（完整收益版）")
        st.markdown("""
        ### 📝 策略规则
        - **买入**：MACD金叉（MACD线从下向上穿越信号线）
        - **卖出**：MACD死叉（MACD线从上向下穿越信号线）
        - **仓位**：全仓买入/全额清仓（按当日收盘价交易）
        - **数据**：后复权日线数据（含分红/拆股，保证收益真实性）
        """)
        st.markdown("---")

    # ---------------------- 3. 回测参数配置 ----------------------
    st.subheader("1. 回测参数配置")
    with st.container():
        col1, col2 = st.columns([1, 2], gap="medium")

        # 股票代码输入
        with col1:
            st.session_state.symbol = st.text_input(
                "📌 股票代码",
                value=st.session_state.symbol,
                help="示例：600000（浦发）、000858（五粮液）",
                placeholder="6位A股代码",
                max_chars=6
            )

        # 日期选择
        with col2:
            date_col1, date_col2 = st.columns(2)
            with date_col1:
                st.session_state.start_date = st.date_input(
                    "开始日期",
                    value=st.session_state.start_date,
                    min_value=datetime(2000, 1, 1),
                    max_value=datetime.now() - pd.Timedelta(days=1)
                )
            with date_col2:
                st.session_state.end_date = st.date_input(
                    "结束日期",
                    value=st.session_state.end_date,
                    min_value=st.session_state.start_date,
                    max_value=datetime.now()
                )

        start_str = st.session_state.start_date.strftime("%Y%m%d")
        end_str = st.session_state.end_date.strftime("%Y%m%d")

    # ---------------------- 4. 获取个股数据 ----------------------
    st.subheader("2. 获取个股历史数据")
    with st.container(border=True):
        get_data_btn = st.button("📥 点击获取数据", type="primary", use_container_width=True)

        if get_data_btn:
            with st.spinner(f"获取 {st.session_state.symbol} 数据中..."):
                st.session_state.stock_df = None
                retry_count = 0
                max_retries = 3
                success = False

                while retry_count < max_retries and not success:
                    try:
                        # 获取后复权数据
                        raw_df = ak.stock_zh_a_hist(
                            symbol=st.session_state.symbol,
                            period="daily",
                            start_date=start_str,
                            end_date=end_str,
                            adjust="hfq"
                        )

                        # 数据校验
                        if len(raw_df) == 0:
                            st.error("未获取到数据，请检查代码或日期范围")
                            break
                        if len(raw_df) < 30:
                            st.warning(f"数据仅{len(raw_df)}条（建议30条以上）")

                        # 标准化日期列
                        if "date" in raw_df.columns:
                            raw_df.rename(columns={"date": "日期"}, inplace=True)
                        if "日期" not in raw_df.columns:
                            st.error("数据缺少日期列，请重试")
                            break
                        raw_df["日期"] = pd.to_datetime(raw_df["日期"])
                        raw_df = raw_df.sort_values("日期").reset_index(drop=True)

                        # 存储并展示结果
                        st.session_state.stock_df = raw_df
                        st.success(
                            f"数据获取成功！\n"
                            f"时间范围：{raw_df['日期'].min().strftime('%Y-%m-%d')} ~ {raw_df['日期'].max().strftime('%Y-%m-%d')}\n"
                            f"记录数：{len(raw_df)} 条"
                        )
                        with st.expander("查看原始数据（前10行）"):
                            st.dataframe(
                                raw_df[["日期", "开盘", "最高", "最低", "收盘", "成交量"]].head(10),
                                hide_index=True
                            )
                        success = True

                    except (ConnectionError, Timeout):
                        retry_count += 1
                        wait = 2 * retry_count
                        if retry_count < max_retries:
                            st.warning(f"网络超时，{wait}秒后第{retry_count + 1}次重试...")
                            time.sleep(wait)
                        else:
                            st.error("3次重试失败，请检查网络")
                    except Exception as e:
                        error_msg = str(e).lower()
                        if any(k in error_msg for k in ["不存在", "无效"]):
                            st.error(f"代码 {st.session_state.symbol} 无效")
                        elif "无数据" in error_msg:
                            st.error("该时段无数据，请调整日期")
                        else:
                            st.error(f"错误：{str(e)}")
                        break

    # ---------------------- 5. 执行回测 ----------------------
    st.subheader("3. 执行回测（含收益分析）")
    with st.container(border=True):
        backtest_btn = st.button(
            "🚀 开始回测",
            type="primary",
            use_container_width=True,
            disabled=st.session_state.stock_df is None
        )

        if backtest_btn:
            try:
                # 步骤1：数据清洗
                st.write("🔧 步骤1/4：数据清洗...")
                df_clean = data_clean.clean1(st.session_state.stock_df)
                st.session_state.df_clean = df_clean
                if len(df_clean) < 30:
                    st.error("数据不足30条，无法计算MACD")
                    raise Exception("数据量不足")

                # 步骤2：特征工程（计算MACD）
                st.write("🔧 步骤2/4：计算MACD指标...")
                feature_df = feature_engineering.feature_engineering(df_clean)
                st.session_state.feature_df = feature_df

                # 步骤3：二次清洗
                st.write("🔧 步骤3/4：数据标准化...")
                df = data_clean.clean2(feature_df)
                st.session_state.df = df

                # 验证必要列
                required_cols = ["MACD", "MACD_Signal", "日期", "收盘"]
                missing = [c for c in required_cols if c not in df.columns]
                if missing:
                    st.error(f"缺少必要列：{', '.join(missing)}")
                    raise Exception("数据格式错误")

                # 步骤4：计算信号与收益
                st.write("📊 步骤4/4：计算信号与收益...")
                df["买卖信号"] = 0  # 0=无，1=买，-1=卖
                df["仓位"] = 0  # 0=空仓，1=满仓
                df["每日收益%"] = 0.0
                df["累计收益倍数"] = 1.0

                # 计算买卖信号
                df.loc[
                    (df["MACD"] > df["MACD_Signal"]) &
                    (df["MACD"].shift(1) <= df["MACD_Signal"].shift(1)),
                    "买卖信号"
                ] = 1  # 买入信号

                df.loc[
                    (df["MACD"] < df["MACD_Signal"]) &
                    (df["MACD"].shift(1) >= df["MACD_Signal"].shift(1)),
                    "买卖信号"
                ] = -1  # 卖出信号

                # 计算仓位与收益
                for i in range(1, len(df)):
                    prev = df.iloc[i - 1]
                    curr_idx = df.index[i]

                    # 更新仓位
                    if prev["仓位"] == 0:
                        df.at[curr_idx, "仓位"] = 1 if df.iloc[i]["买卖信号"] == 1 else 0
                    else:
                        df.at[curr_idx, "仓位"] = 0 if df.iloc[i]["买卖信号"] == -1 else 1

                    # 计算收益
                    if df.at[curr_idx, "仓位"] == 1:
                        daily_return = (df.iloc[i]["收盘"] - prev["收盘"]) / prev["收盘"] * 100
                        df.at[curr_idx, "每日收益%"] = round(daily_return, 2)
                        df.at[curr_idx, "累计收益倍数"] = round(
                            prev["累计收益倍数"] * (1 + daily_return / 100), 4
                        )
                    else:
                        df.at[curr_idx, "累计收益倍数"] = prev["累计收益倍数"]

                # 计算核心指标
                total_return = (df["累计收益倍数"].iloc[-1] - 1) * 100
                max_dd = calculate_max_drawdown(df["累计收益倍数"])
                win_rate = calculate_win_rate(df)
                signal_counts = df["买卖信号"].value_counts().sort_index()
                buy_cnt = signal_counts.get(1, 0)
                sell_cnt = signal_counts.get(-1, 0)

                st.session_state.backtest_result = {
                    "总收益率(%)": round(total_return, 2),
                    "最大回撤(%)": max_dd,
                    "胜率(%)": win_rate,
                    "买入信号": buy_cnt,
                    "卖出信号": sell_cnt,
                    "完整交易": min(buy_cnt, sell_cnt),
                    "累计收益倍数": df["累计收益倍数"].iloc[-1]
                }

                st.success("✅ 回测完成！")
                with st.expander("查看特征工程数据（前5行）"):
                    st.dataframe(
                        df[["日期", "收盘", "MACD", "MACD_Signal", "买卖信号"]].head(),
                        hide_index=True
                    )

            except Exception as e:
                st.error(f"回测失败：{str(e)}")

    # ---------------------- 6. 回测结果展示 ----------------------
    if st.session_state.backtest_result is not None and st.session_state.df is not None:
        st.subheader("4. 回测结果分析")
        result = st.session_state.backtest_result
        df = st.session_state.df

        # 核心指标卡片
        with st.container(border=True):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(
                    "总收益率",
                    f"{result['总收益率(%)']}%",
                    f"本金100元→{100 + result['总收益率(%)']:.2f}元"
                )
            with col2:
                st.metric("最大回撤", f"{result['最大回撤(%)']}%")
            with col3:
                st.metric("胜率", f"{result['胜率(%)']}%")
            with col4:
                st.metric("完整交易次数", result['完整交易'])

        # 收益曲线与价格曲线
        with st.container(border=True):
            st.subheader("累计收益曲线 vs 股票价格")
            chart_df = df.set_index("日期")[["收盘", "累计收益倍数"]].copy()
            # 价格标准化（便于同图对比）
            chart_df["收盘标准化"] = chart_df["收盘"] / chart_df["收盘"].iloc[0]
            st.line_chart(
                chart_df[["收盘标准化", "累计收益倍数"]],
                y_label="倍数（初始值=1）",
                height=400
            )
            st.caption("""
            蓝色：股票价格（标准化为初始值=1）  
            橙色：策略累计收益（初始值=1，代表本金）  
            注：价格标准化仅用于视觉对比，不影响实际收益计算
            """)

        # MACD指标与买卖信号
        with st.container(border=True):
            st.subheader("MACD指标与买卖信号")
            macd_df = df.set_index("日期")[["MACD", "MACD_Signal", "收盘"]]
            st.line_chart(macd_df, height=400)

            # 标记买卖信号点
            buy_signals = df[df["买卖信号"] == 1]
            sell_signals = df[df["买卖信号"] == -1]

            st.caption(f"""
            蓝色：收盘价 | 橙色：MACD线 | 绿色：MACD信号线  
            买入信号：{len(buy_signals)} 个 | 卖出信号：{len(sell_signals)} 个
            """)

        # 交易详情表
        with st.container(border=True):
            st.subheader("交易信号详情")
            signal_df = df[df["买卖信号"] != 0].copy()
            signal_df["信号类型"] = signal_df["买卖信号"].map({1: "买入", -1: "卖出"})
            st.dataframe(
                signal_df[["日期", "收盘", "MACD", "MACD_Signal", "信号类型"]],
                hide_index=True,
                use_container_width=True
            )

        # 收益统计
        with st.container(border=True):
            st.subheader("收益分布")
            profit_days = df[df["每日收益%"] > 0]
            loss_days = df[df["每日收益%"] < 0]
            st.dataframe({
                "类别": ["盈利交易日", "亏损交易日", "平均每日盈利", "平均每日亏损"],
                "数据": [
                    f"{len(profit_days)} 天",
                    f"{len(loss_days)} 天",
                    f"{profit_days['每日收益%'].mean():.2f}%",
                    f"{loss_days['每日收益%'].mean():.2f}%"
                ]
            }, hide_index=True)


if __name__ == "__main__":
    show()
