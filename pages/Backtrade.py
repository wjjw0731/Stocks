import streamlit as st
import akshare as ak
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import time
from requests.exceptions import ConnectionError, Timeout
# 导入自定义工具函数（需确保utils文件夹存在对应文件）
from utils import feature_engineering, data_clean, predict_signal

plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示异常

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
                model_paths = {
                    "static": "../model//model1_static_lgb.pkl",
                    "time": "../model//model2_time_lgb.pkl",
                    "meta": "../model//meta_model_logistic.pkl"
                }

                models = predict_signal.load_models(model_paths)
                # print(st.session_state.df.columns)
                static_features = ['开盘', '收盘', '最高', '最低', '成交量', '换手率',
                                   'MA_5', 'MA_20', 'MACD', 'MACD_Signal', 'RSI_14',
                                   'Volatility_20D', 'BB_Middle', 'ATR_14', 'OBV']

                time_features = ['收盘_5d_mean', '成交量_5d_mean', 'MACD_5d_mean', 'RSI_14_5d_mean',
                                 '最高_5d_max', '最低_5d_min', '股票代码']

                # st.dataframe(st.session_state.df.head(10), hide_index=True)

                df_signal = predict_signal.predict_signal(st.session_state.df, static_features, time_features, models["static"], models["time"], models["meta"])
                st.session_state.df_signal = df_signal

                # st.dataframe(st.session_state.df_signal.head(10), hide_index=True)

            except Exception as e:
                st.error(f"回测失败：{str(e)}")

            st.write("🔧 步骤5/5：执行回测与收益计算...")
            initial_capital = st.text_input("初始资金：", "100000")  # 给个默认值
            try:
                st.session_state.initial_capital = float(initial_capital)
            except ValueError:
                st.error("请输入合法的数字作为初始资金")
                st.stop()

            # 初始化回测列
            st.session_state.df_signal["买卖信号"] = st.session_state.df_signal["pred_signal"]
            st.session_state.df_signal["仓位"] = 0  # 0=空仓，1=满仓
            st.session_state.df_signal["每日收益%"] = 0.0
            st.session_state.df_signal["累计收益倍数"] = 1.0
            st.session_state.df_signal["资金余额"] = st.session_state.initial_capital
            st.session_state.df_signal["持仓数量"] = 0
            st.session_state.df_signal["持仓价值"] = 0

            # 执行回测逻辑
            for i in range(1, len(st.session_state.df_signal)):
                prev = st.session_state.df_signal.iloc[i - 1]
                curr = st.session_state.df_signal.iloc[i]
                curr_idx = st.session_state.df_signal.index[i]
                close_price = curr["收盘"]

                # 更新仓位
                if prev["仓位"] == 0:
                    # 空仓状态
                    if curr["买卖信号"] == 1:
                        # 全仓买入
                        position = int(prev["资金余额"] / close_price)
                        st.session_state.df_signal.at[curr_idx, "仓位"] = 1
                        st.session_state.df_signal.at[curr_idx, "持仓数量"] = position
                        st.session_state.df_signal.at[curr_idx, "持仓价值"] = position * close_price
                        st.session_state.df_signal.at[curr_idx, "资金余额"] = prev["资金余额"] - (position * close_price)
                    else:
                        # 保持空仓
                        st.session_state.df_signal.at[curr_idx, "仓位"] = 0
                        st.session_state.df_signal.at[curr_idx, "持仓数量"] = 0
                        st.session_state.df_signal.at[curr_idx, "持仓价值"] = 0
                        st.session_state.df_signal.at[curr_idx, "资金余额"] = prev["资金余额"]
                else:
                    # 满仓状态
                    if curr["买卖信号"] == -1:
                        # 清仓卖出
                        st.session_state.df_signal.at[curr_idx, "仓位"] = 0
                        st.session_state.df_signal.at[curr_idx, "资金余额"] = prev["资金余额"] + (prev["持仓数量"] * close_price)
                        st.session_state.df_signal.at[curr_idx, "持仓数量"] = 0
                        st.session_state.df_signal.at[curr_idx, "持仓价值"] = 0
                    else:
                        # 保持持仓
                        st.session_state.df_signal.at[curr_idx, "仓位"] = 1
                        st.session_state.df_signal.at[curr_idx, "持仓数量"] = prev["持仓数量"]
                        st.session_state.df_signal.at[curr_idx, "持仓价值"] = prev["持仓数量"] * close_price
                        st.session_state.df_signal.at[curr_idx, "资金余额"] = prev["资金余额"]

                # 计算收益指标
                total_asset_prev = prev["资金余额"] + prev["持仓价值"]
                total_asset_curr = st.session_state.df_signal.at[curr_idx, "资金余额"] + st.session_state.df_signal.at[curr_idx, "持仓价值"]
                daily_return = (
                                           total_asset_curr - total_asset_prev) / total_asset_prev * 100 if total_asset_prev > 0 else 0
                st.session_state.df_signal.at[curr_idx, "每日收益%"] = round(daily_return, 2)
                st.session_state.df_signal.at[curr_idx, "累计收益倍数"] = round(
                    prev["累计收益倍数"] * (1 + daily_return / 100), 4
                )

            # 计算核心回测指标
            final_asset = st.session_state.df_signal["资金余额"].iloc[-1] + st.session_state.df_signal["持仓价值"].iloc[-1]
            total_return = (final_asset - st.session_state.initial_capital) / st.session_state.initial_capital * 100
            max_dd = calculate_max_drawdown(df["累计收益倍数"])
            win_rate = calculate_win_rate(st.session_state.df_signal)
            signal_counts = st.session_state.df_signal["买卖信号"].value_counts().sort_index()
            buy_cnt = signal_counts.get(1, 0)
            sell_cnt = signal_counts.get(-1, 0)

            st.session_state.backtest_result = {
                "总收益率(%)": round(total_return, 2),
                "最大回撤(%)": max_dd,
                "胜率(%)": win_rate,
                "买入信号": buy_cnt,
                "卖出信号": sell_cnt,
                "完整交易": min(buy_cnt, sell_cnt),
                "初始资金(元)": st.session_state.initial_capital,
                "最终资产(元)": round(final_asset, 2)
            }

            st.success("✅ 回测完成！")
            with st.expander("查看回测数据样例（前5行）"):
                st.dataframe(
                    st.session_state.df_signal[["日期", "股票代码", "收盘", "pred_signal", "仓位", "资金余额", "累计收益倍数"]].head(),
                    hide_index=True
                )

        # 4. 结果可视化
        if st.session_state.backtest_result is not None and st.session_state.df is not None:
            st.subheader("4. 回测结果分析")
            df = st.session_state.df_signal
            result = st.session_state.backtest_result

            # 核心指标卡片
            with st.container(border=True):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("总收益率", f"{result['总收益率(%)']}%")
                with col2:
                    st.metric("最大回撤", f"{result['最大回撤(%)']}%")
                with col3:
                    st.metric("胜率", f"{result['胜率(%)']}%")
                with col4:
                    st.metric("最终资产", f"¥{result['最终资产(元)']:.2f}")

            # 图表1：股价+信号标记
            st.write("### 📈 股价走势与模型信号")
            fig1, ax1 = plt.subplots(figsize=(12, 6))
            ax1.plot(df["日期"], df["收盘"], color="#1f77b4", linewidth=1.5, label="Close Price")

            # 标记买入/卖出信号
            buy_signals = df[df["买卖信号"] == 1]
            sell_signals = df[df["买卖信号"] == -1]
            ax1.scatter(buy_signals["日期"], buy_signals["收盘"],
                        color="#2ca02c", marker="^", s=80, label="Buy Signal", zorder=5)
            ax1.scatter(sell_signals["日期"], sell_signals["收盘"],
                        color="#d62728", marker="v", s=80, label="Sell Signal", zorder=5)

            ax1.set_xlabel("日期")
            ax1.set_ylabel("收盘价（元）")
            ax1.set_title(f"Stock Price and Trading Signals({df['日期'].min().strftime('%Y-%m')}-{df['日期'].max().strftime('%Y-%m')})")
            ax1.legend()
            ax1.grid(alpha=0.3)
            plt.xticks(rotation=45)
            st.pyplot(fig1)

            # 图表2：策略收益 vs 持有收益
            st.write("### 📊 策略收益与持有收益对比")
            fig2, ax2 = plt.subplots(figsize=(12, 6))
            df["股价标准化"] = df["收盘"] / df["收盘"].iloc[0]  # 标准化股价
            ax2.plot(df["日期"], df["累计收益倍数"], color="#ff7f0e", linewidth=2, label="Strategy Cumulative Return")
            ax2.plot(df["日期"], df["股价标准化"], color="#1f77b4", linewidth=1.5, linestyle="--", label="Buy-and-Hold Return")

            ax2.set_xlabel("Date")
            ax2.set_ylabel("Return Multiple (Initial=1)")
            ax2.set_title("Strategy vs Buy-and-Hold Returns")
            ax2.legend()
            ax2.grid(alpha=0.3)
            plt.xticks(rotation=45)
            st.pyplot(fig2)

            # 图表3：资金与持仓价值变化
            st.write("### 💰 资金与持仓价值变化")
            fig3, ax3 = plt.subplots(figsize=(12, 6))
            ax3.plot(df["日期"], df["资金余额"], color="#2ca02c", linewidth=2, label="Cash Balance")
            ax3.plot(df["日期"], df["持仓价值"], color="#d62728", linewidth=2, label="Holdings Value")
            ax3.plot(df["日期"], df["资金余额"] + df["持仓价值"],
                     color="#1f77b4", linewidth=2.5, linestyle="--", label="Total Asset")
            ax3.axhline(y=st.session_state.initial_capital, color="#ff7f0e",
                        linestyle=":", linewidth=1.5, label=f"Initial Capital")

            ax3.set_xlabel("Date")
            ax3.set_ylabel("Amount")
            ax3.set_title("Cash and Holdings Value over Time")
            ax3.legend()
            ax3.grid(alpha=0.3)
            plt.xticks(rotation=45)
            st.pyplot(fig3)

            # 图表4：信号分布
            st.write("### 📊 模型信号分布")
            fig4, ax4 = plt.subplots(figsize=(8, 6))
            signal_counts = df["买卖信号"].value_counts()
            labels = ["No Signal", "Buy Signal", "Sell Signal"]
            sizes = [signal_counts.get(0, 0), signal_counts.get(1, 0), signal_counts.get(-1, 0)]
            colors = ["#ffbb78", "#2ca02c", "#d62728"]

            ax4.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%",
                    startangle=90, textprops={"fontsize": 11})
            ax4.set_title(f"Signal Distribution (Total Days: {len(df)})")
            st.pyplot(fig4)

            # 交易详情表
            st.write("### 📋 交易信号详情")
            signal_df = df[df["买卖信号"] != 0].copy()
            signal_df["信号类型"] = signal_df["买卖信号"].map({1: "买入", -1: "卖出"})
            st.dataframe(
                signal_df[["日期", "股票代码", "收盘", "信号类型", "资金余额", "持仓价值"]],
                use_container_width=True
            )

