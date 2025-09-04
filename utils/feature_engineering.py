import pandas as pd
import numpy as np

def feature_engineering(df):
    # 计算不同周期的移动平均线
    df['MA_5'] = df['收盘'].rolling(window=5).mean()  # 5日均线
    df['MA_20'] = df['收盘'].rolling(window=20).mean()  # 20日均线
    df['MA_60'] = df['收盘'].rolling(window=60).mean()  # 60日均线

    # 计算指数移动平均线
    df['EMA_12'] = df['收盘'].ewm(span=12, adjust=False).mean()  # 12日EMA
    df['EMA_26'] = df['收盘'].ewm(span=26, adjust=False).mean()  # 26日EMA

    # MACD (指数平滑异同移动平均线)
    df['MACD'] = df['EMA_12'] - df['EMA_26']  # DIF线
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()  # DEA信号线
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']  # MACD柱状图

    # 计算价格变化
    delta = df['收盘'].diff()

    # 分离上涨和下跌
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)

    # 计算平均增益和平均损失 (通常使用14天周期)
    period = 14
    avg_gain = up.ewm(com=period - 1, adjust=False).mean()
    avg_loss = down.ewm(com=period - 1, adjust=False).mean()

    # 计算RSI
    rs = avg_gain / avg_loss
    df['RSI_14'] = 100 - (100 / (1 + rs))

    # 计算日收益率
    df['Daily_Return'] = df['收盘'].pct_change()

    # 计算20日历史波动率（年化）
    df['Volatility_20D'] = df['Daily_Return'].rolling(window=20).std() * np.sqrt(252)  # 252个交易日

    # 计算中轨（20日MA）、上轨和下轨
    window = 20
    df['BB_Middle'] = df['收盘'].rolling(window=window).mean()
    bb_std = df['收盘'].rolling(window=window).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)

    # 计算TR（真实波幅）
    high_low = df['最高'] - df['最低']
    high_close = np.abs(df['最高'] - df['收盘'].shift())
    low_close = np.abs(df['最低'] - df['收盘'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    # 计算14日ATR
    df['ATR_14'] = tr.rolling(window=14).mean()

    # 计算OBV
    obv = (np.sign(df['收盘'].diff()) * df['成交量']).fillna(0).cumsum()
    df['OBV'] = obv
    return df