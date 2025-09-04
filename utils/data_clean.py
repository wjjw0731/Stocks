import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

#清洗获取实盘数据
def clean1(df):
    df.fillna(value=0, inplace=True)
    df.drop_duplicates(keep='first', inplace=True)
    df['日期'] = pd.to_datetime(df['日期'])

    return df

# 清洗特征工程后的数据
def clean2(df):
    df.bfill(inplace=True)
    df.drop_duplicates(keep='first', inplace=True)

    # 标准化
    features = ['开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌幅', '涨跌额',
    '换手率', 'MA_5', 'MA_20', 'MA_60', 'EMA_12', 'EMA_26', 'MACD',
    'MACD_Signal', 'MACD_Histogram', 'RSI_14', 'Daily_Return',
    'Volatility_20D', 'BB_Middle', 'BB_Upper', 'BB_Lower', 'ATR_14', 'OBV']

    df[features] = df.groupby('股票代码')[features].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-8)
    )

    le = joblib.load('/data/wangjiawei/Downloads/stock_encoder.pkl')
    df['股票代码'] = le.fit_transform(df['股票代码'])

    return df
