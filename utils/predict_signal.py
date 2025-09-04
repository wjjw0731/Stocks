import joblib
import streamlit as st
import numpy as np
import os
def load_models(model_paths):
    """加载模型并返回模型字典"""
    models = {}
    for name, path in model_paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"模型文件不存在: {path}")
        try:
            models[name] = joblib.load(path)
            st.success(f"✅ 成功加载模型: {name}")
        except Exception as e:
            raise Exception(f"加载模型 {name} 失败: {str(e)}")
    return models

def predict_signal(df, static_fea, time_fea, model1, model2, meta_model):
    df['pred_signal'] = 0
    for code, group in df.groupby('股票代码'):
        X1 = group[static_fea + ['股票代码']]
        X2 = group[time_fea]
        probs1 = model1.predict_proba(X1)
        probs2 = model2.predict_proba(X2)
        meta_features = np.hstack([probs1, probs2])
        df.loc[group.index, 'pred_signal'] = meta_model.predict(meta_features)

    return df