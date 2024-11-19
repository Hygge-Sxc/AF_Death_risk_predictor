# app.py

import streamlit as st
import joblib
import pandas as pd

# 设置页面配置
st.set_page_config(
    page_title="住院期间死亡风险预测",
    layout="centered",
    initial_sidebar_state="expanded",
)

# 标题
st.title("🏥 住院期间死亡风险预测应用")

# 描述
st.markdown("""
该应用程序使用**随机森林**、**XGBoost**和**CatBoost**三种机器学习模型，预测住院期间患者死亡的风险。
请在下方输入相应的特征值，然后点击“预测”按钮查看结果。
""")

# 定义特征输入
st.header("📝 输入患者特征")

def user_input_features():
    age = st.number_input('年龄 (岁)', min_value=0, max_value=120, value=50)
    sex = st.selectbox('性别', options=['女性 (0)', '男性 (1)'], index=1)
    cp = st.selectbox('胸痛类型 (cp)', options=[
        '典型心绞痛 (1)',
        '非典型心绞痛 (2)',
        '非心绞痛性疼痛 (3)',
        '无症状 (4)'
    ], index=0)
    trestbps = st.number_input('静息血压 (mm Hg)', min_value=0, value=120)
    chol = st.number_input('血清胆固醇 (mg/dl)', min_value=0, value=200)
    fbs = st.selectbox('空腹血糖 > 120 mg/dl (fbs)', options=['否 (0)', '是 (1)'], index=0)
    restecg = st.selectbox('静息心电图结果 (restecg)', options=[
        '正常 (0)',
        'ST-T波异常 (1)',
        '左室肥厚 (2)'
    ], index=0)
    thalach = st.number_input('最大心率 (thalach)', min_value=0, value=150)
    exang = st.selectbox('运动引起的心绞痛 (exang)', options=['否 (0)', '是 (1)'], index=0)
    oldpeak = st.number_input('运动相对于静息的ST下降 (oldpeak)', min_value=0.0, value=1.0, format="%.1f")
    slope = st.selectbox('ST段的坡度 (slope)', options=[
        '上升 (1)',
        '平坦 (2)',
        '下降 (3)'
    ], index=0)
    ca = st.selectbox('主要血管数目 (ca)', options=['0', '1', '2', '3', '4'], index=0)
    thal = st.selectbox('地中海贫血 (thal)', options=[
        '正常 (3)',
        '固定缺陷 (6)',
        '可逆缺陷 (7)',
        '未知 (-1)'
    ], index=0)

    # 将选择的字符串转换为数值
    sex = int(sex.split('(')[-1].strip(')'))
    cp = int(cp.split('(')[-1].strip(')'))
    fbs = int(fbs.split('(')[-1].strip(')'))
    restecg = int(restecg.split('(')[-1].strip(')'))
    exang = int(exang.split('(')[-1].strip(')'))
    slope = int(slope.split('(')[-1].strip(')'))
    
    thal_option = thal.split('(')[-1].strip(')')
    if thal_option == '-1':
        thal = 0  # 例如，将未知转换为0，或根据实际数据调整
    else:
        thal = int(thal_option)
    
    # 将 'ca' 从字符串转换为整数
    ca = int(ca)

    data = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }

    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# 显示输入的特征
st.subheader('输入的特征')
st.write(input_df)

# 加载模型的函数并缓存
@st.cache_resource
def load_models():
    try:
        rf_model = joblib.load('random_forest_model.pkl')
        xgb_model = joblib.load('xgboost_model.pkl')
        cb_model = joblib.load('catboost_model.pkl')
        return rf_model, xgb_model, cb_model
    except Exception as e:
        st.error(f"加载模型时出错: {e}")
        return None, None, None

# 加载模型
rf_model, xgb_model, cb_model = load_models()

# 如果模型加载失败，停止执行
if rf_model is None or xgb_model is None or cb_model is None:
    st.stop()

# 预测函数（返回预测类别和概率）
def predict_with_proba(models, input_data):
    predictions = {}
    probabilities = {}
    for name, model in models.items():
        pred = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0][1]  # 预测为1的概率
        predictions[name] = '死亡' if pred == 1 else '存活'
        probabilities[name] = proba
    return predictions, probabilities

# 预测按钮
if st.button('预测 🔍'):
    models = {
        '随机森林': rf_model,
        'XGBoost': xgb_model,
        'CatBoost': cb_model
    }
    predictions, probabilities = predict_with_proba(models, input_df)
    st.subheader('预测结果')
    for model_name in models.keys():
        result = predictions[model_name]
        proba = probabilities[model_name]
        st.write(f"**{model_name} 预测结果**: {result} (死亡概率: {proba:.2%})")