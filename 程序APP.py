import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt

# =======================
# 1. 模型加载
# =======================
model = joblib.load('RF-GBDT.pkl')  # 加载RF-GBDT模型

st.title("Disease Prediction System (RF-GBDT Model)")
st.markdown("### 输入患者临床及影像学特征，预测是否患病")

# =======================
# 2. 定义特征输入
# =======================

st.sidebar.header("输入特征值")

# ---- 临床数据 ----
st.sidebar.subheader("Clinical Features")

PR = st.sidebar.number_input("PR", min_value=0, value=1, step=1)
Family_history = st.sidebar.number_input("Family history", min_value=0, value=0, step=1)
Retraction_phenomenon = st.sidebar.number_input("Retraction phenomenon", min_value=0, value=1, step=1)
US_reported_ALN_status = st.sidebar.selectbox(
    "US-reported ALN status",
    options=[0, 1],
    index=0,
    format_func=lambda x: "Negative (0)" if x == 0 else "Positive (1)"
)
Adler_grade = st.sidebar.number_input("Adler grade", min_value=0, value=3, step=1)

# ---- 影像数据 ----
st.sidebar.subheader("Imaging Features")

lbp_3D_m1_glcm_ClusterShade = st.sidebar.number_input("lbp_3D_m1_glcm_ClusterShade", value=-10.372358, format="%.6f")
wavelet_LHH_gldm_DependenceEntropy = st.sidebar.number_input("wavelet_LHH_gldm_DependenceEntropy", value=3.436621, format="%.6f")
wavelet_LHH_firstorder_Maximum = st.sidebar.number_input("wavelet_LHH_firstorder_Maximum", value=3.160000e-13, format="%.6e")
wavelet_HHH_gldm_LargeDependenceLowGrayLevelEmphasis = st.sidebar.number_input("wavelet_HHH_gldm_LargeDependenceLowGrayLevelEmphasis", value=12.28125, format="%.6f")
logarithm_ngtdm_Busyness = st.sidebar.number_input("logarithm_ngtdm_Busyness", value=6.421602, format="%.6f")
lbp_3D_k_glrlm_ShortRunLowGrayLevelEmphasis = st.sidebar.number_input("lbp_3D_k_glrlm_ShortRunLowGrayLevelEmphasis", value=0.270797, format="%.6f")
logarithm_glszm_LargeAreaHighGrayLevelEmphasis = st.sidebar.number_input("logarithm_glszm_LargeAreaHighGrayLevelEmphasis", value=4.343142e+06, format="%.6e")
wavelet_LHH_firstorder_90Percentile = st.sidebar.number_input("wavelet_LHH_firstorder_90Percentile", value=1.670000e-13, format="%.6e")
wavelet_LHL_glcm_ClusterProminence = st.sidebar.number_input("wavelet_LHL_glcm_ClusterProminence", value=3.880000e+11, format="%.6e")

# =======================
# 3. 构造输入特征(模型构建时的输入顺序)
# =======================
feature_names = [
    'PR',
    'lbp_3D_m1_glcm_ClusterShade',
    'wavelet_LHH_gldm_DependenceEntropy',
    'wavelet_LHH_firstorder_Maximum',
    'wavelet_HHH_gldm_LargeDependenceLowGrayLevelEmphasis',
    'Family history',
    'Retraction phenomenon',
    'logarithm_ngtdm_Busyness',
    'lbp_3D_k_glrlm_ShortRunLowGrayLevelEmphasis',
    'US-reported ALN status',
    'logarithm_glszm_LargeAreaHighGrayLevelEmphasis',
    'wavelet_LHH_firstorder_90Percentile',
    'wavelet_LHL_glcm_ClusterProminence',
    'Adler grade'
]

input_values = np.array([[
    PR,
    lbp_3D_m1_glcm_ClusterShade,
    wavelet_LHH_gldm_DependenceEntropy,
    wavelet_LHH_firstorder_Maximum,
    wavelet_HHH_gldm_LargeDependenceLowGrayLevelEmphasis,
    Family_history,
    Retraction_phenomenon,
    logarithm_ngtdm_Busyness,
    lbp_3D_k_glrlm_ShortRunLowGrayLevelEmphasis,
    US_reported_ALN_status,
    logarithm_glszm_LargeAreaHighGrayLevelEmphasis,
    wavelet_LHH_firstorder_90Percentile,
    wavelet_LHL_glcm_ClusterProminence,
    Adler_grade
]])

# =======================
# 4. 模型预测
# =======================
if st.button("开始预测"):
    prediction = model.predict(input_values)[0]
    probas = model.predict_proba(input_values)[0]

    st.markdown(f"### 🩺 预测结果: {'患病' if prediction == 1 else '未患病'}")
    st.write(f"**预测概率:** {probas}")

    # =======================
    # 5. 结果解释与建议
    # =======================
    prob = probas[prediction] * 100
    if prediction == 1:
        advice = (
            f"模型预测您患病的概率为 **{prob:.2f}%**。建议尽快进行进一步临床检查和医生咨询，"
            f"确保获得准确的诊断和个性化治疗方案。"
        )
    else:
        advice = (
            f"模型预测您健康的概率为 **{prob:.2f}%**。请继续保持良好的生活习惯，"
            f"定期体检以确保持续健康。"
        )

    st.info(advice)

    # =======================
    # 6. 可视化预测概率
    # =======================
    plt.figure(figsize=(6, 3))
    bars = plt.barh(['Not sick', 'sick'], [probas[0], probas[1]], color=['#2E86C1', '#E74C3C'])
    plt.xlabel("Predicted probability")
    for i, v in enumerate(probas):
        plt.text(v + 0.001, i, f"{v:.3f}", va='center', fontweight='bold')
    plt.xlim(0, 1)
    plt.tight_layout()
    st.pyplot(plt)

