import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.sidebar.title("Pages")

# 如果 session_state 还没有 page 变量，则默认设为 "主页"
if "page" not in st.session_state:
    st.session_state["page"] = "Page principale"

# 创建导航栏
page = st.sidebar.radio(
    "Sélection de page",
    ["Page principale", "page secondaire", "model"],
    index=["Page principale", "page secondaire", "model"].index(
        st.session_state["page"]
    ),
)

# 更新 session_state 以保持页面状态
st.session_state["page"] = page

# 显示对应的页面内容
if page == "Page principale":
    st.title("Hello, Streamlit!")
    st.write("Test 1 de Streamlit")

    st.title("Application de Transformer sans Transformer")
    st.header("M1 Math&IA 2024-2025")
    st.subheader("Projet de Deep Learning")
    st.text("Jie ZHOU & Youwei ZHEN")
    st.markdown("**Y'a rien encore**")

    name = st.text_input("Entrez votre nom:")
    st.write("Bonjour,", name)

    if st.button("Refresh"):
        st.write("Arrêtez de me cliquer dessus!")
        np.random.seed(42)
        data = np.random.randn(100, 3)
        df = pd.DataFrame(data, columns=["A", "B", "C"])

        st.line_chart(df)
        st.bar_chart(df)

        fig, ax = plt.subplots()
        ax.hist(data[:, 0], bins=20)
        st.pyplot(fig)


elif page == "page secondaire":
    st.write("# Deuxième page 📊")
elif page == "model":
    st.write("# model 🤖")
