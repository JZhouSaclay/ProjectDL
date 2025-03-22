import streamlit as st
from transformers import pipeline


def main():
    st.title("Hugging Face Transformer 示例")
    st.write(
        "这是一个简单的情感分析示例，使用 Hugging Face Transformers 的 pipeline 实现。"
    )

    # 用户输入文本
    user_input = st.text_area(
        "请输入文本：", "我非常喜欢使用 Hugging Face 的 Transformers！"
    )

    # 当用户点击按钮时，加载情感分析模型并预测
    if st.button("开始情感分析"):
        # 加载预训练情感分析 pipeline（模型会自动下载）
        sentiment_analyzer = pipeline("sentiment-analysis")
        result = sentiment_analyzer(user_input)

        st.subheader("预测结果")
        st.write(result)


if __name__ == "__main__":
    main()
