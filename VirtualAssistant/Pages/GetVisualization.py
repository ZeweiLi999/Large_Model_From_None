import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# 数据可视化函数
def data_visualization(uploaded_file, StringMsg):
    try:
        # 读取上传的文件数据
        data = pd.read_csv(uploaded_file)

        if StringMsg == "展示柱状图":
            st.write("### 柱状图展示")
            plt.bar(data.columns, data.iloc[0])
            st.pyplot(plt)

        elif StringMsg == "展示折线图":
            st.write("### 折线图展示")
            plt.plot(data.columns, data.iloc[0])
            st.pyplot(plt)

        else:
            st.write("暂不支持该类型的可视化请求，请重新输入。")

    except Exception as e:
        st.write(f"读取文件或生成可视化时出错：{e}")


# Streamlit应用主程序
if __name__ == "__main__":
    st.title("数据可视化应用")

    uploaded_file = st.file_uploader("请选择要上传的文件（需为CSV格式）", type="csv")
    StringMsg = st.text_input("请输入可视化请求消息（如：展示柱状图、展示折线图）", key="msg_input")

    if uploaded_file and StringMsg:
        data_visualization(uploaded_file, StringMsg)