import streamlit as st

sidebar_logo = "../imgs/img_title_large.png"
main_body_logo = "../imgs/img_title_large.png"

st.logo(sidebar_logo, icon_image=main_body_logo,size="large")

pages = {
    "首页": [
        st.Page("Pages/HomePages.py", title="介绍", icon="👨‍💻"),
 ],
    "基础教学-数据可视化": [
        st.Page("Pages/1_GraphVisual.py", title="计算图可视化", icon="🔠"),
        st.Page("Pages/2_Lr_Iters.py", title="学习率和迭代次数可视化", icon="📊"),
        st.Page("Pages/3_GradDownVisual.py", title="梯度下降可视化", icon="📈"),
    ],
    "大模型教程": [
        st.Page("Pages/7_Colab.py", title="Colab微调教程", icon="🧰"),
        st.Page("Pages/8_ShowLLM.py", title="低参数大模型对比", icon="🖥️"),
    ],
    "应用-虚拟助手": [
        st.Page("Pages/4_CreateRobot.py", title="创建助手", icon="🤖"),
        st.Page("Pages/5_BeginChats.py", title="开始聊天", icon="🤗"),
        st.Page("Pages/6_Explore.py", title="探索助手", icon="🥳"),
    ],
}

pg = st.navigation(pages)
pg.run()
