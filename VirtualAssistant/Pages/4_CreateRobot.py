if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import streamlit as st
import json


@st.dialog("创建角色虚拟助手")
def vote(select_model,StringBotName):
    st.write(f"确定要创建{select_model}嘛?")
    st.write(f"你的助手名字是{StringBotName}")
    col1, col2 , col3 , col4 = st.columns(4)
    with col1:
        yes_button = st.button("确定",use_container_width=True)
    with col4:
        no_button = st.button("取消", use_container_width=True,type="primary")
    if yes_button:
        # 加载JSON文件
        with open('./LLM/History.json', 'r') as f:
            data = json.load(f)
        my_robot = {'name':StringBotName,'model':select_model,"description": "没有描述...","image": "./imgs/test_img.png"}
        data["robot"].append(my_robot)
        with open('./LLM/History.json', 'w') as f:
            json.dump(data, f, indent=2)
        st.rerun()
    if no_button:
        st.rerun()


def Get_path(dir="./LLM"):
    # 将相对路径转换为绝对路径
    dir = os.path.abspath(dir)
    print(dir)
    # 获取目录中的子目录
    return [file for file in os.listdir(dir) if os.path.isdir(os.path.join(dir, file))]

options = Get_path()

st.title("创建助手🤖")

st.write("创建属于自己的角色虚拟助手")


StringBotName = st.text_input("输入你的助手名字")

select_model = st.selectbox("请选择你要创建的模型" + ':star2:', options, index=0)

data = st.file_uploader("上传你的聊天背景")

StringPrompt = st.text_area("输入你的提示词")

create_button = st.button("开始创建")


if create_button:
    if StringBotName:
        if "vote" not in st.session_state:
            vote(select_model,StringBotName)
    else:
        st.error('助手名字不能为空')
