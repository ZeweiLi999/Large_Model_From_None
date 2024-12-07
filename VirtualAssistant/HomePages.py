import streamlit as st

st.title("Welcome to LearnTorch")
st.write("创建属于自己的角色虚拟助手")

# 检查是否已经在st.session_state中存在assistants_dict，如果不存在则初始化一个空字典
if "assistants_dict" not in st.session_state:
    st.session_state.assistants_dict = {}

if "is_creating" not in st.session_state:
    st.session_state.is_creating = False

if st.button("开始创建"):
    st.session_state.is_creating = True

if st.session_state.is_creating:
    StringBotName = st.text_input("输入你的助手名字")
    Instruction = st.text_input("备注")

    if st.button("添加助手"):
        if StringBotName:
            st.session_state.assistants_dict[StringBotName] = Instruction
            st.success(f"添加{StringBotName}成功！")
            st.session_state.is_creating = False
        else:
            st.warning("请输入助手名字")

st.sidebar.title("LearnTorch")
