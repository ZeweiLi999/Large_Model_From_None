# Explore.py

# Streamlit的页面配置（以下是用于显示侧边栏名称和图标）
# 这些注释会被Streamlit读取
# streamlit_page: name: 探索
# streamlit_page: order: 2
# streamlit_page: icon: 🧭

import streamlit as st

st.sidebar.image(r"C:\Users\LiRunze\Downloads\right_icon_blue-removebg-preview.png", use_container_width=True)  # 替换 "your_image_path.png" 为你的图片路径

# 添加导航菜单
st.sidebar.title("导航菜单")
selected_page = st.sidebar.radio(
    "选择页面",
    ["HomePages", "BeginChats", "Explore", "GetVisualization", "UploadFiles"]
)

# 主界面内容
if selected_page == "HomePages":
    st.title("Welcome to LearnTorch")
    st.write("创建属于自己的角色虚拟助手")
elif selected_page == "BeginChats":
    st.title("Begin Chats")
elif selected_page == "Explore":
    st.title("Explore")
elif selected_page == "GetVisualization":
    st.title("Get Visualization")
elif selected_page == "UploadFiles":
    st.title("Upload Files")

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
