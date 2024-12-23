import streamlit as st
import json
import os


def Get_path(dir="./LLM"):
    # 将相对路径转换为绝对路径
    dir = os.path.abspath(dir)
    print(dir)
    # 获取目录中的子目录
    return [file for file in os.listdir(dir) if os.path.isdir(os.path.join(dir, file))]

options = Get_path()

# 获取文件夹中所有的文件和子文件夹
all_files = os.listdir("./imgs/ChatImgs/")

# 只获取文件，不包括文件夹
files = [f for f in all_files if os.path.isfile(os.path.join("./imgs/ChatImgs/", f))]

@st.dialog("删除虚拟助手")
def delete(name):
    st.write(f"确定删除{name}助手嘛？")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        yes_button = st.button("确定", use_container_width=True,type="primary")
    with col4:
        no_button = st.button("取消", use_container_width=True)
    if yes_button:
        del data[name]
        with open('./LLM/History.json', 'w') as f:
            json.dump(data, f, indent=2)
        st.success("删除成功！")
        st.rerun()
    if no_button:
        st.rerun()

@st.dialog("修改虚拟助手信息")
def update(name,model,description,image):
    Assistantname = st.text_input("助手名称：",name)
    model_index = options.index(model)
    Assistantmodel = st.selectbox("模型：" , options, index = model_index)
    Assistantdescription = st.text_area("提示词：",description)
    tarfile = image
    tarfile = os.path.basename(tarfile)
    img_index = files.index(tarfile)
    Assistantimage = st.selectbox("聊天背景" + ':star2:', files, index = img_index)
    st.image("./imgs/ChatImgs/" + Assistantimage)
    col1, col2 , col3 , col4 = st.columns(4)
    with col1:
        yes_button = st.button("确定",use_container_width=True)
    with col4:
        no_button = st.button("取消", use_container_width=True,type="primary")
    if yes_button:
        Assistant_data = data.pop(name)
        data[Assistantname] = Assistant_data
        data[Assistantname]["model"] = Assistantmodel
        data[Assistantname]["description"] = Assistantdescription
        data[Assistantname]["image"] = "./imgs/ChatImgs/" + Assistantimage
        with open('./LLM/History.json', 'w') as f:
            json.dump(data, f, indent=2)
        st.success("修改成功！")
        st.rerun()
    if no_button:
        st.rerun()

with open('./LLM/History.json', 'r') as f:
    data = json.load(f)

# Streamlit 页面设置
st.set_page_config(page_title="Explore Page", page_icon="🤖", layout="wide")

st.title("Explore Your Bots! 🤖")
st.write("选择你喜欢的助手, 去和它聊天吧! 这里提供了默认助手和用户自定义的助手...")

# 搜索框
search_query = st.text_input("搜索你的助手: ", placeholder="打出你的助手名或描述...")

st.markdown("---")

# 筛选逻辑
filtered_helpers = [
    {"name": name, **helper} for name, helper in data.items()
    if search_query.lower() in name.lower() or search_query.lower() in helper["description"].lower()
]

# 展示逻辑
if search_query:
    st.subheader(f"搜索结果：'{search_query}'")
    display_helpers = filtered_helpers
else:
    st.subheader("所有助手：")
    display_helpers = [{"name": name, **helper} for name, helper in data.items()]

# 瀑布流展示区域
with st.container():
    # 计算每行显示多少个助手
    cols_per_row = 2  # 每行展示2个助手
    rows = [display_helpers[i:i + cols_per_row] for i in range(0, len(display_helpers), cols_per_row)]

    for row in rows:
        cols = st.columns(cols_per_row)
        for idx, helper in enumerate(row):
            with cols[idx]:
                # 图片部分
                st.image(helper["image"], use_container_width=True, caption=helper["name"])  # 替换为你的实际图片路径
                # 名称、类型、描述
                st.markdown("基础模型：" + helper['model'])
                st.write("提示词："+ helper["description"])
                # 添加一个按钮，点击后跳转到聊天页面
                cols1,cols2,cols3 = st.columns([1,1,1])
                with cols1:
                    button1 = st.button(f"开始和 {helper['name']} 聊天",key=f"start_{helper['name']}",use_container_width=True)
                with cols2:
                    button2 = st.button(f"修改 {helper['name']} 助手",key=f"update_{helper['name']}",use_container_width=True)
                with cols3:
                    button3 = st.button("删除助手",key=f"delete_{helper['name']}",use_container_width=True,type="primary")
                if button2:
                   update(helper['name'],helper['model'],helper["description"],helper["image"])
                if button3:
                    delete(helper['name'])



# 底部logo部分
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image(r"./imgs/logo_learntorch(1).png", use_container_width=True)
st.write("Powered by Streamlit and Hugging Face.")
st.write("Explore AI Virtual Assistants Page")
