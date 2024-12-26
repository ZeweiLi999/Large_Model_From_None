if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import streamlit as st
from LLM.Chat_reponse import Chat_reponse
import time
import json
import base64

def main_bg(main_bg):
    main_bg_ext = "png"
    st.markdown(
        f"""
         <style>
         .stApp {{
             background: linear-gradient(rgba(255, 255, 255, 0.7), rgba(255, 255, 255, 0.7)), url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()});
             background-size: cover; /* 背景图像铺满整个屏幕 */
             background-position: center center; /* 保证背景居中显示 */
             background-attachment: fixed; /* 背景固定，滚动时背景不移动 */
             height: 100vh; /* 设置高度为视口的高度 */
         }}
         </style>
         """,
        unsafe_allow_html=True
    )

def stream_data(response, wait):
    for word in response:
        yield word
        time.sleep(0.02)
    wait.update(label="生成完毕!", state="complete", expanded=False)

with open('./LLM/History.json', 'r') as f:
    data = json.load(f)

modelindex = 0

if "modelname" in st.session_state:
    modeldata = list(data)
    st.session_state.messages = []

    if st.session_state.modelname in modeldata:
        modelindex = modeldata.index(st.session_state.modelname)

options = data.keys()

select_model = st.selectbox("请选择你的助手" + ':star2:', options, index = modelindex)

# 使用session_state来存储上次选择的值
if "previous_option" not in st.session_state:
    st.session_state.previous_option = select_model  # 初始化

# 检查当前选项与上次选项是否不同
if select_model != st.session_state.previous_option:
    st.session_state.previous_option = select_model  # 更新session_state
    st.session_state.messages = []

col1,col2 = st.columns([5,1])
with col2:
    number = st.number_input(
        "输出限制", min_value= 48 , max_value = 1024 , value = "min", step = 1 ,placeholder=""
    )

main_bg(f"{data[select_model]['image']}")

st.divider()  # Draws a horizontal rule

with st.chat_message("ai"):
    st.write(f"我是助手：{select_model}\n\n")
    st.write(f"{data[select_model]['start']}")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if input := st.chat_input("Say Something"):
    # Display user message in chat message container
    st.chat_message("user").markdown(input)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": input})

    wait = st.status("生成中")
    response = Chat_reponse(input, prompt = f"{data[select_model]['description']}" ,
                            model_dir = "./LLM/" + f"{data[select_model]['model']}" , max_tokens = number)
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.write_stream(stream_data(response, wait))
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

