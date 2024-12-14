import streamlit as st
<<<<<<< HEAD
from LLM.Chat_reponse import Chat_reponse
import time

def stream_data(response):
    for word in response:
        yield word
        time.sleep(0.02)

options = ["英语教师模型","甄嬛模型"]

select_model = st.selectbox("请选择你的模型" + ':star2:', options, index=0)

st.divider()  # Draws a horizontal rule

with st.chat_message("ai"):
    st.write(f"我是{select_model}\n\n你想问我些什么呀？")

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

    response = Chat_reponse(input)
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.write_stream(stream_data(response))
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
=======


prompt = st.chat_input("Say something")
with st.chat_message("user"):
    st.write(f"顾旭东是谁呀")
with st.chat_message("ai"):
    st.write("皇上身边的一个小太监👋")

with st.chat_message("user"):
    st.write(f"李泽威是谁呀")
with st.chat_message("ai"):
    st.write("他是皇上新近带回来的一个小主子，他的母亲是当年被皇上宠幸的宫女，皇上很宠他。")
>>>>>>> 0681856a209f6f3b029501abd7d9ea22b5097a5a
