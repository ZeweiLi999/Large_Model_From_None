import streamlit as st


prompt = st.chat_input("Say something")
with st.chat_message("user"):
    st.write(f"顾旭东是谁呀")
with st.chat_message("ai"):
    st.write("皇上身边的一个小太监👋")

with st.chat_message("user"):
    st.write(f"李泽威是谁呀")
with st.chat_message("ai"):
    st.write("他是皇上新近带回来的一个小主子，他的母亲是当年被皇上宠幸的宫女，皇上很宠他。")