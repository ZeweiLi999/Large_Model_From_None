import streamlit as st

# 确保 st.session_state.assistants_dict 已初始化
if "assistants_dict" not in st.session_state:
    st.session_state.assistants_dict = {
    "助手A": "简介A",
    "助手B": "简介B"
}

st.header("开始和你的助手聊天吧！")
if st.session_state.assistants_dict:
    AssistantBotName = st.sidebar.selectbox("请选择你的助手", list(st.session_state.assistants_dict.keys()))
    if AssistantBotName:
        # 创建一个以助手名字命名的选项卡
        tab = st.tabs([AssistantBotName])[0]
        with tab:
            st.header(f"与 {AssistantBotName} 的聊天对话框")

            # 用于存储与该助手的对话历史
            conversation_history = []

            user_input = st.text_input(f"请输入与 {AssistantBotName} 的对话内容")
            if st.button(f"发送给 {AssistantBotName}"):
                if user_input:
                    # 这里可以添加实际与助手交互的逻辑，比如调用API等
                    StringMsg = f"{AssistantBotName}回应：{user_input}"
                    conversation_history.append(("用户", user_input))
                    conversation_history.append((AssistantBotName, StringMsg))

                    st.write(f"用户：{user_input}")
                    st.write(f"{AssistantBotName}：{StringMsg}")
                else:
                    st.warning(f"请输入与 {AssistantBotName} 的对话内容后再发送。")
else:
    st.sidebar.write("暂无助手可供选择，请先创建助手。")