import streamlit as st
from streamlit_extras.switch_page_button import switch_page  # 使用 Streamlit Extra 插件

# 定义 AI 助手数据
ai_helpers = [
    {
        "name": "助手1",
        "type": "默认提供",
        "description": "没有描述...",
        "image": "./VirtualAssistant/imgs/test_img.png",
    },
    {
        "name": "助手2",
        "type": "默认提供",
        "description": "没有描述...",
        "image": "./VirtualAssistant/imgs/test_img.png",
    },
    {
        "name": "助手3",
        "type": "默认提供",
        "description": "没有描述...",
        "image": "./VirtualAssistant/imgs/test_img.png",
    },
    {
        "name": "可爱的Bot",
        "type": "用户自定义",
        "description": "没有描述...",
        "image": "./VirtualAssistant/imgs/test_img.png",
    },
]

# Streamlit 页面设置
st.set_page_config(page_title="Explore Page", page_icon="🤖", layout="wide")

st.title("Explore Your Bots! 🤖")
st.write("选择你喜欢的助手, 去和它聊天吧! 这里提供了默认助手和用户自定义的助手...")

# 搜索框
search_query = st.text_input("搜索你的助手: ", placeholder="打出你的助手名或描述...")

st.markdown("---")

# 筛选逻辑
filtered_helpers = [
    helper for helper in ai_helpers
    if search_query.lower() in helper["name"].lower() or search_query.lower() in helper["description"].lower()
]

# 展示逻辑
if search_query:
    st.subheader(f"Search Results for '{search_query}':")
    display_helpers = filtered_helpers
else:
    st.subheader("你的助手: ")
    display_helpers = ai_helpers

# 瀑布流展示区域
# 使用st.columns来控制瀑布流的位置
with st.container():
    # 计算一行显示多少个助手
    cols_per_row = 3  # 每行展示列数
    rows = [display_helpers[i:i + cols_per_row] for i in range(0, len(display_helpers), cols_per_row)]

    # 创建3列布局，瀑布流放置在中间列
    col1, col2, col3 = st.columns([1, 2, 1])  # 使中间列占屏幕一半宽度
    with col2:  # 在中间列里展示瀑布流
        for row in rows:
            cols = st.columns(cols_per_row)
            for idx, helper in enumerate(row):
                with cols[idx]:
                    # 图片部分
                    st.image(helper["image"], use_container_width=True, caption=helper["name"])  # 替换为你的实际图片路径
                    # 名称、类型、描述
                    st.markdown(f"**{helper['name']}** ({helper['type']})")
                    st.write(helper["description"])
                    # 添加一个按钮，点击后跳转到聊天页面
                    if st.button(f"开始和 {helper['name']} 聊天", key=helper['name']):
                        # 存储选中的助手信息到会话状态
                        st.session_state['selected_helper'] = helper
                        # 跳转到聊天页面
                        switch_page("beginchats")

# 底部logo部分
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image(r"./VirtualAssistant/imgs/logo_learntorch(1).png", use_container_width=True)
st.write("Powered by Streamlit and Hugging Face.")
st.write("Explore AI Virtual Assistants Page")
