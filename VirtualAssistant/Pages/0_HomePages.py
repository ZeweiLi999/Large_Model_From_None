import streamlit as st

# 标题
st.title("🌟Welcome to Your Adventure!🌟")

st.write(
    """
    ### 🌐 什么是 LearnTorch?
    欢迎来到 LearnTorch，这里是一个虚拟私人助手创建平台，帮助你从头学习深度学习模型的基本原理，并构建属于自己的 AI 模型！

    这个平台是基于《深度学习入门2 自制框架》书籍（作者：斋藤康毅）设计的，旨在以可视化和实践的方式，帮助你深入理解深度学习框架的核心技术。

    无论你是新手还是有一定基础的学习者，这个平台都会成为你探索深度学习世界的完美伴侣！✨
    """
)

# 预览
st.header("🔑 核心功能")

st.write(
    """
    - 🎨 **数据可视化**：探索深度学习的计算图和梯度下降原理。
    - 🤖 **虚拟助手**：创建并训练属于自己的 AI 助手。
    - 📈 **大模型实验**：学习如何在 Colab 上微调大模型。
    - 🧠 **理论与实践结合**：通过实践代码来巩固你的理论知识。
    """
)

# 新手教程(1)
st.header("🎓 新手指南")

st.subheader("步骤 1：构建你的第一个助手")

st.write(
    """
    1. 点击 **“开始创建”** 按钮。
    2. 输入你的助手名字（例如：小智）。
    3. 添加一条简短的描述或备注。
    4. 点击 **“添加助手”**，系统会保存你的助手！

    💡 小贴士：名字和备注可以帮助你个性化你的助手。
    """
)

if st.button("开始创建助手🚀"):
    st.session_state.is_creating = True

if "is_creating" not in st.session_state:
    st.session_state.is_creating = False

if st.session_state.is_creating:
    bot_name = st.text_input("输入你的助手名字")
    bot_description = st.text_input("备注（选填）")

    if st.button("确认创建✨"):
        if bot_name:
            st.session_state.assistants_dict[bot_name] = bot_description
            st.success(f"助手 {bot_name} 创建成功！")
            st.session_state.is_creating = False
        else:
            st.warning("助手名字不能为空哦！")

# 新手教程(2)
st.subheader("步骤 2：深入学习")

st.write(
    """
    想更进一步？探索以下功能：

    - **计算图可视化**：学习如何直观理解神经网络的反向传播机制。
    - **梯度下降原理**：通过图表展示不同学习率和迭代次数的影响。
    - **模型微调**：通过 Colab 学习如何对大模型进行微调训练。

    🔗 在侧边栏选择相应的功能模块，开启深度学习的冒险之旅吧！
    """
)

# 结语
st.header("✨ 开启你的 AI 旅程吧！")

st.write(
    """
    AI 的世界充满了可能性，而你的创造力将是探索的最大动力。希望 LearnTorch 能成为你在深度学习领域的最佳伙伴！🎉

    📘 **建议**：
    - 多多尝试和实践，代码写得越多，收获就越多。🖊
    - 如果在使用过程中有任何问题，欢迎随时反馈！💬
    
    **参考文献**  
    \- [1] [日] 斋藤康毅, *深度学习入门2: 自制框架*, 郑明智, 译. 北京: 人民邮电出版社, 2021, ISBN: 9787115607515.
    """
)
