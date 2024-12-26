import streamlit as st

# 标题
st.title("👨‍💻欢迎使用LearnTorch！")
container1 = st.container(border=True,key=1)
with container1:
    st.subheader("什么是 LearnTorch?")
    st.image("./imgs/logo_learntorch_small.png")
    st.write(
        """
        欢迎使用LearnTorch，这是一个从零开始大模型项目，帮助你从头学习深度学习模型的基本原理，并构建属于自己的 AI 模型！
        """
    )
    st.write(
        """
        LearnTorch主要分为自制深度学习系统和虚拟助手系统，旨在以可视化和实践的方式实现：
        
        - **学习深度学习本质**：1.帮助你深入理解深度学习框架的核心技术
        - **上手大模型**：2.接触最新大模型成果
        
        无论你是新手还是有一定基础的学习者，LearnTorch都尽力帮助你有所提升！✨
        """)

    st.subheader("核心功能")
    st.image("../VirtualAssistant/imgs/0_HomePages/mind.png")
    st.write(
        """
        - **数据可视化**：探索深度学习的本质。
        - **虚拟助手**：创建并训练属于自己的 AI 助手。
        - **大模型实验**：学习如何在国内外社区上微调大模型。
        - **理论与实践结合**：通过实践代码来巩固你的理论知识。
        """
    )
# 预览

container2 = st.container(border=True,key=2)
with container2:
    st.header("🎓 新手指南")
    st.write("🔗 在侧边栏选择相应的功能模块，开启深度学习的冒险之旅吧！")
    st.subheader("步骤 1：理论学习")
    st.write(
        """
    
        - **计算图可视化**：学习如何直观理解神经网络的反向传播机制。
        - **学习率和迭代次数可视化**：通过图表展示不同学习率和迭代次数的影响。
        - **梯度下降可视化**：通过图表展示不同学习率和迭代次数的影响。
    
        """
    )
    st.divider()

    st.subheader("步骤 2：上手大模型教程")

    st.write(
        """
        - **CUDA pytorch环境安装教程**：学习如何部署最常用的环境。
        - **低参数大模型教程**：通过图表展示不同学习率和迭代次数的影响。
        - **魔搭社区微调教程**：通过魔搭社区免费显卡资源对大模型进行微调训练，设计自己模型。
        - **Colab社区微调教程**：通过Colab免费显卡资源对大模型进行微调训练，设计自己模型。
        """
    )
    st.divider()

    st.subheader("步骤 3：实际上手大模型")

    st.write(
        """
        - **创建助手**：创建你想要的助手，包括提示词、开场白、背景图片。
        - **开始聊天**：和你所创建的助手聊天。
        - **探索助手**：管理你所创建的助手。
    
        """
    )
    st.divider()
    # 结语
    st.header("✨ 开启你的大模型旅程吧！")

    st.write(
        """
        大模型的世界充满了可能性，而你的创造力将是探索的最大动力。希望 LearnTorch 帮助减少上手大模型的阻力！🎉
        """
    )
