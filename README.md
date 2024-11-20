# 从零实现大模型/Large_Model_From_None

## 0.项目介绍/Intro ![Static Badge](https://img.shields.io/badge/Intro-项目介绍-4B9D9D) 
我们要实现的最终项目如下图所示：

![image](https://github.com/user-attachments/assets/e9d3637a-8c4f-4f69-951b-d47f2ba8ef1e)



<!-- 
```
Large_Model_From_None
    ├─llama3
        ├─llama3_from_Pytorch    #用Pytorch实现llama3
        ├─llama3_from_LearnTorch    #用LearnTorch实现llama3
    └─LearnTorch
        ├─Core                  #LearnTorch核心代码
        ├─Test                  #LearnTorch测试代码
```
-->

```mermaid
graph TD
    Large_Model_From_None --> llama3
    Large_Model_From_None --> LearnTorch

    llama3 --> llama3_from_Pytorch["llama3_from_Pytorch<br>用Pytorch实现llama3"]
    llama3 --> llama3_from_LearnTorch["llama3_from_LearnTorch<br>用LearnTorch实现llama3"]

    LearnTorch --> Core["Core<br>LearnTorch核心代码"]
    LearnTorch --> Test["Test<br>LearnTorch测试代码"]
```


## 1.自制深度学习框架/LearnTorch ![Static Badge](https://img.shields.io/badge/LearnTorch-自制深度学习框架-0584E3) 


![img_title-removebg](https://github.com/user-attachments/assets/7059a476-6c70-49e5-b327-ef767222f730)


## 2.虚拟私人助手/VirtualAssistant ![Static Badge](https://img.shields.io/badge/VirtualAssistant-虚拟私人助手-7884A4) 
