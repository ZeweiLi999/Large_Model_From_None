# 项目介绍


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
    llama3 --> llama3_from_LearnTorch["llama3_from_Mytorch<br>用LearnTorch实现llama3"]

    LearnTorch --> Core["Core<br>LearnTorch核心代码"]
    LearnTorch --> Test["Test<br>LearnTorch测试代码"]
```
