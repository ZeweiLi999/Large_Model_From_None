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
<br/><br/><br/><br/>
<div align=center>
<img src=".\imgs\img_title-removebg.png">
</div><br/><br/><br/>

### (1)自动微分

<div align="center">
        <img src="./LearnTorch_ALL/TeachImage/Grad/0_2pi_singrad.png" width="48%" height="48%">
        <img src="./LearnTorch_ALL/TeachImage/Grad/0_10_yx4.png" width="48%" height="48%">
</div>

### (2)可视化计算图
<img src="./LearnTorch_ALL/TeachImage/CGMap/sphere_All.png">

### (3)可视化梯度下降
#### A.梯度下降优化对比牛顿法优化
<img src="LearnTorch_ALL/TeachImage/Grad/GradV.S.Newton.mp4iter_200_10_FPS10.gif">

<br/>

#### B.线性回归梯度下降优化可视化
<img src="LearnTorch_ALL/TeachImage/Grad/linear_regression_iter_200_lr_0.1.gif">

<br/>

## 2.虚拟私人助手/VirtualAssistant ![Static Badge](https://img.shields.io/badge/VirtualAssistant-虚拟私人助手-7884A4) 

### (1)自定义助手模型
<br/>

<img src="./VirtualAssistant/images/创建助手.png">

