# 从零实现大模型/Large_Model_From_None

## 0.项目介绍/Intro ![Static Badge](https://img.shields.io/badge/Intro-项目介绍-4B9D9D) 
我们要实现的最终项目如下图所示：

<img src="imgs/规划思维导图.PNG">

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

### (2)可视化学习率和迭代次数
#### (2.1)优化慢
<img src="./LearnTorch_ALL/TeachImage/Grad/Gradient_Underfit_lr0.001_iters200_FPS10.gif">

#### (2.2)优化速度快
<img src="./LearnTorch_ALL/TeachImage/Grad/Gradient_Wellfit_lr0.085_iters200_FPS10.gif">

#### (2.3)反复震荡
<img src="./LearnTorch_ALL/TeachImage/Grad/Gradient_Overfit_lr0.2_iters200_FPS10.gif">

### (3)可视化梯度下降
#### (3.1) 梯度下降优化对比牛顿法优化
<img src="LearnTorch_ALL/TeachImage/Grad/GradV.S.Newton_small_iter_200_10_FPS10.gif">


#### (3.2)线性回归梯度下降优化可视化
<div align="center">
<img src="LearnTorch_ALL/TeachImage/Grad/linear_regression_small_iter_200_lr_0.1.gif">
</div>

<div align="center">
<img src="LearnTorch_ALL/TeachImage/Grad/linear_regression_sin_small_iter_10000_lr_0.2.gif">
</div>

#### (3.3)神经网络线性回归梯度下降优化可视化
##### (3.3.1)欠拟合

<div align="center">
<img src="LearnTorch_ALL/TeachImage/Grad/underfitting_iter_10000_lr_0.5_H1_5_H2_5.gif">
</div>

##### (3.3.2)良好拟合

<div align="center">
<img src="LearnTorch_ALL/TeachImage/Grad/wellfitting_iter_10000_lr_0.5_H1_10_H2_5.gif">
</div>

##### (3.3.3)过拟合

<div align="center">
<img src="LearnTorch_ALL/TeachImage/Grad/overfitting_iter_10000_lr_0.5_H1_15_H2_10.gif">
</div>



## 2.虚拟私人助手/VirtualAssistant ![Static Badge](https://img.shields.io/badge/VirtualAssistant-虚拟私人助手-7884A4) 

### (0)首页
<br/>

<div align="center">
<img src="./VirtualAssistant/imgs/0.首页.PNG">
</div>

### (1)计算图可视化
<br/>

<div align="center">
<img src="./VirtualAssistant/imgs/1.计算图可视化.PNG">
</div>

