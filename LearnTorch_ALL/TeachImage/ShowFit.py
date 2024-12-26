if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import LearnTorch.Functions as F
from LearnTorch import Variable


def mean_squared_error(x0, x1):
    diff = x0 - x1
    return F.sum(diff ** 2) / len(diff)


def linear_regression_visualization(lr, iters, fps, if_save, file_path, hidden_units1, hidden_units2):
    """
    可视化一个多层神经网络的训练过程，目标是拟合 sin(2πx) 函数。

    参数：
    - lr: 学习率
    - iters: 迭代次数
    - fps: 动画保存帧率
    - if_save: 是否保存动画
    - file_path: 动画保存路径
    - hidden_units1: 第一隐藏层的神经元数
    - hidden_units2: 第二隐藏层的神经元数
    """
    # 数据准备
    np.random.seed(0)
    x = np.random.rand(10, 1)
    y = np.sin(2 * np.pi * x) + (np.random.rand(10, 1) * 2 - 1) * 0.5
    x, y = Variable(x), Variable(y)

    # 权重初始化
    I, H1, H2, O = 1, hidden_units1, hidden_units2, 1
    W1 = Variable(0.01 * np.random.randn(I, H1))
    b1 = Variable(np.zeros(H1))
    W2 = Variable(0.01 * np.random.randn(H1, H2))
    b2 = Variable(np.zeros(H2))
    W3 = Variable(0.01 * np.random.randn(H2, O))
    b3 = Variable(np.zeros(O))

    # 存储权重和损失值
    W1_vals, b1_vals, W2_vals, b2_vals, W3_vals, b3_vals, loss_vals = [], [], [], [], [], [], []

    # 平滑 x 曲线
    x_curve = np.linspace(0, 1, 100).reshape(-1, 1)
    x_curve = Variable(x_curve)

    # 定义预测函数（增加一层隐藏层）
    def predict(x, W1, b1, W2, b2, W3, b3):
        y = F.linear(x, W1, b1)
        y = F.sigmoid(y)
        y = F.linear(y, W2, b2)
        y = F.sigmoid(y)
        y = F.linear(y, W3, b3)
        return y

    # 训练过程
    for _ in range(iters):
        y_pred = predict(x, W1, b1, W2, b2, W3, b3)
        loss = mean_squared_error(y, y_pred)

        # 反向传播与更新
        W1.cleargrad()
        b1.cleargrad()
        W2.cleargrad()
        b2.cleargrad()
        W3.cleargrad()
        b3.cleargrad()
        loss.backward()

        W1.data -= lr * W1.grad.data
        b1.data -= lr * b1.grad.data
        W2.data -= lr * W2.grad.data
        b2.data -= lr * b2.grad.data
        W3.data -= lr * W3.grad.data
        b3.data -= lr * b3.grad.data

        # 保存权重和损失
        W1_vals.append(W1.data.copy())
        b1_vals.append(b1.data.copy())
        W2_vals.append(W2.data.copy())
        b2_vals.append(b2.data.copy())
        W3_vals.append(W3.data.copy())
        b3_vals.append(b3.data.copy())
        loss_vals.append(loss.data.copy())

    # 可视化
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(x.data, y.data, label="Data", color="lightblue")
    line, = ax.plot([], [], color="lightcoral", label="Prediction")
    ax.plot(x_curve.data, np.sin(2 * np.pi * x_curve.data), color="lightgreen", label="Target Function")
    annotation = ax.annotate("", xy=(0, 0), xytext=(20, 20),
                              textcoords="offset points",
                              bbox=dict(boxstyle="round", fc="w"),
                              arrowprops=dict(arrowstyle="->"))
    ax.legend()
    ax.grid(True)
    ax.set_title(f"y = sin(2πx) Neural Network Training (H1={hidden_units1}, H2={hidden_units2})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    def update(frame):
        # 使用平滑曲线的预测值
        y_curve_pred = predict(x_curve, W1_vals[frame], b1_vals[frame], W2_vals[frame], b2_vals[frame], W3_vals[frame], b3_vals[frame])
        line.set_data(x_curve.data.flatten(), y_curve_pred.data.flatten())

        # 更新注释
        annotation.set_text(f"Iteration: {frame + 1}\n" +
                            f"Loss: {loss_vals[frame]:.4f}")
        annotation.xy = (x_curve.data.mean(), y_curve_pred.data.mean())  # 注释位置

        return line, annotation

    ani = FuncAnimation(
        fig, update, frames=range(0, iters, 10), interval=0.1, blit=True
    )

    if if_save:
        filename = os.path.splitext(file_path)[0] + f"_iter_{iters}_lr_{lr}_H1_{hidden_units1}_H2_{hidden_units2}" + os.path.splitext(file_path)[-1]
        ani.save(filename, fps=fps, writer="pillow", dpi=85)

    plt.show()


def arg_show(lr, iters, fps, file_path, hidden_units1, hidden_units2):
    """
    可视化一个多层神经网络的训练过程，目标是拟合 sin(2πx) 函数。

    参数：
    - lr: 学习率
    - iters: 迭代次数
    - fps: 动画保存帧率
    - if_save: 是否保存动画
    - file_path: 动画保存路径
    - hidden_units1: 第一隐藏层的神经元数
    - hidden_units2: 第二隐藏层的神经元数
    """
    # 数据准备
    np.random.seed(0)
    x = np.random.rand(10, 1)
    y = np.sin(2 * np.pi * x) + (np.random.rand(10, 1) * 2 - 1) * 0.5
    x, y = Variable(x), Variable(y)

    # 权重初始化
    I, H1, H2, O = 1, hidden_units1, hidden_units2, 1
    W1 = Variable(0.01 * np.random.randn(I, H1))
    b1 = Variable(np.zeros(H1))
    W2 = Variable(0.01 * np.random.randn(H1, H2))
    b2 = Variable(np.zeros(H2))
    W3 = Variable(0.01 * np.random.randn(H2, O))
    b3 = Variable(np.zeros(O))

    # 存储权重和损失值
    W1_vals, b1_vals, W2_vals, b2_vals, W3_vals, b3_vals, loss_vals = [], [], [], [], [], [], []

    # 平滑 x 曲线
    x_curve = np.linspace(0, 1, 100).reshape(-1, 1)
    x_curve = Variable(x_curve)

    # 定义预测函数（增加一层隐藏层）
    def predict(x, W1, b1, W2, b2, W3, b3):
        y = F.linear(x, W1, b1)
        y = F.sigmoid(y)
        y = F.linear(y, W2, b2)
        y = F.sigmoid(y)
        y = F.linear(y, W3, b3)
        return y

    # 训练过程
    for _ in range(iters):
        y_pred = predict(x, W1, b1, W2, b2, W3, b3)
        loss = mean_squared_error(y, y_pred)

        # 反向传播与更新
        W1.cleargrad()
        b1.cleargrad()
        W2.cleargrad()
        b2.cleargrad()
        W3.cleargrad()
        b3.cleargrad()
        loss.backward()

        W1.data -= lr * W1.grad.data
        b1.data -= lr * b1.grad.data
        W2.data -= lr * W2.grad.data
        b2.data -= lr * b2.grad.data
        W3.data -= lr * W3.grad.data
        b3.data -= lr * b3.grad.data

        # 保存权重和损失
        W1_vals.append(W1.data.copy())
        b1_vals.append(b1.data.copy())
        W2_vals.append(W2.data.copy())
        b2_vals.append(b2.data.copy())
        W3_vals.append(W3.data.copy())
        b3_vals.append(b3.data.copy())
        loss_vals.append(loss.data.copy())

    # 可视化
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(x.data, y.data, label="Data", color="lightblue")
    line, = ax.plot([], [], color="lightcoral", label="Prediction")
    ax.plot(x_curve.data, np.sin(2 * np.pi * x_curve.data), color="lightgreen", label="Target Function")
    annotation = ax.annotate("", xy=(0, 0), xytext=(20, 20),
                              textcoords="offset points",
                              bbox=dict(boxstyle="round", fc="w"),
                              arrowprops=dict(arrowstyle="->"))
    ax.legend()
    ax.grid(True)
    ax.set_title(f"y = sin(2πx) Neural Network Training (H1={hidden_units1}, H2={hidden_units2})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    def update(frame):
        # 使用平滑曲线的预测值
        y_curve_pred = predict(x_curve, W1_vals[frame], b1_vals[frame], W2_vals[frame], b2_vals[frame], W3_vals[frame], b3_vals[frame])
        line.set_data(x_curve.data.flatten(), y_curve_pred.data.flatten())

        # 更新注释
        annotation.set_text(f"Iteration: {frame + 1}\n" +
                            f"Loss: {loss_vals[frame]:.4f}")
        annotation.xy = (x_curve.data.mean(), y_curve_pred.data.mean())  # 注释位置

        return line, annotation

    ani = FuncAnimation(
        fig, update, frames=range(0, iters, 10), interval=0.1, blit=True
    )

    filename = os.path.join(file_path , "_iter_{}_lr_{}_H1_{}_H2_{}.gif".format(iters, lr, hidden_units1 , hidden_units2))
    ani.save(filename, fps=fps, writer="pillow", dpi=85)

    return filename

if __name__ == "__main__":
    file_path = os.path.join(".", "Grad", "linear_regression_sin_deep.gif")

    # 欠拟合示例
    # file_path_underfitting = os.path.join(".", "Grad", "underfitting.gif")
    # linear_regression_visualization(
    #     lr=0.5,
    #     iters=10000,
    #     fps=30,
    #     if_save=True,
    #     file_path=file_path_underfitting,
    #     hidden_units1=5,
    #     hidden_units2=5
    # )

    # 过拟合示例
    # file_path_goodfit = os.path.join(".", "Grad", "good_fit.gif")
    # linear_regression_visualization(
    #     lr=0.5,
    #     iters=10000,
    #     fps=30,
    #     if_save=True,
    #     file_path=file_path_goodfit,
    #     hidden_units1=5,
    #     hidden_units2=5
    # )

    # 过拟合示例
    # file_path_overfitting = os.path.join(".", "Grad", "overfitting.gif")
    # linear_regression_visualization(
    #     lr=0.5,
    #     iters=10000,
    #     fps=30,
    #     if_save=True,
    #     file_path=file_path_overfitting,
    #     hidden_units1=15,
    #     hidden_units2=10
    # )
