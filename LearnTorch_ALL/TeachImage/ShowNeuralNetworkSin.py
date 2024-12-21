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

 # 推理函数
def predict(x, W1, b1, W2, b2):
    y = F.linear(x, W1, b1)
    y = F.sigmoid(y)
    y = F.linear(y, W2, b2)
    return y

def linear_regression_visualization(lr, iters, fps, if_save, file_path):
    """
    可视化一个简单神经网络的训练过程，目标是拟合 sin(2πx) 函数。

    参数：
    - lr: 学习率
    - iters: 迭代次数
    - fps: 动画保存帧率
    - if_save: 是否保存动画
    - file_path: 动画保存路径
    """
    # 数据准备
    np.random.seed(0)
    x = np.random.rand(10, 1)
    y = np.sin(2 * np.pi * x) + (np.random.rand(10, 1) * 2 - 1)
    x, y = Variable(x), Variable(y)

    # 权重的初始化
    I, H, O = 1, 100, 1
    # I对应输入层的维度，H对应隐藏层维度，O对应输出层维度，H是超参数
    W1 = Variable(0.01 * np.random.randn(I, H))
    b1 = Variable(np.zeros(H))
    W2 = Variable(0.01 * np.random.randn(H, O))
    b2 = Variable(np.zeros(O))

    # 存储权重和损失值
    W1_vals, b1_vals, W2_vals, b2_vals, loss_vals = [], [], [], [], []

    # 生成平滑的 x 点用于绘制曲线
    x_curve = np.linspace(0, 1, 100).reshape(-1, 1)  # 创建等间距点
    x_curve = Variable(x_curve)

    # 训练过程
    for _ in range(iters):
        y_pred = predict(x, W1, b1, W2, b2)
        loss = mean_squared_error(y, y_pred)

        # 反向传播与更新
        W1.cleargrad()
        b1.cleargrad()
        W2.cleargrad()
        b2.cleargrad()
        loss.backward()

        W1.data -= lr * W1.grad.data
        b1.data -= lr * b1.grad.data
        W2.data -= lr * W2.grad.data
        b2.data -= lr * b2.grad.data

        W1_vals.append(W1.data.copy())
        b1_vals.append(b1.data.copy())
        W2_vals.append(W2.data.copy())
        b2_vals.append(b2.data.copy())
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
    ax.set_title("y = sin(2πx) Neural Network Training")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    def update(frame):

        # 使用平滑曲线的预测值
        y_curve_pred = predict(x_curve, W1_vals[frame], b1_vals[frame], W2_vals[frame], b2_vals[frame])
        line.set_data(x_curve.data.flatten(), y_curve_pred.data.flatten())

        # 更新注释
        annotation.set_text(f"Iteration: {frame + 1}\n" +
                            f"W1: {W1_vals[frame][0].mean():.2f}\n" +
                            f"b1: {b1_vals[frame][0]:.2f}\n" +
                            f"W2: {W2_vals[frame][0].mean():.2f}\n" +
                            f"b2: {b2_vals[frame][0]:.2f}\n" +
                            f"Loss: {loss_vals[frame]:.4f}")
        annotation.xy = (x_curve.data.mean(), y_curve_pred.data.mean())  # 注释位置

        return line, annotation

    ani = FuncAnimation(
        fig, update, frames=range(0, iters, 10), interval=0.1, blit=True
    )

    if if_save:
        filename = os.path.splitext(file_path)[0] + f"_iter_{iters}_lr_{lr}" + os.path.splitext(file_path)[-1]
        ani.save(filename, fps=fps, writer="pillow",dpi=85)

    plt.show()


if __name__ == "__main__":
    file_path = os.path.join(".", "Grad", "linear_regression_sin_small.gif")
    linear_regression_visualization(lr=0.8, iters=1000, fps=30, if_save=False, file_path=file_path)
