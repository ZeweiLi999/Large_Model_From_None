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

def linear_regression_visualization(lr, iters, fps, if_save, file_path):
    # 数据准备
    np.random.seed(0)
    x = np.random.rand(500, 1)
    y = 5 + 2 * x + (np.random.rand(500, 1)*2 -1)
    x, y = Variable(x), Variable(y)

    W = Variable(np.zeros((1, 1)))
    b = Variable(np.zeros(1))

    def predict(x):
        y = F.matmul(x, W) + b
        return y

    W_vals, b_vals, loss_vals = [], [], []

    # 训练过程
    for _ in range(iters):
        y_pred = predict(x)
        loss = mean_squared_error(y, y_pred)

        W.cleargrad()
        b.cleargrad()
        loss.backward()

        W.data -= lr * W.grad.data
        b.data -= lr * b.grad.data

        W_vals.append(W.data.copy())
        b_vals.append(b.data.copy())
        loss_vals.append(loss.data.copy())

    # 可视化
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x.data, y.data, label="Data", color="lightblue")
    line, = ax.plot([], [], color="lightcoral", label="Prediction")
    target_line, = ax.plot(x.data, 5 + 2 * x.data, color="lightgreen", label="Target Function")
    annotation = ax.annotate("", xy=(0, 0), xytext=(20, 20),
                             textcoords="offset points",
                             bbox=dict(boxstyle="round", fc="w"),
                             arrowprops=dict(arrowstyle="->"))
    ax.legend()
    ax.grid(True)
    ax.set_title("y = 2 * x + 5 Linear Regression")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    def update(frame):
        W_frame = W_vals[frame]
        b_frame = b_vals[frame]

        # 更新预测线
        y_pred = x.data @ W_frame + b_frame
        line.set_data(x.data.flatten(), y_pred.flatten())

        # 更新注释文本
        annotation.set_text(f"Iteration: {frame + 1}\n" +
                            f"W: {W_frame[0].mean():.2f}\n" +
                            f"b: {b_frame[0]:.2f}\n" +
                            f"Loss: {loss_vals[frame]:.4f}")
        mean_x, mean_y = x.data.mean(), y_pred.mean()
        annotation.xy = (mean_x, mean_y)  # 注释箭头指向的点

        return line, annotation

    ani = FuncAnimation(
        fig, update, frames=iters, interval=50, blit=True
    )

    if if_save:
        filename = os.path.splitext(file_path)[0] + f"_iter_{iters}_lr_{lr}" + os.path.splitext(file_path)[-1]
        ani.save(filename, fps=fps, writer="pillow")

    plt.show()

if __name__ == "__main__":
    file_path = os.path.join(".", "Grad", "linear_regression_small.gif")
    linear_regression_visualization(lr=0.1, iters=200, fps=10, if_save=True, file_path=file_path)
