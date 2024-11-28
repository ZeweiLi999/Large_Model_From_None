if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from LearnTorch import Variable

def f(x):
    return x ** 4 - 2 * x ** 2

def gx2(x):  # f函数的二阶导数形式
    return 12 * x ** 2 - 4

# 梯度下降与牛顿法数据生成
def gradient_descent(x,lr=0.001, iters=1000):
    # 初始值 学习率 迭代次数
    x_vals, y_vals = [x.data], [f(x.data)]
    for _ in range(iters):
        y = f(x)
        x.cleargrad()
        y.backward()
        x.data = x.data -  lr * x.grad
        x_vals.append(x.data)
        y_vals.append(f(x.data))
    return x_vals, y_vals

def newton_method(x, iters=9):
    # 初始值 迭代次数
    x_vals, y_vals = [x.data], [f(x.data)]
    for _ in range(iters):
        y = f(x)
        x.cleargrad()
        y.backward()
        x.data = x.data -  x.grad / gx2(x.data)
        x_vals.append(x.data)
        y_vals.append(f(x.data))
    return x_vals, y_vals

# 动态绘图
def grad_newton_animate_optimization(lr, iter_grad, iter_newton, fps, if_save, mp4_file_path):
    x_range = np.linspace(-2.5, 2.5, 500)
    y_range = f(x_range)

    # 数据准备
    gd_x_vals, gd_y_vals = gradient_descent(x = Variable(np.array(2.0)),lr = lr,iters=iter_grad)  # 梯度下降优化数据
    nm_x_vals, nm_y_vals = newton_method(x = Variable(np.array(2.0)),iters=iter_newton)  # 牛顿法优化数据

    # 创建图像
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # 梯度下降子图
    axes[0].plot(x_range, y_range, label="f(x) = x^4 - 2x^2", color='blue')
    gd_point, = axes[0].plot([], [], 'go', label="Gradient Descent Point")  # 动态点
    gd_line, = axes[0].plot([], [], 'g--', alpha=0.6)  # 动态轨迹
    # 显示文本代码
    gd_annotation = axes[0].annotate("", xy=(0, 0), xytext=(-50, 0),
                                     textcoords="offset points",
                                     bbox=dict(boxstyle="round", fc="w"),
                                     arrowprops=dict(arrowstyle="->"))
    axes[0].set_title("Gradient Descent Optimization")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("f(x)")
    axes[0].axhline(0, color='black', linewidth=0.5, linestyle='--')
    axes[0].axvline(0, color='black', linewidth=0.5, linestyle='--')
    axes[0].legend()
    axes[0].grid(True)

    # 牛顿法子图
    axes[1].plot(x_range, y_range, label="f(x) = x^4 - 2x^2", color='blue')
    nm_point, = axes[1].plot([], [], 'ro', label="Newton's Method Point")  # 动态点
    nm_line, = axes[1].plot([], [], 'r--', alpha=0.6)  # 动态轨迹
    # 显示文本代码
    nm_annotation = axes[1].annotate("", xy=(0, 0), xytext=(-50, 0),
                                     textcoords="offset points",
                                     bbox=dict(boxstyle="round", fc="w"),
                                     arrowprops=dict(arrowstyle="->"))
    axes[1].set_title("Newton's Method Optimization")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("f(x)")
    axes[1].axhline(0, color='black', linewidth=0.5, linestyle='--')
    axes[1].axvline(0, color='black', linewidth=0.5, linestyle='--')
    axes[1].legend()
    axes[1].grid(True)

    # 更新函数
    # 更新函数
    def update(frame):
        # 更新梯度下降点和轨迹
        if frame < len(gd_x_vals):
            gd_x, gd_y = gd_x_vals[frame], gd_y_vals[frame]
            # 更新梯度下降点和轨迹
            gd_point.set_data([gd_x], [gd_y])  # 修复：将标量值包装为序列
            gd_line.set_data(gd_x_vals[:frame + 1], gd_y_vals[:frame + 1])  # 这部分已经是序列，保留不变
            gd_annotation.set_text(f"itertion{frame}:({gd_x:.2f}, {gd_y:.2f})")
            gd_annotation.set_visible(True)

        # 更新牛顿法点和轨迹
        if frame < len(nm_x_vals):
            nm_x, nm_y = nm_x_vals[frame], nm_y_vals[frame]
            nm_point.set_data([nm_x], [nm_y])  # 修复：将标量值包装为序列
            nm_line.set_data(nm_x_vals[:frame + 1], nm_y_vals[:frame + 1])  # 这部分已经是序列，保留不变
            nm_annotation.set_text(f"itertion{frame}:({nm_x:.2f}, {nm_y:.2f})")
            nm_annotation.set_visible(True)

        return gd_point, gd_line, gd_annotation, nm_point, nm_line, nm_annotation

    # 创建动画
    ani = FuncAnimation(
        fig, update, frames=max(len(gd_x_vals), len(nm_x_vals)),
        interval=500
    )

    # 显示动画
    plt.tight_layout()
    if if_save:
        filename = mp4_file_path.replace(".mp4", "") + "iter_{}_{}".format(iter_grad,iter_newton) + "_FPS{}.mp4".format(fps)
        ani.save(filename = filename, fps = fps, writer = "ffmpeg")
    plt.show()


if __name__ == "__main__":
    mp4_file_path = os.path.join(".", "Grad", "GradV.S.Newton.mp4")

    # 调用函数进行动画演示
    grad_newton_animate_optimization(lr= 0.001, iter_grad=200, iter_newton=10 ,fps=10, if_save=True, mp4_file_path=mp4_file_path)

