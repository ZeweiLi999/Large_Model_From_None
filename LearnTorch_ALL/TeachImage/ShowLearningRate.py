if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import matplotlib.pyplot as plt
import LearnTorch.Functions as F
from matplotlib.animation import FuncAnimation
from LearnTorch import Variable

# 新目标函数，包含多个局部最小值
def f(x):
    y = F.sin(5.0 * x) + x ** 2.0 - 0.5 * x
    return y

# 梯度下降数据生成
def gradient_descent(x, lr=0.001, iters=1000):
    x_vals, y_vals = [x.data.copy()], [f(x.data).data]
    for _ in range(iters):
        y = f(x)
        x.cleargrad()
        y.backward()
        gx = x.grad
        x.data = x.data - lr * gx.data
        x_vals.append(x.data.copy())
        y_vals.append(f(x.data).data)
    return x_vals, y_vals

# 动态绘图函数，专注于梯度下降
def gradient_descent_animation(lr, iters, fps, if_save, file_path):
    x_range = np.linspace(-4, 4, 500)
    y_range = f(x_range)

    # 数据准备
    gd_x_vals, gd_y_vals = gradient_descent(x=Variable(np.array([3.0])), lr=lr, iters=iters)

    # 创建图像
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x_range, y_range.data, label="f(x) = sin(5x) + x^2 - 0.5x", color='lightblue')
    gd_point, = ax.plot([], [], "go", label="Gradient Descent Point")
    gd_line, = ax.plot([], [],color="lightblue", alpha=0.6)

    # 显示文本代码
    gd_annotation = ax.annotate("", xy=(0, 0), xytext=(-50, 0),
                                textcoords="offset points",
                                bbox=dict(boxstyle="round", fc="w"),
                                arrowprops=dict(arrowstyle="->"))

    ax.set_title("Gradient Descent Optimization")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax.axvline(0, color='black', linewidth=0.5, linestyle='--')
    ax.legend()
    ax.grid(True)

    # 更新函数
    def update(frame):
        if frame < len(gd_x_vals):
            gd_x, gd_y = gd_x_vals[frame], gd_y_vals[frame]
            gd_line.set_data(gd_x_vals[:frame + 1], gd_y_vals[:frame + 1])
            gd_point.set_data([gd_x], [gd_y])
            gd_annotation.set_text("iter:({}): \n"
                                   "x:({:.2f})\n "
                                   "y:({:.2f})\n".format(frame, gd_x.item(), gd_y.item()))
            gd_annotation.set_position((gd_x + 20, gd_y + 20))  # 偏移值避免遮挡点
            gd_annotation.xy = (gd_x, gd_y)  # 确保箭头位置同步
            gd_annotation.set_visible(True)

        return gd_point, gd_line, gd_annotation

    # 创建动画
    ani = FuncAnimation(
        fig, update, frames=len(gd_x_vals), interval=100
    )

    # 显示动画
    plt.tight_layout()
    if if_save:
        filename = os.path.splitext(file_path)[0] + f"_lr{lr}_iters{iters}_FPS{fps}" + os.path.splitext(file_path)[-1]
        ani.save(filename=filename, fps=fps, writer="pillow")
    plt.show()


def gradient_descent_show(lr, iters, fps=10, file_path="../imgs"):
    x_range = np.linspace(-4, 4, 500)
    y_range = f(x_range)

    # 数据准备
    gd_x_vals, gd_y_vals = gradient_descent(x=Variable(np.array([3.0])), lr=lr, iters=iters)

    # 创建图像
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x_range, y_range.data, label="f(x) = sin(5x) + x^2 - 0.5x", color='lightblue')
    gd_point, = ax.plot([], [], "go", label="Gradient Descent Point")
    gd_line, = ax.plot([], [], color="lightblue", alpha=0.6)

    # 显示文本代码
    gd_annotation = ax.annotate("", xy=(0, 0), xytext=(-50, 0),
                                textcoords="offset points",
                                bbox=dict(boxstyle="round", fc="w"),
                                arrowprops=dict(arrowstyle="->"))

    ax.set_title("Gradient Descent Optimization")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax.axvline(0, color='black', linewidth=0.5, linestyle='--')
    ax.legend()
    ax.grid(True)

    # 更新函数
    def update(frame):
        if frame < len(gd_x_vals):
            gd_x, gd_y = gd_x_vals[frame], gd_y_vals[frame]
            gd_line.set_data(gd_x_vals[:frame + 1], gd_y_vals[:frame + 1])
            gd_point.set_data([gd_x], [gd_y])
            gd_annotation.set_text("iter:({}): \n"
                                   "x:({:.2f})\n "
                                   "y:({:.2f})\n".format(frame, gd_x.item(), gd_y.item()))
            gd_annotation.set_position((gd_x + 0.2, gd_y + 0.2))  # 避免遮挡点
            gd_annotation.xy = (gd_x, gd_y)  # 箭头位置同步
            gd_annotation.set_visible(True)

        return gd_point, gd_line, gd_annotation

    # 创建动画
    ani = FuncAnimation(fig, update, frames=len(gd_x_vals), interval=100)

    # 保存动画到临时文件
    filename = os.path.join(file_path, "2_lr{}_iters{}_FPS{}.gif".format(lr,iters,fps))
    ani.save(filename=filename, fps=fps, writer="pillow")
    plt.close(fig)  # 关闭 Matplotlib 图形，释放内存
    return filename

if __name__ == "__main__":
    file_dir = os.path.join(".", "Grad")
    os.makedirs(file_dir, exist_ok=True)

    # # 欠拟合：学习率太小
    # # file_path_underfit = os.path.join(file_dir, "Gradient_Underfit.gif")
    # # gradient_descent_animation(lr=0.001, iters=200, fps=10, if_save=True, file_path=file_path_underfit)
    #
    # # 拟合快：学习率适中
    # file_path_wellfit = os.path.join(file_dir, "Gradient_Wellfit.gif")
    # gradient_descent_animation(lr=0.085, iters=200, fps=10, if_save=True, file_path=file_path_wellfit)
    #
    # # 拟合慢：学习率较大但仍合理
    # file_path_slowfit = os.path.join(file_dir, "Gradient_Slowfit.gif")
    # gradient_descent_animation(lr=0.2, iters=200, fps=10, if_save=True, file_path=file_path_slowfit)

    # 迭代次数
    #欠拟合：学习率太小
    file_path_underfit = os.path.join(file_dir, "Gradient_iterlow.gif")
    gradient_descent_animation(lr=0.085, iters=5, fps=10, if_save=True, file_path=file_path_underfit)

    # 拟合快：学习率适中
    file_path_wellfit = os.path.join(file_dir, "Gradient_itermain.gif")
    gradient_descent_animation(lr=0.085, iters=50, fps=10, if_save=True, file_path=file_path_wellfit)

    # 拟合慢：学习率较大但仍合理
    file_path_slowfit = os.path.join(file_dir, "Gradient_iterhigh.gif")
    gradient_descent_animation(lr=0.085, iters=100, fps=10, if_save=True, file_path=file_path_slowfit)