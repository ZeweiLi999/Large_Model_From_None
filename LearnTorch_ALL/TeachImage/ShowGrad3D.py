if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from LearnTorch import Variable
import LearnTorch.Functions as F

def rosenbrock(x, y):
    """Rosenbrock function."""
    return (1 - x)**2 +  (y - x**2)**2

def gradient_descent_rosenbrock(starting_point, learning_rate=0.001, n_iterations=100):
    """Perform gradient descent on the Rosenbrock function."""
    x = Variable(np.array(starting_point[0], dtype=float))
    y = Variable(np.array(starting_point[1], dtype=float))
    history = [(x.data.copy(), y.data.copy())]  # Store the history of points

    for _ in range(n_iterations):
        # Compute the loss
        loss = rosenbrock(x, y)

        # Clear gradients and backpropagate
        x.cleargrad()
        y.cleargrad()
        loss.backward()

        # Update parameters
        x.data -= learning_rate * x.grad.data
        y.data -= learning_rate * y.grad.data

        # Store current position
        history.append((x.data.copy(), y.data.copy()))

    return np.array(history)

def visualize_rosenbrock(lr, iters, file_path, starting_point):
    """
    Visualize gradient descent on the Rosenbrock function and save as a GIF.

    Parameters:
    - lr: Learning rate
    - iters: Number of iterations
    - file_path: Path to save the GIF
    - starting_point: Initial point for gradient descent
    """
    # Run gradient descent
    history = gradient_descent_rosenbrock(starting_point, learning_rate=lr, n_iterations=iters)

    # Create a meshgrid for the surface plot
    x1_range = np.linspace(-5, 5, 100)
    x2_range = np.linspace(-5, 5, 100)
    x1, x2 = np.meshgrid(x1_range, x2_range)

    # Calculate Rosenbrock function values over the grid
    Z = rosenbrock(x1, x2)

    # Visualization setup
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the Rosenbrock function surface
    ax.plot_surface(x1, x2, Z, alpha=0.5, cmap='viridis')
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('Rosenbrock Function Value')

    # Gradient descent path
    line, = ax.plot([], [], [], color='red', label='Gradient Descent Path')
    point, = ax.plot([], [], [], 'bo', label='Current Point')

    def update(frame):
        # Update the path and point for the current frame
        line.set_data(history[:frame + 1, 0], history[:frame + 1, 1])
        line.set_3d_properties(rosenbrock(history[:frame + 1, 0], history[:frame + 1, 1]))

        # Wrap the current point in a sequence (list)
        point.set_data([history[frame, 0]], [history[frame, 1]])
        point.set_3d_properties([rosenbrock(history[frame, 0], history[frame, 1])])
        ax.set_title(f'Iteration: {frame + 1}')
        return line, point

    ani = FuncAnimation(fig, update, frames=len(history), interval=100, blit=True)

    # Save the animation as a GIF
    ani.save(file_path, fps=10, writer='pillow', dpi=85)

    print(f"GIF saved at {file_path}")
    plt.show()


def visualize_rosenbrock_show(lr, iters, file_path, starting_point):
    """
    Visualize gradient descent on the Rosenbrock function and save as a GIF.

    Parameters:
    - lr: Learning rate
    - iters: Number of iterations
    - file_path: Path to save the GIF
    - starting_point: Initial point for gradient descent
    """
    # Run gradient descent
    history = gradient_descent_rosenbrock(starting_point, learning_rate=lr, n_iterations=iters)

    # Create a meshgrid for the surface plot
    x1_range = np.linspace(-5, 5, 100)
    x2_range = np.linspace(-5, 5, 100)
    x1, x2 = np.meshgrid(x1_range, x2_range)

    # Calculate Rosenbrock function values over the grid
    Z = rosenbrock(x1, x2)

    # Visualization setup
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the Rosenbrock function surface
    ax.plot_surface(x1, x2, Z, alpha=0.5, cmap='viridis')
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('Rosenbrock Function Value')

    # Gradient descent path
    line, = ax.plot([], [], [], color='red', label='Gradient Descent Path')
    point, = ax.plot([], [], [], 'bo', label='Current Point')

    def update(frame):
        # Update the path and point for the current frame
        line.set_data(history[:frame + 1, 0], history[:frame + 1, 1])
        line.set_3d_properties(rosenbrock(history[:frame + 1, 0], history[:frame + 1, 1]))

        # Wrap the current point in a sequence (list)
        point.set_data([history[frame, 0]], [history[frame, 1]])
        point.set_3d_properties([rosenbrock(history[frame, 0], history[frame, 1])])
        ax.set_title(f'Iteration: {frame + 1}')
        return line, point

    ani = FuncAnimation(fig, update, frames=len(history), interval=100, blit=True)

    # Save the animation as a GIF
    #ani.save(file_path, fps=10, writer='pillow', dpi=85)

    filename = os.path.join(file_path, "3D_iter_{}_lr_{}_x_{}_y_{}.gif".format(iters, lr, starting_point[0], starting_point[1]))
    ani.save(filename, fps=10, writer="pillow", dpi=85)

    return filename

if __name__ == "__main__":
    file_path = os.path.join("Grad", "gradient_descent_rosenbrock.gif")

    # Example usage
    visualize_rosenbrock(
        lr=0.001,
        iters=100,
        file_path=file_path,
        starting_point=[-3, -3]
    )
