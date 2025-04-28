import tensorflow as tf
from rrt_with_obstacles import RRTWithObstacles
import matplotlib.pyplot as plt

def main():
    # 起点和终点
    start = [0.1, 0.1]
    goal = [0.9, 0.9]
    x_range = (0.0, 1.0)
    y_range = (0.0, 1.0)

    # 定义简单矩形障碍物
    obstacles = [
        [0.3, 0.0, 0.4, 0.6],  # 左下角 (0.3, 0.0), 右上角 (0.4, 0.6)
        [0.6, 0.4, 0.7, 1.0]   # 左下角 (0.6, 0.4), 右上角 (0.7, 1.0)
    ]

    rrt = RRTWithObstacles(start, goal, x_range, y_range, obstacles, step_size=0.05, goal_sample_rate=0.1, max_iterations=5000)
    
    path = rrt.plan()

    if path:
        print("Path found!")
        plot_path(path, obstacles)
    else:
        print("Failed to find a path.")

def plot_path(path, obstacles):
    fig, ax = plt.subplots()
    xs, ys = zip(*path)
    ax.plot(xs, ys, marker='o', color='blue')

    # 绘制障碍物
    for obs in obstacles:
        rect = plt.Rectangle((obs[0], obs[1]), obs[2] - obs[0], obs[3] - obs[1], color='red', alpha=0.5)
        ax.add_patch(rect)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.grid(True)
    plt.title("RRT with Obstacles")
    plt.show()

if __name__ == "__main__":
    main()
