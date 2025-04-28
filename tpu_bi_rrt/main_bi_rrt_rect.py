import tensorflow as tf
from bi_rrt_rect import BiRRTWithRectObstacles
from visualize import plot_path

def main():
    start = [0.1, 0.1]
    goal = [0.9, 0.9]
    x_range = (0.0, 1.0)
    y_range = (0.0, 1.0)

    # 障碍物定义 [xmin, ymin, xmax, ymax]
    obstacles = [
        [0.3, 0.3, 0.6, 0.5],
        [0.6, 0.6, 0.8, 0.9]
    ]

    with tf.device('/TPU:0'):
        planner = BiRRTWithRectObstacles(start, goal, x_range, y_range, obstacles, step_size=0.05, connect_threshold=0.05, max_iterations=5000)
        path = planner.plan()

    plot_path(path, obstacles)

if __name__ == "__main__":
    main()
