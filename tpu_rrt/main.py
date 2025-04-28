import tensorflow as tf
from rrt import RRT

def main():
    # 初始化 RRT
    start = [0.1, 0.1]
    goal = [0.9, 0.9]
    x_range = (0.0, 1.0)
    y_range = (0.0, 1.0)

    rrt = RRT(start, goal, x_range, y_range, step_size=0.05, goal_sample_rate=0.1, max_iterations=5000)
    
    path = rrt.plan()

    if path:
        print("Path found:")
        for p in path:
            print(p)
    else:
        print("Failed to find a path.")

if __name__ == "__main__":
    main()
