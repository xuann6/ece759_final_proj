import matplotlib.pyplot as plt

def plot_path(path, obstacles):
    fig, ax = plt.subplots()

    if path:
        xs, ys = zip(*path)
        ax.plot(xs, ys, marker='o', color='blue', label='Path')
    else:
        print("No path to plot.")

    # 绘制障碍物
    for obs in obstacles:
        rect = plt.Rectangle((obs[0], obs[1]), obs[2] - obs[0], obs[3] - obs[1], color='red', alpha=0.5)
        ax.add_patch(rect)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True)
    ax.set_aspect('equal')
    plt.legend()
    plt.title('Bidirectional RRT with Rectangular Obstacles')
    plt.show()
