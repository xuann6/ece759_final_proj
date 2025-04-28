import matplotlib.pyplot as plt

def plot_path(path):
    xs, ys = zip(*path)
    plt.plot(xs, ys, marker='o')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title("RRT Path")
    plt.grid(True)
    plt.show()
