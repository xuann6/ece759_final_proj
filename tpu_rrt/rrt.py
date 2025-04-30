import tensorflow as tf

class RRT:
    def __init__(self, start, goal, x_range, y_range, step_size=0.05, goal_sample_rate=0.1, max_iterations=1000):
        self.start = tf.constant(start, dtype=tf.float32)
        self.goal = tf.constant(goal, dtype=tf.float32)
        self.x_range = x_range
        self.y_range = y_range
        self.step_size = step_size
        self.goal_sample_rate = goal_sample_rate
        self.max_iterations = max_iterations

        # 节点列表 [N, 2]
        self.nodes = tf.Variable([self.start], dtype=tf.float32)
        # 每个节点的父节点索引
        self.parents = tf.Variable([-1], dtype=tf.int32)

    def distance(self, a, b):
        return tf.norm(a - b, axis=-1)

    def sample(self):
        """随机采样点（有一定概率直接采样目标点）"""
        if tf.random.uniform(()) < self.goal_sample_rate:
            return self.goal
        else:
            x = tf.random.uniform((), self.x_range[0], self.x_range[1])
            y = tf.random.uniform((), self.y_range[0], self.y_range[1])
            return tf.stack([x, y])

    def nearest(self, sample):
        """找到离 sample 最近的节点索引"""
        distances = self.distance(self.nodes, sample)
        return tf.argmin(distances)

    def steer(self, from_node, to_node):
        """从 from_node 朝 to_node 延伸一步"""
        direction = to_node - from_node
        dist = tf.norm(direction)
        if dist <= self.step_size:
            return to_node
        else:
            direction = direction / dist
            return from_node + direction * self.step_size

    def plan(self):
        """主RRT构建逻辑"""
        for _ in range(self.max_iterations):
            random_point = self.sample()
            nearest_idx = self.nearest(random_point)
            nearest_node = self.nodes[nearest_idx]
            new_node = self.steer(nearest_node, random_point)

            # 添加新节点
            self.nodes.assign(tf.concat([self.nodes, [new_node]], axis=0))
            self.parents.assign(tf.concat([self.parents, [nearest_idx]], axis=0))

            # 检查是否到达目标
            if self.distance(new_node, self.goal) <= self.step_size:
                # 最后连接目标
                self.nodes.assign(tf.concat([self.nodes, [self.goal]], axis=0))
                self.parents.assign(tf.concat([self.parents, [len(self.nodes) - 2]], axis=0))
                print(f"Goal reached in {len(self.nodes)} nodes!")
                return self.extract_path()
        
        print("Goal not reached within max iterations.")
        return None

    def extract_path(self):
        """回溯找到完整路径"""
        path = []
        current_idx = len(self.parents) - 1  # 从 goal 开始
        while current_idx != -1:
            path.append(self.nodes[current_idx].numpy())
            current_idx = self.parents[current_idx].numpy()
        path.reverse()
        return path
