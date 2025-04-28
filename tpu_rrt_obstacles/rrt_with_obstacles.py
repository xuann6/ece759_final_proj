import tensorflow as tf

class RRTWithObstacles:
    def __init__(self, start, goal, x_range, y_range, obstacles, step_size=0.05, goal_sample_rate=0.1, max_iterations=1000):
        self.start = tf.constant(start, dtype=tf.float32)
        self.goal = tf.constant(goal, dtype=tf.float32)
        self.x_range = x_range
        self.y_range = y_range
        self.obstacles = tf.constant(obstacles, dtype=tf.float32)  # 障碍物 [N, 4]
        self.step_size = step_size
        self.goal_sample_rate = goal_sample_rate
        self.max_iterations = max_iterations

        # 节点列表
        self.nodes = tf.Variable([self.start], dtype=tf.float32)
        self.parents = tf.Variable([-1], dtype=tf.int32)

    def distance(self, a, b):
        return tf.norm(a - b, axis=-1)

    def sample(self):
        """随机采样一个点"""
        if tf.random.uniform(()) < self.goal_sample_rate:
            return self.goal
        else:
            x = tf.random.uniform((), self.x_range[0], self.x_range[1])
            y = tf.random.uniform((), self.y_range[0], self.y_range[1])
            return tf.stack([x, y])

    def nearest(self, sample):
        """找到最近的已有节点"""
        distances = self.distance(self.nodes, sample)
        return tf.argmin(distances)

    def steer(self, from_node, to_node):
        """朝着 to_node 延伸一小步"""
        direction = to_node - from_node
        dist = tf.norm(direction)
        if dist <= self.step_size:
            return to_node
        else:
            direction = direction / dist
            return from_node + direction * self.step_size

    def collision_check(self, point):
        """检查单点是否落在任何障碍物内"""
        # obstacles: [N, 4] -> [x_min, y_min, x_max, y_max]
        x, y = point[0], point[1]
        inside_x = tf.logical_and(x >= self.obstacles[:, 0], x <= self.obstacles[:, 2])
        inside_y = tf.logical_and(y >= self.obstacles[:, 1], y <= self.obstacles[:, 3])
        inside = tf.logical_and(inside_x, inside_y)
        return tf.reduce_any(inside)

    def plan(self):
        """主RRT逻辑"""
        for _ in range(self.max_iterations):
            random_point = self.sample()
            nearest_idx = self.nearest(random_point)
            nearest_node = self.nodes[nearest_idx]
            new_node = self.steer(nearest_node, random_point)

            if self.collision_check(new_node):
                continue  # 如果新节点撞到障碍物，跳过

            # 添加新节点
            self.nodes.assign(tf.concat([self.nodes, [new_node]], axis=0))
            self.parents.assign(tf.concat([self.parents, [nearest_idx]], axis=0))

            if self.distance(new_node, self.goal) <= self.step_size:
                # 连接到目标
                if not self.collision_check(self.goal):
                    self.nodes.assign(tf.concat([self.nodes, [self.goal]], axis=0))
                    self.parents.assign(tf.concat([self.parents, [len(self.nodes) - 2]], axis=0))
                    print(f"Goal reached in {len(self.nodes)} nodes!")
                    return self.extract_path()

        print("Goal not reached within max iterations.")
        return None

    def extract_path(self):
        """回溯出完整路径"""
        path = []
        current_idx = len(self.parents) - 1  # 从 goal 开始
        while current_idx != -1:
            path.append(self.nodes[current_idx].numpy())
            current_idx = self.parents[current_idx].numpy()
        path.reverse()
        return path
