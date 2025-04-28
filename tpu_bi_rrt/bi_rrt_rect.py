import tensorflow as tf

class BiRRTWithRectObstacles:
    def __init__(self, start, goal, x_range, y_range, obstacles, step_size=0.05, connect_threshold=0.05, max_iterations=5000):
        self.start = tf.constant(start, dtype=tf.float32)
        self.goal = tf.constant(goal, dtype=tf.float32)
        self.x_range = x_range
        self.y_range = y_range
        self.obstacles = tf.constant(obstacles, dtype=tf.float32)  # [N, 4]: (xmin, ymin, xmax, ymax)
        self.step_size = step_size
        self.connect_threshold = connect_threshold
        self.max_iterations = max_iterations
        
        # Trees: lists of (nodes, parents)
        self.start_tree = [self.start.numpy()]
        self.start_parents = [-1]
        self.goal_tree = [self.goal.numpy()]
        self.goal_parents = [-1]

    def distance(self, a, b):
        return tf.norm(a - b, axis=-1)

    def sample(self):
        x = tf.random.uniform((), self.x_range[0], self.x_range[1])
        y = tf.random.uniform((), self.y_range[0], self.y_range[1])
        return tf.stack([x, y])

    def nearest(self, tree, point):
        tree_tensor = tf.convert_to_tensor(tree, dtype=tf.float32)
        distances = self.distance(tree_tensor, point)
        return tf.argmin(distances)

    def steer(self, from_node, to_node):
        direction = to_node - from_node
        dist = tf.norm(direction)
        if dist <= self.step_size:
            return to_node
        else:
            direction = direction / dist
            return from_node + direction * self.step_size

    def collision_check(self, point):
        x, y = point[0], point[1]
        inside_x = tf.logical_and(x >= self.obstacles[:, 0], x <= self.obstacles[:, 2])
        inside_y = tf.logical_and(y >= self.obstacles[:, 1], y <= self.obstacles[:, 3])
        inside = tf.logical_and(inside_x, inside_y)
        return tf.reduce_any(inside)

    def extend(self, tree, parents, random_point):
        nearest_idx = int(self.nearest(tree, random_point))
        nearest_node = tf.constant(tree[nearest_idx], dtype=tf.float32)
        new_node = self.steer(nearest_node, random_point)
        
        if self.collision_check(new_node):
            return False
        
        tree.append(new_node.numpy())
        parents.append(nearest_idx)
        return True

    def trees_connected(self, tree1, tree2):
        node1 = tf.constant(tree1[-1], dtype=tf.float32)
        tree2_tensor = tf.convert_to_tensor(tree2, dtype=tf.float32)
        distances = self.distance(tree2_tensor, node1)
        min_dist = tf.reduce_min(distances)
        min_idx = tf.argmin(distances)
        if min_dist <= self.connect_threshold:
            return True, int(min_idx)
        else:
            return False, -1

    def build_path(self, start_parents, goal_parents, meeting_idx):
        # Trace back start tree
        path = []
        idx = len(self.start_parents) - 1
        while idx != -1:
            path.append(self.start_tree[idx])
            idx = self.start_parents[idx]
        path.reverse()

        # Trace goal tree from meeting node
        idx = meeting_idx
        goal_path = []
        while idx != -1:
            goal_path.append(self.goal_tree[idx])
            idx = self.goal_parents[idx]
        
        path.extend(goal_path)
        return path

    def plan(self):
        use_start_tree = True
        
        for _ in range(self.max_iterations):
            random_point = self.sample()
            
            if use_start_tree:
                success = self.extend(self.start_tree, self.start_parents, random_point)
            else:
                success = self.extend(self.goal_tree, self.goal_parents, random_point)
            
            if success:
                connected, meet_idx = self.trees_connected(self.start_tree, self.goal_tree)
                if connected:
                    print(f"Connected after {len(self.start_tree) + len(self.goal_tree)} nodes.")
                    return self.build_path(self.start_parents, self.goal_parents, meet_idx)

            use_start_tree = not use_start_tree

        print("Failed to connect within max iterations.")
        return None
