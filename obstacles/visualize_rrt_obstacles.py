import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

def read_tree_data(file_path):
    if not os.path.exists(file_path):
        print(f"Tree file {file_path} not found.")
        return None
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error reading tree file: {e}")
        return None

def read_obstacles_data(file_path):
    if not os.path.exists(file_path):
        print(f"Obstacles file {file_path} not found.")
        return None
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error reading obstacles file: {e}")
        return None

def visualize_tree(tree_df, obstacles_df=None, title="RRT Visualization", animate=True, save_animation=False):
    if tree_df is None:
        return
    
    # Create a figure and axes
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Set axis limits and labels
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(title)
    ax.grid(True)
    
    # Draw obstacles if available
    if obstacles_df is not None:
        for _, obstacle in obstacles_df.iterrows():
            rect = patches.Rectangle(
                (obstacle['x'], obstacle['y']), 
                obstacle['width'], 
                obstacle['height'],
                linewidth=1, 
                edgecolor='r', 
                facecolor='r', 
                alpha=0.3
            )
            ax.add_patch(rect)
    
    # Extract start and goal nodes
    start_node = tree_df.iloc[0]
    start_x, start_y = start_node['x'], start_node['y']
    
    # The goal node is the last node with parent_id not -1
    goal_node = tree_df.iloc[-1]
    goal_x, goal_y = goal_node['x'], goal_node['y']
    
    # Plot start and goal positions
    ax.plot(start_x, start_y, 'go', markersize=10, label='Start')
    ax.plot(goal_x, goal_y, 'ro', markersize=10, label='Goal')
    
    # Add legend
    ax.legend()
    
    if animate:
        # Animate tree growth
        max_time = tree_df['time'].max()
        time_steps = np.linspace(0, max_time, 100)
        
        lines = []
        
        for t in time_steps:
            visible_df = tree_df[tree_df['time'] <= t]
            
            # Clear previous lines
            for line in lines:
                line.remove()
            lines = []
            
            # Draw edges
            for _, node in visible_df.iterrows():
                if node['parent_id'] >= 0:
                    parent_idx = int(node['parent_id'])
                    parent = tree_df.iloc[parent_idx]
                    line, = ax.plot([parent['x'], node['x']], [parent['y'], node['y']], 'b-', alpha=0.5)
                    lines.append(line)
            
            # Draw nodes
            nodes_scatter = ax.scatter(visible_df['x'], visible_df['y'], c='b', s=20, alpha=0.5)
            lines.append(nodes_scatter)
            
            # Draw the path if the goal has been reached
            if goal_node['node_id'] in visible_df['node_id'].values:
                path_nodes = []
                current_id = goal_node['node_id']
                
                # Trace back from goal to start
                while current_id != -1:
                    current_node = tree_df[tree_df['node_id'] == current_id].iloc[0]
                    path_nodes.append((current_node['x'], current_node['y']))
                    current_id = current_node['parent_id']
                
                # Extract x and y coordinates
                path_x, path_y = zip(*path_nodes)
                path_line, = ax.plot(path_x, path_y, 'g-', linewidth=2)
                lines.append(path_line)
            
            plt.title(f"{title} - Time: {t:.2f}s")
            plt.pause(0.01)
            
            if save_animation:
                plt.savefig(f"rrt_frame_{len(lines):04d}.png")
    else:
        # Draw the final tree
        for _, node in tree_df.iterrows():
            if node['parent_id'] >= 0:
                parent_idx = int(node['parent_id'])
                parent = tree_df.iloc[parent_idx]
                ax.plot([parent['x'], node['x']], [parent['y'], node['y']], 'b-', alpha=0.5)
        
        # Draw nodes
        ax.scatter(tree_df['x'], tree_df['y'], c='b', s=20, alpha=0.5)
        
        # Extract and draw the final path
        path_nodes = []
        current_id = goal_node['node_id']
        
        # Check if goal was reached
        if current_id in tree_df['node_id'].values:
            # Trace back from goal to start
            while current_id != -1:
                current_node = tree_df[tree_df['node_id'] == current_id].iloc[0]
                path_nodes.append((current_node['x'], current_node['y']))
                current_id = current_node['parent_id']
            
            # Extract x and y coordinates
            path_x, path_y = zip(*path_nodes)
            ax.plot(path_x, path_y, 'g-', linewidth=2, label='Path')
            ax.legend()
    
    plt.show()

def main():
    # File paths for the RRT data
    tree_file = "rrt_obstacles_tree.csv"
    obstacles_file = "rrt_obstacles.csv"
    
    # Read the data
    tree_df = read_tree_data(tree_file)
    obstacles_df = read_obstacles_data(obstacles_file)
    
    if tree_df is not None:
        print(f"Tree data loaded with {len(tree_df)} nodes.")
        
        if obstacles_df is not None:
            print(f"Obstacles data loaded with {len(obstacles_df)} obstacles.")
        
        # Visualize the tree with obstacles
        visualize_tree(tree_df, obstacles_df, title="RRT with Obstacles", animate=True, save_animation=False)

if __name__ == "__main__":
    main() 