import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
from matplotlib.animation import FuncAnimation

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

def preprocess_tree_data(tree_df):
    """Preprocess tree data to remove kernel launch overhead and 
    recalculate timestamps to show only tree building time."""
    if tree_df is None or tree_df.empty:
        return None
    
    # Sort by time to ensure chronological order
    tree_df = tree_df.sort_values('time')
    
    # Identify and remove any kernel launch delays 
    time_diffs = tree_df['time'].diff().fillna(0)
    
    # Calculate median and standard deviation of time differences
    median_diff = time_diffs[time_diffs > 0].median()
    std_diff = time_diffs[time_diffs > 0].std()
    
    # Identify outliers (likely kernel launch overhead)
    threshold = median_diff + 3 * std_diff
    outlier_indices = time_diffs[time_diffs > threshold].index
    
    # Create a new time column with adjusted times
    adjusted_time = tree_df['time'].copy()
    
    # Compress the timeline by removing outlier time gaps
    for idx in outlier_indices:
        if idx in adjusted_time.index:  # Check if index exists
            excess_time = time_diffs[idx] - median_diff
            adjusted_time.loc[adjusted_time.index >= idx] -= excess_time
    
    # Update the tree dataframe
    tree_df['adjusted_time'] = adjusted_time
    
    # Recalculate time from zero
    tree_df['adjusted_time'] = tree_df['adjusted_time'] - tree_df['adjusted_time'].min()
    
    return tree_df

def visualize_cuda_rrt(tree_file="rrt_cuda_tree.csv", obstacles_file=None, 
                      output_file="cuda_rrt_visualization.mp4", animate=True, fps=30):
    # Read tree data
    tree_df = read_tree_data(tree_file)
    if tree_df is None:
        return
    
    # Ensure node_id and parent_id are integers
    tree_df['node_id'] = tree_df['node_id'].astype(int)
    tree_df['parent_id'] = tree_df['parent_id'].astype(int)
    
    # Preprocess tree data to remove kernel launch overhead
    tree_df = preprocess_tree_data(tree_df)
    if tree_df is None:
        return
    
    # Read obstacle data if available
    obstacles_df = None
    if obstacles_file and os.path.exists(obstacles_file):
        obstacles_df = pd.read_csv(obstacles_file)
    else:
        print(f"Obstacles file not found or not specified. Continuing without obstacles.")
    
    # Create figure for visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("CUDA-Accelerated RRT Visualization", fontsize=16)
    
    # Set up the tree visualization plot
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_title("RRT Tree (GPU Accelerated)")
    ax1.grid(True, alpha=0.3)
    
    # Set up the performance plot
    ax2.set_xlabel("Tree Building Time (seconds)")
    ax2.set_ylabel("Number of Nodes")
    ax2.set_title("Tree Growth Rate (Kernel Overhead Removed)")
    ax2.grid(True, alpha=0.3)
    
    # Extract start and goal nodes
    start_node = tree_df.iloc[0]
    goal_node = tree_df.iloc[-1]
    
    # Add obstacles if available
    if obstacles_df is not None:
        for _, obstacle in obstacles_df.iterrows():
            rect = patches.Rectangle(
                (obstacle['x'], obstacle['y']),
                obstacle['width'], obstacle['height'],
                linewidth=1, edgecolor='r', facecolor='r', alpha=0.3
            )
            ax1.add_patch(rect)
    
    # Plot start and goal positions
    start_circle = patches.Circle((start_node.x, start_node.y), 0.02, color='green', alpha=0.7)
    goal_circle = patches.Circle((goal_node.x, goal_node.y), 0.02, color='red', alpha=0.7)
    
    ax1.add_patch(start_circle)
    ax1.add_patch(goal_circle)
    ax1.text(start_node.x + 0.03, start_node.y, 'Start', fontsize=10)
    ax1.text(goal_node.x + 0.03, goal_node.y, 'Goal', fontsize=10)
    
    # Initialize empty plots for animation
    tree_lines, = ax1.plot([], [], 'b-', alpha=0.4)
    tree_nodes = ax1.scatter([], [], c='b', s=10, alpha=0.6)
    growth_line, = ax2.plot([], [], 'g-', linewidth=2)
    path_line, = ax1.plot([], [], 'r-', linewidth=2)
    
    # Time display
    time_text = ax1.text(0.02, 0.97, '', transform=ax1.transAxes, fontsize=10,
                      bbox=dict(facecolor='white', alpha=0.7))
    
    # Maximum adjusted time for normalization
    max_time = tree_df['adjusted_time'].max()
    
    # Path found display
    path_found_text = ax1.text(0.5, 0.97, '', transform=ax1.transAxes, fontsize=12,
                            ha='center', bbox=dict(facecolor='white', alpha=0.7))
    
    # Statistics display
    stats_text = ax2.text(0.05, 0.95, '', transform=ax2.transAxes, verticalalignment='top',
                       bbox=dict(facecolor='white', alpha=0.7))
    
    # Calculate tree statistics
    total_nodes = len(tree_df)
    build_time = max_time
    nodes_per_second = total_nodes / build_time if build_time > 0 else 0
    
    # Update statistics text
    stats_text.set_text(f"Total nodes: {total_nodes}\n"
                     f"Build time: {build_time:.3f}s\n"
                     f"Nodes/second: {nodes_per_second:.1f}")
    
    # Extract the final path
    path_nodes = []
    # Start with the goal node (last node in the dataframe)
    current_node = tree_df.iloc[-1]
    
    # Traverse the path backwards from goal to start
    while current_node.parent_id != -1:
        path_nodes.append((current_node.x, current_node.y))
        # Find the parent node by its node_id matching the current node's parent_id
        parent_row = tree_df[tree_df['node_id'] == current_node.parent_id]
        if parent_row.empty:
            print(f"Warning: Could not find parent node with node_id {current_node.parent_id}")
            break
        current_node = parent_row.iloc[0]
    
    # Add the start node
    path_nodes.append((current_node.x, current_node.y))
    
    # Reverse to get path from start to goal
    path_nodes.reverse()
    path_x, path_y = zip(*path_nodes) if path_nodes else ([], [])
    
    # Function to update the animation
    def update(frame):
        # Scale frame to time
        current_time = frame * max_time / 100 if animate else max_time
        
        # Get nodes up to current time
        visible_df = tree_df[tree_df['adjusted_time'] <= current_time]
        
        if visible_df.empty:
            return tree_lines, tree_nodes, growth_line, path_line, time_text, path_found_text
        
        # Update tree edges
        x_lines = []
        y_lines = []
        
        for _, node in visible_df.iterrows():
            if node.parent_id >= 0:
                # Find parent by node_id
                parent_rows = tree_df[tree_df['node_id'] == node.parent_id]
                if not parent_rows.empty:
                    parent = parent_rows.iloc[0]
                    x_lines.extend([parent.x, node.x, None])
                    y_lines.extend([parent.y, node.y, None])
        
        tree_lines.set_data(x_lines, y_lines)
        
        # Update tree nodes
        tree_nodes.set_offsets(visible_df[['x', 'y']].values)
        
        # Update growth line
        time_points = visible_df['adjusted_time'].values
        node_counts = np.arange(1, len(visible_df) + 1)
        growth_line.set_data(time_points, node_counts)
        
        # Adjust growth plot limits
        if len(time_points) > 0:
            ax2.set_xlim(0, max(time_points.max() * 1.1, 0.1))
            ax2.set_ylim(0, node_counts.max() * 1.1)
        
        # Update time text
        time_text.set_text(f'Build time: {current_time:.3f}s')
        
        # Show path if reached the end
        if frame >= 99 or not animate:
            if path_x and path_y:
                path_line.set_data(path_x, path_y)
                path_found_text.set_text(f'Path found: {len(path_nodes)} nodes')
        
        return tree_lines, tree_nodes, growth_line, path_line, time_text, path_found_text
    
    # Create animation
    if animate:
        ani = FuncAnimation(fig, update, frames=100, interval=50, blit=True)
        
        # Save animation to file
        if output_file:
            print(f"Saving animation to {output_file}...")
            try:
                ani.save(output_file, writer='ffmpeg', fps=fps, dpi=150)
                print(f"Animation saved successfully to {output_file}")
            except Exception as e:
                print(f"Failed to save animation: {e}")
                print("Try installing ffmpeg or using a different writer")
    else:
        # Just show final state
        update(100)
    
    # Also create a static final image
    fig_final, ax_final = plt.subplots(figsize=(10, 8))
    ax_final.set_xlim(0, 1)
    ax_final.set_ylim(0, 1)
    ax_final.set_xlabel("X")
    ax_final.set_ylabel("Y")
    ax_final.set_title("CUDA RRT: Final Tree and Path (Kernel Overhead Removed)")
    ax_final.grid(True, alpha=0.3)
    
    # Add obstacles
    if obstacles_df is not None:
        for _, obstacle in obstacles_df.iterrows():
            rect = patches.Rectangle(
                (obstacle['x'], obstacle['y']),
                obstacle['width'], obstacle['height'],
                linewidth=1, edgecolor='r', facecolor='r', alpha=0.3
            )
            ax_final.add_patch(rect)
    
    # Plot start and goal
    ax_final.add_patch(patches.Circle((start_node.x, start_node.y), 0.02, color='green', alpha=0.7))
    ax_final.add_patch(patches.Circle((goal_node.x, goal_node.y), 0.02, color='red', alpha=0.7))
    ax_final.text(start_node.x + 0.03, start_node.y, 'Start', fontsize=10)
    ax_final.text(goal_node.x + 0.03, goal_node.y, 'Goal', fontsize=10)
    
    # Plot full tree
    for _, node in tree_df.iterrows():
        if node.parent_id >= 0:
            # Find parent by node_id
            parent_rows = tree_df[tree_df['node_id'] == node.parent_id]
            if not parent_rows.empty:
                parent = parent_rows.iloc[0]
                ax_final.plot([parent.x, node.x], [parent.y, node.y], 'b-', alpha=0.3)
    
    # Plot all nodes
    ax_final.scatter(tree_df.x, tree_df.y, c='b', s=5, alpha=0.5)
    
    # Plot path
    if path_x and path_y:
        ax_final.plot(path_x, path_y, 'r-', linewidth=2, label=f'Path: {len(path_nodes)} nodes')
        ax_final.legend()
    
    # Add statistics
    stats = (
        f"Nodes: {total_nodes}\n"
        f"Build time: {build_time:.3f}s\n"
        f"Nodes/second: {nodes_per_second:.1f}"
    )
    ax_final.text(0.05, 0.95, stats, transform=ax_final.transAxes, 
                verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))
    
    # Save final image
    plt.tight_layout()
    fig_final.savefig("cuda_rrt_final.png", dpi=150)
    print("Final visualization saved to cuda_rrt_final.png")
    
    plt.show()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize CUDA RRT results with kernel overhead removed.')
    parser.add_argument('--tree', default='rrt_cuda_tree.csv', help='Path to the tree CSV file')
    parser.add_argument('--obstacles', default=None, help='Path to the obstacles CSV file (optional)')
    parser.add_argument('--output', default='cuda_rrt_animation.mp4', help='Output file path for animation')
    parser.add_argument('--no-animate', action='store_true', help='Skip animation and show only the final result')
    
    args = parser.parse_args()
    
    visualize_cuda_rrt(args.tree, args.obstacles, args.output, not args.no_animate)