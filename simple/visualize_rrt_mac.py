#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from matplotlib.patches import Circle


def save_frames_as_gif(ani, filename='rrt_animation', frames=100):
    """Save animation frames as a GIF using imageio"""
    import imageio.v2 as imageio
    import os
    
    frames_dir = "temp_frames"
    os.makedirs(frames_dir, exist_ok=True)
    
    print(f"Generating {frames} frames...")
    # Save each frame as an image
    for i in range(frames):
        ani._draw_frame(i)
        plt.savefig(f"{frames_dir}/frame_{i:03d}.png")
        if i % 10 == 0:
            print(f"Generated frame {i}/{frames}")
    
    # Collect all frames
    images = []
    for i in range(frames):
        images.append(imageio.imread(f"{frames_dir}/frame_{i:03d}.png"))
    
    # Save as GIF
    print("Creating GIF...")
    imageio.mimsave(f'{filename}.gif', images, duration=0.05)
    print(f"Animation saved to {filename}.gif")
    
    # Clean up frames
    for i in range(frames):
        os.remove(f"{frames_dir}/frame_{i:03d}.png")
    os.rmdir(frames_dir)
    
    return f'{filename}.gif'

def visualize_rrt(tree_file="rrt_star_tree.csv"):
    # Load the RRT tree data
    try:
        df = pd.read_csv(tree_file)
    except FileNotFoundError:
        print(f"Error: Cannot find the tree data file '{tree_file}'")
        print("Please run the RRT algorithm first to generate the tree data.")
        return
    
    # Ensure parent_id is an integer type
    df['parent_id'] = df['parent_id'].astype(int)
    
    print(f"Loaded {len(df)} nodes from {tree_file}")
    
    # Create figure and axes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Set up the tree visualization plot
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_title("RRT Tree Growth")
    ax1.grid(True)
    
    # Add start and goal markers
    start_node = df.iloc[0]
    goal_node = df.iloc[-1]
    
    start_circle = Circle((start_node.x, start_node.y), 0.02, color='green', alpha=0.7)
    goal_circle = Circle((goal_node.x, goal_node.y), 0.02, color='red', alpha=0.7)
    
    ax1.add_patch(start_circle)
    ax1.add_patch(goal_circle)
    ax1.text(start_node.x + 0.03, start_node.y, 'Start', fontsize=10)
    ax1.text(goal_node.x + 0.03, goal_node.y, 'Goal', fontsize=10)
    
    # Set up the performance plot
    ax2.set_xlabel("Time (milliseconds)")
    ax2.set_ylabel("Number of Nodes")
    ax2.set_title("Tree Growth Rate")
    ax2.grid(True)
    
    # Create empty line objects for the animation
    tree_lines, = ax1.plot([], [], 'b-', alpha=0.5)
    tree_points, = ax1.plot([], [], 'bo', markersize=2, alpha=0.5)
    final_path, = ax1.plot([], [], 'r-', linewidth=2)  # Add dedicated artist for final path
    perf_line, = ax2.plot([], [], 'g-')
    
    # Get the max time for animation
    max_time = df['time'].max()
    time_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes)
    
    def init():
        tree_lines.set_data([], [])
        tree_points.set_data([], [])
        final_path.set_data([], [])  # Initialize final path
        perf_line.set_data([], [])
        time_text.set_text('')
        return tree_lines, tree_points, final_path, perf_line, time_text
    
    def animate(frame):
        # Scale frame to time
        current_time = frame * (max_time / 100)
        
        # Get all nodes up to current time
        current_df = df[df['time'] <= current_time]
        
        # No nodes yet
        if len(current_df) == 0:
            return tree_lines, tree_points, final_path, perf_line, time_text
        
        # Plot nodes and connections
        x_points = current_df['x'].values
        y_points = current_df['y'].values
        
        # Plot all edges
        x_lines = []
        y_lines = []
        
        for _, node in current_df.iterrows():
            if node['parent_id'] >= 0:
                parent = df.iloc[int(node['parent_id'])]
                x_lines.extend([parent['x'], node['x'], None])
                y_lines.extend([parent['y'], node['y'], None])
        
        tree_lines.set_data(x_lines, y_lines)
        tree_points.set_data(x_points, y_points)
        
        # Update performance plot
        times = df['time'].values
        times = times[times <= current_time]
        
        # Node count over time - convert times to milliseconds for display
        time_ms = times * 1000
        node_counts = np.arange(1, len(times) + 1)
        
        perf_line.set_data(time_ms, node_counts)
        ax2.relim()
        ax2.autoscale_view()
        
        # Update time text - display in milliseconds
        time_ms = current_time * 1000
        time_text.set_text(f'Time: {time_ms:.1f}ms, Nodes: {len(current_df)}')
        
        # On the last frame, show the final path in red
        if frame == 99 and df.iloc[-1]['parent_id'] >= 0:  # Last frame and goal was reached
            path_nodes = []
            current_idx = len(df) - 1
            
            while current_idx >= 0:
                path_nodes.append(current_idx)
                current_idx = int(df.iloc[current_idx]['parent_id'])
            
            path_nodes.reverse()
            path_df = df.iloc[path_nodes]
            
            # Update the path artist with the final path data
            final_path.set_data(path_df['x'], path_df['y'])
        else:
            # Clear the path for all other frames
            final_path.set_data([], [])
        
        return tree_lines, tree_points, final_path, perf_line, time_text
    
    # Create animation
    ani = animation.FuncAnimation(
        fig, animate, frames=100, init_func=init, blit=True, interval=50, repeat=False
    )
    
    # Also create a static final view
    fig_final, (ax1_final, ax2_final) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Final tree visualization
    ax1_final.set_xlim(0, 1)
    ax1_final.set_ylim(0, 1)
    ax1_final.set_xlabel("X")
    ax1_final.set_ylabel("Y")
    ax1_final.set_title("Final RRT Tree")
    ax1_final.grid(True)
    
    # Add start and goal markers
    start_circle_final = Circle((start_node.x, start_node.y), 0.02, color='green', alpha=0.7)
    goal_circle_final = Circle((goal_node.x, goal_node.y), 0.02, color='red', alpha=0.7)
    
    ax1_final.add_patch(start_circle_final)
    ax1_final.add_patch(goal_circle_final)
    ax1_final.text(start_node.x + 0.03, start_node.y, 'Start', fontsize=10)
    ax1_final.text(goal_node.x + 0.03, goal_node.y, 'Goal', fontsize=10)
    
    # Plot all edges in the final tree
    for idx, node in df.iterrows():
        if node['parent_id'] >= 0:
            parent = df.iloc[int(node['parent_id'])]

            was_rewired = node['parent_id'] > idx
            color = 'black' if was_rewired else 'blue'

            ax1_final.plot([parent['x'], node['x']], [parent['y'], node['y']], 
                     color=color, alpha=0.5, linestyle='-')
    
    # Plot all nodes
    ax1_final.plot(df['x'], df['y'], 'bo', markersize=2, alpha=0.5)
    
    # Find the final path nodes
    if df.iloc[-1]['parent_id'] >= 0:  # If goal was reached
        path_nodes = []
        current_idx = len(df) - 1
        
        while current_idx >= 0:
            path_nodes.append(current_idx)
            current_idx = int(df.iloc[current_idx]['parent_id'])
        
        path_nodes.reverse()
        path_df = df.iloc[path_nodes]
        
        # Plot the path
        ax1_final.plot(path_df['x'], path_df['y'], 'r-', linewidth=2, label='Path')
    
    # Plot performance data
    ax2_final.set_xlabel("Time (milliseconds)")
    ax2_final.set_ylabel("Number of Nodes")
    ax2_final.set_title("Tree Growth Rate")
    ax2_final.grid(True)
    
    # Node count over time for final plot - convert to milliseconds
    time_ms = df['time'] * 1000
    ax2_final.plot(time_ms, np.arange(1, len(df) + 1), 'g-')
    
    # Add statistics to the plot
    max_time_ms = max_time * 1000
    stats_text = (
        f"Total nodes: {len(df)}\n"
        f"Total time: {max_time_ms:.1f}ms\n"
        f"Nodes/millisecond: {len(df)/max_time_ms:.3f}\n"
    )
    
    ax2_final.text(0.05, 0.95, stats_text, transform=ax2_final.transAxes, 
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    
    # Save animation to file
    output_file = save_frames_as_gif(ani)
    
    # Save final view to image
    fig_final.savefig('rrt_final.png', dpi=150)
    
    print("Visualization complete!")
    print("Animation saved to 'rrt_animation.mp4'")
    print("Final tree image saved to 'rrt_final.png'")
    
    # Close the static figure before showing the animation
    plt.close(fig_final)
    
    # Show only the animation figure
    plt.figure(fig.number)
    plt.show()

def visualize_rrt_bi(tree_file="rrt_tree.csv"):
    # Load the RRT tree data
    try:
        df = pd.read_csv(tree_file)
    except FileNotFoundError:
        print(f"Error: Cannot find the tree data file '{tree_file}'")
        print("Please run the RRT algorithm first to generate the tree data.")
        return
    
    # Ensure parent_id is an integer type
    df['parent_id'] = df['parent_id'].astype(int)
    
    print(f"Loaded {len(df)} nodes from {tree_file}")
    
    # Create figure and axes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Set up the tree visualization plot
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_title("RRT Tree Growth")
    ax1.grid(True)
    
    # Add start and goal markers
    start_indices = df.index[df['parent_id'] == -1].tolist()
    if len(start_indices) < 2:
        print("Error: The data does not contain two trees. Make sure there are two nodes with parent_id = -1")
        return
    
    start_idx = start_indices[0]
    goal_idx = start_indices[1]
    
    start_node = df.iloc[start_idx]
    goal_node = df.iloc[goal_idx]
    
    start_circle = Circle((start_node.x, start_node.y), 0.02, color='green', alpha=0.7)
    goal_circle = Circle((goal_node.x, goal_node.y), 0.02, color='red', alpha=0.7)
    
    ax1.add_patch(start_circle)
    ax1.add_patch(goal_circle)
    ax1.text(start_node.x + 0.03, start_node.y, 'Start', fontsize=10)
    ax1.text(goal_node.x + 0.03, goal_node.y, 'Goal', fontsize=10)
    
    # Set up the performance plot
    ax2.set_xlabel("Time (milliseconds)")
    ax2.set_ylabel("Number of Nodes")
    ax2.set_title("Tree Growth Rate")
    ax2.grid(True)
    
    # Create empty line objects for the animation
    tree_lines, = ax1.plot([], [], 'b-', alpha=0.5)
    tree_points, = ax1.plot([], [], 'bo', markersize=2, alpha=0.5)
    final_path, = ax1.plot([], [], 'r-', linewidth=2)  # Add dedicated artist for final path
    perf_line, = ax2.plot([], [], 'g-')
    
    # Get the max time for animation
    max_time = df['time'].max()
    time_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes)
    
    def init():
        tree_lines.set_data([], [])
        tree_points.set_data([], [])
        final_path.set_data([], [])  # Initialize final path
        perf_line.set_data([], [])
        time_text.set_text('')
        return tree_lines, tree_points, final_path, perf_line, time_text
    
    def animate(frame):
        # Scale frame to time
        current_time = frame * (max_time / 100)
        
        # Get all nodes up to current time
        current_df = df[df['time'] <= current_time]
        
        # No nodes yet
        if len(current_df) == 0:
            return tree_lines, tree_points, final_path, perf_line, time_text
        
        # Plot nodes and connections
        x_points = current_df['x'].values
        y_points = current_df['y'].values
        
        # Plot all edges
        x_lines = []
        y_lines = []
        
        for _, node in current_df.iterrows():
            if node['parent_id'] >= 0:
                parent = df.iloc[int(node['parent_id'])]
                x_lines.extend([parent['x'], node['x'], None])
                y_lines.extend([parent['y'], node['y'], None])
        
        tree_lines.set_data(x_lines, y_lines)
        tree_points.set_data(x_points, y_points)
        
        # Update performance plot
        times = df['time'].values
        times = times[times <= current_time]
        
        # Node count over time - convert times to milliseconds for display
        time_ms = times * 1000
        node_counts = np.arange(1, len(times) + 1)
        
        perf_line.set_data(time_ms, node_counts)
        ax2.relim()
        ax2.autoscale_view()
        
        # Update time text - display in milliseconds
        time_ms = current_time * 1000
        time_text.set_text(f'Time: {time_ms:.1f}ms, Nodes: {len(current_df)}')
        
        # On the last frame, show the final path in red
        if frame == 99 and df.iloc[-1]['parent_id'] >= 0:  # Last frame and goal was reached
            path_nodes = []
            current_idx = len(df) - 1
            
            while current_idx >= 0:
                path_nodes.append(current_idx)
                current_idx = int(df.iloc[current_idx]['parent_id'])
            
            path_nodes.reverse()

            current_idx = goal_idx - 1
            while current_idx >= 0:
                path_nodes.append(current_idx)
                current_idx = int(df.iloc[current_idx]['parent_id'])

            path_df = df.iloc[path_nodes]
            
            # Update the path artist with the final path data
            final_path.set_data(path_df['x'], path_df['y'])
        else:
            # Clear the path for all other frames
            final_path.set_data([], [])
        
        return tree_lines, tree_points, final_path, perf_line, time_text
    
    # Create animation
    ani = animation.FuncAnimation(
        fig, animate, frames=100, init_func=init, blit=True, interval=50, repeat=False
    )
    
    # Also create a static final view
    fig_final, (ax1_final, ax2_final) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Final tree visualization
    ax1_final.set_xlim(0, 1)
    ax1_final.set_ylim(0, 1)
    ax1_final.set_xlabel("X")
    ax1_final.set_ylabel("Y")
    ax1_final.set_title("Final RRT-Bi Tree")
    ax1_final.grid(True)
    
    # Add start and goal markers
    start_circle_final = Circle((start_node.x, start_node.y), 0.02, color='green', alpha=0.7)
    goal_circle_final = Circle((goal_node.x, goal_node.y), 0.02, color='red', alpha=0.7)
    
    ax1_final.add_patch(start_circle_final)
    ax1_final.add_patch(goal_circle_final)
    ax1_final.text(start_node.x + 0.03, start_node.y, 'Start', fontsize=10)
    ax1_final.text(goal_node.x + 0.03, goal_node.y, 'Goal', fontsize=10)
    
    # Plot all edges in the final tree
    for _, node in df.iterrows():
        if node['parent_id'] >= 0:
            parent = df.iloc[int(node['parent_id'])]
            ax1_final.plot([parent['x'], node['x']], [parent['y'], node['y']], 'b-', alpha=0.5)
    
    # Plot all nodes
    ax1_final.plot(df['x'], df['y'], 'bo', markersize=2, alpha=0.5)
    
    # Find the final path nodes
    if df.iloc[-1]['parent_id'] >= 0:  # If goal was reached
        path_nodes = []
        current_idx = len(df) - 1
        
        while current_idx >= 0:
            path_nodes.append(current_idx)
            current_idx = int(df.iloc[current_idx]['parent_id'])
        
        path_nodes.reverse()
        current_idx = goal_idx - 1
        while current_idx >= 0:
            path_nodes.append(current_idx)
            current_idx = int(df.iloc[current_idx]['parent_id'])

        path_df = df.iloc[path_nodes]
        
        # Plot the path
        ax1_final.plot(path_df['x'], path_df['y'], 'r-', linewidth=2, label='Path')
    
    # Plot performance data
    ax2_final.set_xlabel("Time (milliseconds)")
    ax2_final.set_ylabel("Number of Nodes")
    ax2_final.set_title("Tree Growth Rate")
    ax2_final.grid(True)
    
    # Node count over time for final plot - convert to milliseconds
    time_ms = df['time'] * 1000
    ax2_final.plot(time_ms, np.arange(1, len(df) + 1), 'g-')
    
    # Add statistics to the plot
    max_time_ms = max_time * 1000
    stats_text = (
        f"Total nodes: {len(df)}\n"
        f"Total time: {max_time_ms:.1f}ms\n"
        f"Nodes/millisecond: {len(df)/max_time_ms:.3f}\n"
    )
    
    ax2_final.text(0.05, 0.95, stats_text, transform=ax2_final.transAxes, 
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    
    # Save animation to file
    output_file = save_frames_as_gif(ani)
    
    # Save final view to image
    fig_final.savefig('rrt_bi_final.png', dpi=150)
    
    print("Visualization complete!")
    print("Animation saved to 'rrt_bi_animation.mp4'")
    print("Final tree image saved to 'rrt_bi_final.png'")
    
    # Close the static figure before showing the animation
    plt.close(fig_final)
    
    # Show only the animation figure
    plt.figure(fig.number)
    plt.show()
    
if __name__ == "__main__":
    visualize_rrt()
    #visualize_rrt_bi("rrt_bidirect_tree.csv")