import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.animation as animation
import os

def visualize_rrt_animation():
    # Load data
    try:
        nodes_df = pd.read_csv('rrt_nodes.csv')
        world_df = pd.read_csv('rrt_world.csv')
    except FileNotFoundError:
        print("Error: CSV files not found. Make sure to run the CUDA code first.")
        return
    
    # Set the node ID as the index for easier lookups
    nodes_df = nodes_df.set_index('id')
    
    print(f"Loaded {len(nodes_df)} nodes")
    
    # Extract world info
    world_info = world_df[world_df['type'] == 'world'].iloc[0]
    world_width, world_height = world_info['x2'], world_info['y2']
    
    # Extract obstacles, start and goal
    obstacles = world_df[world_df['type'] == 'obstacle']
    start = world_df[world_df['type'] == 'start'].iloc[0]
    goal = world_df[world_df['type'] == 'goal'].iloc[0]
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Function to setup the plot
    def init():
        ax.clear()
        ax.set_xlim(0, world_width)
        ax.set_ylim(0, world_height)
        ax.set_title('RRT Algorithm Over Time')
        
        # Draw obstacles
        for _, obstacle in obstacles.iterrows():
            rect = Rectangle((obstacle['x1'], obstacle['y1']), 
                             obstacle['x2'], obstacle['y2'], 
                             fill=True, color='red', alpha=0.5)
            ax.add_patch(rect)
        
        # Draw start and goal
        ax.scatter(start['x1'], start['y1'], s=100, color='green', marker='*', label='Start')
        ax.scatter(goal['x1'], goal['y1'], s=100, color='red', marker='*', label='Goal')
        ax.legend()
        
        return []
    
    # Animation frames
    total_nodes = len(nodes_df)
    frames = min(300, total_nodes)  # Limit to 300 frames max for performance
    step_size = max(1, total_nodes // frames)  # Calculate step size
    
    # Function to update plot for each frame
    def update(frame):
        # Calculate the number of nodes to show in this frame
        num_nodes = min(total_nodes, (frame + 1) * step_size)
        
        # Get the subset of nodes to display
        current_nodes = nodes_df.iloc[:num_nodes]
        
        # Text showing progress
        frame_text = ax.text(0.02, 0.96, f"Nodes: {num_nodes}/{total_nodes}", 
                            transform=ax.transAxes, fontsize=10,
                            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # Plot edges (connections)
        edges = []
        path_edges = []
        for idx, node in current_nodes.iterrows():
            parent_id = node['parent']
            if parent_id >= 0:  # Skip the root node
                # Check if parent exists in the dataframe
                if parent_id in current_nodes.index:
                    parent = current_nodes.loc[parent_id]
                    if node['on_path'] == 1 and parent['on_path'] == 1:
                        # Path edge
                        path_edge, = ax.plot([node['x'], parent['x']], [node['y'], parent['y']], 
                                 color='green', linewidth=2.5)
                        path_edges.append(path_edge)
                    else:
                        # Regular edge
                        edge, = ax.plot([node['x'], parent['x']], [node['y'], parent['y']], 
                                 color='lightgray', alpha=0.3, linewidth=0.5)
                        edges.append(edge)
        
        # Plot regular nodes and path nodes
        regular_nodes = current_nodes[current_nodes['on_path'] == 0]
        path_nodes = current_nodes[current_nodes['on_path'] == 1]
        
        # Plot nodes
        scatter_regular = ax.scatter(regular_nodes['x'], regular_nodes['y'], 
                          s=5, alpha=0.5, color='blue', zorder=5)
        
        scatter_path = ax.scatter(path_nodes['x'], path_nodes['y'], 
                        s=20, color='green', zorder=10)
        
        return [frame_text, scatter_regular, scatter_path] + edges + path_edges
    
    # Create animation
    print("Creating animation...")
    anim = animation.FuncAnimation(fig, update, frames=frames, 
                                   init_func=init, blit=True, repeat=False)
    
    # Save animation
    print("Saving animation as rrt_animation.mp4...")
    anim.save('rrt_animation.mp4', writer='ffmpeg', fps=30, dpi=100)
    
    # Also save a GIF for easier viewing
    print("Saving animation as rrt_animation.gif...")
    anim.save('rrt_animation.gif', writer='pillow', fps=15, dpi=80)
    
    print("Animation saved!")
    
    # Show the last frame as a static image
    plt.figure(figsize=(12, 10))
    
    # Draw obstacles
    for _, obstacle in obstacles.iterrows():
        rect = Rectangle((obstacle['x1'], obstacle['y1']), 
                         obstacle['x2'], obstacle['y2'], 
                         fill=True, color='red', alpha=0.5)
        plt.gca().add_patch(rect)
    
    # Draw all nodes and connections
    for idx, node in nodes_df.iterrows():
        parent_id = node['parent']
        if parent_id >= 0:  # Skip the root node
            if parent_id in nodes_df.index:
                parent = nodes_df.loc[parent_id]
                if node['on_path'] == 1 and parent['on_path'] == 1:
                    plt.plot([node['x'], parent['x']], [node['y'], parent['y']], 
                             color='green', linewidth=2.5)
                else:
                    plt.plot([node['x'], parent['x']], [node['y'], parent['y']], 
                             color='lightgray', alpha=0.3, linewidth=0.5)
    
    # Plot nodes
    regular_nodes = nodes_df[nodes_df['on_path'] == 0]
    path_nodes = nodes_df[nodes_df['on_path'] == 1]
    
    plt.scatter(regular_nodes['x'], regular_nodes['y'], 
                s=5, alpha=0.5, color='blue', label='RRT Tree')
    plt.scatter(path_nodes['x'], path_nodes['y'], 
                s=20, color='green', label='Path')
    
    # Draw start and goal
    plt.scatter(start['x1'], start['y1'], s=100, color='green', marker='*', label='Start')
    plt.scatter(goal['x1'], goal['y1'], s=100, color='red', marker='*', label='Goal')
    
    plt.xlim(0, world_width)
    plt.ylim(0, world_height)
    plt.title('RRT Tree - Final State')
    plt.legend()
    plt.savefig('rrt_tree_visualise.png')
    
    print("Final tree visualization saved as rrt_tree_visualise.png")

if __name__ == "__main__":
    visualize_rrt_animation()