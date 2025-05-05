import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.animation as animation
import os

def visualize_rrt_animation_fast():
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
    
    total_nodes = len(nodes_df)
    
    # Calculate optimal number of frames based on total nodes
    # For 1500 nodes, use fewer frames (around 40-60)
    if total_nodes > 1000:
        frames = 50  # Much fewer frames for large node counts
    else:
        frames = min(100, total_nodes)  # Fewer frames than original
        
    step_size = max(1, total_nodes // frames)
    print(f"Creating animation with {frames} frames (step size: {step_size})")
    
    # Pre-process all the frames data to avoid recalculation during animation
    print("Pre-processing frames data...")
    frame_data = []
    
    # Draw fixed elements once
    # Draw world boundaries
    ax.set_xlim(0, world_width)
    ax.set_ylim(0, world_height)
    ax.set_title('RRT Algorithm Progress')
    
    # Draw obstacles
    for _, obstacle in obstacles.iterrows():
        rect = Rectangle((obstacle['x1'], obstacle['y1']), 
                         obstacle['x2'], obstacle['y2'], 
                         fill=True, color='red', alpha=0.5)
        ax.add_patch(rect)
    
    # Draw start and goal
    ax.scatter(start['x1'], start['y1'], s=100, color='green', marker='*', label='Start')
    ax.scatter(goal['x1'], goal['y1'], s=100, color='red', marker='*', label='Goal')
    ax.legend(loc='upper left')
    
    # Store all edge data for each frame
    for frame in range(frames):
        num_nodes = min(total_nodes, (frame + 1) * step_size)
        current_nodes = nodes_df.iloc[:num_nodes]
        
        # Regular edges
        regular_edges = []
        # Path edges
        path_edges = []
        
        # Store node data
        regular_nodes = current_nodes[current_nodes['on_path'] == 0]
        path_nodes = current_nodes[current_nodes['on_path'] == 1]
        
        # Store edge data
        for idx, node in current_nodes.iterrows():
            parent_id = node['parent']
            if parent_id >= 0 and parent_id in current_nodes.index:
                parent = current_nodes.loc[parent_id]
                if node['on_path'] == 1 and parent['on_path'] == 1:
                    path_edges.append(([node['x'], parent['x']], [node['y'], parent['y']]))
                else:
                    regular_edges.append(([node['x'], parent['x']], [node['y'], parent['y']]))
        
        frame_data.append({
            'num_nodes': num_nodes,
            'regular_edges': regular_edges,
            'path_edges': path_edges,
            'regular_nodes_x': regular_nodes['x'].values,
            'regular_nodes_y': regular_nodes['y'].values,
            'path_nodes_x': path_nodes['x'].values,
            'path_nodes_y': path_nodes['y'].values
        })
    
    # Keep references to the artists that need to be updated
    paths_artists = []
    regular_scatter = None
    path_scatter = None
    node_text = None
    
    # Function to initialize plot
    def init():
        nonlocal regular_scatter, path_scatter, node_text
        regular_scatter = ax.scatter([], [], s=5, alpha=0.5, color='blue', zorder=5)
        path_scatter = ax.scatter([], [], s=20, color='green', zorder=10)
        node_text = ax.text(0.02, 0.96, "", transform=ax.transAxes, fontsize=10,
                          verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        return [regular_scatter, path_scatter, node_text]
    
    # Function to update for each frame
    def update(frame):
        nonlocal regular_scatter, path_scatter, node_text, paths_artists
        
        # Remove previous paths
        for artist in paths_artists:
            artist.remove()
        paths_artists = []
        
        data = frame_data[frame]
        
        # Update text
        node_text.set_text(f"Nodes: {data['num_nodes']}/{total_nodes}")
        
        # Update nodes
        regular_scatter.set_offsets(np.column_stack((data['regular_nodes_x'], data['regular_nodes_y'])))
        path_scatter.set_offsets(np.column_stack((data['path_nodes_x'], data['path_nodes_y'])))
        
        # Update edges
        for x, y in data['regular_edges']:
            line, = ax.plot(x, y, color='lightgray', alpha=0.3, linewidth=0.5)
            paths_artists.append(line)
            
        for x, y in data['path_edges']:
            line, = ax.plot(x, y, color='green', linewidth=2.5)
            paths_artists.append(line)
        
        return [regular_scatter, path_scatter, node_text] + paths_artists
    
    # Create animation
    print("Creating animation...")
    anim = animation.FuncAnimation(fig, update, frames=frames, 
                                  init_func=init, blit=True, repeat=False)
    
    # Save animation
    print("Saving animation as rrt_animation.mp4...")
    anim.save('rrt_animation.mp4', writer='ffmpeg', fps=20, dpi=100)
    
    print("Animation saved!")
    
    # Show final state as static image
    # Use the visualization code to create a final image
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
                            color='lightgray', alpha=0.2, linewidth=0.5)
    
    # Plot nodes
    regular_nodes = nodes_df[nodes_df['on_path'] == 0]
    path_nodes = nodes_df[nodes_df['on_path'] == 1]
    
    plt.scatter(regular_nodes['x'], regular_nodes['y'], 
                s=5, alpha=0.5, color='blue', label='RRT Tree', zorder=5)
    plt.scatter(path_nodes['x'], path_nodes['y'], 
                s=20, color='green', label='Path', zorder=10)
    
    # Draw start and goal
    plt.scatter(start['x1'], start['y1'], s=100, color='green', marker='*', label='Start')
    plt.scatter(goal['x1'], goal['y1'], s=100, color='red', marker='*', label='Goal')
    
    plt.xlim(0, world_width)
    plt.ylim(0, world_height)
    plt.title('RRT Final Tree')
    plt.legend()
    plt.savefig('rrt_tree.png')
    
    print("Final tree visualization saved as rrt_tree.png")

if __name__ == "__main__":
    visualize_rrt_animation_fast()