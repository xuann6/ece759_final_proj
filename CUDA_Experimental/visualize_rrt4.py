#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
import time
import matplotlib.cm as cm

def load_data(max_nodes=20000):
    """Load and preprocess data with optimization for large datasets"""
    # Load world data
    world_df = pd.read_csv('birrt_world_optimized.csv')
    
    # Try to load performance data if available
    try:
        perf_df = pd.read_csv('birrt_optimized_performance.csv')
        performance = {row['metric']: row['value'] for _, row in perf_df.iterrows()}
    except:
        performance = {}
    
    # Optimized data loading with proper dtypes for faster processing
    dtype_dict = {
        'global_id': np.int32,
        'x': np.float32,
        'y': np.float32,
        'parent_global_id': np.int32,
        'tree_type': np.int8,
        'on_path': np.int8
    }
    
    # Load node data with optimized dtypes
    nodes_df = pd.read_csv('birrt_nodes_optimized.csv', 
                          dtype=dtype_dict,
                          usecols=list(dtype_dict.keys()))
    
    # Extract path nodes (typically a small subset)
    path_nodes = nodes_df[nodes_df['on_path'] == 1]
    
    # If dataset is too large, sample it
    if len(nodes_df) > max_nodes:
        # Keep all path nodes
        # Sample non-path nodes
        non_path_nodes = nodes_df[nodes_df['on_path'] == 0]
        
        # Sample separately from each tree to maintain balance
        start_tree = non_path_nodes[non_path_nodes['tree_type'] == 0]
        goal_tree = non_path_nodes[non_path_nodes['tree_type'] == 1]
        
        # Calculate sampling fraction for each tree
        start_sampling_fraction = (max_nodes // 2) / max(len(start_tree), 1)
        goal_sampling_fraction = (max_nodes // 2) / max(len(goal_tree), 1)
        
        # Sample from each tree
        sampled_start = start_tree.sample(frac=start_sampling_fraction) if len(start_tree) > 0 else start_tree
        sampled_goal = goal_tree.sample(frac=goal_sampling_fraction) if len(goal_tree) > 0 else goal_tree
        
        # Combine sampled nodes with path nodes
        nodes_df = pd.concat([path_nodes, sampled_start, sampled_goal])
        print(f"Sampled dataset from {len(start_tree) + len(goal_tree)} to {len(nodes_df)} nodes")
    
    return nodes_df, world_df, performance

def visualize_bidirectional_rrt(nodes_df, world_df, performance):
    """Create a visualization of the bidirectional RRT algorithm"""
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Set up world boundaries
    world_row = world_df[world_df['type'] == 'world'].iloc[0]
    world_x, world_y = float(world_row['x']), float(world_row['y'])
    world_width, world_height = float(world_row['width']), float(world_row['height_or_threshold'])
    
    # Set plot limits
    ax.set_xlim(world_x, world_x + world_width)
    ax.set_ylim(world_y, world_y + world_height)
    
    # Draw obstacles
    for _, row in world_df[world_df['type'] == 'obstacle'].iterrows():
        x, y = float(row['x']), float(row['y'])
        width, height = float(row['width']), float(row['height_or_threshold'])
        rect = patches.Rectangle((x, y), width, height, linewidth=2, edgecolor='k', facecolor='gray', alpha=0.6)
        ax.add_patch(rect)
    
    # Extract start and goal positions
    start_row = world_df[world_df['type'] == 'start'].iloc[0]
    goal_row = world_df[world_df['type'] == 'goal'].iloc[0]
    start_x, start_y = float(start_row['x']), float(start_row['y'])
    goal_x, goal_y = float(goal_row['x']), float(goal_row['y'])
    
    # Draw start and goal
    ax.plot(start_x, start_y, 'go', markersize=10, label='Start')
    ax.plot(goal_x, goal_y, 'ro', markersize=10, label='Goal')
    
    # Extract tree data
    start_tree_nodes = nodes_df[nodes_df['tree_type'] == 0]
    goal_tree_nodes = nodes_df[nodes_df['tree_type'] == 1]
    path_nodes = nodes_df[nodes_df['on_path'] == 1]
    
    print(f"Processing tree segments for {len(start_tree_nodes)} start nodes and {len(goal_tree_nodes)} goal nodes...")
    start_time = time.time()
    
    # Optimize lookups with a dictionary for faster access
    node_dict = nodes_df.set_index('global_id')[['x', 'y']].to_dict('index')
    
    # Create lines for the start tree
    start_tree_segments = []
    for _, node in start_tree_nodes.iterrows():
        if node['parent_global_id'] >= 0:  # Skip root nodes
            parent_id = node['parent_global_id']
            if parent_id in node_dict:
                parent = node_dict[parent_id]
                start_tree_segments.append([(node['x'], node['y']), (parent['x'], parent['y'])])
    
    # Create lines for the goal tree
    goal_tree_segments = []
    for _, node in goal_tree_nodes.iterrows():
        if node['parent_global_id'] >= 0:  # Skip root nodes
            parent_id = node['parent_global_id']
            if parent_id in node_dict:
                parent = node_dict[parent_id]
                goal_tree_segments.append([(node['x'], node['y']), (parent['x'], parent['y'])])
    
    print(f"Tree segments generated in {time.time() - start_time:.2f} seconds")
    print(f"Start tree: {len(start_tree_segments)} segments, Goal tree: {len(goal_tree_segments)} segments")
    
    # Draw trees with alpha blending for better visualization
    lc_start = LineCollection(start_tree_segments, colors='blue', alpha=0.4, linewidth=0.5, label='Start Tree')
    lc_goal = LineCollection(goal_tree_segments, colors='green', alpha=0.4, linewidth=0.5, label='Goal Tree')
    ax.add_collection(lc_start)
    ax.add_collection(lc_goal)
    
    # Create path segments (if path was found)
    if len(path_nodes) > 0:
        print(f"Found path with {len(path_nodes)} nodes")
        
        # Sort path nodes by their global_id to ensure correct ordering
        # For bi-directional RRT, we need special handling
        start_path_nodes = path_nodes[path_nodes['tree_type'] == 0].sort_values('global_id')
        goal_path_nodes = path_nodes[path_nodes['tree_type'] == 1].sort_values('global_id', ascending=False)
        
        # Extract coordinates
        start_path_coords = [(node['x'], node['y']) for _, node in start_path_nodes.iterrows()]
        goal_path_coords = [(node['x'], node['y']) for _, node in goal_path_nodes.iterrows()]
        
        # Combine path coordinates
        path_coords = start_path_coords + goal_path_coords
        
        # Create path segments
        path_segments = []
        for i in range(len(path_coords) - 1):
            path_segments.append([path_coords[i], path_coords[i + 1]])
        
        # Draw the path with a thicker line
        lc_path = LineCollection(path_segments, colors='red', linewidth=2.5, label='Path')
        ax.add_collection(lc_path)
        
        # Add path nodes with a different marker
        path_x = [coord[0] for coord in path_coords]
        path_y = [coord[1] for coord in path_coords]
        ax.scatter(path_x, path_y, color='red', s=30, alpha=0.7)
    else:
        print("No path found in the data")
    
    # Add performance details if available
    performance_text = ""
    if performance:
        gpu_time_ms = performance.get('total_gpu_time_ms', 'N/A')
        cpu_time_ms = performance.get('total_cpu_time_ms', 'N/A')
        iterations = performance.get('iterations', 'N/A')
        start_nodes = performance.get('start_tree_nodes', 'N/A')
        goal_nodes = performance.get('goal_tree_nodes', 'N/A')
        path_length = performance.get('path_length', 'N/A')
        blocks = performance.get('blocks', 'N/A')
        threads = performance.get('threads_per_block', 'N/A')
        
        performance_text = (
            f"GPU Runtime: {gpu_time_ms} ms\n"
            f"CPU Runtime: {cpu_time_ms} ms\n"
            f"Iterations: {iterations}\n"
            f"Start Tree: {start_nodes} nodes\n"
            f"Goal Tree: {goal_nodes} nodes\n"
            f"Path Length: {path_length} nodes\n"
            f"CUDA Blocks: {blocks}, Threads/Block: {threads}"
        )
    
    # Add title and labels
    plt.title('Bi-directional RRT (rrt4)', fontsize=14)
    plt.xlabel('X', fontsize=12)
    plt.ylabel('Y', fontsize=12)
    
    # Add performance text
    plt.figtext(0.02, 0.02, performance_text, fontsize=10, 
                bbox=dict(facecolor='white', alpha=0.8))
    
    # Add legend with proxy artists for collections
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='blue', lw=1, alpha=0.7, label='Start Tree'),
        Line2D([0], [0], color='green', lw=1, alpha=0.7, label='Goal Tree'),
        Line2D([0], [0], color='red', lw=2, label='Path'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='g', markersize=8, label='Start'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='r', markersize=8, label='Goal')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    # Enhance the appearance
    ax.set_aspect('equal')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the visualization
    plt.savefig('rrt4_visualization.png', dpi=300, bbox_inches='tight')
    print("Visualization saved as 'rrt4_visualization.png'")
    
    # Create a heatmap visualization
    create_heatmap(nodes_df, world_df, world_width, world_height)

def create_heatmap(nodes_df, world_df, world_width, world_height):
    """Create a heatmap showing node density"""
    plt.figure(figsize=(12, 10))
    
    # Separate data by tree
    start_tree_nodes = nodes_df[nodes_df['tree_type'] == 0]
    goal_tree_nodes = nodes_df[nodes_df['tree_type'] == 1]
    
    # Generate heatmap grid
    grid_size = 50
    heatmap = np.zeros((grid_size, grid_size))
    
    # Map coordinates to grid cells
    start_x_indices = np.floor(start_tree_nodes['x'] / world_width * (grid_size-1)).astype(int)
    start_y_indices = np.floor(start_tree_nodes['y'] / world_height * (grid_size-1)).astype(int)
    
    goal_x_indices = np.floor(goal_tree_nodes['x'] / world_width * (grid_size-1)).astype(int)
    goal_y_indices = np.floor(goal_tree_nodes['y'] / world_height * (grid_size-1)).astype(int)
    
    # Count nodes in each grid cell
    for x, y in zip(start_x_indices, start_y_indices):
        if 0 <= x < grid_size and 0 <= y < grid_size:
            heatmap[y, x] += 1
            
    for x, y in zip(goal_x_indices, goal_y_indices):
        if 0 <= x < grid_size and 0 <= y < grid_size:
            heatmap[y, x] += 1
    
    # Create a logarithmic scale for better visualization of density differences
    heatmap = np.log1p(heatmap)  # log(1+x) to handle zeros
    
    # Plot heatmap
    plt.imshow(heatmap, extent=[0, world_width, 0, world_height], 
               origin='lower', cmap='viridis', interpolation='bilinear')
    plt.colorbar(label='Log(Node Count + 1)')
    
    # Draw obstacles on heatmap
    for _, row in world_df[world_df['type'] == 'obstacle'].iterrows():
        x, y = float(row['x']), float(row['y'])
        width, height = float(row['width']), float(row['height_or_threshold'])
        rect = patches.Rectangle((x, y), width, height, linewidth=1, 
                                 edgecolor='red', facecolor='none', alpha=0.8)
        plt.gca().add_patch(rect)
    
    # Extract start and goal positions
    start_row = world_df[world_df['type'] == 'start'].iloc[0]
    goal_row = world_df[world_df['type'] == 'goal'].iloc[0]
    start_x, start_y = float(start_row['x']), float(start_row['y'])
    goal_x, goal_y = float(goal_row['x']), float(goal_row['y'])
    
    # Draw start and goal
    plt.plot(start_x, start_y, 'go', markersize=10, label='Start')
    plt.plot(goal_x, goal_y, 'ro', markersize=10, label='Goal')
    
    plt.title('Bi-directional RRT Node Density Heatmap', fontsize=14)
    plt.xlabel('X', fontsize=12)
    plt.ylabel('Y', fontsize=12)
    plt.legend(loc='upper right')
    plt.tight_layout()
    
    # Save the heatmap visualization
    plt.savefig('rrt4_heatmap.png', dpi=300, bbox_inches='tight')
    print("Heatmap saved as 'rrt4_heatmap.png'")

def main():
    """Main function to run the visualization"""
    start_time = time.time()
    print("Loading RRT data...")
    nodes_df, world_df, performance = load_data(max_nodes=50000)
    print(f"Loaded {len(nodes_df)} nodes, {len(world_df)} world elements in {time.time() - start_time:.2f} seconds")
    
    print("Generating visualization...")
    vis_start = time.time()
    visualize_bidirectional_rrt(nodes_df, world_df, performance)
    print(f"Visualization completed in {time.time() - vis_start:.2f} seconds")
    print(f"Total runtime: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()