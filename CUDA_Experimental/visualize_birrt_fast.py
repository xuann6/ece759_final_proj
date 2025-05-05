#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
import matplotlib.cm as cm
import time

# Load the RRT data with optimizations
def load_data(max_nodes=10000):
    # Load world data (small file, load first)
    world_df = pd.read_csv('birrt_world.csv')
    
    # Extract performance data (small file)
    try:
        perf_df = pd.read_csv('birrt_performance.csv')
        performance = {row['metric']: row['value'] for _, row in perf_df.iterrows()}
    except:
        performance = {}
    
    # For node data, use optimized loading with proper dtypes
    dtype_dict = {
        'global_id': np.int32,
        'tree_type': np.int8,
        'x': np.float32,
        'y': np.float32,
        'parent_global_id': np.int32,
        'on_path': np.int8
    }
    
    # Only read the columns we need with optimized dtypes
    nodes_df = pd.read_csv('birrt_nodes.csv', 
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
        
        # Calculate sampling fraction
        sampling_fraction = (max_nodes // 2) / max(len(start_tree), 1)
        
        # Sample from each tree
        sampled_start = start_tree.sample(frac=sampling_fraction) if len(start_tree) > 0 else start_tree
        sampled_goal = goal_tree.sample(frac=sampling_fraction) if len(goal_tree) > 0 else goal_tree
        
        # Combine sampled nodes with path nodes
        nodes_df = pd.concat([path_nodes, sampled_start, sampled_goal])
        print(f"Sampled dataset from {len(start_tree) + len(goal_tree)} to {len(nodes_df)} nodes")
    
    return nodes_df, world_df, performance

def visualize_trees_and_path(nodes_df, world_df, performance):
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
    
    # Optimize lookups with a dictionary
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
    lc_start = LineCollection(start_tree_segments, colors='blue', alpha=0.4, linewidth=0.5)
    lc_goal = LineCollection(goal_tree_segments, colors='green', alpha=0.4, linewidth=0.5)
    ax.add_collection(lc_start)
    ax.add_collection(lc_goal)
    
    # Create path segments (if path was found)
    if len(path_nodes) > 0:
        # Sort path nodes to ensure correct order
        path_nodes = path_nodes.sort_values(by='global_id')
        
        # Create line segments for the path
        path_segments = []
        path_coords = []
        
        # Extract ordered coordinates for the path
        for _, node in path_nodes.iterrows():
            path_coords.append((node['x'], node['y']))
        
        # Create segments between consecutive points
        for i in range(len(path_coords) - 1):
            path_segments.append([path_coords[i], path_coords[i + 1]])
        
        # Draw the path with a thicker line
        lc_path = LineCollection(path_segments, colors='red', linewidth=2.5)
        ax.add_collection(lc_path)
    
    # Add performance details if available
    performance_text = ""
    if performance:
        runtime = performance.get('runtime_ms', 'N/A')
        iterations = performance.get('iterations', 'N/A')
        start_nodes = performance.get('start_nodes', 'N/A')
        goal_nodes = performance.get('goal_nodes', 'N/A')
        path_length = performance.get('path_length', 'N/A')
        
        performance_text = (
            f"Runtime: {runtime} ms\n"
            f"Iterations: {iterations}\n"
            f"Start Tree Nodes: {start_nodes}\n"
            f"Goal Tree Nodes: {goal_nodes}\n"
            f"Path Length: {path_length}"
        )
    
    # Add title and labels
    plt.title('Optimized Bi-directional RRT', fontsize=14)
    plt.xlabel('X', fontsize=12)
    plt.ylabel('Y', fontsize=12)
    
    # Add performance text
    plt.figtext(0.02, 0.02, performance_text, fontsize=10, 
                bbox=dict(facecolor='white', alpha=0.8))
    
    # Add legend
    plt.legend(loc='upper right')
    
    # Enhance the appearance
    ax.set_aspect('equal')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the visualization
    plt.savefig('birrt_optimized_visualization.png', dpi=300, bbox_inches='tight')
    print("Visualization saved as 'birrt_optimized_visualization.png'")
    
    plt.show()

def main():
    start_time = time.time()
    print("Loading RRT data...")
    nodes_df, world_df, performance = load_data(max_nodes=10000)  # Use the same sample size as original
    print(f"Loaded {len(nodes_df)} nodes, {len(world_df)} world elements in {time.time() - start_time:.2f} seconds")
    print(f"Performance metrics: {performance}")
    
    print("Generating visualization...")
    vis_start = time.time()
    visualize_trees_and_path(nodes_df, world_df, performance)
    print(f"Visualization completed in {time.time() - vis_start:.2f} seconds")
    print(f"Total runtime: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()