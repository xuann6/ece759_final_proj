#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
import matplotlib.animation as animation
import time
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm

def load_data(max_nodes=None):
    # Define optimized dtypes
    dtype_dict = {
        'global_id': np.int32,
        'tree_type': np.int8,
        'x': np.float32,
        'y': np.float32,
        'parent_global_id': np.int32,
        'on_path': np.int8
    }
    
    # Load node data - don't sample for animation
    print("Loading node data...")
    nodes_df = pd.read_csv('birrt_nodes.csv', dtype=dtype_dict)
    
    # Load world data
    world_df = pd.read_csv('birrt_world.csv')
    
    # Extract performance data
    try:
        perf_df = pd.read_csv('birrt_performance.csv')
        performance = {row['metric']: row['value'] for _, row in perf_df.iterrows()}
    except:
        performance = {}
    
    if max_nodes and len(nodes_df) > max_nodes:
        # Keep all path nodes
        path_nodes = nodes_df[nodes_df['on_path'] == 1]
        
        # Sample non-path nodes
        non_path_nodes = nodes_df[nodes_df['on_path'] == 0]
        
        # Stratified sampling while preserving order
        start_tree = non_path_nodes[non_path_nodes['tree_type'] == 0]
        goal_tree = non_path_nodes[non_path_nodes['tree_type'] == 1]
        
        # Sample with equal spacing to preserve timeline
        start_step = max(1, len(start_tree) // (max_nodes // 2))
        goal_step = max(1, len(goal_tree) // (max_nodes // 2))
        
        sampled_start = start_tree.iloc[::start_step].copy()
        sampled_goal = goal_tree.iloc[::goal_step].copy()
        
        # Combine sampled nodes with path nodes
        nodes_df = pd.concat([path_nodes, sampled_start, sampled_goal])
        nodes_df = nodes_df.sort_values(by='global_id') # Ensure proper order
        print(f"Sampled dataset from {len(start_tree) + len(goal_tree)} to {len(nodes_df)} nodes")
    
    return nodes_df, world_df, performance

def create_tree_animation(nodes_df, world_df, performance, fps=30, max_frames=600):
    # Setup figure
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
    start_tree_nodes = nodes_df[nodes_df['tree_type'] == 0].sort_values(by='global_id')
    goal_tree_nodes = nodes_df[nodes_df['tree_type'] == 1].sort_values(by='global_id')
    path_nodes = nodes_df[nodes_df['on_path'] == 1].sort_values(by='global_id')
    
    # Create progress points based on node IDs
    max_start_id = start_tree_nodes['global_id'].max()
    max_goal_id = goal_tree_nodes['global_id'].max()
    
    # Create lookup dictionary for fast parent access
    node_dict = nodes_df.set_index('global_id')[['x', 'y', 'parent_global_id', 'tree_type']].to_dict('index')
    
    # Create colormap for time progression
    start_cmap = plt.cm.Blues
    goal_cmap = plt.cm.Greens
    
    # Create collections for dynamic updates
    start_tree_collection = LineCollection([], linewidth=0.5, cmap=start_cmap)
    goal_tree_collection = LineCollection([], linewidth=0.5, cmap=goal_cmap)
    path_collection = LineCollection([], colors='red', linewidth=2.5)
    
    ax.add_collection(start_tree_collection)
    ax.add_collection(goal_tree_collection)
    ax.add_collection(path_collection)
    
    # Determine when the path was found
    path_found_id = 0
    if len(path_nodes) > 0:
        # Assuming the path is added after it's found, the lowest global_id with on_path=1
        path_found_id = path_nodes['global_id'].min()
    
    # Determine number of frames
    total_nodes = len(start_tree_nodes) + len(goal_tree_nodes)
    if total_nodes > max_frames:
        # Scale down to max_frames while maintaining even distribution
        frame_step = total_nodes // max_frames
    else:
        frame_step = 1
    
    # Create frames that show evolution over time
    frames = []
    total_ids = max(max_start_id, max_goal_id)
    
    # Determine frame IDs with even distribution
    frame_ids = list(range(0, total_ids, frame_step))
    if total_ids not in frame_ids:
        frame_ids.append(total_ids)
    
    print(f"Creating {len(frame_ids)} animation frames...")
    
    # Add title and legend elements that won't change
    plt.title('Bi-directional RRT Growth Animation', fontsize=14)
    plt.xlabel('X', fontsize=12)
    plt.ylabel('Y', fontsize=12)
    plt.legend(loc='upper right')
    ax.set_aspect('equal')
    plt.grid(True, alpha=0.3)
    
    # Performance text (static)
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
    
    perf_text = plt.figtext(0.02, 0.02, performance_text, fontsize=10, 
                bbox=dict(facecolor='white', alpha=0.8))
    
    # Time counter text (will be updated)
    time_text = ax.text(0.5, 0.95, '', transform=ax.transAxes, ha='center', fontsize=12,
                      bbox=dict(facecolor='white', alpha=0.8))
    
    # Precompute all frames
    all_frames = []
    
    for i, current_id in enumerate(tqdm(frame_ids)):
        # Nodes up to current ID
        current_start_nodes = start_tree_nodes[start_tree_nodes['global_id'] <= current_id]
        current_goal_nodes = goal_tree_nodes[goal_tree_nodes['global_id'] <= current_id]
        
        # Create start tree segments
        start_segments = []
        start_colors = []
        for _, node in current_start_nodes.iterrows():
            if node['parent_global_id'] >= 0:  # Skip root nodes
                parent_id = node['parent_global_id']
                if parent_id in node_dict:
                    parent = node_dict[parent_id]
                    start_segments.append([(node['x'], node['y']), (parent['x'], parent['y'])])
                    # Color based on ID (normalized to 0-1)
                    color_val = node['global_id'] / max_start_id if max_start_id > 0 else 0
                    start_colors.append(color_val)
        
        # Create goal tree segments
        goal_segments = []
        goal_colors = []
        for _, node in current_goal_nodes.iterrows():
            if node['parent_global_id'] >= 0:  # Skip root nodes
                parent_id = node['parent_global_id']
                if parent_id in node_dict:
                    parent = node_dict[parent_id]
                    goal_segments.append([(node['x'], node['y']), (parent['x'], parent['y'])])
                    # Color based on ID (normalized to 0-1)
                    color_val = node['global_id'] / max_goal_id if max_goal_id > 0 else 0
                    goal_colors.append(color_val)
        
        # Create path segments if path found
        path_segments = []
        if current_id >= path_found_id and len(path_nodes) > 0:
            path_coords = [(node['x'], node['y']) for _, node in path_nodes.iterrows()]
            path_segments = [[path_coords[i], path_coords[i + 1]] for i in range(len(path_coords) - 1)]
        
        # Save the frame data
        all_frames.append({
            'start_segments': start_segments.copy() if start_segments else [],
            'start_colors': start_colors.copy() if start_colors else [],
            'goal_segments': goal_segments.copy() if goal_segments else [],
            'goal_colors': goal_colors.copy() if goal_colors else [],
            'path_segments': path_segments.copy() if path_segments else [],
            'progress': i / len(frame_ids)
        })
    
    # Animation function
    def update(frame):
        # Update start tree
        if frame['start_segments']:
            start_tree_collection.set_segments(frame['start_segments'])
            if frame['start_colors']:
                start_tree_collection.set_array(np.array(frame['start_colors']))
            start_tree_collection.set_color('blue')
        else:
            start_tree_collection.set_segments([])
        
        # Update goal tree
        if frame['goal_segments']:
            goal_tree_collection.set_segments(frame['goal_segments'])
            if frame['goal_colors']:
                goal_tree_collection.set_array(np.array(frame['goal_colors']))
            goal_tree_collection.set_color('green')
        else:
            goal_tree_collection.set_segments([])
        
        # Update path
        if frame['path_segments']:
            path_collection.set_segments(frame['path_segments'])
            path_collection.set_color('red')
        else:
            path_collection.set_segments([])
        
        # Update progress text
        progress_pct = int(frame['progress'] * 100)
        time_text.set_text(f"Progress: {progress_pct}%")
        
        return start_tree_collection, goal_tree_collection, path_collection, time_text
    
    # Create animation
    ani = animation.FuncAnimation(
        fig, 
        update, 
        frames=all_frames,
        interval=1000/fps,  # In milliseconds
        blit=True
    )
    
    # Save animation
    print("Saving animation...")
    ani.save('birrt_animation.mp4', writer='ffmpeg', fps=fps, dpi=200)
    print("Animation saved as 'birrt_animation.mp4'")
    
    # Also save a high-quality still of the final frame
    plt.savefig('birrt_final_frame.png', dpi=300)
    print("Final frame saved as 'birrt_final_frame.png'")
    
    return ani

def main():
    start_time = time.time()
    print("Loading data...")
    
    # Load data - use sampling for more manageable animation with even distribution
    nodes_df, world_df, performance = load_data(max_nodes=200000)
    
    print(f"Data loaded in {time.time() - start_time:.2f} seconds")
    print(f"Creating animation for {len(nodes_df)} nodes...")
    
    # Create animation with 30fps and maximum of 600 frames (20 sec video)
    ani = create_tree_animation(nodes_df, world_df, performance, fps=30, max_frames=600)
    
    print(f"Total time: {time.time() - start_time:.2f} seconds")
    
if __name__ == "__main__":
    main()